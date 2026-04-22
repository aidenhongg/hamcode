"""MLP head. 2 hidden layers + dropout. CPU-friendly; also works on CUDA."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from .base import HeadRegistry


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


@HeadRegistry.register("mlp")
class MLPHead:
    def __init__(
        self,
        seed: int = 42,
        hidden: int = 128,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 8,            # tight cap — head sweep budget
        batch_size: int = 256,
        patience: int = 3,
        device: str | None = None,
    ) -> None:
        self.hp = dict(
            seed=seed, hidden=hidden, dropout=dropout, lr=lr,
            weight_decay=weight_decay, epochs=epochs, batch_size=batch_size,
            patience=patience,
        )
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._input_dim: int | None = None
        self.model: _MLP | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weight=None):
        seed = self.hp["seed"]
        torch.manual_seed(seed); np.random.seed(seed)

        self._input_dim = int(X_train.shape[1])
        self.model = _MLP(self._input_dim, self.hp["hidden"], self.hp["dropout"]).to(self.device)

        class_w = torch.ones(2, dtype=torch.float32, device=self.device)
        if class_weight is not None:
            class_w = torch.tensor([class_weight[0], class_weight[1]],
                                    dtype=torch.float32, device=self.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_w)
        opt = torch.optim.AdamW(self.model.parameters(),
                                 lr=self.hp["lr"],
                                 weight_decay=self.hp["weight_decay"])

        Xt = torch.from_numpy(X_train.astype(np.float32))
        yt = torch.from_numpy(y_train.astype(np.int64))
        if X_val is not None and y_val is not None:
            Xv = torch.from_numpy(X_val.astype(np.float32)).to(self.device)
            yv = torch.from_numpy(y_val.astype(np.int64)).to(self.device)
        else:
            Xv = yv = None

        bs = self.hp["batch_size"]
        n = Xt.shape[0]
        best_val = -1.0
        best_state = None
        patience_left = self.hp["patience"]
        history: list[dict] = []

        for epoch in range(self.hp["epochs"]):
            self.model.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                xb = Xt[idx].to(self.device)
                yb = yt[idx].to(self.device)
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                epoch_loss += float(loss.item()) * xb.shape[0]
            epoch_loss /= max(1, n)

            val_acc: float | None = None
            if Xv is not None:
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(Xv)
                    preds = logits.argmax(dim=-1)
                    val_acc = float((preds == yv).float().mean().item())
                history.append({"epoch": epoch, "train_loss": epoch_loss, "val_acc": val_acc})
                if val_acc > best_val + 1e-4:
                    best_val = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_left = self.hp["patience"]
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break
            else:
                history.append({"epoch": epoch, "train_loss": epoch_loss})

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return {"epochs_ran": len(history), "best_val_acc": best_val, "history": history}

    def _forward(self, X):
        assert self.model is not None
        self.model.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X):
        return np.argmax(self._forward(X), axis=-1)

    def predict_proba(self, X):
        return self._forward(X)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        assert self.model is not None
        torch.save(self.model.state_dict(), out_dir / "mlp.pt")
        joblib.dump({"hp": self.hp, "input_dim": self._input_dim},
                    out_dir / "mlp_meta.pkl")

    @classmethod
    def load(cls, out_dir: Path):
        meta = joblib.load(out_dir / "mlp_meta.pkl")
        inst = cls(**meta["hp"])
        inst._input_dim = meta["input_dim"]
        inst.model = _MLP(inst._input_dim, inst.hp["hidden"], inst.hp["dropout"])
        inst.model.load_state_dict(torch.load(out_dir / "mlp.pt", map_location="cpu"))
        inst.model.to(inst.device).eval()
        return inst

    def feature_importance(self) -> dict[str, float] | None:
        # For MLPs, permutation importance is more principled but slow. Skip here;
        # leave it to sweep to optionally compute on the winning head.
        return None
