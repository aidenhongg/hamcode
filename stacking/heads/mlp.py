"""MLP head. Configurable depth/width/activation/optimizer for HP search.

Architecture sketch:
    input -> [Linear(in, h) -> Act -> (LayerNorm?) -> Dropout] x L -> Linear(h, 2)

Variable knobs:
    hidden_layers   : number of hidden blocks L  (1..5 typical)
    hidden_dim      : width of each hidden layer (64..512 typical)
    activation      : 'relu' | 'gelu' | 'silu'
    dropout         : 0.0..0.5
    layer_norm      : bool — add LayerNorm before dropout in each block
    optimizer       : 'adam' | 'adamw'
    lr / weight_decay: standard
    epochs / batch_size / patience: training control

Fits on CPU or CUDA. For the head sweep we keep it on CPU (fast enough at
~20K rows and <300 feature dims); if you want GPU pass device="cuda".
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
import torch.nn as nn

from .base import HeadRegistry


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def _widen_activation(name: str) -> type[nn.Module]:
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"unknown activation {name!r}; expected one of {list(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[name]


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 2,
        hidden_dim: int = 128,
        activation: str = "relu",
        dropout: float = 0.3,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        act_cls = _widen_activation(activation)
        layers: list[nn.Module] = []
        prev = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(act_cls())
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@HeadRegistry.register("mlp")
class MLPHead:
    def __init__(
        self,
        seed: int = 42,
        hidden_layers: int = 2,
        hidden_dim: int = 128,
        activation: str = "relu",
        dropout: float = 0.3,
        layer_norm: bool = False,
        optimizer: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 40,            # with patience=5, typical stop at ~10-15
        batch_size: int = 256,
        patience: int = 5,
        grad_clip: float = 1.0,
        device: str | None = None,
    ) -> None:
        self.hp = dict(
            seed=seed,
            hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
            layer_norm=layer_norm,
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            grad_clip=grad_clip,
        )
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._input_dim: int | None = None
        self.model: _MLP | None = None

    def _make_optimizer(self, params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
        name = self.hp["optimizer"].lower()
        lr = self.hp["lr"]
        wd = self.hp["weight_decay"]
        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        raise ValueError(f"unknown optimizer {name!r}; expected adam or adamw")

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weight=None):
        seed = self.hp["seed"]
        torch.manual_seed(seed); np.random.seed(seed)

        self._input_dim = int(X_train.shape[1])
        self.model = _MLP(
            self._input_dim,
            hidden_layers=self.hp["hidden_layers"],
            hidden_dim=self.hp["hidden_dim"],
            activation=self.hp["activation"],
            dropout=self.hp["dropout"],
            layer_norm=self.hp["layer_norm"],
        ).to(self.device)

        class_w = torch.ones(2, dtype=torch.float32, device=self.device)
        if class_weight is not None:
            class_w = torch.tensor([class_weight[0], class_weight[1]],
                                    dtype=torch.float32, device=self.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_w)
        opt = self._make_optimizer(self.model.parameters())

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
                if self.hp["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.hp["grad_clip"],
                    )
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
        inst.model = _MLP(
            inst._input_dim,
            hidden_layers=inst.hp["hidden_layers"],
            hidden_dim=inst.hp["hidden_dim"],
            activation=inst.hp["activation"],
            dropout=inst.hp["dropout"],
            layer_norm=inst.hp["layer_norm"],
        )
        inst.model.load_state_dict(torch.load(out_dir / "mlp.pt", map_location="cpu"))
        inst.model.to(inst.device).eval()
        return inst

    def feature_importance(self) -> dict[str, float] | None:
        # For MLPs, permutation importance is more principled but slow.
        # Left for a separate post-hoc explain step on the winning head.
        return None
