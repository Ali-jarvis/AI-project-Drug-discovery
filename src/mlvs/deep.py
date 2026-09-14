"""PyTorch deep network exposed through the scikit-learn estimator API.

Kept in its own module so the package imports cleanly when PyTorch is absent;
nothing here is imported unless the ``dnn`` model is selected.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class TorchDNNClassifier(BaseEstimator, ClassifierMixin):
    """Feed-forward network with dropout, early stopping and class weighting.

    Implements ``fit``/``predict``/``predict_proba`` so it drops straight into
    ``GridSearchCV``, ``cross_val_predict`` and the imbalanced-learn pipeline
    used everywhere else in this package.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (512, 256),
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 100,
        patience: int = 10,
        weight_decay: float = 0.0,
        class_weight: str | None = "balanced",
        device: str | None = None,
        random_state: int = 17,
        verbose: bool = False,
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.class_weight = class_weight
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # -- internals ---------------------------------------------------------

    def _build(self, n_features: int):
        from torch import nn

        layers: list[nn.Module] = []
        prev = n_features
        for size in self.hidden_sizes:
            layers += [nn.Linear(prev, size), nn.BatchNorm1d(size), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = size
        layers.append(nn.Linear(prev, 2))
        return nn.Sequential(*layers)

    def _resolve_device(self):
        import torch

        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- sklearn API -------------------------------------------------------

    def fit(self, X, y):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("TorchDNNClassifier expects exactly two classes.")
        y_idx = np.searchsorted(self.classes_, y).astype(np.int64)

        device = self._resolve_device()
        self.device_ = str(device)
        self.model_ = self._build(X.shape[1]).to(device)

        weight = None
        if self.class_weight == "balanced":
            counts = np.bincount(y_idx, minlength=2).astype(np.float32)
            weight = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), device=device)

        criterion = nn.CrossEntropyLoss(weight=weight)
        optimiser = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Hold out a stratified slice for early stopping.
        rng = np.random.default_rng(self.random_state)
        val_idx = np.concatenate(
            [
                rng.permutation(np.flatnonzero(y_idx == c))[: max(1, int(0.1 * (y_idx == c).sum()))]
                for c in (0, 1)
            ]
        )
        train_mask = np.ones(len(y_idx), dtype=bool)
        train_mask[val_idx] = False

        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X[train_mask]), torch.from_numpy(y_idx[train_mask])
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=bool(train_mask.sum() > self.batch_size),
        )
        X_val = torch.from_numpy(X[~train_mask]).to(device)
        y_val = torch.from_numpy(y_idx[~train_mask]).to(device)

        best_loss, best_state, waited = float("inf"), None, 0
        for epoch in range(self.max_epochs):
            self.model_.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimiser.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                optimiser.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = float(criterion(self.model_(X_val), y_val))
            if self.verbose:
                print(f"epoch {epoch:3d}  val_loss {val_loss:.4f}")

            if val_loss < best_loss - 1e-4:
                best_loss, waited = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_loss_ = best_loss
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        import torch

        check_is_fitted(self, "model_")
        X = np.asarray(X, dtype=np.float32)
        device = torch.device(self.device_)
        self.model_.eval()

        out = []
        with torch.no_grad():
            for start in range(0, len(X), 4096):
                batch = torch.from_numpy(X[start : start + 4096]).to(device)
                out.append(torch.softmax(self.model_(batch), dim=1).cpu().numpy())
        return np.vstack(out)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags
