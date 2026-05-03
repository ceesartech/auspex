"""Neural network model with PyTorch."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from .base_model import BaseModel
from .model_config import ModelConfig

logger = logging.getLogger(__name__)


class MatchDataset(Dataset):
    """PyTorch Dataset for match data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MatchPredictorNN(nn.Module):
    """Neural network architecture for match prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        dropout_rate: float = 0.3,
        num_classes: int = 3,
        batch_norm: bool = True,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralNetworkMatchPredictor(BaseModel):
    """Neural network predictor wrapper."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
    ) -> tuple:
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in features:
                features.remove(target)

        self.feature_names = features

        X = df[features].fillna(df[features].median())
        X_scaled = self.scaler.fit_transform(X)
        y = self.label_encoder.fit_transform(df[target])

        return X_scaled, y

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        features: Optional[List[str]] = None,
        target: str = "match_outcome",
        **kwargs,
    ) -> Dict[str, Any]:
        logger.info("Starting Neural Network training...")

        X_train, y_train = self.prepare_data(train_df, target, features)

        train_dataset = MatchDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.hyperparameters.get("batch_size", 256),
            shuffle=True,
        )

        val_loader = None
        if val_df is not None:
            X_val = self.scaler.transform(
                val_df[self.feature_names].fillna(
                    val_df[self.feature_names].median()
                )
            )
            y_val = self.label_encoder.transform(val_df[target])
            val_dataset = MatchDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=256)

        num_classes = len(self.label_encoder.classes_)
        self.model = MatchPredictorNN(
            input_size=X_train.shape[1],
            hidden_layers=self.config.hyperparameters.get(
                "hidden_layers", [256, 128, 64, 32]
            ),
            dropout_rate=self.config.hyperparameters.get("dropout_rate", 0.3),
            num_classes=num_classes,
            batch_norm=self.config.hyperparameters.get("batch_norm", True),
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.hyperparameters.get("learning_rate", 0.001),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.config.training_config.get("reduce_lr_patience", 5),
            factor=0.5,
        )

        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_val_loss = float("inf")
        patience = self.config.training_config.get("early_stopping_patience", 15)
        patience_counter = 0
        best_model_state = None

        epochs = self.config.hyperparameters.get("epochs", 100)
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validate
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += y_batch.size(0)
                        correct += (predicted == y_batch).sum().item()

                val_loss /= len(val_loader)
                val_accuracy = correct / total
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_accuracy)

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    if best_model_state:
                        self.model.load_state_dict(best_model_state)
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs}: "
                        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                        f"val_acc={val_accuracy:.4f}"
                    )

        self.is_fitted = True
        self.training_history = history
        logger.info("Neural Network training completed")

        return {"training_history": history, "best_val_loss": best_val_loss}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler.transform(
                X[self.feature_names].fillna(X[self.feature_names].median())
            )
        else:
            X_scaled = self.scaler.transform(X)

        dataset = MatchDataset(X_scaled, np.zeros(len(X_scaled)))
        loader = DataLoader(dataset, batch_size=256)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                proba = torch.softmax(outputs, dim=1)
                predictions.append(proba.cpu().numpy())

        return np.vstack(predictions)

    def save(self, path: str) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {"model_state": self.model.state_dict(), "config": self.config.to_dict()},
            path,
        )

        preprocessor_path = path.parent / f"{path.stem}_preprocessors.pkl"
        joblib.dump(
            {
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "feature_names": self.feature_names,
            },
            preprocessor_path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        import joblib

        path = Path(path)

        checkpoint = torch.load(path, weights_only=False)

        preprocessor_path = path.parent / f"{path.stem}_preprocessors.pkl"
        preprocessors = joblib.load(preprocessor_path)
        self.scaler = preprocessors["scaler"]
        self.label_encoder = preprocessors["label_encoder"]
        self.feature_names = preprocessors["feature_names"]

        num_classes = len(self.label_encoder.classes_)
        self.model = MatchPredictorNN(
            input_size=len(self.feature_names),
            hidden_layers=self.config.hyperparameters.get(
                "hidden_layers", [256, 128, 64, 32]
            ),
            dropout_rate=self.config.hyperparameters.get("dropout_rate", 0.3),
            num_classes=num_classes,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")
