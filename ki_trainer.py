"""
KI TRAINER STUDIO - Echter Python Trainer
Erstellt und trainiert Modelle mit PyTorch & Scikit-Learn
Author: Meta AI
"""

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm


# ================= CONFIG =================
@dataclass
class TrainerConfig:
    model_type: Literal["mlp", "cnn", "transformer", "random_forest", "linear"] = "mlp"
    task: Literal["classification", "regression"] = "classification"
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    optimizer: Literal["adam", "sgd", "adamw"] = "adam"
    hidden_dims: list = None
    dropout: float = 0.2
    activation: Literal["relu", "gelu", "tanh"] = "relu"
    test_split: float = 0.2
    device: str = "auto"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ================= MODELLE =================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2, activation="relu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=128, nhead=4, num_layers=2, num_classes=2, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.max_len = max_len
        
    def forward(self, x):
        # x: [B, seq_len] token ids
        seq_len = x.size(1)
        emb = self.embedding(x) + self.pos_enc[:, :seq_len, :]
        out = self.transformer(emb)
        pooled = out.mean(dim=1)  # mean pooling
        return self.classifier(pooled)


# ================= DATASET LOADER =================
class DataManager:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.label_enc = LabelEncoder()
        self.input_dim = None
        self.output_dim = None
        
    def load_csv(self, path: str, target_col: str) -> Tuple[DataLoader, DataLoader]:
        df = pd.read_csv(path)
        print(f"[DATA] CSV geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Nur numerische Spalten für dieses Beispiel
        X = X.select_dtypes(include=[np.number]).fillna(0)
        
        if self.config.task == "classification":
            y = self.label_enc.fit_transform(y)
            self.output_dim = len(np.unique(y))
        else:
            self.output_dim = 1
            
        self.input_dim = X.shape[1]
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.test_split, random_state=42
        )
        
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train) if self.config.task=="classification" else torch.FloatTensor(y_train))
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test) if self.config.task=="classification" else torch.FloatTensor(y_test))
        
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.config.batch_size)
        
        print(f"[DATA] Train: {len(train_ds)} | Test: {len(test_ds)} | Input Dim: {self.input_dim}")
        return train_loader, test_loader

    def load_demo(self, name: str = "iris") -> Tuple[DataLoader, DataLoader]:
        from sklearn.datasets import load_iris, fetch_california_housing
        if name == "iris":
            data = load_iris()
            X, y = data.data, data.target
            self.config.task = "classification"
            self.input_dim = 4
            self.output_dim = 3
        elif name == "california":
            data = fetch_california_housing()
            X, y = data.data, data.target
            self.config.task = "regression"
            self.input_dim = 8
            self.output_dim = 1
        else:
            raise ValueError("Demo: iris, california")
        
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=self.config.test_split, random_state=42)
        
        if self.config.task == "classification":
            train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        else:
            train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1,1))
            test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1,1))
            
        return DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True), DataLoader(test_ds, batch_size=self.config.batch_size)


# ================= TRAINER =================
class KiTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        print(f"[TRAINER] Device: {self.device} | Config: {asdict(config)}")
    
    def build_model(self, input_dim: int, output_dim: int):
        if self.config.model_type == "mlp":
            self.model = MLP(input_dim, self.config.hidden_dims, output_dim, self.config.dropout, self.config.activation)
        elif self.config.model_type == "cnn":
            self.model = SimpleCNN(num_classes=output_dim, input_channels=1)
        elif self.config.model_type == "transformer":
            self.model = SimpleTransformer(num_classes=output_dim)
        elif self.config.model_type in ["random_forest", "linear"]:
            self.model = None  # sklearn wird separat behandelt
            return
        else:
            raise ValueError(f"Unbekanntes Modell: {self.config.model_type}")
        
        self.model.to(self.device)
        params = sum(p.numel() for p in self.model.parameters())
        print(f"[MODEL] {self.config.model_type.upper()} erstellt | Parameter: {params:,}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, input_dim: int, output_dim: int):
        # Sklearn Pfad
        if self.config.model_type == "random_forest":
            return self._train_sklearn(train_loader, val_loader, is_rf=True)
        if self.config.model_type == "linear":
            return self._train_sklearn(train_loader, val_loader, is_rf=False)
        
        # PyTorch Pfad
        self.build_model(input_dim, output_dim)
        
        criterion = nn.CrossEntropyLoss() if self.config.task == "classification" else nn.MSELoss()
        
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        elif self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=1e-2)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        best_val = float('inf')
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss, correct, total = 0, 0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            for X, y in pbar:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                if self.config.task == "classification":
                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})
                else:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Validation
            self.model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    out = self.model(X)
                    loss = criterion(out, y)
                    val_loss += loss.item()
                    if self.config.task == "classification":
                        pred = out.argmax(dim=1)
                        val_correct += (pred == y).sum().item()
                        val_total += y.size(0)
                        all_preds.extend(pred.cpu().numpy())
                        all_labels.extend(y.cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = correct / total if total > 0 else 0
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            
            scheduler.step(avg_val_loss)
            
            print(f"-> Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} Acc={train_acc:.4f} | Val Loss={avg_val_loss:.4f} Acc={val_acc:.4f}")
            
            # Best Model speichern
            if avg_val_loss < best_val:
                best_val = avg_val_loss
                torch.save(self.model.state_dict(), "best_model.pt")
                print("   [CHECKPOINT] Bestes Modell gespeichert -> best_model.pt")
        
        print("\n[TRAINING ABGESCHLOSSEN]")
        self.plot_history()
        if self.config.task == "classification" and len(all_labels) > 0:
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds))
            print("Confusion Matrix:")
            print(confusion_matrix(all_labels, all_preds))
        
        return self.model
    
    def _train_sklearn(self, train_loader, val_loader, is_rf=True):
        # Daten aus Loadern holen
        X_train = torch.cat([b[0] for b in train_loader]).numpy()
        y_train = torch.cat([b[1] for b in train_loader]).numpy()
        X_test = torch.cat([b[0] for b in val_loader]).numpy()
        y_test = torch.cat([b[1] for b in val_loader]).numpy()
        
        if is_rf:
            model = RandomForestClassifier(n_estimators=100, random_state=42) if self.config.task=="classification" else RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000) if self.config.task=="classification" else LinearRegression()
        
        print(f"[SKLEARN] Trainiere {model.__class__.__name__}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        if self.config.task == "classification":
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            print(f"[ERGEBNIS] Accuracy: {acc:.4f} | F1: {f1:.4f}")
            print(classification_report(y_test, preds))
        else:
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            print(f"[ERGEBNIS] MSE: {mse:.4f} | R2: {r2:.4f}")
        
        import joblib
        joblib.dump(model, "best_model_sklearn.pkl")
        print("[CHECKPOINT] Modell gespeichert -> best_model_sklearn.pkl")
        self.model = model
        return model

    def plot_history(self):
        if len(self.history["train_loss"]) == 0:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        ax1.plot(self.history["train_loss"], label="Train Loss")
        ax1.plot(self.history["val_loss"], label="Val Loss")
        ax1.set_title("Loss Verlauf")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        
        if self.config.task == "classification":
            ax2.plot(self.history["train_acc"], label="Train Acc")
            ax2.plot(self.history["val_acc"], label="Val Acc")
            ax2.set_title("Accuracy Verlauf")
            ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("training_history.png", dpi=150)
        print("[PLOT] Verlauf gespeichert -> training_history.png")
        plt.close()

    def predict(self, x_input):
        self.model.eval()
        with torch.no_grad():
            if isinstance(x_input, np.ndarray):
                x_input = torch.FloatTensor(x_input).to(self.device)
            out = self.model(x_input)
            if self.config.task == "classification":
                return torch.softmax(out, dim=1).cpu().numpy()
            return out.cpu().numpy()

    def export_onnx(self, input_dim, filename="model.onnx"):
        if self.model is None or isinstance(self.model, (RandomForestClassifier, RandomForestRegressor, LinearRegression, LogisticRegression)):
            print("[EXPORT] ONNX Export nur für PyTorch Modelle")
            return
        dummy = torch.randn(1, input_dim).to(self.device)
        torch.onnx.export(self.model, dummy, filename, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        print(f"[EXPORT] ONNX Modell gespeichert -> {filename}")


# ================= BEISPIEL USAGE =================
if __name__ == "__main__":
    print("=== KI TRAINER STUDIO - DEMO ===\n")
    
    # 1. Config erstellen
    config = TrainerConfig(
        model_type="mlp",          # mlp, cnn, transformer, random_forest, linear
        task="classification",      # classification oder regression
        epochs=15,
        batch_size=16,
        lr=0.001,
        optimizer="adam",
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        activation="relu"
    )
    
    # 2. Daten laden (Demo)
    data_manager = DataManager(config)
    train_loader, test_loader = data_manager.load_demo("iris")  # oder "california" für Regression
    # Alternativ eigene CSV: train_loader, test_loader = data_manager.load_csv("deine_daten.csv", target_col="label")
    
    # 3. Trainer starten
    trainer = KiTrainer(config)
    model = trainer.train(train_loader, test_loader, data_manager.input_dim, data_manager.output_dim)
    
    # 4. Export
    trainer.export_onnx(data_manager.input_dim)
    
    print("\nFertig! Modelle liegen in:")
    print("- best_model.pt (PyTorch)")
    print("- training_history.png (Plot)")
    print("- model.onnx (für Produktion)")
