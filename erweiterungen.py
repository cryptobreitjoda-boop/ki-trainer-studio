"""
Erweiterte Trainer: Bild-Klassifizierung + Text-Klassifizierung + AutoML
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ki_trainer import TrainerConfig, KiTrainer, DataManager
import os

# ============ 1. BILD TRAINER (MNIST / CIFAR10) ============
def train_image_model(dataset_name="mnist", epochs=5):
    print(f"\n=== IMAGE TRAINER: {dataset_name.upper()} ===")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if dataset_name=="mnist" else transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    
    if dataset_name == "mnist":
        train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST("./data", train=False, transform=transform)
        in_channels = 1
        num_classes = 10
    else:
        train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10("./data", train=False, transform=transform)
        in_channels = 3
        num_classes = 10
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    # CNN Modell direkt bauen
    from ki_trainer import SimpleCNN
    config = TrainerConfig(model_type="cnn", task="classification", epochs=epochs, lr=1e-3, batch_size=64)
    trainer = KiTrainer(config)
    
    # Custom build für CNN
    from ki_trainer import SimpleCNN
    trainer.model = SimpleCNN(num_classes=num_classes, input_channels=in_channels).to(trainer.device)
    
    # Training Loop für Bilder
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainer.model.parameters(), lr=config.lr)
    
    for epoch in range(epochs):
        trainer.model.train()
        running_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(trainer.device), labels.to(trainer.device)
            optimizer.zero_grad()
            outputs = trainer.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Acc: {100.*correct/total:.2f}%")
    
    # Evaluieren
    trainer.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(trainer.device), labels.to(trainer.device)
            outputs = trainer.model(images)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    print(f"TEST Accuracy: {100.*correct/total:.2f}%")
    torch.save(trainer.model.state_dict(), f"best_{dataset_name}_cnn.pt")
    return trainer.model


# ============ 2. TEXT TRAINER (IMDB Sentiment) ============
def train_text_model():
    print("\n=== TEXT TRAINER: IMDB Sentiment ===")
    print("Installiere dafür: pip install datasets transformers")
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        # Kleines Demo-Tokenizing ohne HF Download falls offline
        # Fallback: Dummy Text Daten
        print("Lade IMDB Demo (gekürzt auf 1000 Samples für schnelles Training)...")
        dataset = load_dataset("imdb", split="train[:1%]")  # nur 1% für Demo
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        def tokenize(batch):
            return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
        
        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'label'])
        
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        config = TrainerConfig(model_type="transformer", task="classification", epochs=3, lr=2e-4)
        from ki_trainer import SimpleTransformer
        trainer = KiTrainer(config)
        trainer.model = SimpleTransformer(vocab_size=tokenizer.vocab_size, num_classes=2).to(trainer.device)
        
        # Training hier gleich wie oben...
        print("Text-Modell bereit zum Trainieren - siehe ki_trainer.py für Loop")
        return trainer
        
    except Exception as e:
        print(f"Text-Training benötigt Internet + transformers: {e}")
        print("Nutze stattdessen die Transformer Klasse aus ki_trainer.py mit eigenen Texten.")


# ============ 3. AUTOML - FINDET BESTES MODELL AUTOMATISCH ============
def automl_search(csv_path: str, target_col: str):
    print("\n=== AUTOML SEARCH ===")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col]).select_dtypes(include=['number']).fillna(0)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "MLP (PyTorch)": "pytorch"
    }
    
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        if name == "MLP (PyTorch)":
            # PyTorch MLP schnell testen
            config = TrainerConfig(model_type="mlp", epochs=10, hidden_dims=[64,32])
            dm = DataManager(config)
            # reuse loaders aus X_train
            import torch
            from torch.utils.data import TensorDataset, DataLoader
            train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train.values if hasattr(y_train, 'values') else y_train))
            test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test.values if hasattr(y_test, 'values') else y_test))
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=32)
            trainer = KiTrainer(config)
            trainer.train(train_loader, test_loader, X_train.shape[1], len(set(y)))
            score = max(trainer.history["val_acc"]) if trainer.history["val_acc"] else 0
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
        
        print(f"{name}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_name = name
    
    print(f"\n[BEST] {best_name} mit {best_score:.4f} Accuracy")
    return best_name, best_score


if __name__ == "__main__":
    # Wähle was du trainieren willst:
    # train_image_model("mnist", epochs=5)   # Bild-Klassifizierung
    # train_text_model()                      # Text-Klassifizierung
    # automl_search("deine_daten.csv", "label") # AutoML

    print("Beispiele in dieser Datei - entkommentiere die gewünschte Funktion.")
