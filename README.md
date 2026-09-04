# 🤖 KI Trainer Studio

Komplettes Framework zum Erstellen und Trainieren von KI-Modellen - lokal, mit Docker, oder als API.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## ✨ Features

- **5 Modelltypen**: MLP, CNN, Transformer, Random Forest, Linear
- **2 Tasks**: Klassifizierung & Regression
- **3 Interfaces**: Python Script, Jupyter Notebook, REST API
- **Export**: PyTorch (.pt), ONNX, Scikit-Learn, Production Package
- **Docker**: API + Jupyter + TensorBoard mit einem Befehl

## 🚀 Schnellstart

### Option 1: Docker (empfohlen)

```bash
git clone https://github.com/deinname/ki-trainer-studio.git
cd ki-trainer-studio
docker-compose up
```

- API: http://localhost:8000/docs
- Jupyter: http://localhost:8888
- TensorBoard: http://localhost:6006

### Option 2: Lokal

```bash
./setup.sh
source venv/bin/activate
python ki_trainer.py
```

### Option 3: Notebook

```bash
jupyter lab KI_Trainer_Notebook.ipynb
```

## 📖 Nutzung

### Eigene CSV trainieren

```python
from ki_trainer import TrainerConfig, KiTrainer, DataManager

config = TrainerConfig(model_type="mlp", task="classification", epochs=20)
dm = DataManager(config)
train_loader, test_loader = dm.load_csv("daten.csv", target_col="label")

trainer = KiTrainer(config)
trainer.train(train_loader, test_loader, dm.input_dim, dm.output_dim)
```

### Bilder trainieren (MNIST/CIFAR10)

```python
from erweiterungen import train_image_model
train_image_model("mnist", epochs=5)
```

### API nutzen

```bash
# Upload
curl -X POST http://localhost:8000/upload -F "file=@daten.csv" -F "target_col=label"

# Training
curl -X POST http://localhost:8000/train -F "model_type=mlp" -F "epochs=20"

# Predict
curl -X POST http://localhost:8000/predict -F "values=5.1,3.5,1.4,0.2"
```

## 📁 Struktur

```
ki-trainer-studio/
├── ki_trainer.py              # Haupt-Trainer (PyTorch + Sklearn)
├── erweiterungen.py           # Bild, Text, AutoML
├── api_server.py              # FastAPI Server
├── KI_Trainer_Notebook.ipynb  # Interaktives Lab
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.sh
└── data/ uploads/ models/
```

## 🛠️ Config

```python
TrainerConfig(
    model_type="mlp",          # mlp, cnn, transformer, random_forest, linear
    task="classification",     # classification, regression
    epochs=20,
    batch_size=32,
    lr=0.001,
    hidden_dims=[128, 64, 32],
    dropout=0.2,
    optimizer="adam"           # adam, adamw, sgd
)
```

## 📦 Export

- `best_model.pt` - PyTorch
- `model.onnx` - Für Web/Mobile/ONNX Runtime
- `production_package.pkl` - Inkl. Scaler & Config

## Lizenz

MIT - frei nutzbar.

---
Erstellt mit Meta AI 🤖

