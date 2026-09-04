#!/bin/bash
set -e

echo "=== KI Trainer Studio Setup ==="

# Ordner erstellen
mkdir -p data uploads models runs

# Python Env check
if ! command -v python3 &> /dev/null; then
    echo "Python3 nicht gefunden!"
    exit 1
fi

echo "[1/4] Virtuelle Umgebung..."
python3 -m venv venv
source venv/bin/activate

echo "[2/4] Dependencies installieren..."
pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn jupyter ipykernel joblib

echo "[3/4] Jupyter Kernel registrieren..."
python -m ipykernel install --user --name=ki-trainer --display-name="KI Trainer Studio"

echo "[4/4] Demo Daten erstellen..."
python -c "
import pandas as pd
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv('data/iris_demo.csv', index=False)
print('Demo CSV erstellt: data/iris_demo.csv')
"

echo ""
echo "=== Fertig! ==="
echo "Starte mit:"
echo "  source venv/bin/activate"
echo "  python ki_trainer.py          # Training"
echo "  uvicorn api_server:app --reload  # API"
echo "  jupyter lab                   # Notebook"
echo ""
echo "Oder mit Docker:"
echo "  docker-compose up"
