"""
KI TRAINER STUDIO - API Server
Start: uvicorn api_server:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from ki_trainer import TrainerConfig, KiTrainer, DataManager
from pathlib import Path

app = FastAPI(title="KI Trainer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

trainer_instance: KiTrainer = None
data_manager: DataManager = None
train_loader = None
test_loader = None

@app.get("/")
def root():
    return {"status": "KI Trainer läuft", "endpoints": ["/upload", "/train", "/predict", "/history"]}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...), target_col: str = Form(...), task: str = Form("classification")):
    global data_manager, train_loader, test_loader
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Speichern
    Path("uploads").mkdir(exist_ok=True)
    save_path = f"uploads/{file.filename}"
    with open(save_path, "wb") as f:
        f.write(contents)
    
    config = TrainerConfig(task=task)
    data_manager = DataManager(config)
    train_loader, test_loader = data_manager.load_csv(save_path, target_col)
    
    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "input_dim": data_manager.input_dim,
        "output_dim": data_manager.output_dim,
        "preview": df.head(5).to_dict(orient="records")
    }

@app.post("/train")
async def start_training(
    model_type: str = Form("mlp"),
    epochs: int = Form(10),
    lr: float = Form(0.001),
    hidden_dims: str = Form("128,64")
):
    global trainer_instance
    if train_loader is None:
        return {"error": "Erst Daten hochladen via /upload"}
    
    hd = [int(x) for x in hidden_dims.split(",")]
    config = TrainerConfig(model_type=model_type, epochs=epochs, lr=lr, hidden_dims=hd, task=data_manager.config.task)
    trainer_instance = KiTrainer(config)
    model = trainer_instance.train(train_loader, test_loader, data_manager.input_dim, data_manager.output_dim)
    
    return {
        "status": "done",
        "history": trainer_instance.history,
        "model_type": model_type,
        "best_model": "best_model.pt"
    }

@app.get("/history")
def get_history():
    if trainer_instance is None:
        return {"error": "Noch kein Training"}
    return trainer_instance.history

@app.post("/predict")
async def predict(values: str = Form(...)):
    # values: "5.1,3.5,1.4,0.2" als komma-getrennt
    import numpy as np
    if trainer_instance is None or trainer_instance.model is None:
        return {"error": "Erst trainieren"}
    arr = np.array([[float(x) for x in values.split(",")]])
    arr_scaled = data_manager.scaler.transform(arr)
    pred = trainer_instance.predict(arr_scaled)
    return {"input": values, "prediction": pred.tolist(), "predicted_class": int(pred.argmax()) if trainer_instance.config.task=="classification" else float(pred[0])}

# Start: uvicorn api_server:app --reload
