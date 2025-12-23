# FastAPI / TorchScript inference

from fastapi import FastAPI

app = FastAPI()

@app.post("/inference")
def predict(text: str):
    return {"entities": [...]}
