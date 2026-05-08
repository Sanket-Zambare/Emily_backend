from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path

# Load .env once here — applies to entire app
env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

from routers import outfit

app = FastAPI(title="EMILY AI Backend")

# CORS — allows Flutter app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(outfit.router, prefix="/outfit", tags=["outfit"])

@app.get("/")
def root():
    return {"status": "EMILY backend running"}