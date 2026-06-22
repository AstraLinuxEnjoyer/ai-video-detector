from fastapi import FastAPI
from backend.core.config import settings
from backend.api.routes.upload import router as upload_router
from backend.api.routes.task import router as task_router

app = FastAPI(
    title="AI Video Detector",
    description="Backend for AI-generated video detection system",
    version="0.1.0"
)

app.include_router(upload_router)
app.include_router(task_router)

@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }

@app.get("/health")
async def health():
    return {
        "status": "ok"
    }