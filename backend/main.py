from fastapi import FastAPI

app = FastAPI(
    title="AI Video Detector",
    description="Backend for AI-generated video detection system",
    version="0.1.0"
)


@app.get("/")
async def root():
    return {
        "name": "AI Video Detector",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok"
    }