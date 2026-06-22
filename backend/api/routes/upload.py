from fastapi import APIRouter, UploadFile, File

from backend.services import (
    storage_service,
    task_service
)

router = APIRouter()


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # 1. создаем задачу
    task_id = storage_service.create_task()

    # 2. читаем файл
    file_bytes = await file.read()

    # 3. сохраняем видео
    video_path = storage_service.save_video(
        task_id=task_id,
        file_bytes=file_bytes,
        filename="input.mp4"
    )

    task_service.create(
    task_id=task_id,
    filename=file.filename
)

    return {
        "task_id": task_id,
        "filename": file.filename,
        "saved_to": str(video_path)
    }