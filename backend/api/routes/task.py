from fastapi import APIRouter, HTTPException

from backend.services import task_service

router = APIRouter()


@router.get("/task/{task_id}")
async def get_task(task_id: str):

    try:
        task = task_service.load(task_id)

        return task

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )