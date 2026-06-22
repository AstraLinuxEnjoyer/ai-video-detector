from pydantic import BaseModel
from datetime import datetime


class TaskSchema(BaseModel):
    task_id: str

    status: str

    progress: int

    filename: str | None = None

    created_at: datetime

    updated_at: datetime

    result: float | None = None

    error: str | None = None