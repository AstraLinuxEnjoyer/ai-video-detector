"""Loopback-only demo server with serialized inference and durable results."""
import asyncio
import copy
import json
import logging
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from filelock import FileLock

from training.spatial.common import atomic_json
from .analysis import Cancelled, MAX_SECONDS, analyze

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT/'demo_runtime'
MAX_BYTES = 400*1024*1024
ALLOWED = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}


class Job:
    def __init__(self, directory, name):
        self.directory, self.id = directory, directory.name
        self.lock, self.cancel = threading.RLock(), threading.Event()
        self.data = dict(id=self.id, filename=name, status='queued', phase='queued', progress=0,
                         frames=[], clips=[], warnings=[], message='Ожидает обработки')

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.data)

    def update(self, **values):
        with self.lock:
            self.data.update(copy.deepcopy(values))
            atomic_json(self.directory/'status.json', self.data)

    def check(self):
        if self.cancel.is_set():
            raise Cancelled()


class Service:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='trace-inference')
        self.jobs, self.detector = {}, None
        self.lock = threading.Lock()

    def process(self, job):
        try:
            job.check()
            job.update(status='running', phase='models', progress=2, message='Подключаем обученные веса')
            if self.detector is None:
                from .inference import Detector
                self.detector = Detector()
            job.update(models=self.detector.describe())
            analyze(job, self.detector)
        except Cancelled:
            job.update(status='cancelled', message='Анализ остановлен')
        except Exception as exc:
            logging.getLogger('trace').exception('Analysis failed: %s', job.id)
            job.update(status='error', message=str(exc), error=type(exc).__name__)


service = Service()


@asynccontextmanager
async def lifespan(_app):
    RUNTIME.mkdir(exist_ok=True)
    lock = FileLock(RUNTIME/'server.lock', timeout=0)
    lock.acquire()
    log = logging.getLogger('trace')
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.FileHandler(RUNTIME/'demo.log', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        log.addHandler(handler)
        log.addHandler(logging.StreamHandler())
    for path in RUNTIME.glob('*/status.json'):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('status') in ('queued', 'running'):
            data.update(status='error', message='Сервер был перезапущен; загрузите видео повторно')
            atomic_json(path, data)
    try:
        yield
    finally:
        for job in service.jobs.values():
            job.cancel.set()
        service.executor.shutdown(wait=True, cancel_futures=True)
        lock.release()


app = FastAPI(title='TRACE / Video Forensics', lifespan=lifespan)
app.mount('/assets', StaticFiles(directory=ROOT/'demo/web'), name='assets')


@app.middleware('http')
async def local_security(request: Request, call_next):
    # No LAN hosting or arbitrary external origin may submit private videos.
    host = request.headers.get('host', '').split(':')[0]
    if host not in ('127.0.0.1', 'localhost', 'testserver'):
        return JSONResponse({'detail': 'Localhost only'}, status_code=403)
    origin = request.headers.get('origin')
    if origin and origin != f'{request.url.scheme}://{request.headers.get("host")}':
        return JSONResponse({'detail': 'Cross-origin requests are not allowed'}, status_code=403)
    if request.method == 'POST':
        try:
            if int(request.headers.get('content-length', 0)) > MAX_BYTES+1024*1024:
                return JSONResponse({'detail': 'Maximum upload: 400 MB'}, status_code=413)
        except ValueError:
            return JSONResponse({'detail': 'Invalid content length'}, status_code=400)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
    return response


@app.get('/')
def index():
    return FileResponse(ROOT/'demo/web/index.html')


@app.get('/api/health')
def health():
    from .inference import model_sources
    try:
        info = model_sources()
        return dict(status='ready', device=os.environ.get('TRACE_DEVICE', 'cpu'), max_seconds=MAX_SECONDS,
                    max_mb=400, models={k: dict(epoch=v['epoch'], sha256=v['sha256']) for k, v in info.items()},
                    ffmpeg=bool(shutil.which('ffmpeg') and shutil.which('ffprobe')))
    except (OSError, ValueError, KeyError) as exc:
        return dict(status='unavailable', message=str(exc), max_seconds=MAX_SECONDS, max_mb=400)


@app.post('/api/jobs', status_code=202)
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, 'Поддерживаются MP4, MOV, MKV, AVI, WebM и M4V')
    with service.lock:
        if any(j.snapshot()['status'] in ('queued', 'running') for j in service.jobs.values()):
            raise HTTPException(409, 'Уже идет анализ. Дождитесь его окончания или остановите его.')
        if shutil.disk_usage(RUNTIME).free < MAX_BYTES+512*1024*1024:
            raise HTTPException(507, 'Для обработки требуется не менее 1 ГБ свободного места')
        directory = RUNTIME/uuid.uuid4().hex
        directory.mkdir()
        job = Job(directory, Path(file.filename).name)
        service.jobs[job.id] = job
    size = 0
    try:
        with (directory/'input.video').open('wb') as stream:
            while chunk := await file.read(1024*1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, 'Максимальный размер файла: 400 МБ')
                stream.write(chunk)
        if not size:
            raise HTTPException(400, 'Пустой файл')
        job.update(bytes=size)
        service.executor.submit(service.process, job)
        return dict(id=job.id)
    except BaseException:
        service.jobs.pop(job.id, None)
        shutil.rmtree(directory)
        raise
    finally:
        await file.close()


def job_dir(job_id):
    if not re_id(job_id):
        raise HTTPException(404, 'Задача не найдена')
    path = RUNTIME/job_id
    if not path.is_dir():
        raise HTTPException(404, 'Задача не найдена')
    return path


def re_id(value):
    return len(value) == 32 and all(char in '0123456789abcdef' for char in value)


@app.get('/api/jobs/{job_id}')
def status(job_id: str):
    directory = job_dir(job_id)
    if job_id in service.jobs:
        return service.jobs[job_id].snapshot()
    path = directory/'status.json'
    if not path.exists():
        raise HTTPException(404, 'Задача не найдена')
    return json.loads(path.read_text(encoding='utf-8'))


@app.post('/api/jobs/{job_id}/cancel')
def cancel(job_id: str):
    job_dir(job_id)
    job = service.jobs.get(job_id)
    if job and job.snapshot()['status'] in ('queued', 'running'):
        job.cancel.set()
    return dict(ok=True)


@app.get('/api/jobs/{job_id}/files/{filename}')
def artifact(job_id: str, filename: str):
    directory = job_dir(job_id)
    allowed = filename in {'preview.mp4', 'report.json', 'frames.csv'} or (
        filename.startswith(('frame_', 'heat_', 'flow_')) and Path(filename).suffix in {'.jpg', '.gif'})
    path = directory/filename
    if not allowed or path.parent.resolve() != directory.resolve() or not path.is_file():
        raise HTTPException(404, 'Файл не найден')
    download = filename in {'report.json', 'frames.csv'}
    return FileResponse(path, filename=filename if download else None,
                        content_disposition_type='attachment' if download else 'inline')
