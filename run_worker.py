from backend.workers.video_worker import VideoWorker

if __name__ == "__main__":
    worker = VideoWorker()
    worker.run()