"""Asynchronous FIFO Job Queue Manager for Fotogenie.

Guarantees sequential GPU generation on a single RTX 3060 without blocking
FastAPI HTTP workers. Provides real-time queue position tracking, elapsed time,
and non-blocking polling endpoints.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Optional

from backend.logging import get_logger

logger = get_logger("fotogenie.queue_manager")


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    job_type: str  # "generate" or "dress-swap"
    handler: Callable[..., Coroutine[Any, Any, Any]]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    status: JobStatus = JobStatus.QUEUED
    position: int = 0
    created_at: float = field(default_factory=time.monotonic)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    error_status: int = 500
    event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at if self.completed_at is not None else time.monotonic()
        return round(end - self.started_at, 2)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "position": self.position,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
            "result": self.result if self.status == JobStatus.COMPLETED else None,
        }


class QueueManager:
    _instance: Optional[QueueManager] = None

    @classmethod
    def get_instance(cls) -> QueueManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._jobs: Dict[str, Job] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._current_job: Optional[Job] = None
        self._lock = asyncio.Lock()
        self._running = False

    def start(self) -> None:
        """Start the background queue consumer worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("QueueManager started with background worker.")

    async def stop(self) -> None:
        """Gracefully stop the background worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        logger.info("QueueManager stopped.")

    def _update_positions(self) -> None:
        """Recalculate 1-based positions for all queued jobs."""
        queued_jobs = [j for j in self._jobs.values() if j.status == JobStatus.QUEUED]
        for idx, job in enumerate(queued_jobs, start=1):
            job.position = idx

    async def submit(
        self,
        job_type: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Job:
        """Submit a job to the FIFO queue."""
        if not self._running:
            self.start()
        job_id = uuid.uuid4().hex
        job = Job(
            job_id=job_id,
            job_type=job_type,
            handler=handler,
            args=args,
            kwargs=kwargs,
        )
        self._jobs[job_id] = job
        await self._queue.put(job)
        self._update_positions()
        logger.info(
            "Submitted job %s (type=%s, position=%d)",
            job_id,
            job_type,
            job.position,
        )
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve job by ID."""
        return self._jobs.get(job_id)

    def get_stats(self) -> dict:
        """Return current queue metrics."""
        queued = sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)
        busy = self._current_job is not None
        return {
            "busy": busy,
            "queue_count": queued + (1 if busy else 0),
            "current_job_id": self._current_job.job_id if self._current_job else None,
            "total_completed": sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED),
            "total_failed": sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED),
        }

    async def _worker_loop(self) -> None:
        """Sequential single-generation GPU execution loop."""
        logger.info("Queue worker loop active.")
        while self._running:
            try:
                job = await self._queue.get()
            except asyncio.CancelledError:
                break

            self._current_job = job
            job.status = JobStatus.PROCESSING
            job.position = 0
            job.started_at = time.monotonic()
            self._update_positions()

            logger.info("Processing job %s (%s)...", job.job_id, job.job_type)
            try:
                result = await job.handler(*job.args, **job.kwargs)
                job.result = result
                job.status = JobStatus.COMPLETED
                logger.info(
                    "Job %s completed in %.2fs",
                    job.job_id,
                    job.elapsed_seconds,
                )
            except Exception as exc:
                if hasattr(exc, "status_code") and hasattr(exc, "detail"):
                    job.error = str(exc.detail)
                    job.error_status = int(exc.status_code)
                else:
                    job.error = str(exc)
                    job.error_status = 500
                job.status = JobStatus.FAILED
                logger.error("Job %s failed after %.2fs: %s", job.job_id, job.elapsed_seconds, exc)
            finally:
                job.completed_at = time.monotonic()
                job.event.set()
                self._current_job = None
                self._queue.task_done()
                self._update_positions()

                # Clean up completed jobs older than 30 minutes to prevent memory leaks
                self._prune_stale_jobs()

    def _prune_stale_jobs(self) -> None:
        now = time.monotonic()
        to_del = [
            jid
            for jid, j in self._jobs.items()
            if j.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            and j.completed_at is not None
            and (now - j.completed_at > 1800)
        ]
        for jid in to_del:
            self._jobs.pop(jid, None)
