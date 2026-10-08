"""Background optimization runs: one thread per run, serialised by a semaphore, fanned out to
WebSocket subscribers through asyncio queues (`loop.call_soon_threadsafe`)."""

from __future__ import annotations

import asyncio
import logging
import struct
import threading
import time

import numpy as np
from fastapi import Request

from topop.core.optimize import optimize
from topop.core.problem import IterationInfo
from topop.server.build import BuiltProblem, run_params
from topop.server.schemas import IterationRecord, ParamsSpec, ProgressMsg, RunInfo, StatusMsg
from topop.server.store import TERMINAL, RunRecord, Store, now_iso, store_of

log = logging.getLogger(__name__)

CLOSE = None  # queue sentinel: no more messages for this subscriber
RESULT_STATUS = {
    "converged": "done",
    "max_iter": "done",
    "cancelled": "cancelled",
    "error": "error",
}


def density_frame(it: int, rho: np.ndarray, active: np.ndarray | None = None) -> bytes:
    """Binary WS frame: <u4 it, <u4 nx, <u4 ny, <u4 nz, then u8 round(rho*255), C-order."""
    q = np.clip(np.rint(np.nan_to_num(np.asarray(rho, dtype=np.float64)) * 255.0), 0, 255)
    q = q.astype(np.uint8)
    if active is not None:
        q[~np.asarray(active, dtype=bool)] = 0
    nx, ny, nz = q.shape
    return struct.pack("<4I", int(it), nx, ny, nz) + np.ascontiguousarray(q).tobytes()


def _status_msg(kind: str, info: RunInfo, message: str | None = None) -> dict:
    return StatusMsg(type=kind, message=message, run=info).model_dump(mode="json")


class RunManager:
    def __init__(self, store: Store):
        self.store = store
        self.semaphore = threading.Semaphore(1)  # one run at a time per process
        self._threads: dict[str, threading.Thread] = {}

    # ---- lifecycle ------------------------------------------------------------------------------

    def start(self, run_id: str, built: BuiltProblem, params: ParamsSpec) -> None:
        rec = self.store.get_run(run_id)
        thread = threading.Thread(
            target=self._worker, args=(rec, built, params), name=f"topop-run-{run_id}", daemon=True
        )
        self._threads[run_id] = thread
        thread.start()

    def cancel(self, run_id: str) -> RunInfo:
        rec = self.store.get_run(run_id)
        rec.cancel.set()
        with rec.lock:
            queued = rec.info.status == "queued"
        if queued:  # never started: finish now; the worker sees it and skips the run
            self._finish(rec, "cancelled", "cancelled while queued")
        return rec.snapshot()

    def shutdown(self, timeout: float = 10.0) -> None:
        for rec in self.store.list_runs():
            if not rec.finished:
                rec.cancel.set()
        deadline = time.monotonic() + timeout
        for thread in list(self._threads.values()):
            thread.join(max(0.0, deadline - time.monotonic()))

    def _worker(self, rec: RunRecord, built: BuiltProblem, params: ParamsSpec) -> None:
        self.semaphore.acquire()
        try:
            with rec.lock:  # atomic with cancel(): a run cancelled while queued never starts
                start = rec.info.status == "queued" and not rec.cancel.is_set()
                if start:
                    rec.info.status = "running"
                    self._push_locked(rec, _status_msg("started", rec.info.model_copy(deep=True)))
            if not start:
                self._finish(rec, "cancelled", "cancelled while queued")  # no-op if finished
                return
            self._optimize(rec, built, params)
        except MemoryError as exc:
            self._finish(rec, "error", str(exc) or "out of memory; lower the resolution")
        except Exception as exc:
            log.exception("run %s failed", rec.info.id)
            self._finish(rec, "error", str(exc) or type(exc).__name__)
        finally:
            self.semaphore.release()
            self._threads.pop(rec.info.id, None)

    def _optimize(self, rec: RunRecord, built: BuiltProblem, params: ParamsSpec) -> None:
        every = max(1, int(params.density_every))
        active = built.active

        def callback(info: IterationInfo, rho: np.ndarray) -> bool:
            record = IterationRecord(
                it=int(info.it),
                compliance=float(info.compliance),
                volume=float(info.volume),
                change=float(info.change),
                t_iter=float(info.t_iter),
            )
            progress = ProgressMsg(**record.model_dump()).model_dump(mode="json")
            frame = density_frame(info.it, rho, active) if info.it % every == 0 else None
            with rec.lock:
                rec.info.history.append(record)
                self._push_locked(rec, progress)
                if frame is not None:
                    rec.latest_frame, rec.latest_frame_it = frame, int(info.it)
                    self._push_locked(rec, frame)
            return not rec.cancel.is_set()

        result = optimize(built.problem, run_params(params), callback, cancel=rec.cancel.is_set)
        status = RESULT_STATUS.get(result.status, "error")
        rho = result.rho if result.history else None
        self._finish(rec, status, result.message or None, rho=rho)

    def _finish(
        self, rec: RunRecord, status: str, message: str | None, rho: np.ndarray | None = None
    ) -> None:
        final_frame = None
        with rec.lock:
            if rec.info.status in TERMINAL:
                return
            rec.info.status = status
            rec.info.finished_at = now_iso()
            rec.message = message
            if status == "error":
                rec.info.error = message or "run failed"
            if rho is not None:
                rec.rho = rho
                last_it = rec.info.history[-1].it if rec.info.history else 0
                if rec.latest_frame_it != last_it:
                    active = rec.built.active if rec.built is not None else None
                    final_frame = density_frame(last_it, rho, active)
                    rec.latest_frame, rec.latest_frame_it = final_frame, last_it
        self.store.persist_run(rec)  # before `done` goes out, so exports exist on disk
        with rec.lock:
            if final_frame is not None:
                self._push_locked(rec, final_frame)
            self._push_locked(rec, _status_msg(status, rec.info.model_copy(deep=True), message))
            self._push_locked(rec, CLOSE)
            rec.subscribers.clear()

    # ---- WebSocket fan-out ----------------------------------------------------------------------

    @staticmethod
    def _push_locked(rec: RunRecord, msg: dict | bytes | None) -> None:
        alive = []
        for loop, queue in rec.subscribers:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, msg)
            except RuntimeError:  # the subscriber's event loop is closed
                continue
            alive.append((loop, queue))
        rec.subscribers[:] = alive

    def subscribe(
        self, rec: RunRecord, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue
    ) -> tuple[RunInfo, bytes | None, dict | None]:
        """(run info so far, latest density frame, final status message if already finished).

        Atomic with respect to the worker: every message after the snapshot reaches `queue`.
        """
        with rec.lock:
            info = rec.info.model_copy(deep=True)
            frame = rec.latest_frame
            if rec.info.status not in TERMINAL:
                rec.subscribers.append((loop, queue))
                return info, frame, None
        message = rec.message if rec.message is not None else info.error
        if frame is None:  # finished before a restart: rebuild the last frame from runs/{id}.npz
            res = self.store.run_result(rec)
            if res is not None:
                rho, _, active, _ = res
                frame = density_frame(info.history[-1].it if info.history else 0, rho, active)
        return info, frame, _status_msg(info.status, info, message)

    @staticmethod
    def unsubscribe(rec: RunRecord, queue: asyncio.Queue) -> None:
        with rec.lock:
            rec.subscribers[:] = [s for s in rec.subscribers if s[1] is not queue]


def runs_of(app) -> RunManager:
    """The app's run manager. The lifespan creates it; fall back to creating it lazily."""
    mgr = getattr(app.state, "runs", None)
    if mgr is None:
        mgr = app.state.runs = RunManager(store_of(app))
    return mgr


def get_runs(request: Request) -> RunManager:
    return runs_of(request.app)
