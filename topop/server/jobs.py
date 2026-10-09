"""Background optimization runs: one thread per run, serialised by a semaphore, fanned out to
WebSocket subscribers through bounded mailboxes (`Mailbox`)."""

from __future__ import annotations

import asyncio
import logging
import os
import struct
import threading
import time
from collections import deque

import numpy as np
from fastapi import Request

from topop.core.optimize import optimize
from topop.core.problem import IterationInfo
from topop.server.build import BuiltProblem, run_params
from topop.server.schemas import IterationRecord, ParamsSpec, ProgressMsg, RunInfo, StatusMsg
from topop.server.store import TERMINAL, RunRecord, Store, now_iso, store_of

log = logging.getLogger(__name__)

CLOSE = None  # mailbox sentinel: no more messages for this subscriber
GONE = object()  # mailbox marker: the subscriber was dropped (stalled) or disconnected
STALL_SECONDS = 30.0  # a subscriber that takes nothing for this long while messages wait is dropped
DEFAULT_MAX_QUEUED = 4
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


def iteration_record(info: IterationInfo) -> IterationRecord:
    return IterationRecord(
        it=int(info.it),
        compliance=float(info.compliance),
        volume=float(info.volume),
        change=float(info.change),
        t_iter=float(info.t_iter),
        stress_max=None if info.stress_max is None else float(info.stress_max),
        constraint=None if info.constraint is None else float(info.constraint),
    )


def _status_msg(kind: str, info: RunInfo, message: str | None = None) -> dict:
    return StatusMsg(type=kind, message=message, run=info).model_dump(mode="json")


def max_queued() -> int:
    """TOPOP_MAX_QUEUED (default 4): runs allowed to wait behind the running one."""
    try:
        return max(0, int(os.environ.get("TOPOP_MAX_QUEUED") or DEFAULT_MAX_QUEUED))
    except ValueError:
        return DEFAULT_MAX_QUEUED


class Mailbox:
    """One stream subscriber: every JSON message in order, but at most one density frame (a new
    frame replaces one not sent yet). Filled from any thread, drained on the subscriber's loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self._lock = threading.Lock()
        self._items: deque = deque()
        self._has_frame = False
        self._waiting_since: float | None = None  # since when something waits and nothing is taken
        self._gone = False
        self._wake = asyncio.Event()  # set/cleared/awaited on `loop` only

    def put(self, msg: dict | bytes | None) -> bool:
        """Queue a JSON message, a density frame or CLOSE. False once the subscriber is gone."""
        with self._lock:
            if self._gone:
                return False
            if self._waiting_since is None:
                self._waiting_since = time.monotonic()
            if isinstance(msg, bytes):
                if self._has_frame:  # keep only the newest frame, after the progress it belongs to
                    for i in range(len(self._items) - 1, -1, -1):
                        if isinstance(self._items[i], bytes):
                            del self._items[i]
                            break
                self._has_frame = True
            self._items.append(msg)
        return self._notify()

    def stalled(self, limit: float) -> bool:
        with self._lock:
            since = self._waiting_since
        return since is not None and time.monotonic() - since > limit

    def drop(self) -> None:
        """Discard what is queued; `get` returns GONE from now on."""
        with self._lock:
            self._gone = True
            self._items.clear()
            self._has_frame = False
            self._waiting_since = None
        self._notify()

    def pending(self) -> list:
        with self._lock:
            return list(self._items)

    def _notify(self) -> bool:
        try:
            self.loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:  # the subscriber's event loop is closed
            with self._lock:
                self._gone = True
                self._items.clear()
            return False
        return True

    async def get(self) -> dict | bytes | None | object:
        while True:
            with self._lock:
                if self._gone:
                    return GONE
                if self._items:
                    msg = self._items.popleft()
                    if isinstance(msg, bytes):
                        self._has_frame = False
                    self._waiting_since = time.monotonic() if self._items else None
                    return msg
                self._wake.clear()
            await self._wake.wait()


class RunManager:
    def __init__(self, store: Store):
        self.store = store
        self.semaphore = threading.Semaphore(1)  # one run at a time per process
        self._threads: dict[str, tuple[threading.Thread, RunRecord]] = {}
        self._threads_lock = threading.Lock()

    # ---- lifecycle ------------------------------------------------------------------------------

    def start(self, run_id: str, built: BuiltProblem, params: ParamsSpec) -> None:
        rec = self.store.get_run(run_id)
        thread = threading.Thread(
            target=self._worker, args=(rec, built, params), name=f"topop-run-{run_id}", daemon=True
        )
        with self._threads_lock:
            self._threads[run_id] = (thread, rec)
        thread.start()

    def queued(self) -> int:
        """Runs of this manager still waiting for the semaphore."""
        with self._threads_lock:
            recs = [rec for _, rec in self._threads.values()]
        return sum(1 for rec in recs if rec.info.status == "queued")

    def cancel(self, run_id: str) -> RunInfo:
        rec = self.store.get_run(run_id)
        rec.cancel.set()
        with rec.lock:
            queued = rec.info.status == "queued"
        if queued:  # never started: finish now; the worker sees it and skips the run
            self._finish(rec, "cancelled", "cancelled while queued")
        return rec.snapshot()

    def shutdown(self, timeout: float = 10.0) -> None:
        with self._threads_lock:
            live = list(self._threads.values())
        for _, rec in live:
            rec.cancel.set()
        deadline = time.monotonic() + timeout
        for thread, _ in live:
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
        except BaseException as exc:  # SystemExit, KeyboardInterrupt, ...: never leave it running
            log.exception("run %s aborted", rec.info.id)
            why = f": {exc}" if str(exc) else ""
            self._finish(rec, "error", f"run aborted ({type(exc).__name__}{why})")
            if isinstance(exc, KeyboardInterrupt | SystemExit):
                raise
        finally:
            self.semaphore.release()
            with self._threads_lock:
                self._threads.pop(rec.info.id, None)

    def _optimize(self, rec: RunRecord, built: BuiltProblem, params: ParamsSpec) -> None:
        every = max(1, int(params.density_every))
        active = built.active

        def callback(info: IterationInfo, rho: np.ndarray) -> bool:
            record = iteration_record(info)
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
        self._finish(
            rec,
            status,
            result.message or None,
            rho=rho,
            outcome=result.status,
            stress=result.stress,
        )

    def _finish(
        self,
        rec: RunRecord,
        status: str,
        message: str | None,
        rho: np.ndarray | None = None,
        outcome: str | None = None,
        stress: np.ndarray | None = None,
    ) -> None:
        final_frame = None
        with rec.lock:
            if rec.info.status in TERMINAL:
                return
            rec.info.status = status
            rec.info.finished_at = now_iso()
            rec.message = message
            rec.info.message = message
            rec.info.outcome = outcome or ("cancelled" if status == "cancelled" else "error")
            if status == "error":
                rec.info.error = message or "run failed"
            if rho is not None:
                rec.rho, rec.stress = rho, stress
                last_it = rec.info.history[-1].it if rec.info.history else 0
                if rec.latest_frame_it != last_it:
                    active = rec.built.active if rec.built is not None else None
                    final_frame = density_frame(last_it, rho, active)
                    rec.latest_frame, rec.latest_frame_it = final_frame, last_it
        try:
            # before `done` goes out, so exports exist on disk; a failure is noted in the message
            self.store.finish_run(rec)
        finally:  # whatever happened, subscribers get the final status and are closed
            with rec.lock:
                if final_frame is not None:
                    self._push_locked(rec, final_frame)
                info = rec.info.model_copy(deep=True)
                self._push_locked(rec, _status_msg(status, info, rec.message))
                self._push_locked(rec, CLOSE)
                rec.subscribers.clear()

    # ---- WebSocket fan-out ----------------------------------------------------------------------

    @staticmethod
    def _push_locked(rec: RunRecord, msg: dict | bytes | None) -> None:
        alive = []
        for box in rec.subscribers:
            if not box.put(msg):  # disconnected, dropped, or its loop is closed
                continue
            if msg is not CLOSE and box.stalled(STALL_SECONDS):
                log.warning(
                    "run %s: dropping a stream subscriber that read nothing for %.0f s",
                    rec.info.id,
                    STALL_SECONDS,
                )
                box.drop()
                continue
            alive.append(box)
        rec.subscribers[:] = alive

    def subscribe(self, rec: RunRecord, box: Mailbox) -> tuple[RunInfo, bytes | None, dict | None]:
        """(run info so far, latest density frame, final status message if already finished).

        Atomic with respect to the worker: every message after the snapshot reaches `box`. A
        finished run's frame is rebuilt from runs/{id}.npz, so call this off the event loop.
        """
        with rec.lock:
            info = rec.info.model_copy(deep=True)
            frame = rec.latest_frame
            if rec.info.status not in TERMINAL:
                rec.subscribers.append(box)
                return info, frame, None
            message = rec.message if rec.message is not None else info.error
        if frame is None:  # released after persisting, or finished before a restart
            res = self.store.run_result(rec)
            if res is not None:
                rho, _, active, _ = res
                frame = density_frame(info.history[-1].it if info.history else 0, rho, active)
        return info, frame, _status_msg(info.status, info, message)

    @staticmethod
    def unsubscribe(rec: RunRecord, box: Mailbox) -> None:
        with rec.lock:
            rec.subscribers[:] = [b for b in rec.subscribers if b is not box]


def runs_of(app) -> RunManager:
    """The app's run manager. The lifespan creates it; fall back to creating it lazily."""
    mgr = getattr(app.state, "runs", None)
    if mgr is None:
        mgr = app.state.runs = RunManager(store_of(app))
    return mgr


def get_runs(request: Request) -> RunManager:
    return runs_of(request.app)
