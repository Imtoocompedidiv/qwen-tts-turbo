"""ServerMonitor — request tracking, health checks, deadlock watchdog."""

import os
import threading
import time


class ServerMonitor:
    """Encapsulates runtime monitoring, auto-fallback, and deadlock watchdog.

    All mutable state is instance attributes — no module globals.
    """

    # Constants
    FRAME_TIMEOUT = float(os.environ.get("FRAME_TIMEOUT", "10"))
    GENERATION_TIMEOUT = float(os.environ.get("GENERATION_TIMEOUT", "60"))
    MK_FAILURE_THRESHOLD = 3

    def __init__(self, engine):
        """Initialize monitor with a reference to the TTSEngine.

        Args:
            engine: TTSEngine instance (used by _auto_disable_megakernel).
        """
        self.engine = engine

        # Counters
        self.request_count = 0
        self.error_count = 0
        self.mk_failure_count = 0

        # TTFP history (rolling 100 samples)
        self._ttfp_history = []
        self._TTFP_HISTORY_MAX = 100

        # Generation tracking
        self.generation_active = [False]
        self.last_frame_time = [time.monotonic()]

        # Degraded flag
        self.server_degraded = False

        # Start deadlock watchdog thread
        self._watchdog_thread = threading.Thread(
            target=self._deadlock_watchdog, daemon=True, name="deadlock-watchdog"
        )
        self._watchdog_thread.start()

    def _auto_disable_megakernel(self):
        """Disable predictor megakernel at runtime after repeated failures."""
        if self.engine.mk_predictor is not None:
            print(
                f"AUTO-FALLBACK: Disabling megakernel predictor after "
                f"{self.MK_FAILURE_THRESHOLD} runtime failures",
                flush=True,
            )
            self.engine.mk_predictor = None
            self.server_degraded = True

    def health_status(self):
        """Compute health status: ok / degraded / unhealthy."""
        if self.generation_active[0] and (
            time.monotonic() - self.last_frame_time[0]
        ) > self.FRAME_TIMEOUT:
            return "unhealthy"  # Likely deadlocked right now
        if (
            self.server_degraded
            or self.mk_failure_count >= self.MK_FAILURE_THRESHOLD
        ):
            return "degraded"  # Megakernel was auto-disabled
        return "ok"

    def ttfp_percentiles(self):
        """Compute TTFP percentiles from recent history."""
        if not self._ttfp_history:
            return {}
        s = sorted(self._ttfp_history)
        n = len(s)
        return {
            "p50": round(s[n // 2], 1),
            "p90": round(s[int(n * 0.9)], 1),
            "p99": round(s[min(int(n * 0.99), n - 1)], 1),
            "min": round(s[0], 1),
            "max": round(s[-1], 1),
            "samples": n,
        }

    def record_ttfp(self, ttfp_ms):
        """Record a TTFP measurement for percentile tracking."""
        if ttfp_ms > 0:
            self._ttfp_history.append(ttfp_ms)
            if len(self._ttfp_history) > self._TTFP_HISTORY_MAX:
                self._ttfp_history.pop(0)

    def _deadlock_watchdog(self):
        """Background thread: force-exit if generation is stuck beyond timeout.

        This catches cases where the async timeout (couche 1) is bypassed,
        e.g. if the event loop itself is blocked or the executor thread
        holds the GIL during a CUDA deadlock.
        """
        GRACE = 5  # extra seconds beyond FRAME_TIMEOUT before hard kill
        while True:
            time.sleep(2)
            if self.generation_active[0]:
                elapsed = time.monotonic() - self.last_frame_time[0]
                if elapsed > self.FRAME_TIMEOUT + GRACE:
                    print(
                        f"WATCHDOG: Generation stuck {elapsed:.0f}s "
                        f"(limit {self.FRAME_TIMEOUT + GRACE:.0f}s). Force exit.",
                        flush=True,
                    )
                    os._exit(1)  # Hard exit — start.sh restart loop picks up
