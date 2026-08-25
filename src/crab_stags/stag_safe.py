"""Crash-safe wrapper around stag.detectMarkers.

stag-python 1.1.1 segfaults on some real-world frames: the edge-drawing stage
(JoinAnchorPointsUsingSortedAnchors, via EDPF/EDLines) overruns a fixed-size
buffer when a frame produces too many edge segments. It takes the whole
interpreter with it, so in a notebook the kernel dies mid-loop and the capture
device is left open. Known upstream and unfixed as of 1.1.1:
https://github.com/manfredstoiber/stag-python/issues/8

The crash is content-dependent, not resolution-dependent: downscaling changes
how often it happens but does not order it (on one 12-frame sample, 0.8x
crashed more often than 1.0x). So there is no input transform to trust here.
Instead we run the detector in a separate process. When it dies we lose that
one frame, respawn, and carry on.

Usage mirrors stag.detectMarkers:

    from crab_stags.stag_safe import StagDetector

    with StagDetector(libraryHD=15) as detector:
        while True:
            ret, image = cap.read()
            corners, ids, rejected = detector.detect(image)

A frame that kills the worker comes back as an empty detection, exactly like a
frame with nothing in it, and its index is recorded in ``detector.skipped``.
Check that list rather than assuming an empty result means an empty scene.

Restarting costs roughly 200 ms (a new interpreter plus ``import stag``), paid
only on frames that actually crash.

Why subprocess and not multiprocessing
--------------------------------------
multiprocessing's "spawn" re-imports the parent's __main__ in the child. In
VS Code's interactive window __main__ is the notebook .py file itself, so the
child re-runs the whole notebook -- reopening the camera and recursively
spawning again. subprocess exec'ing this file directly has no main-module
fixup, so it behaves identically from a notebook, a script, or a REPL, with no
``if __name__ == "__main__"`` guard required of the caller.

Frames go to the worker over a length-prefixed pickle protocol on its stdin,
and results come back the same way on a private copy of its stdout. The child
points sys.stdout at stderr immediately, so a stray print (or the objc
duplicate-class warnings) cannot corrupt the stream.
"""

from __future__ import annotations

import os
import pickle
import select
import struct
import subprocess
import sys

import numpy as np

__all__ = ["StagDetector", "EMPTY_DETECTION"]

# What stag.detectMarkers returns for a frame containing no markers. Matching it
# exactly means a skipped frame flows through caller code on the ordinary path.
EMPTY_DETECTION = ((), np.empty((0, 1), dtype=np.int32), ())

_HEADER = struct.Struct("<Q")


def _read_exactly(stream, n):
    """Read n bytes, or raise EOFError if the stream ends first."""
    chunks = []
    remaining = n
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("worker closed the pipe")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send(stream, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _recv(stream):
    (length,) = _HEADER.unpack(_read_exactly(stream, _HEADER.size))
    return pickle.loads(_read_exactly(stream, length))


class StagDetector:
    """Detect STag markers in a subprocess, surviving detector segfaults.

    Parameters
    ----------
    libraryHD : int
        STag library/family, one of [11, 13, 15, 17, 19, 21, 23].
    errorCorrection : int, optional
        Passed through to stag.detectMarkers; -1 selects the library maximum.
    timeout : float
        Seconds to wait for one frame's result. Guards against a hang rather
        than a crash (a crash is detected immediately, via the closed pipe).
        Generous by default: this is a deadlock backstop, not a latency budget.
        Implemented with select() on a pipe, which is POSIX-only; on Windows
        this would need a reader thread instead. Everything else here is
        portable.
    max_restarts : int or None
        Give up after this many worker deaths, so a systematically fatal input
        cannot spin forever. None for no limit.
    verbose : bool
        Print a line each time a worker dies.
    """

    def __init__(
        self,
        libraryHD,
        errorCorrection=-1,
        timeout=30.0,
        max_restarts=None,
        verbose=True,
    ):
        self.libraryHD = libraryHD
        self.errorCorrection = errorCorrection
        self.timeout = timeout
        self.max_restarts = max_restarts
        self.verbose = verbose

        # Indices (into the sequence of detect() calls) whose frame killed the
        # worker, and how many restarts that has cost us so far.
        self.skipped = []
        self.restarts = 0

        self._calls = 0
        self._proc = None

    # -- lifecycle ---------------------------------------------------------

    def _start(self):
        self._proc = subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                str(self.libraryHD),
                str(self.errorCorrection),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr is inherited, so the worker's own noise lands in the same
            # place as ours instead of silently filling a pipe buffer.
        )

    def _reap(self):
        """Tear down a worker that has died or is about to be replaced."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return None

        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                try:
                    stream.close()
                except (BrokenPipeError, OSError):
                    pass
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
        return proc.returncode

    def close(self):
        """Shut the worker down cleanly. Safe to call more than once."""
        self._reap()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def __del__(self):
        # Best effort: a notebook that never reaches close() still shouldn't
        # leave an orphan behind.
        try:
            self.close()
        except Exception:
            pass

    # -- detection ---------------------------------------------------------

    def detect(self, image):
        """Detect markers in ``image``.

        Returns (corners, ids, rejected), as stag.detectMarkers does. If the
        worker dies on this frame, returns EMPTY_DETECTION and records the call
        index in ``self.skipped``.
        """
        call_idx = self._calls
        self._calls += 1

        if self._proc is None:
            self._start()

        try:
            _send(self._proc.stdin, image)
            # Only the wait for the first byte is bounded; once the worker has
            # started answering, the rest of the message follows immediately.
            ready, _, _ = select.select([self._proc.stdout], [], [], self.timeout)
            if not ready:
                raise TimeoutError(f"worker did not answer within {self.timeout}s")
            return _recv(self._proc.stdout)
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError, TimeoutError) as exc:
            self._on_worker_lost(call_idx, exc)
            return EMPTY_DETECTION

    def _on_worker_lost(self, call_idx, exc):
        returncode = self._reap()
        self.skipped.append(call_idx)
        self.restarts += 1

        if self.verbose:
            # A segfault surfaces as returncode -11 (killed by SIGSEGV), which
            # is the signature of the upstream buffer overrun. Anything else --
            # notably a positive exit code -- means the worker failed for some
            # other reason and is worth investigating rather than ignoring.
            detail = f"returncode {returncode}" if returncode is not None else str(exc)
            note = "" if returncode == -11 else "  <- not a segfault, check stderr"
            print(
                f"stag_safe: detector died on frame {call_idx} ({detail}); "
                f"skipping it and restarting [{self.restarts} so far]{note}"
            )

        if self.max_restarts is not None and self.restarts > self.max_restarts:
            raise RuntimeError(
                f"stag detector died {self.restarts} times, exceeding "
                f"max_restarts={self.max_restarts}. The input is probably "
                f"triggering the upstream crash on nearly every frame."
            )


# -- worker ----------------------------------------------------------------
# Reached only when this file is exec'd as a script by _start() above.


def _worker_main(library_hd, error_correction):
    # Take a private copy of stdout for the protocol, then point fd 1 (and
    # sys.stdout) at stderr, so nothing else can write into the data stream.
    data_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    data_out = os.fdopen(data_fd, "wb")
    data_in = sys.stdin.buffer

    import stag

    try:
        while True:
            image = _recv(data_in)
            _send(data_out, stag.detectMarkers(image, library_hd, error_correction))
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        # Parent went away or interrupted; exiting quietly is the right response.
        pass


if __name__ == "__main__":
    _worker_main(int(sys.argv[1]), int(sys.argv[2]))
