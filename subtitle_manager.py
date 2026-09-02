"""
subtitle_manager.py
-------------------
Holds subtitle state: the latest original/translated pair plus a rolling
history of everything said this session.

It does NOT touch the GUI directly (that stays on the main thread). The
pipeline updates this manager and then tells the GUI what to show. Keeping the
data here means we can later export the transcript (e.g. to CSV with pandas)
without changing anything else.

A small lock makes it safe to read/write from the worker threads.
"""

import threading
import time


class SubtitleManager:
    def __init__(self, max_history=200):
        self._lock = threading.Lock()
        self._history = []          # list of entry dicts
        self._max_history = max_history
        self._seq = 0               # fallback ordering when no seq is supplied

    def add(self, original, translated, detected_name="", seq=None):
        """
        Record a new translated line and return the entry.

        `seq` orders the line in the transcript. The web app processes chunks
        concurrently, so they can FINISH out of order; passing the moment the
        chunk was received (which is in speech order) keeps the exported
        transcript in the order things were actually said. Callers that don't
        care get insertion order.
        """
        with self._lock:
            self._seq += 1
            entry = {
                "time": time.strftime("%H:%M:%S"),
                "original": original or "",
                "translated": translated or "",
                "detected": detected_name or "",
                "seq": self._seq if seq is None else seq,
            }
            self._history.append(entry)
            # Trim old entries so memory stays bounded during long sessions.
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        return entry

    def _ordered(self):
        """History in speech order. Caller must hold the lock."""
        return sorted(self._history, key=lambda e: e["seq"])

    def current(self):
        """Return the most recent entry, or None."""
        with self._lock:
            return self._ordered()[-1] if self._history else None

    def history(self):
        """Return a copy of the full history (safe to iterate)."""
        with self._lock:
            return self._ordered()

    def clear(self):
        with self._lock:
            self._history.clear()

    def to_rows(self):
        """
        Return history as a list of (time, detected, original, translated)
        tuples - convenient for building a pandas DataFrame / CSV export later.
        """
        with self._lock:
            return [(e["time"], e["detected"], e["original"], e["translated"])
                    for e in self._ordered()]
