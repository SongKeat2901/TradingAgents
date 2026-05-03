"""Stdout progress reporter for tradingresearch.

Each node-completion line is consumed by an OpenClaw skill agent and
forwarded to Telegram (rate-limited on the OpenClaw side).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import IO


@dataclass
class ProgressCallback:
    stream: IO[str] = field(default_factory=lambda: sys.stdout)
    _node_starts: dict[str, float] = field(default_factory=dict)

    def on_node_start(self, node: str) -> None:
        self._node_starts[node] = time.monotonic()
        print(f"[{node}] start", file=self.stream, flush=True)

    def on_node_done(self, node: str, duration_s: float | None = None) -> None:
        if duration_s is None:
            started = self._node_starts.get(node, time.monotonic())
            duration_s = time.monotonic() - started
        print(f"[{node}] done ({duration_s:.1f}s)", file=self.stream, flush=True)
