"""Plain-line console helpers used by the fake-world test actuator."""
from __future__ import annotations

import sys
import time


class ConsolePrompt:
    """Line-based confirmation and throttled progress for command handlers."""

    def __init__(self):
        self._last_progress = 0.0

    def confirm(self, prompt: str, default: bool = False, body: list[str] | None = None) -> bool:
        for line in body or []:
            print(line)
        if not sys.stdin.isatty():
            print(f'확인 필요: {prompt} (비대화형 입력이라 취소)', file=sys.stderr)
            return False
        suffix = 'Y/n' if default else 'y/N'
        try:
            answer = input(f'{prompt} [{suffix}] ').strip().lower()
        except EOFError:
            return False
        if not answer:
            return default
        if answer in ('y', 'yes'):
            return True
        if answer in ('n', 'no'):
            return False
        print('y 또는 n을 입력하세요', file=sys.stderr)
        return False

    def step(self, text: str) -> None:
        print(text, file=sys.stderr, flush=True)

    def progress(self, text: str) -> None:
        now = time.monotonic()
        if now - self._last_progress < 1.0:
            return
        self._last_progress = now
        print(text, file=sys.stderr, flush=True)
