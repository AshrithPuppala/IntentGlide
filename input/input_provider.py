"""Input source interfaces and Pygame mouse event adapter."""
from __future__ import annotations

import time
from typing import Iterable, Optional

from shared.contracts import InputSample


class InputProvider:
    """Minimal provider interface used by the application."""

    def poll(self) -> list[InputSample]:
        raise NotImplementedError


class PygameMouseProvider(InputProvider):
    """Convert Pygame mouse-motion events into timestamped samples.

    Call poll(events) with the event batch already obtained by the UI loop.
    Samples are timestamped as consumed; the physical mouse rate is not assumed.
    """

    def __init__(self):
        self._last_position: Optional[tuple[int, int]] = None

    def poll(self, events: Optional[Iterable] = None) -> list[InputSample]:
        if events is None:
            try:
                import pygame
            except ImportError as exc:
                raise RuntimeError("Install pygame to use mouse input") from exc
            events = pygame.event.get()

        samples = []
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("Install pygame to use mouse input") from exc

        for event in events:
            if event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                samples.append(
                    InputSample(
                        timestamp=time.monotonic(),
                        x=float(x),
                        y=float(y),
                        source="mouse",
                    )
                )
                self._last_position = (x, y)
        return samples
