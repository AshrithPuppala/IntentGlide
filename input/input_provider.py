import time
import pygame
from shared.contracts import InputSample

class MouseProvider:
    """Person 4 owns pygame.init(), the window and the event loop."""
    def get_sample(self) -> InputSample:
        x, y = pygame.mouse.get_pos()
        return InputSample(timestamp=time.monotonic(),
                           x=float(x), y=float(y), source="mouse")

class InputManager:
    def __init__(self, **sources):
        self.sources = sources            # e.g. synthetic=..., mouse=...
        self.active = next(iter(sources))

    def set_source(self, name: str):
        if name not in self.sources:
            raise KeyError(f"Unknown source: {name}")
        self.active = name

    def get_sample(self) -> InputSample:
        return self.sources[self.active].get_sample()
