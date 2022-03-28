from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List


@dataclass
class Extraction:
    start: int
    end: int
    type: str
    text: str
    confidence: float = 0.


@dataclass
class NeToken:
    start: int
    end: int
    text: str
    bio: str
    confidence: float


class NeExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> List[Extraction]:
        pass
