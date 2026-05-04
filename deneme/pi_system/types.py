from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class LaneResult:
    found: bool
    lane_center_x: Optional[int]
    frame_center_x: int
    steering_error: int
    direction_hint: str
    debug_mask: Optional[np.ndarray] = None


@dataclass
class TrafficLightResult:
    state: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class SignResult:
    label: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    kind: str = "YOK"
    symbol_mask: Optional[np.ndarray] = None
    area: int = 0
    match_score: float = 0.0


@dataclass
class ParkingResult:
    active: bool
    phase: str
    target_speed: float
    target_turn: float


@dataclass
class ControlCommand:
    mode: str
    speed: float
    steering: float
    reason: str


@dataclass
class PerceptionSnapshot:
    lane: Optional[LaneResult] = None
    traffic_light: Optional[TrafficLightResult] = None
    sign: Optional[SignResult] = None
    parking: Optional[ParkingResult] = None
    frame_index: int = 0
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)
