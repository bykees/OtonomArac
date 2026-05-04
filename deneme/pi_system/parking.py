from typing import Optional

from .config import VehicleConfig
from .types import ParkingResult, SignResult


class ParkingController:
    def __init__(self, config: VehicleConfig):
        self.config = config
        self.park_mode = False

    def process(self, sign: Optional[SignResult]) -> ParkingResult:
        if sign is not None and sign.label.startswith("park_yeri"):
            self.park_mode = True

        if not self.park_mode:
            return ParkingResult(
                active=False,
                phase="IDLE",
                target_speed=self.config.cruise_speed,
                target_turn=0.0,
            )

        return ParkingResult(
            active=True,
            phase="SEARCH_SLOT",
            target_speed=self.config.parking_speed,
            target_turn=0.15,
        )
