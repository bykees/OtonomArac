from .types import ControlCommand, PerceptionSnapshot


class VehicleStateMachine:
    def __init__(self):
        self.state = "LANE_FOLLOW"

    def update(self, snapshot: PerceptionSnapshot) -> ControlCommand:
        if snapshot.traffic_light and snapshot.traffic_light.state == "KIRMIZI":
            self.state = "STOP_AT_RED"
            return ControlCommand(
                mode=self.state,
                speed=0.0,
                steering=0.0,
                reason="Kirmizi isik algilandi",
            )

        if snapshot.parking and snapshot.parking.active:
            self.state = "PARKING"
            return ControlCommand(
                mode=self.state,
                speed=snapshot.parking.target_speed,
                steering=snapshot.parking.target_turn,
                reason=f"Park modu: {snapshot.parking.phase}",
            )

        if snapshot.sign and snapshot.sign.label != "YOK" and snapshot.sign.confidence >= 0.40:
            self.state = "SIGN_REACTION"
            return ControlCommand(
                mode=self.state,
                speed=0.35,
                steering=0.0,
                reason=f"Tabela algilandi: {snapshot.sign.label}",
            )

        self.state = "LANE_FOLLOW"
        lane = snapshot.lane
        steering = 0.0
        if lane is not None and lane.frame_center_x != 0:
            steering = max(-1.0, min(1.0, lane.steering_error / 120.0))

        return ControlCommand(
            mode=self.state,
            speed=0.45,
            steering=steering,
            reason="Normal serit takibi",
        )
