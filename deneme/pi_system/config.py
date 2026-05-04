from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    camera_index: int = 0
    use_picamera2: bool = True
    use_directshow: bool = False
    fps: int = 30
    buffer_size: int = 1
    color_format: str = "RGB888"
    color_conversion: str = "NONE"
    flip_horizontal: bool = False
    flip_vertical: bool = False


@dataclass
class SchedulerConfig:
    lane_every_n_frames: int = 1
    light_every_n_frames: int = 2
    sign_every_n_frames: int = 2
    parking_every_n_frames: int = 2


@dataclass
class DetectionConfig:
    lane_roi_start_ratio: float = 0.62
    lane_white_min_value: int = 115
    lane_white_max_saturation: int = 125
    lane_adaptive_std_coef: float = 0.45
    lane_black_max_value: int = 115
    lane_min_dark_context_ratio: float = 0.12
    lane_roi_top_width_ratio: float = 0.58
    lane_roi_bottom_width_ratio: float = 0.96
    red_light_roi_end_ratio: float = 0.55
    min_sign_area_ratio: float = 0.010
    min_red_light_area: int = 120
    min_red_light_fill_ratio: float = 0.45
    max_red_light_center_offset_ratio: float = 0.42
    sign_match_threshold: float = 0.50
    sign_strong_accept_score: float = 0.65
    sign_min_accept_score: float = 0.42
    sign_ambiguous_margin: float = 0.10
    sign_vote_threshold: float = 0.40
    sign_min_match_margin: float = 0.04
    sign_min_geometry_score: float = 0.55
    sign_min_symbol_ratio: float = 0.020
    sign_max_symbol_ratio: float = 0.42
    sign_min_border_ratio: float = 0.030
    sign_vote_window: int = 7
    sign_vote_min_count: int = 3
    sign_template_size: Tuple[int, int] = (96, 96)
    lane_min_area: int = 120
    lane_max_area_ratio: float = 0.18
    lane_min_height_ratio: float = 0.08
    lane_max_width_ratio: float = 0.58
    lane_min_aspect_ratio: float = 1.25
    lane_max_center_jump: int = 85
    lane_default_width: int = 180
    lane_turn_threshold: int = 28
    lane_smooth_alpha: float = 0.22
    sign_confirm_frames: int = 3
    sign_hold_frames: int = 2
    sign_debug_terminal: bool = True
    sign_debug_interval: int = 10
    sign_red_min_saturation: int = 120
    sign_red_min_value: int = 110
    sign_blue_min_saturation: int = 70
    sign_blue_min_value: int = 50
    traffic_red_min_saturation: int = 150
    traffic_red_min_value: int = 150
    lane_histogram_rows: int = 12


@dataclass
class DebugConfig:
    enabled: bool = True
    show_mask: bool = True
    overlay_symbol_preview: bool = True


@dataclass
class VehicleConfig:
    cruise_speed: float = 0.45
    turn_speed: float = 0.35
    stop_speed: float = 0.0
    parking_speed: float = 0.25


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
