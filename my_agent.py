from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import threading
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
from PIL import Image, ImageDraw

try:
    from arcengine import GameAction, GameState
except Exception:
    class GameState(str, Enum):
        NOT_PLAYED = "NOT_PLAYED"
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"

    class GameAction(Enum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

        @classmethod
        def from_id(cls, value: int) -> "GameAction":
            for action in cls:
                if int(action.value) == int(value):
                    return action
            raise ValueError(value)

        def set_data(self, data: dict[str, Any]) -> "ActionWithData":
            return ActionWithData(self, dict(data))


@dataclass
class ActionWithData:
    id: Any
    action_data: dict[str, int]
    reasoning: dict[str, Any] = field(default_factory=dict)


try:
    from agents.agent import Agent as _BaseAgent  # type: ignore
except Exception:
    try:
        from arc_agi.agent import Agent as _BaseAgent  # type: ignore
    except Exception:
        class _BaseAgent:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.arc_env = kwargs.get("arc_env")
                self.game_id = kwargs.get("game_id", "unknown")

            @property
            def name(self) -> str:
                return type(self).__name__


PALETTE_16 = np.asarray([
    (0, 0, 0), (230, 25, 75), (60, 180, 75), (255, 225, 25),
    (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
    (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128),
    (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 128, 128),
], dtype=np.uint8)


def _short(value: Any, limit: int = 300) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()[:limit]


def _bool_env(names: Sequence[str], default: bool) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _int_env(names: Sequence[str], default: int, low: int | None = None, high: int | None = None) -> int:
    value = default
    for name in names:
        raw = os.getenv(name)
        if raw not in (None, ""):
            try:
                value = int(raw)
            except Exception:
                value = default
            break
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _float_env(names: Sequence[str], default: float, low: float | None = None, high: float | None = None) -> float:
    value = default
    for name in names:
        raw = os.getenv(name)
        if raw not in (None, ""):
            try:
                value = float(raw)
            except Exception:
                value = default
            break
    if not math.isfinite(value):
        value = default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _str_env(names: Sequence[str], default: str | None = None) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw not in (None, ""):
            return raw
    return default


def _clamp01(value: Any) -> float:
    try:
        f = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, f)) if math.isfinite(f) else 0.0


@dataclass(frozen=True)
class AgentConfig:
    agent_version: str = "v2.0-outcome-aware-frontier"
    enable_vlm: bool = True
    model_path: str | None = None
    image_size: int = 512
    max_actions: int = 240
    max_vlm_calls_per_level: int = 24
    vlm_min_action_gap: int = 2
    vlm_max_new_tokens: int = 1800
    max_chunk_steps: int = 4
    max_plan_steps: int = 32
    max_prompt_objects: int = 32
    navigation_max_depth: int = 96
    nonterminal_reset_allowed: bool = False
    log_dir: str | None = "./runs/agent_v2_0"
    log_vlm_io: bool = False
    vlm_backend: str = "local"
    vlm_api_base_url: str | None = None
    vlm_api_key: str | None = None
    vlm_api_model: str | None = None
    vlm_api_timeout_s: float = 60.0

    avoid_action7: bool = True
    action7_noop_quarantine_after: int = 1
    global_noop_quarantine_after: int = 6
    state_noop_quarantine_after: int = 1
    action_repeat_cooldown: int = 2
    quarantine_steps: int = 9
    low_resource_ratio: float = 0.25
    critical_resource_ratio: float = 0.12
    max_click_points_per_object: int = 14
    max_object_click_noops: int = 5
    max_clicks_per_coord_total: int = 4
    max_clicks_per_region_total: int = 12
    transform_mode_threshold: int = 2
    max_same_transform_action: int = 2
    max_same_click_run: int = 3
    max_vlm_consecutive_same_action: int = 4
    max_vlm_same_click_target: int = 2
    resource_plan_max_steps: int = 4
    vlm_repeat_bottleneck_cooldown: int = 3
    force_source_fuse_after: int = 20
    churn_click_fuse_after: int = 4
    churn_state_visit_cap: int = 3
    # Pace the per-level VLM budget across the whole episode instead of letting
    # bottleneck storms burn it in the first ~25 actions (ls20/ft09 pattern).
    vlm_budget_burst: int = 4
    # Cells that change in >= this fraction of observed frame-to-frame transitions
    # are treated as self-drifting animation and excluded from the navigation hash.
    drift_cell_ratio: float = 0.6
    escape_commit_max_run: int = 4
    # 阶段性决策：ACTION7 视为回退、完全回避（先追求通关）。置 1 可恢复
    # “小词表且声明含 ACTION7 时允许 2 次有界探测”（bp35 曾探出 transform 机制）。
    action7_space_probe: bool = False
    # V2.7: do not permanently lock geometry navigation until the attempt has had
    # enough directed exploration. ls20 previously escalated ~step 30 (fuel still
    # high) off a short transform_state_cycle and then rejected geometry for the
    # rest of the level. Direct escalate() calls / tests can set total_action_count.
    min_hypothesis_escalation_actions: int = 48

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            enable_vlm=_bool_env(("ARC_V20_ENABLE_VLM", "ARC_V19_ENABLE_VLM", "ARC_V18_ENABLE_VLM", "ARC_V15_ENABLE_VLM"), True),
            model_path=_str_env(("ARC_V20_MODEL_PATH", "ARC_V19_MODEL_PATH", "ARC_V18_MODEL_PATH", "ARC_V15_MODEL_PATH", "QWEN_MODEL_PATH")),
            image_size=_int_env(("ARC_V20_IMAGE_SIZE", "ARC_V19_IMAGE_SIZE", "ARC_V18_IMAGE_SIZE", "ARC_V15_IMAGE_SIZE"), 512, 64, 1024),
            max_actions=_int_env(("ARC_V20_MAX_ACTIONS", "ARC_V19_MAX_ACTIONS", "ARC_V18_MAX_ACTIONS", "ARC_V15_MAX_ACTIONS"), 240, 1, 10000),
            max_vlm_calls_per_level=_int_env(("ARC_V20_VLM_CALLS_PER_LEVEL", "ARC_V19_VLM_CALLS_PER_LEVEL", "ARC_V18_VLM_CALLS_PER_LEVEL", "ARC_V15_VLM_CALLS_PER_LEVEL"), 24, 0, 1000),
            vlm_min_action_gap=_int_env(("ARC_V20_VLM_MIN_ACTION_GAP", "ARC_V19_VLM_MIN_ACTION_GAP", "ARC_V18_VLM_MIN_ACTION_GAP", "ARC_V15_VLM_MIN_ACTION_GAP"), 2, 0, 32),
            vlm_max_new_tokens=_int_env(("ARC_V20_VLM_MAX_NEW_TOKENS", "ARC_V19_VLM_MAX_NEW_TOKENS", "ARC_V18_VLM_MAX_NEW_TOKENS", "ARC_V15_VLM_MAX_NEW_TOKENS"), 1800, 256, 4096),
            max_chunk_steps=_int_env(("ARC_V20_MAX_CHUNK_STEPS", "ARC_V19_MAX_CHUNK_STEPS", "ARC_V18_MAX_CHUNK_STEPS"), 4, 1, 12),
            max_plan_steps=_int_env(("ARC_V20_MAX_PLAN_STEPS", "ARC_V19_MAX_PLAN_STEPS", "ARC_V18_MAX_PLAN_STEPS"), 32, 1, 32),
            max_prompt_objects=_int_env(("ARC_V20_MAX_PROMPT_OBJECTS", "ARC_V19_MAX_PROMPT_OBJECTS", "ARC_V18_MAX_PROMPT_OBJECTS"), 32, 4, 40),
            navigation_max_depth=_int_env(("ARC_V20_NAVIGATION_MAX_DEPTH", "ARC_V19_NAVIGATION_MAX_DEPTH", "ARC_V18_NAVIGATION_MAX_DEPTH", "ARC_V15_NAVIGATION_MAX_DEPTH"), 96, 8, 512),
            nonterminal_reset_allowed=_bool_env(("ARC_V20_ALLOW_NONTERMINAL_RESET", "ARC_V19_ALLOW_NONTERMINAL_RESET", "ARC_V18_ALLOW_NONTERMINAL_RESET"), False),
            log_dir=_str_env(("ARC_V20_LOG_DIR", "ARC_V19_LOG_DIR", "ARC_V18_LOG_DIR", "ARC_V15_LOG_DIR"), "./runs/agent_v2_0"),
            log_vlm_io=_bool_env(("ARC_V20_LOG_VLM_IO",), False),
            vlm_backend=_str_env(("ARC_V20_VLM_BACKEND", "ARC_V19_VLM_BACKEND", "ARC_V18_VLM_BACKEND", "ARC_V15_VLM_BACKEND"), "local") or "local",
            vlm_api_base_url=_str_env(("ARC_V20_VLM_API_BASE_URL", "ARC_V19_VLM_API_BASE_URL", "ARC_V18_VLM_API_BASE_URL", "ARC_V15_VLM_API_BASE_URL")),
            vlm_api_key=_str_env(("ARC_V20_VLM_API_KEY", "ARC_V19_VLM_API_KEY", "ARC_V18_VLM_API_KEY", "ARC_V15_VLM_API_KEY", "INF_API_KEY")),
            vlm_api_model=_str_env(("ARC_V20_VLM_API_MODEL", "ARC_V19_VLM_API_MODEL", "ARC_V18_VLM_API_MODEL", "ARC_V15_VLM_API_MODEL")),
            vlm_api_timeout_s=_float_env(("ARC_V20_VLM_API_TIMEOUT_S", "ARC_V19_VLM_API_TIMEOUT_S", "ARC_V18_VLM_API_TIMEOUT_S", "ARC_V15_VLM_API_TIMEOUT_S"), 60.0, 1.0, 600.0),
            avoid_action7=_bool_env(("ARC_V20_AVOID_ACTION7", "ARC_V19_AVOID_ACTION7"), True),
            action7_noop_quarantine_after=_int_env(("ARC_V20_ACTION7_NOOP_QUARANTINE_AFTER",), 1, 1, 20),
            global_noop_quarantine_after=_int_env(("ARC_V20_GLOBAL_NOOP_QUARANTINE_AFTER",), 6, 2, 100),
            state_noop_quarantine_after=_int_env(("ARC_V20_STATE_NOOP_QUARANTINE_AFTER",), 1, 1, 20),
            action_repeat_cooldown=_int_env(("ARC_V20_ACTION_REPEAT_COOLDOWN",), 2, 0, 20),
            quarantine_steps=_int_env(("ARC_V20_QUARANTINE_STEPS",), 9, 1, 80),
            low_resource_ratio=_float_env(("ARC_V20_LOW_RESOURCE_RATIO",), 0.25, 0.01, 0.95),
            critical_resource_ratio=_float_env(("ARC_V20_CRITICAL_RESOURCE_RATIO",), 0.12, 0.01, 0.95),
            max_click_points_per_object=_int_env(("ARC_V20_MAX_CLICK_POINTS_PER_OBJECT",), 14, 4, 40),
            max_object_click_noops=_int_env(("ARC_V20_MAX_OBJECT_CLICK_NOOPS",), 5, 1, 40),
            max_clicks_per_coord_total=_int_env(("ARC_V20_MAX_CLICKS_PER_COORD_TOTAL",), 4, 1, 100),
            max_clicks_per_region_total=_int_env(("ARC_V20_MAX_CLICKS_PER_REGION_TOTAL",), 12, 1, 200),
            transform_mode_threshold=_int_env(("ARC_V20_TRANSFORM_MODE_THRESHOLD",), 2, 1, 20),
            max_same_transform_action=_int_env(("ARC_V20_MAX_SAME_TRANSFORM_ACTION",), 2, 1, 20),
            max_same_click_run=_int_env(("ARC_V20_MAX_SAME_CLICK_RUN",), 3, 1, 20),
            max_vlm_consecutive_same_action=_int_env(("ARC_V20_MAX_VLM_CONSECUTIVE_SAME_ACTION",), 4, 1, 24),
            max_vlm_same_click_target=_int_env(("ARC_V20_MAX_VLM_SAME_CLICK_TARGET",), 2, 1, 12),
            resource_plan_max_steps=_int_env(("ARC_V20_RESOURCE_PLAN_MAX_STEPS",), 4, 1, 16),
            vlm_repeat_bottleneck_cooldown=_int_env(("ARC_V20_VLM_REPEAT_BOTTLENECK_COOLDOWN",), 3, 1, 20),
            force_source_fuse_after=_int_env(("ARC_V20_FORCE_SOURCE_FUSE_AFTER",), 20, 4, 200),
            churn_click_fuse_after=_int_env(("ARC_V20_CHURN_CLICK_FUSE_AFTER",), 4, 2, 50),
            churn_state_visit_cap=_int_env(("ARC_V20_CHURN_STATE_VISIT_CAP",), 3, 2, 50),
            vlm_budget_burst=_int_env(("ARC_V20_VLM_BUDGET_BURST",), 4, 1, 100),
            drift_cell_ratio=_float_env(("ARC_V20_DRIFT_CELL_RATIO",), 0.6, 0.2, 1.0),
            escape_commit_max_run=_int_env(("ARC_V20_ESCAPE_COMMIT_MAX_RUN",), 4, 1, 20),
            action7_space_probe=_bool_env(("ARC_V20_ACTION7_SPACE_PROBE",), False),
            min_hypothesis_escalation_actions=_int_env(("ARC_V20_MIN_HYPOTHESIS_ESCALATION_ACTIONS",), 48, 0, 200),
        )


class ObservationError(RuntimeError):
    pass


class VLMMode(str, Enum):
    LEVEL_INIT = "LEVEL_INIT"
    EVALUATE_CHUNK = "EVALUATE_CHUNK"
    BOTTLENECK = "BOTTLENECK"
    SUCCESS_REFLECT = "SUCCESS_REFLECT"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [_json_safe(v) for v in list(value)]
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        return {str(k): _json_safe(getattr(value, k)) for k in value.__dataclass_fields__}
    return value


def _vector_key(vec: tuple[int, int]) -> str:
    return f"{int(vec[0])},{int(vec[1])}"


def _vector_from_key(key: str) -> tuple[int, int] | None:
    try:
        left, right = str(key).split(",", 1)
        return int(left), int(right)
    except Exception:
        return None


def _vector_axis_signature(vec: tuple[int, int] | list[int] | None) -> tuple[str, int, int]:
    if not vec or len(vec) != 2:
        return ("none", 0, 0)
    dx, dy = int(vec[0]), int(vec[1])
    sx = 1 if dx > 0 else -1 if dx < 0 else 0
    sy = 1 if dy > 0 else -1 if dy < 0 else 0
    if dx and not dy:
        return ("x", sx, 0)
    if dy and not dx:
        return ("y", 0, sy)
    if abs(dx) >= abs(dy) * 2 and dx:
        return ("x", sx, 0)
    if abs(dy) >= abs(dx) * 2 and dy:
        return ("y", 0, sy)
    return ("diag", sx, sy)


def _axis_aligned_vector(vec: tuple[int, int] | list[int] | None) -> bool:
    if not vec or len(vec) != 2:
        return False
    dx, dy = int(vec[0]), int(vec[1])
    return bool(dx == 0 and dy != 0) or bool(dy == 0 and dx != 0)


def _dominant_vector(votes: Counter[str]) -> tuple[tuple[int, int], int] | None:
    for key, count in votes.most_common():
        vec = _vector_from_key(key)
        if vec is not None:
            return vec, int(count)
    return None


_LOCAL_OBJECT_ID_RE = re.compile(r"\bO\d+\b")


def _strip_local_object_ids(value: Any) -> Any:
    if isinstance(value, str):
        return _LOCAL_OBJECT_ID_RE.sub("a local object", value)
    if isinstance(value, dict):
        return {(_strip_local_object_ids(k) if isinstance(k, str) else k): _strip_local_object_ids(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [_strip_local_object_ids(v) for v in list(value)]
    return value


def _stable_hash(grid: tuple[tuple[int, ...], ...]) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(bytes([len(grid) % 256, (len(grid[0]) if grid else 0) % 256]))
    for row in grid:
        h.update(bytes(int(v) % 256 for v in row))
    return h.hexdigest()


def _bbox_area(b: tuple[int, int, int, int]) -> int:
    return max(0, b[2] - b[0] + 1) * max(0, b[3] - b[1] + 1)


def _bbox_inter(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)


def _bbox_contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    gx = max(0, max(a[0], b[0]) - min(a[2], b[2]) - 1)
    gy = max(0, max(a[1], b[1]) - min(a[3], b[3]) - 1)
    return gx, gy


def _hex(c: int) -> str:
    return "0123456789ABCDEF"[int(c) % 16]


def render_grid(grid: Sequence[Sequence[int]], size: int = 512) -> Image.Image:
    arr = np.asarray(grid, dtype=np.uint8)
    img = Image.fromarray(PALETTE_16[arr % 16].astype(np.uint8), mode="RGB")
    return img.resize((size, size), Image.Resampling.NEAREST)


@dataclass(frozen=True)
class ComponentObservation:
    color: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    touches_border: bool
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ObjectObservation:
    track_id: str
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int
    colors: tuple[int, ...]
    color_areas: tuple[tuple[int, int], ...]
    component_count: int
    cells: tuple[tuple[int, int, int], ...]
    intrinsic_signature: str
    shape_label: str
    pattern: str
    inner_pattern: str
    frame_color: int | None
    near_edge: bool
    salience: float

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def type_key(self) -> str:
        return self.intrinsic_signature[:16]


@dataclass
class SceneSnapshot:
    grid: tuple[tuple[int, ...], ...]
    world_grid: tuple[tuple[int, ...], ...]
    width: int
    height: int
    state_hash: str
    full_state_hash: str
    background_candidate: int
    structural_colors: tuple[int, ...]
    components: tuple[ComponentObservation, ...]
    objects: tuple[ObjectObservation, ...]
    volatile_cells: frozenset[tuple[int, int]]
    hud_panel_bbox: tuple[int, int, int, int] | None
    counter_bbox: tuple[int, int, int, int] | None
    counter_value: int | None
    counter_capacity: int | None
    counter_ratio: float | None
    life_count: int | None
    template_relations: tuple[dict[str, Any], ...]
    summary: str
    rgb: Image.Image | None = None
    annotated_rgb: Image.Image | None = None

    def object_by_id(self, track_id: str | None) -> ObjectObservation | None:
        if not track_id:
            return None
        return next((obj for obj in self.objects if obj.track_id == track_id), None)


@dataclass
class TransitionReport:
    changed_cell_count: int
    world_changed_cell_count: int
    hud_changed_cell_count: int
    changed_bbox: tuple[int, int, int, int] | None
    world_noop: bool
    full_visual_noop: bool
    effective_noop: bool
    counter_delta: int | None
    counter_ratio_after: float | None
    life_delta: int | None
    retry_detected: bool
    moved_objects: list[dict[str, Any]] = field(default_factory=list)
    transformed_objects: list[dict[str, Any]] = field(default_factory=list)
    appeared_object_ids: list[str] = field(default_factory=list)
    disappeared_object_ids: list[str] = field(default_factory=list)
    controlled_candidate_id: str | None = None
    is_simple_translation: bool = False
    interaction_event: bool = False
    action_key: str = ""
    action_source: str = ""
    summary: str = ""
    previous_rgb: Image.Image | None = None
    annotated_rgb: Image.Image | None = None


class Observer:
    DRIFT_MIN_SAMPLES = 8

    def __init__(self, image_size: int = 512, drift_cell_ratio: float = 0.6):
        self.image_size = image_size
        self.drift_cell_ratio = drift_cell_ratio
        self._last_grid: tuple[tuple[int, ...], ...] | None = None
        self._previous_objects: tuple[ObjectObservation, ...] = ()
        self._next_track_id = 1
        self._hud_panel_bbox: tuple[int, int, int, int] | None = None
        self._counter_bbox: tuple[int, int, int, int] | None = None
        self._counter_fill_color: int | None = None
        self._counter_capacity: int | None = None
        self._life_slots: list[tuple[tuple[int, int, int, int], int]] = []
        self._structural_colors: set[int] = set()
        self._drift_prev_grid: tuple[tuple[int, ...], ...] | None = None
        self._drift_change_counts: Counter[tuple[int, int]] = Counter()
        self._drift_samples = 0
        self.drift_cells: frozenset[tuple[int, int]] = frozenset()
        self._bar_counter_candidates: dict[tuple[int, int, int], dict[str, Any]] = {}

    def reset_level(self) -> None:
        self._previous_objects = ()
        self._next_track_id = 1
        self._hud_panel_bbox = None
        self._counter_bbox = None
        self._counter_fill_color = None
        self._counter_capacity = None
        self._life_slots = []
        self._structural_colors = set()
        self._drift_prev_grid = None
        self._drift_change_counts = Counter()
        self._drift_samples = 0
        self.drift_cells = frozenset()
        self._bar_counter_candidates = {}

    @staticmethod
    def normalize_grid_value(raw_grid: Any) -> tuple[tuple[int, ...], ...]:
        arr = np.asarray(raw_grid)
        if arr.ndim != 2:
            raise ObservationError(f"grid must be 2-D, got {arr.shape}")
        h, w = int(arr.shape[0]), int(arr.shape[1])
        if h < 1 or w < 1 or h > 64 or w > 64:
            raise ObservationError(f"grid shape must be within 64x64, got {arr.shape}")
        rows: list[tuple[int, ...]] = []
        for y in range(h):
            row = []
            for x in range(w):
                v = int(arr[y, x])
                if not 0 <= v <= 15:
                    raise ObservationError("grid values must be 0..15")
                row.append(v)
            rows.append(tuple(row))
        return tuple(rows)

    def scene_from_frame(self, latest_frame: Any) -> SceneSnapshot:
        raw_frames = list(getattr(latest_frame, "frame", []) or [])
        raw_grid = None
        for raw in reversed(raw_frames):
            arr = np.asarray(raw)
            if arr.ndim == 2:
                raw_grid = arr
                break
        if raw_grid is None:
            if self._last_grid is None:
                raise ObservationError("empty frame sequence")
            grid = self._last_grid
        else:
            grid = self.normalize_grid_value(raw_grid)
            self._last_grid = grid
        return self.analyze_grid(grid, render_grid(grid, self.image_size))

    def _update_drift_model(self, grid: tuple[tuple[int, ...], ...]) -> None:
        prev = self._drift_prev_grid
        self._drift_prev_grid = grid
        if prev is None or len(prev) != len(grid) or len(prev[0]) != len(grid[0]):
            return
        if prev is not grid:
            self._drift_samples += 1
            if prev != grid:
                for y, (old_row, new_row) in enumerate(zip(prev, grid)):
                    if old_row == new_row:
                        continue
                    for x, (old, new) in enumerate(zip(old_row, new_row)):
                        if old != new:
                            self._drift_change_counts[(x, y)] += 1
        if self._drift_samples >= self.DRIFT_MIN_SAMPLES:
            threshold = max(1, int(math.ceil(self._drift_samples * self.drift_cell_ratio)))
            detected = {cell for cell, n in self._drift_change_counts.items() if n >= threshold}
            # Sticky union: once a cell is classified as ambient drift keep it for the
            # rest of the level. Re-classifying every frame makes the set oscillate
            # around the threshold and churns every state-hash-keyed memory.
            if detected - self.drift_cells:
                self.drift_cells = frozenset(self.drift_cells | detected)

    def analyze_grid(self, grid: tuple[tuple[int, ...], ...], rgb: Image.Image | None = None) -> SceneSnapshot:
        h, w = len(grid), len(grid[0])
        counts = Counter(v for row in grid for v in row)
        background = min(c for c, n in counts.items() if n == max(counts.values()))
        raw_components = self._components(grid, background)
        self._update_hud_model(grid, raw_components, background)
        self._update_bar_counter_model(grid, raw_components)
        self._update_drift_model(grid)
        volatile = self._volatile_cells(w, h)
        world_rows = [list(row) for row in grid]
        for x, y in volatile:
            if 0 <= x < w and 0 <= y < h:
                world_rows[y][x] = background
        world_grid = tuple(tuple(r) for r in world_rows)
        comps = self._components(world_grid, background)
        structural: set[int] = set()
        structural_colors: Counter[int] = Counter()
        for i, comp in enumerate(comps):
            in_hud = self._hud_panel_bbox is not None and _bbox_inter(comp.bbox, self._hud_panel_bbox) >= max(1, comp.area // 2)
            large = comp.area >= max(64, int(w * h * 0.035)) or _bbox_area(comp.bbox) >= int(w * h * 0.18) or (comp.touches_border and comp.area >= max(32, int(w * h * 0.02)))
            compact_frame = self._component_frame_score(comp) >= 0.62 and max(comp.bbox[2] - comp.bbox[0] + 1, comp.bbox[3] - comp.bbox[1] + 1) <= 16
            if large and not in_hud and not compact_frame:
                self._structural_colors.add(comp.color)
            if in_hud or large or (comp.color in self._structural_colors and comp.area >= 5 and not compact_frame):
                structural.add(i)
                structural_colors[comp.color] += comp.area
        objects = self._assign_track_ids(self._build_objects(world_grid, comps, structural))
        counter = self._measure_counter(grid)
        if counter is not None:
            self._counter_capacity = max(self._counter_capacity or 0, counter)
        capacity = self._counter_capacity
        ratio = max(0.0, min(1.0, counter / capacity)) if counter is not None and capacity else None
        lives = self._measure_lives(grid)
        # Navigation hash ignores learned self-drifting animation cells so that
        # revisit/loop/progress logic keeps working in games with ambient motion
        # (tr87 conveyor, ar25 hazards). Objects/world_grid stay untouched.
        if self.drift_cells:
            nav_rows = [list(row) for row in world_grid]
            for x, y in self.drift_cells:
                if 0 <= x < w and 0 <= y < h:
                    nav_rows[y][x] = background
            nav_hash = _stable_hash(tuple(tuple(r) for r in nav_rows))
        else:
            nav_hash = _stable_hash(world_grid)
        scene = SceneSnapshot(
            grid, world_grid, w, h, nav_hash, _stable_hash(grid), background,
            tuple(c for c, _ in structural_colors.most_common()), comps, objects, frozenset(volatile),
            self._hud_panel_bbox, self._counter_bbox, counter, capacity, ratio, lives,
            self._template_relations(objects), "", rgb or render_grid(grid, self.image_size)
        )
        scene.summary = self._scene_summary(scene)
        scene.annotated_rgb = self._annotated_image(scene)
        self._previous_objects = objects
        return scene

    def _components(self, grid: tuple[tuple[int, ...], ...], background: int) -> tuple[ComponentObservation, ...]:
        h, w = len(grid), len(grid[0])
        seen: set[tuple[int, int]] = set()
        comps: list[ComponentObservation] = []
        for y in range(h):
            for x in range(w):
                if (x, y) in seen or grid[y][x] == background:
                    continue
                color = grid[y][x]
                q = deque([(x, y)])
                seen.add((x, y))
                cells: list[tuple[int, int]] = []
                while q:
                    cx, cy = q.popleft()
                    cells.append((cx, cy))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen and grid[ny][nx] == color:
                            seen.add((nx, ny))
                            q.append((nx, ny))
                xs, ys = [p[0] for p in cells], [p[1] for p in cells]
                box = (min(xs), min(ys), max(xs), max(ys))
                comps.append(ComponentObservation(color, len(cells), box, (sum(xs) / len(cells), sum(ys) / len(cells)), box[0] == 0 or box[1] == 0 or box[2] == w - 1 or box[3] == h - 1, tuple(sorted(cells, key=lambda p: (p[1], p[0])))))
        comps.sort(key=lambda c: (-c.area, c.color, c.bbox[1], c.bbox[0]))
        return tuple(comps)

    def _update_hud_model(self, grid: tuple[tuple[int, ...], ...], comps: Sequence[ComponentObservation], background: int) -> None:
        h, w = len(grid), len(grid[0])
        candidates = []
        for comp in comps:
            x0, y0, x1, y1 = comp.bbox
            cw, ch = x1 - x0 + 1, y1 - y0 + 1
            if y1 >= h - 1 and y0 >= int(h * 0.80) and cw >= max(12, int(w * 0.40)) and ch <= max(8, h // 7) and cw >= 3 * max(1, ch):
                candidates.append((cw * 4 + comp.area - ch, comp))
        if not candidates:
            return
        panel = max(candidates, key=lambda x: x[0])[1]
        self._hud_panel_bbox = panel.bbox
        if self._counter_fill_color is None:
            self._init_counter(grid, panel, background)
        if not self._life_slots:
            self._init_lives(comps, panel)

    def _init_counter(self, grid: tuple[tuple[int, ...], ...], panel: ComponentObservation, background: int) -> None:
        x0, y0, x1, y1 = panel.bbox
        groups: dict[tuple[int, int, int], list[int]] = {}
        for y in range(min(y1, y0 + 1), max(y0, y1 - 1) + 1):
            start = min(x1, x0 + 1)
            end_limit = max(x0, x1 - 1)
            while start <= end_limit:
                color = grid[y][start]
                end = start
                while end + 1 <= end_limit and grid[y][end + 1] == color:
                    end += 1
                if color not in {panel.color, background} and end - start + 1 >= max(4, (x1 - x0 + 1) // 6):
                    groups.setdefault((color, start, end), []).append(y)
                start = end + 1
        if groups:
            (fill, rx0, rx1), rows = max(groups.items(), key=lambda kv: ((kv[0][2] - kv[0][1] + 1) * len(kv[1]), len(kv[1])))
            self._counter_fill_color = fill
            self._counter_bbox = (rx0, min(rows), rx1, max(rows))
            self._counter_capacity = rx1 - rx0 + 1

    def _init_lives(self, comps: Sequence[ComponentObservation], panel: ComponentObservation) -> None:
        groups: dict[tuple[int, int], list[ComponentObservation]] = {}
        for comp in comps:
            if comp.color in {panel.color, self._counter_fill_color} or _bbox_inter(comp.bbox, panel.bbox) < comp.area:
                continue
            w, h = comp.bbox[2] - comp.bbox[0] + 1, comp.bbox[3] - comp.bbox[1] + 1
            if 1 <= w <= 5 and 1 <= h <= 5 and 1 <= comp.area <= 20:
                groups.setdefault((comp.color, comp.area), []).append(comp)
        repeated = [v for v in groups.values() if len(v) >= 2]
        if repeated:
            slots = max(repeated, key=lambda items: (len(items), sum(c.area for c in items)))
            slots.sort(key=lambda c: (c.bbox[0], c.bbox[1]))
            self._life_slots = [(c.bbox, c.color) for c in slots[:8]]

    def _update_bar_counter_model(self, grid: tuple[tuple[int, ...], ...], comps: Sequence[ComponentObservation]) -> None:
        """V2.6: recognize a bare shrinking bar as a resource counter (sb26 energy
        bar: full-width 1px strip at y=53 that loses one cell per energy spend).
        _init_counter only handles fill bars nested inside a bottom-edge HUD panel,
        so sb26's bar was invisible to the resource model: the agent spent its
        first life learning the budget via game-over back-inference and none of the
        low/critical-resource brakes engaged before that. A bar only qualifies
        after it has been observed shrinking twice anchored at one end and never
        growing, which keeps arbitrary walls/dividers (static) and moving platforms
        (both ends shift) out. The HUD-panel counter path keeps precedence."""
        if self._counter_bbox is not None:
            return
        h, w = len(grid), len(grid[0])
        for comp in comps:
            x0, y0, x1, y1 = comp.bbox
            cw, ch = x1 - x0 + 1, y1 - y0 + 1
            if ch > 2 or cw < max(8, w // 2):
                continue
            if comp.area < int(0.9 * cw * ch):
                continue
            key = (y0, y1, comp.color)
            cand = self._bar_counter_candidates.get(key)
            if cand is None:
                self._bar_counter_candidates[key] = {"full_x0": x0, "full_x1": x1, "last_w": cw, "shrinks": 0, "invalid": False}
                continue
            if cand["invalid"]:
                continue
            if cw > cand["last_w"]:
                # A refilling/pulsing bar is not a monotone budget; drop it for good.
                cand["invalid"] = True
                continue
            if cw < cand["last_w"]:
                if x0 == cand["full_x0"] or x1 == cand["full_x1"]:
                    cand["shrinks"] += 1
                else:
                    cand["invalid"] = True
                    continue
            cand["last_w"] = cw
            full_w = cand["full_x1"] - cand["full_x0"] + 1
            if cand["shrinks"] >= 2 and full_w - cw >= 2:
                self._counter_fill_color = comp.color
                self._counter_bbox = (cand["full_x0"], y0, cand["full_x1"], y1)
                self._counter_capacity = full_w
                self._bar_counter_candidates = {}
                return

    def _measure_counter(self, grid: tuple[tuple[int, ...], ...]) -> int | None:
        if self._counter_bbox is None or self._counter_fill_color is None:
            return None
        x0, y0, x1, y1 = self._counter_bbox
        best = 0
        for y in range(max(0, y0), min(len(grid) - 1, y1) + 1):
            cur = 0
            for x in range(max(0, x0), min(len(grid[0]) - 1, x1) + 1):
                if grid[y][x] == self._counter_fill_color:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
        return best

    def _measure_lives(self, grid: tuple[tuple[int, ...], ...]) -> int | None:
        if not self._life_slots:
            return None
        count = 0
        for (x0, y0, x1, y1), color in self._life_slots:
            if any(grid[y][x] == color for y in range(max(0, y0), min(len(grid) - 1, y1) + 1) for x in range(max(0, x0), min(len(grid[0]) - 1, x1) + 1)):
                count += 1
        return count

    def _volatile_cells(self, w: int, h: int) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        boxes = []
        if self._counter_bbox is not None:
            boxes.append(self._counter_bbox)
        boxes.extend(box for box, _ in self._life_slots)
        for x0, y0, x1, y1 in boxes:
            for y in range(max(0, y0), min(h - 1, y1) + 1):
                for x in range(max(0, x0), min(w - 1, x1) + 1):
                    cells.add((x, y))
        return cells

    @staticmethod
    def _component_frame_score(comp: ComponentObservation) -> float:
        x0, y0, x1, y1 = comp.bbox
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w < 5 or h < 5 or max(w, h) > 16 or max(w, h) / max(1, min(w, h)) > 1.9:
            return 0.0
        perim = {(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1) if x in {x0, x1} or y in {y0, y1}}
        return len(perim.intersection(comp.cells)) / max(1, len(perim))

    def _build_objects(self, grid: tuple[tuple[int, ...], ...], comps: Sequence[ComponentObservation], structural: set[int]) -> tuple[ObjectObservation, ...]:
        candidate = [i for i in range(len(comps)) if i not in structural]
        unassigned = set(candidate)
        groups: list[list[int]] = []
        frames = sorted((i for i in candidate if self._component_frame_score(comps[i]) >= 0.62), key=lambda i: (-comps[i].area, comps[i].bbox))
        for fi in frames:
            if fi not in unassigned:
                continue
            group = [fi]
            frame = comps[fi]
            for oi in sorted(unassigned):
                if oi != fi and _bbox_contains(frame.bbox, comps[oi].bbox) and _bbox_inter(frame.bbox, comps[oi].bbox) >= max(1, int(comps[oi].area * 0.8)):
                    group.append(oi)
            for i in group:
                unassigned.discard(i)
            groups.append(group)
        while unassigned:
            root = max(unassigned, key=lambda i: (comps[i].area, -comps[i].bbox[1]))
            group = [root]
            unassigned.remove(root)
            box = comps[root].bbox
            changed = True
            while changed:
                changed = False
                for oi in sorted(unassigned, key=lambda i: (-comps[i].area, comps[i].bbox)):
                    gx, gy = _bbox_gap(box, comps[oi].bbox)
                    union = (min(box[0], comps[oi].bbox[0]), min(box[1], comps[oi].bbox[1]), max(box[2], comps[oi].bbox[2]), max(box[3], comps[oi].bbox[3]))
                    if gx == 0 and gy == 0 and union[2] - union[0] + 1 <= 8 and union[3] - union[1] + 1 <= 8 and _bbox_area(union) <= 64:
                        group.append(oi)
                        unassigned.remove(oi)
                        box = union
                        changed = True
                        break
            groups.append(group)
        objs = []
        for group in groups:
            parts = [comps[i] for i in group]
            cells = tuple(sorted(((x, y, c.color) for c in parts for x, y in c.cells), key=lambda t: (t[1], t[0], t[2])))
            if not cells:
                continue
            xs, ys = [x for x, _, _ in cells], [y for _, y, _ in cells]
            box = (min(xs), min(ys), max(xs), max(ys))
            centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
            ccount = Counter(c for _, _, c in cells)
            colors = tuple(sorted(ccount))
            color_areas = tuple(sorted(ccount.items(), key=lambda kv: (-kv[1], kv[0])))
            frame_color = self._frame_color_for_object(cells, box, parts)
            pattern = self._object_pattern(cells, box)
            inner = self._inner_pattern(cells, box, frame_color)
            shape = self._shape_label(cells, box, len(colors), frame_color)
            near_edge = box[0] == 0 or box[1] == 0 or box[2] == len(grid[0]) - 1 or box[3] == len(grid) - 1
            signature = hashlib.blake2b((shape + "|" + pattern + "|" + inner + "|" + str(colors)).encode("utf-8"), digest_size=16).hexdigest()
            salience = float(len(cells)) + (60.0 if frame_color is not None else 0.0) + (35.0 if inner else 0.0) + (10.0 if len(colors) >= 2 else 0.0) - (8.0 if near_edge else 0.0)
            objs.append(ObjectObservation("", box, centroid, len(cells), colors, color_areas, len(parts), cells, signature, shape, pattern, inner, frame_color, near_edge, salience))
        objs.sort(key=lambda o: (-o.salience, o.bbox[1], o.bbox[0]))
        return tuple(objs)

    @staticmethod
    def _frame_color_for_object(cells: Sequence[tuple[int, int, int]], box: tuple[int, int, int, int], parts: Sequence[ComponentObservation]) -> int | None:
        x0, y0, x1, y1 = box
        perim = {(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1) if x in {x0, x1} or y in {y0, y1}}
        if len(perim) < 12:
            return None
        by_color: dict[int, set[tuple[int, int]]] = {}
        for x, y, c in cells:
            by_color.setdefault(c, set()).add((x, y))
        best_color, best_score = None, 0.0
        for c, pts in by_color.items():
            score = len(perim.intersection(pts)) / max(1, len(perim))
            if score > best_score:
                best_color, best_score = c, score
        return best_color if best_score >= 0.62 else None

    @staticmethod
    def _object_pattern(cells: Sequence[tuple[int, int, int]], box: tuple[int, int, int, int]) -> str:
        x0, y0, x1, y1 = box
        pts = {(x, y): c for x, y, c in cells}
        rows = []
        for y in range(y0, y1 + 1):
            row = []
            for x in range(x0, x1 + 1):
                row.append(_hex(pts[(x, y)]) if (x, y) in pts else ".")
            rows.append("".join(row))
        if len(rows) > 16 or max((len(r) for r in rows), default=0) > 16:
            return f"bbox={x1-x0+1}x{y1-y0+1};cells={len(cells)};colors=" + "".join(_hex(c) for c in sorted({c for _, _, c in cells}))
        return "/".join(rows)

    @staticmethod
    def _inner_pattern(cells: Sequence[tuple[int, int, int]], box: tuple[int, int, int, int], frame_color: int | None) -> str:
        if frame_color is None:
            return ""
        inner = [(x, y, c) for x, y, c in cells if c != frame_color]
        if not inner:
            return ""
        xs, ys = [x for x, _, _ in inner], [y for _, y, _ in inner]
        return Observer._object_pattern(inner, (min(xs), min(ys), max(xs), max(ys)))

    @staticmethod
    def _shape_label(cells: Sequence[tuple[int, int, int]], box: tuple[int, int, int, int], color_count: int, frame_color: int | None) -> str:
        if frame_color is not None:
            return "frame_with_inner_pattern"
        w, h = box[2] - box[0] + 1, box[3] - box[1] + 1
        fill = len({(x, y) for x, y, _ in cells}) / max(1, w * h)
        if color_count >= 2 and fill >= 0.75:
            return "compact_multicolor_block"
        if color_count >= 2:
            return "compact_multicolor_glyph"
        return "single_color_glyph"

    def _assign_track_ids(self, objects: Sequence[ObjectObservation]) -> tuple[ObjectObservation, ...]:
        previous = list(self._previous_objects)
        used: set[int] = set()
        assigned = []
        for obj in objects:
            best_i, best_score = None, -1.0
            for i, old in enumerate(previous):
                if i in used:
                    continue
                inter = _bbox_inter(obj.bbox, old.bbox)
                union = _bbox_area(obj.bbox) + _bbox_area(old.bbox) - inter
                color_overlap = len(set(obj.colors).intersection(old.colors)) / max(1, len(set(obj.colors).union(old.colors)))
                sig_bonus = 2.0 if obj.intrinsic_signature == old.intrinsic_signature else 0.0
                center_dist = abs(obj.centroid[0] - old.centroid[0]) + abs(obj.centroid[1] - old.centroid[1])
                score = sig_bonus + 2.0 * inter / max(1, union) + color_overlap - 0.03 * center_dist
                if score > best_score:
                    best_i, best_score = i, score
            if best_i is not None and best_score >= 1.15:
                tid = previous[best_i].track_id
                used.add(best_i)
            else:
                tid = f"O{self._next_track_id}"
                self._next_track_id += 1
            assigned.append(ObjectObservation(tid, obj.bbox, obj.centroid, obj.area, obj.colors, obj.color_areas, obj.component_count, obj.cells, obj.intrinsic_signature, obj.shape_label, obj.pattern, obj.inner_pattern, obj.frame_color, obj.near_edge, obj.salience))
        assigned.sort(key=lambda o: int(o.track_id[1:]) if o.track_id[1:].isdigit() else 9999)
        return tuple(assigned)

    @staticmethod
    def _template_relations(objects: Sequence[ObjectObservation]) -> tuple[dict[str, Any], ...]:
        framed = [o for o in objects if o.frame_color is not None and o.inner_pattern]
        rels = []
        # V2.3: pairing below was exact-inner-match only, so two distinct framed+
        # inner-pattern objects that never happen to line up exactly (ls20: a large
        # corner status frame + a small cycling glyph frame, different sizes) never
        # got surfaced as related at all. Add a weaker "possible_selector_pair" hint
        # for the non-exact case so the VLM can test whether one predicts/tracks the
        # other. Guarded to a small framed-object count: this is O(n^2) and most
        # games have very few frame_with_inner_pattern objects, but some tile/grid
        # puzzles could have many, and this must not blow up prompt size there.
        small_enough = len(framed) <= 6
        for i, a in enumerate(framed):
            a_rows = [r for r in a.inner_pattern.split("/") if r]
            a_bin = tuple("".join("#" if ch != "." else "." for ch in row) for row in a_rows)
            for b in framed[i + 1:]:
                b_rows = [r for r in b.inner_pattern.split("/") if r]
                b_bin = tuple("".join("#" if ch != "." else "." for ch in row) for row in b_rows)
                if a_bin and b_bin and a_bin == b_bin:
                    rels.append({"left": a.track_id, "right": b.track_id, "same_shape_under_rotation": True, "quarter_turns_left_to_right": 0, "same_inner_colors": a.inner_pattern == b.inner_pattern, "exact_inner_match": a.inner_pattern == b.inner_pattern, "edge_vs_world": a.near_edge != b.near_edge})
                elif small_enough:
                    rels.append({"left": a.track_id, "right": b.track_id, "same_shape_under_rotation": False, "exact_inner_match": False, "possible_selector_pair": True, "edge_vs_world": a.near_edge != b.near_edge})
        return tuple(rels)

    def compare(self, before: SceneSnapshot, after: SceneSnapshot) -> TransitionReport:
        changed, world_changed = [], []
        volatile = set(before.volatile_cells) | set(after.volatile_cells)
        for y in range(max(before.height, after.height)):
            for x in range(max(before.width, after.width)):
                old = before.grid[y][x] if y < before.height and x < before.width else None
                new = after.grid[y][x] if y < after.height and x < after.width else None
                if old != new:
                    changed.append((x, y))
                    if (x, y) not in volatile:
                        world_changed.append((x, y))
        box = None
        if changed:
            xs, ys = [x for x, _ in changed], [y for _, y in changed]
            box = (min(xs), min(ys), max(xs), max(ys))
        before_by, after_by = {o.track_id: o for o in before.objects}, {o.track_id: o for o in after.objects}
        moved, transformed = [], []
        for oid in sorted(set(before_by).intersection(after_by)):
            old, new = before_by[oid], after_by[oid]
            dx, dy = int(round(new.centroid[0] - old.centroid[0])), int(round(new.centroid[1] - old.centroid[1]))
            if dx or dy:
                moved.append({"object_id": oid, "from_bbox": old.bbox, "to_bbox": new.bbox, "dx": dx, "dy": dy, "shape": new.shape_label})
            if old.intrinsic_signature != new.intrinsic_signature:
                transformed.append({"object_id": oid, "before_type": old.shape_label, "after_type": new.shape_label, "before_pattern": old.pattern, "after_pattern": new.pattern})
        appeared, disappeared = sorted(set(after_by) - set(before_by)), sorted(set(before_by) - set(after_by))
        cdelta = after.counter_value - before.counter_value if before.counter_value is not None and after.counter_value is not None else None
        ldelta = after.life_count - before.life_count if before.life_count is not None and after.life_count is not None else None
        candidates = [m for m in moved if not after_by[m["object_id"]].near_edge and after_by[m["object_id"]].area <= 220]
        if not candidates and moved:
            fallback = [m for m in moved if not after_by[m["object_id"]].near_edge]
            if fallback:
                fallback.sort(key=lambda m: (-abs(int(m["dx"])) - abs(int(m["dy"])), after_by[m["object_id"]].area))
                candidates = [fallback[0]]
        controlled = None
        if candidates:
            candidates.sort(key=lambda m: (-abs(int(m["dx"])) - abs(int(m["dy"])), after_by[m["object_id"]].area))
            controlled = candidates[0]["object_id"]
        simple_translation = bool(controlled) and all(m["object_id"] == controlled or after_by[m["object_id"]].near_edge for m in moved) and not transformed and not appeared and not disappeared
        world_noop = before.world_grid == after.world_grid
        full_noop = before.grid == after.grid
        # Treat pure resource consumption as a noop for search purposes: it did not advance the world.
        effective_noop = world_noop and (cdelta is None or cdelta <= 0) and ldelta in (None, 0)
        retry = bool(cdelta is not None and before.counter_capacity is not None and cdelta >= max(2, int(before.counter_capacity * 0.5)) and ldelta is not None and ldelta < 0)
        interaction = bool(retry or (cdelta is not None and cdelta > 1) or ldelta not in (None, 0) or transformed or appeared or disappeared) and not simple_translation
        parts = [f"changed_cells={len(changed)}", f"world_changed_cells={len(world_changed)}", f"effective_noop={str(effective_noop).lower()}", f"simple_translation={str(simple_translation).lower()}"]
        if controlled:
            parts.append(f"controlled_candidate={controlled}")
        if moved:
            parts.append("moved=" + ",".join(f"{m['object_id']}({m['dx']},{m['dy']})" for m in moved[:6]))
        if transformed:
            parts.append("transformed=" + ",".join(t["object_id"] for t in transformed[:6]))
        if appeared:
            parts.append("appeared=" + ",".join(appeared[:6]))
        if disappeared:
            parts.append("disappeared=" + ",".join(disappeared[:6]))
        if cdelta is not None:
            parts.append(f"counter_delta={cdelta}")
        if ldelta is not None:
            parts.append(f"life_delta={ldelta}")
        if retry:
            parts.append("retry_detected=true")
        return TransitionReport(len(changed), len(world_changed), len(changed) - len(world_changed), box, world_noop, full_noop, effective_noop, cdelta, after.counter_ratio, ldelta, retry, moved, transformed, appeared, disappeared, controlled, simple_translation, interaction, summary="; ".join(parts))

    def _scene_summary(self, scene: SceneSnapshot) -> str:
        lines = [f"grid={scene.width}x{scene.height}", f"navigation_state_hash={scene.state_hash[:12]}", f"background_candidate={scene.background_candidate}", f"structural_colors={list(scene.structural_colors[:4])}"]
        if scene.counter_value is not None:
            lines.append(f"step_counter_like={scene.counter_value}/{scene.counter_capacity} ratio={scene.counter_ratio}")
        if scene.life_count is not None:
            lines.append(f"life_like_slots={scene.life_count}")
        for obj in sorted(scene.objects, key=lambda o: -o.salience)[:32]:
            lines.append(f"{obj.track_id}: type={obj.shape_label} bbox={obj.bbox} centroid=({obj.centroid[0]:.1f},{obj.centroid[1]:.1f}) colors={dict(obj.color_areas)} area={obj.area} near_edge={obj.near_edge} pattern={obj.pattern}" + (f" inner={obj.inner_pattern}" if obj.inner_pattern else ""))
        if scene.template_relations:
            lines.append(f"framed_template_relations={list(scene.template_relations)}")
        return "\n".join(lines)

    def _annotated_image(self, scene: SceneSnapshot) -> Image.Image:
        img = (scene.rgb or render_grid(scene.grid, self.image_size)).copy()
        draw = ImageDraw.Draw(img)
        sx, sy = img.width / scene.width, img.height / scene.height
        for obj in scene.objects:
            x0, y0, x1, y1 = obj.bbox
            box = (int(x0 * sx), int(y0 * sy), int((x1 + 1) * sx - 1), int((y1 + 1) * sy - 1))
            draw.rectangle(box, outline=(255, 255, 255), width=max(1, int(min(sx, sy) // 3)))
            label = f"{obj.track_id}:{obj.shape_label[:8]}"
            draw.text((box[0] + 1, max(0, box[1] - 12)), label, fill=(255, 255, 255))
        return img


@dataclass
class VisualDescriptor:
    shape_label: str = ""
    colors: list[int] = field(default_factory=list)
    color_areas: dict[str, int] = field(default_factory=dict)
    pattern: str = ""
    inner_pattern: str = ""
    frame_color: int | None = None
    size_bucket: str = ""
    relation_tags: list[str] = field(default_factory=list)
    near_edge: bool | None = None
    type_key: str = ""


@dataclass
class ActionMeaning:
    action: str
    meaning_nl: str = ""
    kind: str = "unknown"
    vector: tuple[int, int] | None = None
    vector_votes: Counter[str] = field(default_factory=Counter)
    resource_delta: int | None = None
    confidence: float = 0.0
    evidence_events: list[int] = field(default_factory=list)
    attempts: int = 0
    noops: int = 0
    movements: int = 0
    transforms: int = 0
    changed_cells_total: int = 0
    world_changed_cells_total: int = 0
    small_transforms: int = 0
    interactions: int = 0
    retries: int = 0
    life_losses: int = 0

    @property
    def noop_ratio(self) -> float:
        return self.noops / max(1, self.attempts)

    @property
    def positive_count(self) -> int:
        return self.movements + self.transforms + self.interactions

    def as_prompt(self) -> dict[str, Any]:
        return _json_safe(self)


@dataclass
class WinConditionMemory:
    description_nl: str = ""
    visual_roles: dict[str, VisualDescriptor] = field(default_factory=dict)
    confirmed_levels: list[int] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ObjectEffectMemory:
    visual_descriptor: VisualDescriptor
    effect_nl: str
    confidence: float = 0.0
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ResourceModel:
    description_nl: str = ""
    visible_bar: bool = False
    last_value: int | None = None
    capacity: int | None = None
    last_lives: int | None = None
    steady_cost_per_action: int | None = None
    refill_descriptors: list[VisualDescriptor] = field(default_factory=list)
    hazard_descriptors: list[VisualDescriptor] = field(default_factory=list)
    hidden_action_budget_capacity: int | None = None
    hidden_budget_observations: list[int] = field(default_factory=list)
    hidden_budget_source: str = ""


@dataclass
class CompactEvent:
    event_id: int
    level_index: int
    step_index: int
    action_key: str
    source: str
    before_state: str
    after_state: str
    outcome: str
    summary: str
    transition_delta: dict[str, Any] = field(default_factory=dict)

    def as_prompt(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "level": self.level_index,
            "step": self.step_index,
            "action": self.action_key,
            "source": self.source,
            "outcome": self.outcome,
            "before": self.before_state[:12],
            "after": self.after_state[:12],
            "summary": self.summary[:420],
            "delta": self.transition_delta,
        }


@dataclass
class PlanStep:
    step_type: str = "probe_action"
    action: str = ""
    target_role: str = ""
    target_object_id: str = ""
    purpose: str = ""
    stop_condition: str = ""
    expected_predicates: list[dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    attempts: int = 0
    max_attempts: int = 2
    raw: dict[str, Any] = field(default_factory=dict)

    def as_prompt(self) -> dict[str, Any]:
        return {
            "type": self.step_type,
            "action": self.action,
            "target_role": self.target_role,
            "target_object_id": self.target_object_id,
            "purpose": self.purpose[:240],
            "stop_condition": self.stop_condition[:240],
            "expected_predicates": self.expected_predicates[:6],
            "status": self.status,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
        }


@dataclass
class PendingAction:
    name: str
    x: int | None
    y: int | None
    purpose: str
    expected_change: str
    target_object_id: str
    scene_before: SceneSnapshot
    source_frame_id: int
    issued_call: int
    source: str = "executor"
    expected_predicates: list[dict[str, Any]] = field(default_factory=list)
    source_frame_ref: Any | None = None
    vlm_mode: str = ""

    def action_key(self) -> str:
        return f"ACTION6:{self.x},{self.y}" if self.name == "ACTION6" else self.name


@dataclass
class GameMemoryV20:
    action_meanings: dict[str, ActionMeaning] = field(default_factory=dict)
    win_condition: WinConditionMemory | None = None
    object_effects: list[ObjectEffectMemory] = field(default_factory=list)
    mechanics_nl: list[str] = field(default_factory=list)
    resource_model: ResourceModel = field(default_factory=ResourceModel)
    solved_level_summaries: list[dict[str, Any]] = field(default_factory=list)
    advisory_notes: str = ""
    vlm_calls_total: int = 0
    # Fixed action set for the whole game, captured once from the first legal
    # frame. Empty tuple means it was never readable -> keep legacy per-frame legal.
    game_action_space: tuple[int, ...] = ()

    def as_prompt(self) -> dict[str, Any]:
        prompt = {
            "action_meanings": {k: v.as_prompt() for k, v in sorted(self.action_meanings.items())},
            "win_condition": _json_safe(self.win_condition),
            "object_effects": [_json_safe(x) for x in self.object_effects[-12:]],
            "mechanics_nl": self.mechanics_nl[-12:],
            "resource_model": _json_safe(self.resource_model),
            "solved_level_summaries": self.solved_level_summaries[-6:],
            "advisory_notes": self.advisory_notes[:900],
        }
        return _strip_local_object_ids(prompt)


@dataclass
class LevelMemoryV20:
    level_index: int = 0
    levels_completed_at_start: int = 0
    attempt_index: int = 0
    attempt_started_at_action_count: int = 0
    initial_scene: SceneSnapshot | None = None
    current_scene: SceneSnapshot | None = None
    pre_success_scene: SceneSnapshot | None = None
    local_bindings: dict[str, str] = field(default_factory=dict)
    current_plan: list[PlanStep] = field(default_factory=list)
    plan_cursor: int = 0
    plan_goal: str = ""
    recent_events: deque[CompactEvent] = field(default_factory=lambda: deque(maxlen=24))
    known_noops_by_state: dict[str, set[str]] = field(default_factory=dict)
    tried_actions_by_state: dict[str, set[str]] = field(default_factory=dict)
    transition_graph: dict[str, dict[str, str]] = field(default_factory=dict)
    state_action_outcomes: dict[str, dict[str, Counter[str]]] = field(default_factory=dict)
    action_outcomes: dict[str, Counter[str]] = field(default_factory=dict)
    quarantine_until_by_state: dict[str, dict[str, int]] = field(default_factory=dict)
    global_quarantine_until: dict[str, int] = field(default_factory=dict)
    controlled_object_id: str = ""
    tentative_controlled_object_id: str = ""
    actor_votes: Counter[str] = field(default_factory=Counter)
    actor_bbox_history: deque[tuple[int, int, int, int]] = field(default_factory=lambda: deque(maxlen=48))
    walkable_color_votes: Counter[int] = field(default_factory=Counter)
    resource_state: dict[str, Any] = field(default_factory=dict)
    hidden_resource_used_this_attempt: int = 0
    total_action_count: int = 0
    actions_since_vlm: int = 999
    chunk_action_count: int = 0
    vlm_calls_this_level: int = 0
    last_vlm_mode: str = ""
    pending_action: PendingAction | None = None
    last_resolved_pending_action: PendingAction | None = None
    awaiting_reset: bool = False
    bottleneck_reason: str = ""
    notes_for_next_call: str = ""
    recent_state_hashes: deque[str] = field(default_factory=lambda: deque(maxlen=24))
    recent_action_keys: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    initial_vlm_done: bool = False
    initial_vlm_attempted: bool = False
    success_reflected: bool = False
    consecutive_nonterminal_resets: int = 0
    transform_pressure: int = 0
    repeated_transform_streak: int = 0
    repeated_noop_streak: int = 0
    resource_crisis: bool = False
    vlm_issue_signature: str = ""
    vlm_issue_repeat_count: int = 0
    # Window of recent issue signatures. A ping-pong between two states produces
    # two alternating signatures; comparing only the previous one never detects it.
    vlm_issue_recent_signatures: deque[str] = field(default_factory=lambda: deque(maxlen=6))
    # Consecutive VLM calls that returned no usable plan: parsed-but-empty plans AND
    # totally unparseable/INVALID responses both count (see _request_vlm_once).
    # Reset to 0 as soon as a plan is returned. Drives temperature: at temp=0 a bad
    # answer reproduces deterministically (observed r11l run of 9 empty plans; ls20
    # separately produced a repeating non-JSON ramble), so we must raise temperature
    # to sample out of it rather than lower it.
    consecutive_empty_plans: int = 0
    transform_state_visits: Counter[str] = field(default_factory=Counter)
    # V2.5: subset of transform_state_visits contributed by a directed action
    # source (plan_executor/deterministic_navigation/transform_controller/...)
    # rather than aimless fallback probing. See _UNDIRECTED_ACTION_SOURCES.
    transform_state_directed_visits: Counter[str] = field(default_factory=Counter)
    click_noops_by_object: Counter[str] = field(default_factory=Counter)
    click_success_by_object: Counter[str] = field(default_factory=Counter)
    click_success_coords: Counter[str] = field(default_factory=Counter)
    click_success_regions: Counter[str] = field(default_factory=Counter)
    click_coord_counts: Counter[str] = field(default_factory=Counter)
    click_region_counts: Counter[str] = field(default_factory=Counter)
    bad_action_suffixes: list[tuple[str, ...]] = field(default_factory=list)
    last_vlm_io: dict[str, Any] | None = None
    last_vlm_io_call: int = 0
    flushed_last_vlm_io_call: int = 0
    action_recovery_contract: bool = False
    rejected_vlm_plan_feedback: list[dict[str, Any]] = field(default_factory=list)
    last_loop_break_signature: str = ""
    last_loop_break_logged_at: int = -999999
    # VLM/local shared execution contract. These masks are state-scoped and must
    # be honored by every executor/controller path.
    contract_forbidden_by_state: dict[str, dict[str, str]] = field(default_factory=dict)
    fallback_guard_block_until: int = 0
    click_fuse_block_until: int = 0
    # State-conditioned click memory for click-sequence games.
    successful_clicks_by_state: dict[str, Counter[str]] = field(default_factory=dict)
    click_edges_by_state: dict[str, dict[str, str]] = field(default_factory=dict)
    # Progress evidence used to prevent raw transform loops.
    action_progress_scores: dict[str, deque[float]] = field(default_factory=dict)
    # V2.1 real-progress vs meaningless-churn separation.
    # all_state_visits: how many turns each state_hash was observed this level (no
    # window limit). Used to tell a genuinely new state from a cycled/decorative one.
    all_state_visits: Counter[str] = field(default_factory=Counter)
    # Per-click-coordinate count of "changed pixels but made no real progress"
    # transforms, and coords fully banned once that count crosses the fuse.
    nonprogress_transform_coords: Counter[str] = field(default_factory=Counter)
    global_forbidden_click_coords: dict[str, str] = field(default_factory=dict)
    # V2.2 ticker detection: objects that transform on (nearly) every transition
    # regardless of which action ran are timer bars / ambient tickers (tn36/vc33/
    # lf52 shrink-bars). Their per-step tick must not count as a "transform"
    # outcome, otherwise every action looks structural and nonprogress contracts
    # explode. Streaks reset when the object skips a transition.
    object_transform_streaks: Counter[str] = field(default_factory=Counter)
    object_transform_streak_actions: dict[str, set[str]] = field(default_factory=dict)
    object_transform_streak_moves: Counter[str] = field(default_factory=Counter)
    ticker_object_ids: set[str] = field(default_factory=set)
    # V2.2 cross-state nonprogress accounting: forbid an action only after it
    # produced nonprogress transforms in >=2 distinct states, not on first sight.
    nonprogress_transform_actions: Counter[str] = field(default_factory=Counter)
    nonprogress_transform_states: dict[str, set[str]] = field(default_factory=dict)
    # V2.2 navigation-rejection dedupe + VLM feedback (ar25 logged 250 identical
    # rejections per episode without ever telling the VLM).
    nav_reject_counts: Counter[str] = field(default_factory=Counter)
    # Consecutive transitions with counter_delta == -1. A long streak means the
    # counter is a per-step tax (ls20): every action costs 1, so "save resources"
    # guards are meaningless and must stand down. Sticky once confirmed.
    counter_tick_streak: int = 0
    counter_is_step_tax: bool = False
    # Set when a lone VLM movement step was intentionally delegated to the local
    # navigator this turn; _apply_vlm_result must treat the resulting empty plan as
    # a successful handoff, not as "vlm_unusable_plan" (which would re-trigger a
    # bottleneck VLM call every other action).
    fuel_nav_delegated_at: int = -999
    # V2.3: object ids that have shown persistent-id transform evidence (pattern/type
    # changed while keeping the same track_id), e.g. ls20's O1 corner frame cycling
    # its inner pattern. A pure bbox/position heuristic alone (_looks_like_
    # nonplayfield_status_cue) previously mislabelled such objects as decorative HUD
    # cues forever; this lets that heuristic stand down once real evidence exists.
    mechanism_evidence_object_ids: set[str] = field(default_factory=set)
    # V2.4: object ids tagged as possible_selector_pair from framed-template relations
    # at level start (before any transform evidence). Used to pre-label mechanism
    # candidates instead of waiting for the first measured transform on O1.
    selector_pair_object_ids: set[str] = field(default_factory=set)
    # Cap mechanism-hypothesis rewrites per attempt to avoid prompt churn loops.
    hypothesis_escalation_count: int = 0

    def active_step(self) -> PlanStep | None:
        while self.plan_cursor < len(self.current_plan) and self.current_plan[self.plan_cursor].status in {"done", "skipped", "failed"}:
            self.plan_cursor += 1
        return self.current_plan[self.plan_cursor] if self.plan_cursor < len(self.current_plan) else None

    def actor_motion_bbox(self) -> tuple[int, int, int, int] | None:
        if not self.actor_bbox_history:
            return None
        return (
            min(b[0] for b in self.actor_bbox_history),
            min(b[1] for b in self.actor_bbox_history),
            max(b[2] for b in self.actor_bbox_history),
            max(b[3] for b in self.actor_bbox_history),
        )

    def as_prompt(self, scene: SceneSnapshot | None = None) -> dict[str, Any]:
        state = scene.state_hash if scene else ""
        return {
            "level_index": self.level_index,
            "attempt_index": self.attempt_index,
            "local_bindings": dict(self.local_bindings),
            "controlled_object_id": self.controlled_object_id,
            "tentative_controlled_object_id": self.tentative_controlled_object_id,
            "plan_goal": self.plan_goal[:360],
            "plan_cursor": self.plan_cursor,
            "current_plan": [s.as_prompt() for s in self.current_plan[:12]],
            "recent_events": [e.as_prompt() for e in list(self.recent_events)[-12:]],
            "known_noops_here": sorted(self.known_noops_by_state.get(state, set())),
            "tried_here": sorted(self.tried_actions_by_state.get(state, set())),
            "outcomes_here": _json_safe(self.state_action_outcomes.get(state, {})),
            "quarantined_here": {k: v for k, v in self.quarantine_until_by_state.get(state, {}).items() if v > self.total_action_count},
            "terrain_model": {"walkable_color_votes": dict(self.walkable_color_votes), "actor_motion_bbox": self.actor_motion_bbox()},
            "resource_state": dict(self.resource_state),
            "hidden_resource_state": {"used_this_attempt": self.hidden_resource_used_this_attempt},
            "actions_since_vlm": self.actions_since_vlm,
            "chunk_action_count": self.chunk_action_count,
            "vlm_calls_this_level": self.vlm_calls_this_level,
            "bottleneck_reason": self.bottleneck_reason,
            "transform_pressure": self.transform_pressure,
            "resource_crisis": self.resource_crisis,
            "vlm_issue_repeat_count": self.vlm_issue_repeat_count,
            "transform_state_visits": {k[:12]: v for k, v in self.transform_state_visits.most_common(6)},
            "click_noops_by_object": dict(self.click_noops_by_object),
            "click_success_by_object": dict(self.click_success_by_object),
            "action_recovery_contract": self.action_recovery_contract,
            "rejected_vlm_plan_feedback": _json_safe(self.rejected_vlm_plan_feedback[-6:]),
            "click_coord_counts": dict(self.click_coord_counts.most_common(12)),
            "click_region_counts": dict(self.click_region_counts.most_common(12)),
            "click_success_coords": dict(self.click_success_coords.most_common(12)),
            "click_success_regions": dict(self.click_success_regions.most_common(12)),
            "contract_forbidden_here": dict(self.contract_forbidden_by_state.get(state, {})),
            "fallback_guard_block_until": self.fallback_guard_block_until,
            "click_fuse_block_until": self.click_fuse_block_until,
            "successful_clicks_here": dict(self.successful_clicks_by_state.get(state, Counter()).most_common(8)),
            "action_progress_scores": {k: [round(float(v), 3) for v in list(vals)[-4:]] for k, vals in self.action_progress_scores.items()},
            "terminal_bad_action_suffixes": [list(s) for s in self.bad_action_suffixes[-6:]],
            "notes_for_next_call": self.notes_for_next_call[:700],
        }


@dataclass
class RuntimeMemoryV20:
    game: GameMemoryV20 = field(default_factory=GameMemoryV20)
    level: LevelMemoryV20 = field(default_factory=LevelMemoryV20)
    next_event_id: int = 1


@dataclass
class VLMRequest:
    text_prompt: str
    current_rgb: Image.Image
    previous_rgb: Image.Image | None = None
    analysis_rgb: Image.Image | None = None
    max_new_tokens: int = 1100
    # Raised above 0 when the same issue keeps repeating, so a deterministic
    # backend stops returning the exact same rejected plan every call.
    temperature: float = 0.0


class VLMBackend(Protocol):
    @property
    def available(self) -> bool: ...
    def decide(self, request: VLMRequest) -> Any: ...


class DecisionLogger:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.path: Path | None = None
        if config.log_dir:
            try:
                root = Path(config.log_dir)
                root.mkdir(parents=True, exist_ok=True)
                self.path = root / f"agent_v2_0_{int(time.time())}_{os.getpid()}.jsonl"
            except Exception:
                self.path = None

    def log_event(self, event: str, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"event": event, "time": round(time.time(), 3), **_json_safe(payload)}, ensure_ascii=True, default=str) + "\n")
        except Exception:
            pass

    def log_exception(self, exc: BaseException) -> None:
        self.log_event("exception", {"type": type(exc).__name__, "message": str(exc)[:500], "trace": traceback.format_exc()[-2500:]})


_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


class Qwen35Backend:
    def __init__(self, config: AgentConfig, logger: DecisionLogger):
        self.config = config
        self.logger = logger
        self._available = bool(config.enable_vlm and config.model_path)
        self._errors = 0
        if config.enable_vlm and not config.model_path:
            self.disable_for_episode("missing QWEN_MODEL_PATH")

    @property
    def available(self) -> bool:
        return self._available

    def disable_for_episode(self, reason: str) -> None:
        self._available = False
        self.logger.log_event("vlm_disabled", {"reason": reason})

    def ensure_loaded(self) -> bool:
        if not self._available or not self.config.model_path:
            return False
        path = self.config.model_path
        if not Path(path).exists():
            self.disable_for_episode(f"model path missing: {path}")
            return False
        with _MODEL_CACHE_LOCK:
            if path in _MODEL_CACHE:
                return True
            try:
                import torch
                from transformers import AutoProcessor
                try:
                    from transformers import AutoModelForMultimodalLM as ModelClass
                except Exception:
                    try:
                        from transformers import AutoModelForImageTextToText as ModelClass
                    except Exception:
                        from transformers import AutoModelForCausalLM as ModelClass
                if not torch.cuda.is_available() and not _bool_env(("ARC_V20_ALLOW_CPU_VLM", "ARC_V19_ALLOW_CPU_VLM", "ARC_V18_ALLOW_CPU_VLM", "ARC_V15_ALLOW_CPU_VLM"), False):
                    self.disable_for_episode("CUDA unavailable and CPU VLM disabled")
                    return False
                processor = AutoProcessor.from_pretrained(path, local_files_only=True)
                kwargs = dict(local_files_only=True, device_map="auto", low_cpu_mem_usage=True)
                kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
                try:
                    model = ModelClass.from_pretrained(path, **kwargs)
                except TypeError:
                    dtype = kwargs.pop("torch_dtype", None)
                    if dtype is not None:
                        kwargs["dtype"] = dtype
                    model = ModelClass.from_pretrained(path, **kwargs)
                model.eval()
                _MODEL_CACHE[path] = (processor, model)
                return True
            except Exception as exc:
                self._handle_error(f"model load error: {exc}")
                return False

    def _handle_error(self, reason: str) -> None:
        self._errors += 1
        self.logger.log_event("vlm_error", {"reason": reason[:500], "count": self._errors})
        if self._errors >= 2:
            self.disable_for_episode(reason)

    def decide(self, request: VLMRequest) -> str | None:
        if not self.ensure_loaded() or not self.config.model_path:
            return None
        processor, model = _MODEL_CACHE[self.config.model_path]
        try:
            import torch
            content: list[dict[str, Any]] = []
            if request.previous_rgb is not None:
                content.append({"type": "image", "image": request.previous_rgb})
            content.append({"type": "image", "image": request.current_rgb})
            if request.analysis_rgb is not None:
                content.append({"type": "image", "image": request.analysis_rgb})
            content.append({"type": "text", "text": request.text_prompt})
            messages = [{"role": "system", "content": [{"type": "text", "text": V20_SYSTEM_PROMPT}]}, {"role": "user", "content": content}]
            kwargs = dict(conversation=messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            try:
                inputs = processor.apply_chat_template(**kwargs, enable_thinking=False)
            except TypeError:
                inputs = processor.apply_chat_template(**kwargs)
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            input_len = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=request.max_new_tokens, do_sample=False, use_cache=True)
            return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        except Exception as exc:
            self._handle_error(f"generate error: {exc}")
            return None


def _pil_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def vlm_uses_remote_api(config: AgentConfig | None = None) -> bool:
    mode = ((config or AgentConfig.from_env()).vlm_backend or "local").lower().strip()
    return mode in {"api", "openai", "remote", "http"}


class OpenAICompatibleBackend:
    def __init__(self, config: AgentConfig, logger: DecisionLogger):
        self.config = config
        self.logger = logger
        self._client: Any | None = None
        self._error_count = 0
        missing = [name for name, value in (("ARC_V20_VLM_API_BASE_URL", config.vlm_api_base_url), ("ARC_V20_VLM_API_KEY/INF_API_KEY", config.vlm_api_key), ("ARC_V20_VLM_API_MODEL", config.vlm_api_model)) if not value]
        self._available = bool(config.enable_vlm and not missing)
        if config.enable_vlm and missing:
            self.disable_for_episode("missing remote VLM config: " + ", ".join(missing))

    @property
    def available(self) -> bool:
        return self._available

    def disable_for_episode(self, reason: str) -> None:
        self._available = False
        self.logger.log_event("vlm_disabled", {"reason": reason})

    def _client_instance(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.config.vlm_api_key, base_url=self.config.vlm_api_base_url, timeout=self.config.vlm_api_timeout_s)
        return self._client

    def decide(self, request: VLMRequest) -> str | None:
        if not self._available:
            return None
        content = []
        if request.previous_rgb is not None:
            content.append({"type": "image_url", "image_url": {"url": _pil_url(request.previous_rgb)}})
        content.append({"type": "image_url", "image_url": {"url": _pil_url(request.current_rgb)}})
        if request.analysis_rgb is not None:
            content.append({"type": "image_url", "image_url": {"url": _pil_url(request.analysis_rgb)}})
        content.append({"type": "text", "text": request.text_prompt})
        try:
            response = self._client_instance().chat.completions.create(
                model=self.config.vlm_api_model,
                messages=[{"role": "system", "content": V20_SYSTEM_PROMPT}, {"role": "user", "content": content}],
                max_tokens=request.max_new_tokens,
                temperature=max(0.0, min(1.0, float(request.temperature or 0.0))),
                timeout=self.config.vlm_api_timeout_s,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            self._error_count = 0
            message = response.choices[0].message
            content_text = getattr(message, "content", None)
            if isinstance(content_text, list):
                content_text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content_text)
            if content_text and str(content_text).strip():
                return str(content_text)
            dump = message.model_dump() if hasattr(message, "model_dump") else {}
            for key in ("reasoning_content", "reasoning", "refusal"):
                if dump.get(key):
                    return str(dump[key])
            return None
        except Exception as exc:
            self._error_count += 1
            self.logger.log_event("vlm_error", {"reason": str(exc)[:500], "consecutive_errors": self._error_count})
            if self._error_count >= 2:
                self.disable_for_episode("remote VLM repeated errors")
            return None


def make_vlm_backend(config: AgentConfig, logger: DecisionLogger) -> VLMBackend:
    return OpenAICompatibleBackend(config, logger) if vlm_uses_remote_api(config) else Qwen35Backend(config, logger)


V20_ARC_ACTION_REFERENCE: dict[str, str] = {
    "RESET": "Start or restart the game. Agent policy: do not propose unless terminal or no non-reset legal action exists.",
    "ACTION1-ACTION5": "Simple actions (e.g., move up/down/left/right, interact). Per-game mapping is unknown until transition evidence; probe to learn.",
    "ACTION6": "Complex action requiring (x, y) coordinates.",
    "ACTION7": "Additional simple action. In many human-25 games this is undo/back; avoid unless positive evidence proves it advances this game.",
}

V20_LEVEL_INIT_SIMPLE_ACTION_HINT = (
    "LEVEL_INIT: ARC official action prior — ACTION1-ACTION5 are simple actions "
    "(e.g., move up/down/left/right, interact). Do not assume direction, interact target, "
    "or win condition until transition evidence exists. Probe untested simple actions in "
    "legal_actions_now before multi-step navigation."
)

V20_SYSTEM_PROMPT = """You control an abstract ARC-AGI-3 grid game.
Use the raw image, annotated image, object table, transition evidence, action outcome memory, and resource state together.
Object ids like O1/O2 are LOCAL to the current level. Game-level memory must not store bare O-ids; convert them to visual descriptors and roles.
ARC action types: RESET starts/restarts; ACTION1-ACTION5 are simple actions (e.g., move up/down/left/right, interact); ACTION6 requires (x,y) coordinates; ACTION7 is an additional simple action.
ACTION7 is often an undo/back action in public human-25 games. Do not propose ACTION7 unless recent evidence proves it advances this game; prefer another interaction/probe.
When an action causes transforms rather than simple translation, do not keep treating it as a movement vector. Explain the transformed object and propose the next different test.
For ACTION6 click games, give target_object_id and, when relevant, click_hint=center|edge|corner|outside-adjacent or explicit x,y.
Respect resource counters/lives: if each noop consumes counter, use high-information actions only and stop repeating guards.
game_action_space, when present, is the complete fixed set of actions usable for the whole game; never propose or plan actions outside it. When it is null/absent, rely on legal_actions_now instead.
Return exactly one JSON object. No markdown or prose outside JSON. Do not propose RESET unless terminal."""


def _image_for_vlm_log(name: str, img: Image.Image | None) -> dict[str, Any] | None:
    if img is None:
        return None
    try:
        rgb = img.convert("RGB")
        return {"name": name, "mode": rgb.mode, "size": list(rgb.size), "format": "png_data_url", "data_url": _pil_url(rgb)}
    except Exception as exc:
        return {"name": name, "error": str(exc)[:300]}


def _vlm_request_for_log(request: VLMRequest) -> dict[str, Any]:
    images = []
    for name, img in (("previous_rgb", request.previous_rgb), ("current_rgb", request.current_rgb), ("analysis_rgb", request.analysis_rgb)):
        payload = _image_for_vlm_log(name, img)
        if payload is not None:
            images.append(payload)
    return {
        "system_prompt": V20_SYSTEM_PROMPT,
        "text_prompt": request.text_prompt,
        "max_new_tokens": request.max_new_tokens,
        "message_image_order": [img["name"] for img in images],
        "images": images,
    }


def _vlm_raw_for_log(raw: Any) -> Any:
    if isinstance(raw, (str, int, float, bool)) or raw is None:
        return raw
    if isinstance(raw, (dict, list, tuple)):
        return _json_safe(raw)
    try:
        if hasattr(raw, "model_dump"):
            return _json_safe(raw.model_dump())
    except Exception:
        pass
    return str(raw)


def _vlm_result_for_log(result: "V20VLMResult | None") -> dict[str, Any] | None:
    return _json_safe(result) if result is not None else None


@dataclass
class V20VLMResult:
    mode: str = ""
    action_meaning_updates: list[dict[str, Any]] = field(default_factory=list)
    role_bindings: dict[str, str] = field(default_factory=dict)
    win_condition_update: dict[str, Any] = field(default_factory=dict)
    object_effect_updates: list[dict[str, Any]] = field(default_factory=list)
    mechanics_updates: list[str] = field(default_factory=list)
    resource_update: dict[str, Any] = field(default_factory=dict)
    plan_goal: str = ""
    plan: list[dict[str, Any]] = field(default_factory=list)
    bottleneck_analysis: str = ""
    notes_for_next_call: str = ""
    raw_invalid_excerpt: str = ""


def _balanced_objects(text: str) -> list[str]:
    out, in_string, quote, escape, depth, start = [], False, "", False, 0, -1
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue
        if ch in {'"', "'"}:
            in_string, quote = True, ch
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start:i + 1])
    return out


def _repair_truncated_json(text: str) -> dict[str, Any] | None:
    # VLM 输出常因 max_new_tokens 在 JSON 中途截断，导致 _balanced_objects 抓不到任何完整顶层对象。
    # 这里扫描找到最后一个"安全截断点"（完整闭合的 } ] 或顶层/浅层逗号），截断后补全未闭合括号。
    # 先剥掉 markdown 代码块标记（闭合或未闭合的 ```json ... ```），否则 json.loads 会因前缀失败。
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```\s*$", "", stripped)
    text = stripped
    in_string = False
    escape = False
    depth = 0
    last_safe = -1
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            if depth > 0:
                depth -= 1
                last_safe = i + 1
        elif ch == ",":
            # 任意深度的逗号（字符串外）都是安全截断点：逗号之前必然是完整的
            # 数组元素或完整的 key-value 对，截断后补闭合括号即可得到合法 JSON。
            # 典型场景：模型在 action_sequence 里退化复读 "ACTION2","ACTION2",...
            # 直到 max_tokens 截断——旧逻辑只认顶层逗号，会把整个 plan 丢掉。
            last_safe = i
    if last_safe <= 0:
        return None
    prefix = re.sub(r",\s*$", "", text[:last_safe].rstrip()).rstrip()
    if not prefix:
        return None
    # 重新扫描 prefix 确定其内部未闭合括号（不能用全扫描的栈，因为 prefix 已截断）
    in_string = False
    escape = False
    pstack: list[str] = []
    for ch in prefix:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            pstack.append(ch)
        elif ch in "}]":
            if pstack:
                pstack.pop()
    if not pstack:
        return None
    closing = "".join("}" if c == "{" else "]" for c in reversed(pstack))
    for cand in (prefix + closing, prefix + "," + closing):
        cand = re.sub(r",\s*([}\]])", r"\1", cand)
        try:
            val = json.loads(cand)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
    return None


def _parse_payload(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    candidates = re.findall(r"```json\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE) + _balanced_objects(cleaned)
    if not candidates:
        candidates = [cleaned]
    first_echo: dict[str, Any] | None = None
    for cand in sorted(candidates, key=len, reverse=True):
        for text in (cand.strip(), re.sub(r",\s*([}\]])", r"\1", cand.strip())):
            try:
                val = json.loads(text)
                if isinstance(val, dict):
                    if _schema_echo_score(val) >= 4:
                        first_echo = first_echo or val
                        continue
                    return val
            except Exception:
                pass
    repaired = _repair_truncated_json(cleaned)
    if repaired is not None:
        if _schema_echo_score(repaired) < 4:
            return repaired
        first_echo = first_echo or repaired
    return first_echo



_SCHEMA_ECHO_EXACT_TEXTS = {
    "short current-level goal",
    "transferable rule with no bare o-ids",
    "cite event/observation",
    "why",
    "event or visual clue",
    "short natural-language mechanic claim",
    "counter/life behavior",
    "moves actor up or transforms pattern",
    "<action-name>",
    "<click-action-name>",
    "<current-o-id>",
    "<role-name>",
    "<short-observed-effect>",
    "<observed-transition-evidence>",
    "<transferable-rule-no-bare-o-ids>",
    "<short-reusable-mechanic>",
    "<resource-life-counter-behavior>",
    "<current-level-goal>",
    "<why-this-action-is-next>",
    "<why-this-sequence-is-next>",
    "<why-this-click-is-next>",
    "<stop-after-observable-change>",
}

_SCHEMA_ECHO_PLAN_PURPOSES = {
    "approach target",
    "activate transformer",
    "complete goal",
    "<why-this-action-is-next>",
    "<why-this-sequence-is-next>",
    "<why-this-click-is-next>",
}


def _schema_text(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip().lower().replace("_", "-")


def _schema_key(value: Any) -> str:
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip().lower()
    text = re.sub(r"[\s_/]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


_RECOVERY_TEMPLATE_PURPOSE_KEYS = {
    "test-one-concrete-action",
    "test-transform-action-sequence",
    "test-exact-click-coordinates",
}


TRUSTED_ROUTE_REASONS = (
    "geometry",
    "coupled_carrier_geometry",
    "selector_probe",
    "selector_calibration",
    "selector_cycle_learn",
    "selector_pattern_rule",
    "mechanism_probe",
    "resource_recovery",
)


def _is_schema_echo_text(value: Any) -> bool:
    text = _schema_text(value)
    if not text:
        return False
    key = _schema_key(value)
    exact_keys = {_schema_key(x) for x in _SCHEMA_ECHO_EXACT_TEXTS}
    return text in _SCHEMA_ECHO_EXACT_TEXTS or text.replace("-", " ") in _SCHEMA_ECHO_EXACT_TEXTS or key in exact_keys


def _is_schema_echo_action_update(update: dict[str, Any]) -> bool:
    kind = _schema_text(update.get("kind"))
    if "|" in kind and {"movement", "interact-or-transform", "click-or-select", "undo", "resource", "unknown-or-blocked"}.issubset(set(kind.split("|"))):
        return True
    return any(_is_schema_echo_text(update.get(key)) for key in ("meaning_nl", "meaning", "evidence"))


def _is_schema_echo_win_update(update: dict[str, Any]) -> bool:
    if not update:
        return False
    if any(_is_schema_echo_text(update.get(key)) for key in ("description_nl", "description", "claim", "evidence")):
        return True
    roles = update.get("visual_roles") if isinstance(update.get("visual_roles"), dict) else {}
    for raw_desc in roles.values():
        if isinstance(raw_desc, dict) and _schema_text(raw_desc.get("inner_pattern")) == "...":
            return True
    return False


def _is_schema_echo_object_effect(update: dict[str, Any]) -> bool:
    return any(_is_schema_echo_text(update.get(key)) for key in ("effect_nl", "effect", "evidence"))


def _is_schema_echo_plan_step(step: dict[str, Any]) -> bool:
    purpose = _schema_text(step.get("purpose") or step.get("why"))
    purpose_key = _schema_key(step.get("purpose") or step.get("why"))
    target = _schema_text(step.get("target_object_id") or step.get("object_id"))
    role = _schema_text(step.get("target_role") or step.get("role"))
    action = _schema_text(step.get("action") or step.get("name"))
    plan_purpose_keys = {_schema_key(x) for x in _SCHEMA_ECHO_PLAN_PURPOSES}
    if any(_is_schema_echo_text(step.get(key)) for key in ("purpose", "why", "stop_condition", "expected_change", "route_reason")):
        return True
    if action in {"<action-name>", "<click-action-name>"} or target == "<current-o-id>" or role == "<role-name>":
        return True
    if (purpose in _SCHEMA_ECHO_PLAN_PURPOSES or purpose_key in plan_purpose_keys) and target in {"o1", "o4", ""} and role in {"target-frame", "transformer", ""}:
        return True
    return purpose_key == "complete-goal" and action == "action5"


def _schema_echo_score(payload: dict[str, Any]) -> int:
    score = 0
    if isinstance(payload.get("output_schema"), dict):
        score += 4
    if _is_schema_echo_text(payload.get("plan_goal") or payload.get("goal")):
        score += 1
    for update in payload.get("action_meaning_updates") if isinstance(payload.get("action_meaning_updates"), list) else []:
        if isinstance(update, dict) and _is_schema_echo_action_update(update):
            score += 2
    win = payload.get("win_condition_update") if isinstance(payload.get("win_condition_update"), dict) else {}
    if _is_schema_echo_win_update(win):
        score += 2
    for update in payload.get("object_effect_updates") if isinstance(payload.get("object_effect_updates"), list) else []:
        if isinstance(update, dict) and _is_schema_echo_object_effect(update):
            score += 1
    for claim in payload.get("mechanics_updates") if isinstance(payload.get("mechanics_updates"), list) else []:
        if _is_schema_echo_text(claim):
            score += 1
    res = payload.get("resource_update") if isinstance(payload.get("resource_update"), dict) else {}
    if _is_schema_echo_text(res.get("description_nl") or res.get("claim")):
        score += 1
    for step in payload.get("plan") if isinstance(payload.get("plan"), list) else []:
        if isinstance(step, dict) and _is_schema_echo_plan_step(step):
            score += 1
    return score


def _invalid_vlm_result(raw: Any, reason: str) -> V20VLMResult:
    excerpt = _short(raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=True, default=str), 1500)
    return V20VLMResult(raw_invalid_excerpt=f"{reason}: {excerpt}", mode="INVALID")


def parse_vlm_result(raw: Any) -> V20VLMResult | None:
    payload = _parse_payload(raw)
    if payload is None:
        return V20VLMResult(raw_invalid_excerpt=_short(raw, 1500), mode="INVALID") if isinstance(raw, str) and _short(raw, 1) else None

    echo_score = _schema_echo_score(payload)

    def dict_list(key: str, limit: int) -> list[dict[str, Any]]:
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(x) for x in value[:limit] if isinstance(x, dict)]
        if isinstance(value, dict):
            out: list[dict[str, Any]] = []
            for raw_key, raw_item in value.items():
                if not isinstance(raw_item, dict):
                    continue
                item = dict(raw_item)
                if key == "action_meaning_updates" and not item.get("action"):
                    item["action"] = raw_key
                out.append(item)
                if len(out) >= limit:
                    break
            return out
        return []

    def str_list(key: str, limit: int) -> list[str]:
        value = payload.get(key)
        return [_short(x, 360) for x in value[:limit] if _short(x, 360) and not _is_schema_echo_text(x)] if isinstance(value, list) else []

    action_updates = [x for x in dict_list("action_meaning_updates", 16) if not _is_schema_echo_action_update(x)]
    object_effects = [x for x in dict_list("object_effect_updates", 16) if not _is_schema_echo_object_effect(x)]
    raw_steps = dict_list("steps", 24)
    raw_plan = dict_list("plan", 24)
    step_candidates = [x for x in raw_steps if not _is_schema_echo_plan_step(x)]
    plan_candidates = [x for x in raw_plan if not _is_schema_echo_plan_step(x)]
    plan_steps = step_candidates or plan_candidates
    raw_win_update = payload.get("win_condition_update")
    if isinstance(raw_win_update, dict):
        win_update = dict(raw_win_update)
    elif _short(raw_win_update, 600):
        win_update = {"description_nl": _short(raw_win_update, 600), "confidence": 0.35}
    else:
        win_update = {}
    if _is_schema_echo_win_update(win_update):
        win_update = {}
    resource_update = dict(payload.get("resource_update")) if isinstance(payload.get("resource_update"), dict) else {}
    if _is_schema_echo_text(resource_update.get("description_nl") or resource_update.get("claim")):
        resource_update = {}
    plan_goal = _short(payload.get("plan_goal") or payload.get("goal"), 420)
    if _is_schema_echo_text(plan_goal):
        plan_goal = ""
    mechanics = str_list("mechanics_updates", 16)
    rb = payload.get("role_bindings") if isinstance(payload.get("role_bindings"), dict) else {}
    role_bindings = {_short(k, 80): _short(v, 24).upper() for k, v in rb.items() if _short(k, 80) and _short(v, 24)}
    bottleneck = _short(payload.get("bottleneck_analysis"), 900)
    notes = _short(payload.get("notes_for_next_call"), 900)
    useful_without_roles = any([action_updates, win_update, object_effects, mechanics, resource_update, plan_goal, plan_steps, bottleneck, notes])
    useful = useful_without_roles or bool(role_bindings)
    if (echo_score >= 3 and not useful_without_roles) or (echo_score and not useful):
        return _invalid_vlm_result(raw, "schema_echo_only")
    return V20VLMResult(
        mode=_short(payload.get("mode"), 60),
        action_meaning_updates=action_updates,
        role_bindings=role_bindings,
        win_condition_update=win_update,
        object_effect_updates=object_effects,
        mechanics_updates=mechanics,
        resource_update=resource_update,
        plan_goal=plan_goal,
        plan=plan_steps,
        bottleneck_analysis=bottleneck,
        notes_for_next_call=notes,
    )


def action_name(action: Any) -> str:
    if hasattr(action, "name"):
        return str(action.name)
    if hasattr(action, "id") and hasattr(action.id, "name"):
        return str(action.id.name)
    if hasattr(action, "x") and hasattr(action, "y"):
        return "ACTION6"
    for attr in ("action_data", "data"):
        data = getattr(action, attr, None)
        if isinstance(data, dict) and "x" in data and "y" in data:
            return "ACTION6"
    return str(action)


def action_value(action: Any) -> int:
    for obj in (action, getattr(action, "id", None)):
        if hasattr(obj, "value"):
            try:
                return int(obj.value)
            except Exception:
                pass
    if hasattr(action, "x") and hasattr(action, "y"):
        return 6
    m = re.search(r"ACTION(\d+)", action_name(action).upper())
    if m:
        return int(m.group(1))
    if action_name(action).upper() == "RESET":
        return 0
    try:
        return int(action)
    except Exception:
        return 999


def action6_data(action: Any) -> dict[str, int] | None:
    x, y = getattr(action, "x", None), getattr(action, "y", None)
    if x is not None and y is not None:
        try:
            return {"x": int(x), "y": int(y)}
        except Exception:
            return None
    for attr in ("action_data", "data"):
        data = getattr(action, attr, None)
        if isinstance(data, dict) and "x" in data and "y" in data:
            try:
                return {"x": int(data["x"]), "y": int(data["y"])}
            except Exception:
                return None
    return None


def state_name(state: Any) -> str:
    return str(getattr(state, "name", getattr(state, "value", state)))


def get_action_by_id(action_id: int) -> Any | None:
    if hasattr(GameAction, "from_id"):
        try:
            return GameAction.from_id(int(action_id))
        except Exception:
            pass
    try:
        for action in GameAction:
            if action_value(action) == int(action_id):
                return action
    except Exception:
        pass
    return None


def get_action_by_name(name: str) -> Any | None:
    clean = str(name).split(".")[-1].strip().upper()
    if clean.isdigit():
        return get_action_by_id(int(clean))
    try:
        return getattr(GameAction, clean)
    except Exception:
        m = re.fullmatch(r"ACTION(\d+)", clean)
        return get_action_by_id(int(m.group(1))) if m else None


def normalize_one_action(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return normalize_one_action(value.get("id", value.get("name")))
    if isinstance(value, str):
        return get_action_by_name(value)
    if isinstance(value, int):
        return get_action_by_id(value)
    if hasattr(value, "value") and action_name(value).upper().startswith(("ACTION", "RESET")):
        return value
    if hasattr(value, "id"):
        return normalize_one_action(value.id)
    return None


def normalize_legal_actions(frame_actions: Any, env_action_space: Iterable[Any] | None = None, *, allow_env_fallback: bool = True) -> tuple[Any, ...]:
    empty = frame_actions is None or (isinstance(frame_actions, (list, tuple, set)) and len(frame_actions) == 0)
    source = env_action_space if empty and allow_env_fallback and frame_actions is None else frame_actions
    if source is None:
        return ()
    raw_values = [source] if isinstance(source, (str, bytes)) or not isinstance(source, Iterable) else list(source)
    parsed: dict[int, Any] = {}
    for raw in raw_values:
        action = normalize_one_action(raw)
        if action is not None:
            parsed[action_value(action)] = action
    return tuple(parsed[k] for k in sorted(parsed))


def simple_action_names(legal: Sequence[Any]) -> list[str]:
    out = []
    for action in legal:
        name = action_name(action).upper()
        if name != "RESET" and name != "ACTION6" and re.fullmatch(r"ACTION\d+", name):
            out.append(name)
    return sorted(set(out), key=lambda n: int(re.search(r"\d+", n).group(0)))


class MyAgent(_BaseAgent):
    """Outcome-aware ARC-AGI-3 agent.

    V2.0 keeps the V1.9 observer/VLM scaffold, but changes control to be
    state-action-outcome driven: no-op evidence is never erased, repeated
    bad actions are quarantined, transform-heavy games stop using movement
    navigation, and ACTION6 click games explore structured click points.
    """

    MAX_ACTIONS = _int_env(("ARC_V20_MAX_ACTIONS", "ARC_V19_MAX_ACTIONS", "ARC_V18_MAX_ACTIONS", "ARC_V15_MAX_ACTIONS"), 240, 1, 10000)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        backend = kwargs.pop("backend", None)
        config = kwargs.pop("config", None)
        game_id = kwargs.pop("game_id", None)
        super().__init__(*args, **kwargs)
        if game_id is not None:
            self.game_id = game_id
        elif not hasattr(self, "game_id"):
            self.game_id = str(getattr(getattr(self, "arc_env", None), "game_id", "unknown") or "unknown")
        self.config: AgentConfig = config or AgentConfig.from_env()
        self.MAX_ACTIONS = self.config.max_actions
        self.observer = Observer(self.config.image_size, self.config.drift_cell_ratio)
        self.logger = DecisionLogger(self.config)
        self.backend = backend if backend is not None else make_vlm_backend(self.config, self.logger)
        self.memory = RuntimeMemoryV20()
        self._call_index = 0
        self._done = False
        self._last_drift_cell_count = 0

    @property
    def name(self) -> str:
        try:
            base = super().name
        except Exception:
            base = type(self).__name__
        return f"{base}.{self.config.agent_version}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        done = (
            state_name(getattr(latest_frame, "state", None)) == "WIN"
            or self._done
            or self.memory.level.total_action_count >= self.MAX_ACTIONS
            or self.memory.level.consecutive_nonterminal_resets >= 50
        )
        if done:
            self._flush_last_vlm_io("is_done")
        return done

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        self._call_index += 1
        state = state_name(getattr(latest_frame, "state", None))
        levels_completed = int(getattr(latest_frame, "levels_completed", 0) or 0)
        legal = normalize_legal_actions(getattr(latest_frame, "available_actions", None), self._env_action_space(), allow_env_fallback=(state == "NOT_PLAYED"))
        legal = self._reconcile_game_action_space(legal, state)
        try:
            if self.memory.level.total_action_count >= self.MAX_ACTIONS:
                self._flush_last_vlm_io("max_actions")
                self._done = True
                scene = self.memory.level.current_scene
                if scene is None and state != "NOT_PLAYED":
                    scene = self.observer.scene_from_frame(latest_frame)
                selected = self._fallback_nonreset_action(scene, legal, "max actions reached; return non-reset sentinel") if scene is not None else None
                if selected is not None:
                    action, proposal, source = selected
                    self._record_returned_action(action, proposal, scene, latest_frame, source="max_actions_sentinel")
                    return action
                return self._reset_action(reason="max_actions", state=state, scene=scene, legal=legal)

            if state == "NOT_PLAYED":
                return self._reset_action(reason="not_played", state=state, legal=legal)

            scene = self.observer.scene_from_frame(latest_frame)
            level = self.memory.level
            transition: TransitionReport | None = None
            just_started = level.initial_scene is None
            advanced = level.initial_scene is not None and levels_completed > level.levels_completed_at_start

            if just_started:
                self.observer.reset_level()
                scene = self.observer.analyze_grid(scene.grid, scene.rgb)
                self._start_new_level(levels_completed, scene, legal)
            elif advanced:
                transition = self._process_pending_transition(scene, latest_frame, cross_level=True)
                self._record_level_success(scene, levels_completed)
                if state == "WIN":
                    self._done = True
                    return self._reset_action(reason="win", state=state, scene=scene, legal=legal)
                self.observer.reset_level()
                scene = self.observer.analyze_grid(scene.grid, scene.rgb)
                self._start_new_level(levels_completed, scene, legal)
            elif level.awaiting_reset and state not in {"GAME_OVER", "NOT_PLAYED", "WIN"}:
                self.observer.reset_level()
                scene = self.observer.analyze_grid(scene.grid, scene.rgb)
                self._start_new_attempt(levels_completed, scene, legal)
            else:
                transition = self._process_pending_transition(scene, latest_frame)
                self.memory.level.current_scene = scene

            if state == "WIN":
                self._record_level_success(scene, max(levels_completed, self.memory.level.level_index + 1))
                self._done = True
                return self._reset_action(reason="win", state=state, scene=scene, legal=legal)
            if state == "GAME_OVER":
                self._record_game_over(scene, transition)
                return self._reset_action(reason="game_over", state=state, scene=scene, legal=legal)

            self._remember_current_state(scene)
            self._refresh_strategic_flags(scene, transition)
            self._maybe_break_loop(scene)
            if self.memory.level.bottleneck_reason:
                self._update_vlm_issue_state(scene, transition)

            vlm_calls_at_turn_start = self.memory.level.vlm_calls_this_level
            selected: tuple[Any, dict[str, Any], str] | None = None

            # V2.0: let current VLM plan execute before deterministic navigation.
            if self._should_call_vlm(scene, transition, legal):
                self._maybe_call_vlm(scene, transition, legal)
            selected = self._execute_next_plan_action(scene, legal)

            called_vlm_this_turn = self.memory.level.vlm_calls_this_level > vlm_calls_at_turn_start
            repeat_cooldown_active = (
                self.memory.level.vlm_issue_repeat_count >= 2
                and self.memory.level.actions_since_vlm < self.config.vlm_repeat_bottleneck_cooldown
            )
            fallback_guard_temporarily_blocked = (
                self.memory.level.fallback_guard_block_until > self.memory.level.total_action_count
                and self._vlm_available()
                and self.memory.level.vlm_calls_this_level < self.config.max_vlm_calls_per_level
            )
            if selected is None and fallback_guard_temporarily_blocked and not called_vlm_this_turn and not repeat_cooldown_active:
                self.memory.level.bottleneck_reason = self.memory.level.bottleneck_reason or "vlm_zero_step_needs_recovery_contract"
                self._request_vlm_once(scene, transition, legal, VLMMode.BOTTLENECK)
                selected = self._execute_next_plan_action(scene, legal)

            if selected is None:
                selected = self._confirm_probe_action(scene, legal)
            if selected is None and self._in_transform_mode(scene):
                selected = self._transform_controller_action(scene, legal)
            if selected is None:
                selected = self._deterministic_navigation_action(scene, legal)

            called_vlm_this_turn = self.memory.level.vlm_calls_this_level > vlm_calls_at_turn_start
            repeat_cooldown_active = (
                self.memory.level.vlm_issue_repeat_count >= 2
                and self.memory.level.actions_since_vlm < self.config.vlm_repeat_bottleneck_cooldown
            )
            should_ask_before_click = self._click_only_legal(legal) or self.memory.level.bottleneck_reason or self.memory.level.action_recovery_contract
            if selected is None and should_ask_before_click and not called_vlm_this_turn and not repeat_cooldown_active and self._should_call_vlm(scene, transition, legal, force_bottleneck=True):
                self.memory.level.bottleneck_reason = self.memory.level.bottleneck_reason or "no_executable_local_action"
                self._request_vlm_once(scene, transition, legal, VLMMode.BOTTLENECK)
                selected = self._execute_next_plan_action(scene, legal)

            if selected is None:
                selected = self._deterministic_click_action(scene, legal)
            if selected is None and not self._in_transform_mode(scene):
                selected = self._transform_controller_action(scene, legal)

            called_vlm_this_turn = self.memory.level.vlm_calls_this_level > vlm_calls_at_turn_start
            repeat_cooldown_active = (
                self.memory.level.vlm_issue_repeat_count >= 2
                and self.memory.level.actions_since_vlm < self.config.vlm_repeat_bottleneck_cooldown
            )
            if selected is None and not called_vlm_this_turn and not repeat_cooldown_active and self._should_call_vlm(scene, transition, legal, force_bottleneck=True):
                self.memory.level.bottleneck_reason = self.memory.level.bottleneck_reason or "no_executable_local_action"
                self._request_vlm_once(scene, transition, legal, VLMMode.BOTTLENECK)
                selected = self._execute_next_plan_action(scene, legal)

            if selected is None:
                selected = self._frontier_probe_action(scene, legal)
            if selected is None:
                fallback_guard_temporarily_blocked = (
                    self.memory.level.fallback_guard_block_until > self.memory.level.total_action_count
                    and self._vlm_available()
                    and self.memory.level.vlm_calls_this_level < self.config.max_vlm_calls_per_level
                )
                allow_guard = (
                    not fallback_guard_temporarily_blocked
                    and (
                        self.memory.level.vlm_calls_this_level > vlm_calls_at_turn_start
                        or not self._vlm_available()
                        or self.memory.level.vlm_calls_this_level >= self.config.max_vlm_calls_per_level
                    )
                )
                selected = self._fallback_nonreset_action(scene, legal, "safe non-reset fallback after outcome-aware planner exhaustion", allow_guard=allow_guard)
            if selected is None:
                if state not in {"WIN", "GAME_OVER", "NOT_PLAYED"} and not self.config.nonterminal_reset_allowed:
                    selected = self._least_bad_nonreset_action(scene, legal)
                    if selected is None:
                        selected = self._absolute_nonreset_action(scene, legal, "final non-reset fallback after all guards exhausted")
                if selected is None and state not in {"WIN", "GAME_OVER", "NOT_PLAYED"} and not self.config.nonterminal_reset_allowed:
                    selected = self._evidence_exhausted_nonreset_action(scene, legal, "active-state sentinel after all non-reset guards were blocked")
                if selected is None:
                    return self._reset_action(reason="no_nonreset_legal", state=state, scene=scene, legal=legal)

            action, proposal, source = selected
            self._record_returned_action(action, proposal, scene, latest_frame, source=source)
            return action
        except Exception as exc:
            self.logger.log_exception(exc)
            return self._emergency_action(latest_frame, legal)

    def _env_action_space(self) -> Iterable[Any] | None:
        return getattr(getattr(self, "arc_env", None), "action_space", None)

    def _game_space_actions(self) -> tuple[Any, ...]:
        """Materialize the captured game-level action space as action objects."""
        out: list[Any] = []
        for value in self.memory.game.game_action_space:
            action = normalize_one_action(int(value))
            if action is not None:
                out.append(action)
        return tuple(out)

    def _reconcile_game_action_space(self, legal: Sequence[Any], state: str) -> tuple[Any, ...]:
        """Capture the whole-game action space once, then constrain exploration to it.

        - First readable non-reset legal set is frozen as the game action space.
        - Once frozen, every later frame's legal is intersected with it (RESET kept),
          so all downstream exploration only considers these actions.
        - If a frame yields no in-space action (empty/corrupt frame), fall back to the
          frozen space so the agent still has candidates.
        - If it was never readable, return legal unchanged (legacy behaviour).
        """
        legal = tuple(legal)
        game = self.memory.game
        frame_values = tuple(sorted({action_value(a) for a in legal if action_value(a) >= 1}))
        if not game.game_action_space:
            if frame_values:
                game.game_action_space = frame_values
                self.logger.log_event("game_action_space_captured_v20", {"action_space": [f"ACTION{v}" for v in frame_values], "state": state, "level": self.memory.level.level_index})
        else:
            # 离线引擎的 available_actions 整局恒定，这里永不触发；防御的是远程 API
            # 按状态扩充动作集的情形——新动作并入而不是被交集永久丢弃。
            unseen = [v for v in frame_values if v not in game.game_action_space]
            if unseen:
                game.game_action_space = tuple(sorted(set(game.game_action_space) | set(unseen)))
                self.logger.log_event("game_action_space_expanded_v20", {"added": [f"ACTION{v}" for v in unseen], "action_space": [f"ACTION{v}" for v in game.game_action_space], "state": state, "level": self.memory.level.level_index})
        if not game.game_action_space:
            return legal
        space = set(game.game_action_space)
        constrained = tuple(a for a in legal if action_value(a) in space or action_value(a) == 0)
        if constrained:
            return constrained
        return self._game_space_actions()

    def _action7_probe_allowed(self) -> bool:
        """Opening knowledge from the declared whole-game action space: when the
        space is small (<=3 non-click actions) and explicitly includes ACTION7,
        the designer chose it as part of a tiny vocabulary, so it is likely a core
        mechanic rather than undo (sb26 space is 5/6/7, su15 is 6/7). Grant a
        bounded number of real probes before the global ACTION7 avoidance applies.

        Disabled by default (config.action7_space_probe): current stage treats
        ACTION7 strictly as undo and prioritizes completion first."""
        if not self.config.action7_space_probe:
            return False
        space = self.memory.game.game_action_space
        if not space or 7 not in space:
            return False
        if len([v for v in space if v != 6]) > 3:
            return False
        meaning = self.memory.game.action_meanings.get("ACTION7")
        if meaning is None:
            return True
        if meaning.retries or meaning.life_losses or meaning.kind == "undo":
            return False
        return meaning.attempts < 2

    def _game_id(self) -> str:
        return str(getattr(self, "game_id", "unknown"))

    def _resource_state_from_scene(self, scene: SceneSnapshot) -> dict[str, Any]:
        return {"counter": scene.counter_value, "capacity": scene.counter_capacity, "ratio": scene.counter_ratio, "lives": scene.life_count}

    def _start_new_level(self, levels_completed: int, scene: SceneSnapshot, legal: Sequence[Any]) -> None:
        if self.memory.level.initial_scene is not None:
            self._flush_last_vlm_io("before_new_level")
        self.memory.level = LevelMemoryV20(
            level_index=levels_completed,
            levels_completed_at_start=levels_completed,
            initial_scene=scene,
            current_scene=scene,
            resource_state=self._resource_state_from_scene(scene),
        )
        self._append_recent_state_hash(scene.state_hash)
        self._update_resource_model_from_scene(scene)
        self._seed_selector_pair_hints(scene)
        self._seed_initial_plan(scene, legal)
        self.logger.log_event("new_level_v20", {"level": levels_completed, "state": scene.state_hash[:12], "objects": [o.track_id for o in scene.objects], "counter": scene.counter_value, "lives": scene.life_count})


    def _start_new_attempt(self, levels_completed: int, scene: SceneSnapshot, legal: Sequence[Any]) -> None:
        old = self.memory.level
        attempt_index = old.attempt_index + 1
        total_actions = old.total_action_count
        self.memory.level = LevelMemoryV20(
            level_index=old.level_index,
            levels_completed_at_start=levels_completed,
            attempt_index=attempt_index,
            attempt_started_at_action_count=total_actions,
            initial_scene=scene,
            current_scene=scene,
            # Floor colour is a property of the level's rendering, not of any tracked
            # object id, so (unlike actor/ticker evidence, which is keyed by track_id
            # and reset_level() reassigns ids from scratch) it stays valid across a
            # same-level reset and is safe to carry over.
            walkable_color_votes=old.walkable_color_votes,
            recent_events=old.recent_events,
            known_noops_by_state=old.known_noops_by_state,
            tried_actions_by_state=old.tried_actions_by_state,
            transition_graph=old.transition_graph,
            state_action_outcomes=old.state_action_outcomes,
            action_outcomes=old.action_outcomes,
            quarantine_until_by_state=old.quarantine_until_by_state,
            global_quarantine_until=old.global_quarantine_until,
            total_action_count=total_actions,
            resource_state=self._resource_state_from_scene(scene),
            click_noops_by_object=old.click_noops_by_object,
            click_success_by_object=old.click_success_by_object,
            click_success_coords=old.click_success_coords,
            click_success_regions=old.click_success_regions,
            click_coord_counts=old.click_coord_counts,
            click_region_counts=old.click_region_counts,
            bad_action_suffixes=list(old.bad_action_suffixes),
            contract_forbidden_by_state=old.contract_forbidden_by_state,
            fallback_guard_block_until=old.fallback_guard_block_until,
            click_fuse_block_until=old.click_fuse_block_until,
            successful_clicks_by_state=old.successful_clicks_by_state,
            click_edges_by_state=old.click_edges_by_state,
            action_progress_scores=old.action_progress_scores,
            global_forbidden_click_coords=old.global_forbidden_click_coords,
            nonprogress_transform_coords=old.nonprogress_transform_coords,
            nonprogress_transform_actions=old.nonprogress_transform_actions,
            nonprogress_transform_states=old.nonprogress_transform_states,
            counter_is_step_tax=old.counter_is_step_tax,
            # V2.4: a game-over reset is the SAME level/puzzle with fresh object ids,
            # not a new mechanism. ls20 regression: this reset to 0 previously, so
            # attempt 2's LEVEL_INIT immediately re-issued a banned geometry route
            # because the "locked" check looked like a fresh, un-escalated level.
            hypothesis_escalation_count=old.hypothesis_escalation_count,
        )
        level = self.memory.level
        self._append_recent_state_hash(scene.state_hash)
        level.bottleneck_reason = "new_attempt_after_game_over"
        self._update_resource_model_from_scene(scene)
        self._seed_selector_pair_hints(scene)
        if self._mechanism_hypothesis_locked():
            # Re-bind status_template_frame/selector_frame to this attempt's fresh
            # O-ids without incrementing the escalation counter or re-clearing plan
            # state; the hypothesis itself (game.win_condition) already persists.
            self._bind_selector_pair_roles(scene)
        self._seed_initial_plan(scene, legal)
        self.logger.log_event("new_attempt_v20", {"level": level.level_index, "attempt": attempt_index, "state": scene.state_hash[:12], "total_actions": total_actions, "known_noops_preserved": sum(len(v) for v in level.known_noops_by_state.values()), "click_coord_counts": dict(level.click_coord_counts.most_common(8))})

    def _remember_current_state(self, scene: SceneSnapshot) -> None:
        level = self.memory.level
        level.current_scene = scene
        self._append_recent_state_hash(scene.state_hash)
        level.resource_state = self._resource_state_from_scene(scene)
        drift_count = len(self.observer.drift_cells)
        if drift_count != self._last_drift_cell_count:
            self.logger.log_event("auto_drift_cells_v20", {"level": level.level_index, "cells": drift_count, "previous": self._last_drift_cell_count, "sample": sorted(self.observer.drift_cells)[:12]})
            self._last_drift_cell_count = drift_count

    def _append_recent_state_hash(self, state_hash: str) -> None:
        level = self.memory.level
        if level.recent_state_hashes and level.recent_state_hashes[-1] == state_hash:
            return
        level.recent_state_hashes.append(state_hash)
        level.all_state_visits[state_hash] += 1

    def _productive_motion_event(self, event: CompactEvent) -> bool:
        if event.outcome in {"noop", "retry"}:
            return False
        try:
            progress = float(event.transition_delta.get("progress_score", 1.0))
        except Exception:
            progress = 1.0
        if progress <= 0.25:
            return False
        if event.outcome in {"movement", "movement_with_transform", "interaction", "level_advanced"}:
            return True
        delta = event.transition_delta if isinstance(event.transition_delta, dict) else {}
        if event.outcome == "transform":
            # A transform is only "productive" if it did not just bounce us back into a
            # state we keep revisiting. ft09-style decorative flicker cycles among a few
            # already-seen states with no real progress and must NOT count as a route.
            if self.memory.level.all_state_visits.get(event.after_state, 0) >= self.config.churn_state_visit_cap:
                return False
            if (
                delta.get("moved_objects")
                or delta.get("appeared")
                or delta.get("disappeared")
                or delta.get("transformed_objects")
            ):
                return True
        return False

    def _recent_productive_route(self, *, min_len: int = 3) -> bool:
        recent = list(self.memory.level.recent_events)[-min_len:]
        return len(recent) >= min_len and all(self._productive_motion_event(event) for event in recent)

    def _unproductive_state_loop_signal(self, scene: SceneSnapshot) -> tuple[bool, int, str]:
        level = self.memory.level
        hashes = list(level.recent_state_hashes)
        if len(hashes) < 6:
            return False, 0, ""
        recent_events = list(level.recent_events)[-8:]
        if len(recent_events) < 4:
            return False, 0, ""
        if self._recent_productive_route(min_len=3):
            return False, 0, ""
        if recent_events and self._productive_motion_event(recent_events[-1]):
            return False, 0, ""
        state_visits = Counter(hashes)
        max_visits = max(state_visits.values()) if state_visits else 0
        if max_visits < 3:
            return False, 0, ""
        current_visits = state_visits.get(scene.state_hash, 0)
        if current_visits < 2:
            return False, 0, ""
        # Count both hard no-ops/retries and "changed pixels but no real progress"
        # transforms as unproductive, so decorative churn loops (ft09/tr87) are caught.
        unproductive = sum(1 for event in recent_events if not self._productive_motion_event(event))
        if unproductive < max(3, int(len(recent_events) * 0.55)):
            return False, 0, ""
        unique_recent_states = len(set(hashes[-6:]))
        if unique_recent_states <= 2 and current_visits >= 2:
            return True, max_visits, "state_ping_pong_without_progress"
        if current_visits >= 3 and unproductive >= len(recent_events) - 1:
            return True, max_visits, "state_loop_without_erasing_noops"
        return False, 0, ""

    def _maybe_break_loop(self, scene: SceneSnapshot) -> None:
        level = self.memory.level
        loop_active, max_visits, loop_reason = self._unproductive_state_loop_signal(scene)
        if not loop_active:
            return
        # V1.9 cleared known_noops here. V2.0 keeps evidence and quarantines recent no-op actions instead.
        q_before = dict(level.quarantine_until_by_state.get(scene.state_hash, {}))
        q = level.quarantine_until_by_state.setdefault(scene.state_hash, {})
        for key in list(level.known_noops_by_state.get(scene.state_hash, set()))[-8:]:
            q[key] = max(q.get(key, 0), level.total_action_count + self.config.quarantine_steps)
        if level.recent_action_keys and level.recent_events:
            key = level.recent_action_keys[-1]
            last_event = level.recent_events[-1]
            # Only quarantine the last action when it actually failed here. Revisiting a
            # state via productive movement/navigation is normal and must not block it.
            if last_event.outcome in {"noop", "retry"} and not key.startswith("ACTION6:"):
                q[key] = max(q.get(key, 0), level.total_action_count + max(2, self.config.action_repeat_cooldown))
        if q == q_before and not q:
            return
        level.bottleneck_reason = level.bottleneck_reason or loop_reason
        # tr87-style: when the loop is dominated by transforms, tell the VLM to check
        # whether a selector/actor pattern already matches the target and, if so, stop
        # cycling and commit instead of transforming forever.
        if not level.notes_for_next_call:
            recent_tr = sum(1 for e in list(level.recent_events)[-8:] if e.outcome in {"transform", "movement_with_transform"})
            if recent_tr >= 4:
                level.notes_for_next_call = "Repeated transforms are cycling among a few states without progress. If a selector/actor pattern already matches its target, STOP transforming and commit/confirm; otherwise switch to a different action or target."
        signature = f"{scene.state_hash}:{max_visits}:{loop_reason}:{','.join(sorted(q))}"
        if signature != level.last_loop_break_signature or level.total_action_count - level.last_loop_break_logged_at >= self.config.quarantine_steps:
            level.last_loop_break_signature = signature
            level.last_loop_break_logged_at = level.total_action_count
            self.logger.log_event("loop_break_v20", {"state": scene.state_hash[:12], "max_state_visits": max_visits, "reason": loop_reason, "known_noops_kept": sorted(level.known_noops_by_state.get(scene.state_hash, set())), "quarantined": q})

    def _process_pending_transition(self, current_scene: SceneSnapshot, latest_frame: Any, *, cross_level: bool = False) -> TransitionReport | None:
        pending = self.memory.level.pending_action
        if pending is None:
            return None
        if id(latest_frame) == pending.source_frame_id and current_scene.state_hash == pending.scene_before.state_hash:
            return None
        report = self.observer.compare(pending.scene_before, current_scene)
        report.previous_rgb = pending.scene_before.rgb
        report.annotated_rgb = current_scene.annotated_rgb
        report.action_key = pending.action_key()
        report.action_source = pending.source
        if not cross_level:
            self._filter_ticker_transforms(pending, report)
        self.memory.level.last_resolved_pending_action = pending
        self.memory.level.pending_action = None
        self._record_transition(pending, report, current_scene, cross_level=cross_level)
        return report

    TICKER_STREAK_THRESHOLD = 6
    TICKER_MIN_DISTINCT_ACTIONS = 3

    def _filter_ticker_transforms(self, pending: PendingAction, report: TransitionReport) -> None:
        """Detect timer-bar/ambient-ticker objects and strip them from the transform view.

        tn36/vc33/lf52/s5i5 have a bar that ticks on every transition regardless of the
        action, so every step classified as "transform" with progress<=0 and the
        nonprogress contract/bottleneck machinery fired on literally every action. An
        object is a ticker once it transformed on TICKER_STREAK_THRESHOLD consecutive
        transitions under at least TICKER_MIN_DISTINCT_ACTIONS different action keys
        (the action diversity requirement keeps a puzzle piece hammered by one repeated
        action from being misclassified)."""
        level = self.memory.level
        transformed_ids = {str(t.get("object_id") or "") for t in report.transformed_objects}
        transformed_ids.discard("")
        moved_ids = {str(m.get("object_id") or "") for m in report.moved_objects}
        for oid in list(level.object_transform_streaks):
            if oid not in transformed_ids:
                level.object_transform_streaks.pop(oid, None)
                level.object_transform_streak_actions.pop(oid, None)
                level.object_transform_streak_moves.pop(oid, None)
        action_key = pending.action_key()
        for oid in transformed_ids:
            level.object_transform_streaks[oid] += 1
            actions_seen = level.object_transform_streak_actions.setdefault(oid, set())
            actions_seen.add(action_key)
            if oid in moved_ids:
                level.object_transform_streak_moves[oid] += 1
            streak = level.object_transform_streaks[oid]
            moves = level.object_transform_streak_moves.get(oid, 0)
            # A real ticker changes without being carried around. sp80's coupled frames
            # O4/O6 transform AND translate on ~97% of steps (they are the actor pair),
            # while vc33/tn36/lf52 timer bars transform every step but translate on only
            # ~0-30% (a shrinking bar drifts its centroid occasionally). Displacement
            # magnitude does not separate them (both move ~1 cell), but move RATE does:
            # require the streak to be dominated by non-moving transforms so controlled/
            # coupled movers are never mislabeled tickers, without weakening bar detection.
            if (
                oid not in level.ticker_object_ids
                and oid != level.controlled_object_id
                and streak >= self.TICKER_STREAK_THRESHOLD
                and len(actions_seen) >= self.TICKER_MIN_DISTINCT_ACTIONS
                and moves * 2 <= streak
            ):
                level.ticker_object_ids.add(oid)
                self.logger.log_event(
                    "ticker_object_detected_v20",
                    {"level": level.level_index, "object": oid, "streak": level.object_transform_streaks[oid], "distinct_actions": len(actions_seen)},
                )
                level.notes_for_next_call = _short(
                    f"Object {oid} changes on EVERY step regardless of the action (timer/ticker). Ignore its changes when judging action effects; do not treat its tick as progress or as a transform mechanic. "
                    + (level.notes_for_next_call or ""),
                    700,
                )
        if not level.ticker_object_ids:
            return
        kept = [t for t in report.transformed_objects if str(t.get("object_id") or "") not in level.ticker_object_ids]
        if len(kept) == len(report.transformed_objects):
            return
        report.transformed_objects = kept
        structural_left = bool(kept or report.appeared_object_ids or report.disappeared_object_ids)
        # Re-derive interaction_event with the ticker removed (compare() had set it
        # from the unfiltered transform list).
        if report.interaction_event and not (
            report.retry_detected
            or (report.counter_delta is not None and report.counter_delta > 1)
            or report.life_delta not in (None, 0)
            or structural_left
        ):
            report.interaction_event = False
        # If nothing but the ticker changed, the action was effectively a no-op.
        if (
            not structural_left
            and not report.moved_objects
            and (report.counter_delta is None or report.counter_delta <= 0)
            and report.life_delta in (None, 0)
            and not report.retry_detected
        ):
            report.effective_noop = True
            report.summary = report.summary + "; ticker_only=true"

    def _transition_delta_dict(self, report: TransitionReport) -> dict[str, Any]:
        return {
            "changed_cell_count": report.changed_cell_count,
            "world_changed_cell_count": report.world_changed_cell_count,
            "effective_noop": report.effective_noop,
            "counter_delta": report.counter_delta,
            "counter_ratio_after": report.counter_ratio_after,
            "life_delta": report.life_delta,
            "retry_detected": report.retry_detected,
            "moved_objects": report.moved_objects[:6],
            "transformed_objects": report.transformed_objects[:6],
            "appeared": report.appeared_object_ids[:6],
            "disappeared": report.disappeared_object_ids[:6],
            "controlled_candidate_id": report.controlled_candidate_id,
            "simple_translation": report.is_simple_translation,
            "interaction_event": report.interaction_event,
        }

    @staticmethod
    def _controlled_motion(report: TransitionReport, preferred_id: str | None = None) -> dict[str, Any] | None:
        candidate_ids = []
        for cid in (preferred_id, report.controlled_candidate_id):
            if cid and cid not in candidate_ids:
                candidate_ids.append(cid)
        for cid in candidate_ids:
            for move in report.moved_objects:
                if str(move.get("object_id") or "") != cid:
                    continue
                try:
                    dx, dy = int(move.get("dx") or 0), int(move.get("dy") or 0)
                except (TypeError, ValueError):
                    continue
                if dx or dy:
                    return move
        return None

    def _minor_structural_churn(self, report: TransitionReport, visits_before: int | None = None) -> bool:
        if not report.transformed_objects:
            return False
        if report.appeared_object_ids or report.disappeared_object_ids:
            return False
        if report.is_simple_translation or self._controlled_motion(report) is not None:
            return False
        if visits_before == 0:
            # V2.6: sb26 regression - a small pixel footprint alone is not evidence of
            # decorative churn when the resulting state has never been seen before this
            # level (e.g. a selector bar that shrinks by exactly one cell per press:
            # world_changed_cell_count==1 every time, but each press lands on a
            # genuinely new state, not a repeating flicker). Only treat a small
            # transform as churn once it actually revisits an already-seen state;
            # callers that cannot determine novelty keep passing None, which preserves
            # the old always-penalize behavior.
            return False
        return report.world_changed_cell_count <= 2

    @staticmethod
    def _predicates_expect_structural_change(predicates: Sequence[dict[str, Any]] | None) -> bool:
        """True only if the step itself declared it expected a transform/appear/
        disappear. Steps that only ask for not_noop/no_retry/no_life_loss/
        controlled_motion are a pure-movement bet: controlled_motion is satisfied by
        transform-coupled motion too (an actor merging into a container still has
        dx/dy != 0), so those steps must NOT be treated as having anticipated a
        structural change just because they also pass."""
        for pred in predicates or ():
            if not isinstance(pred, dict):
                continue
            kind = _short(pred.get("type") or pred.get("predicate"), 80).lower()
            if kind in {"structural_change", "appeared_or_disappeared"}:
                return True
        return False

    def _classify_outcome(self, report: TransitionReport) -> str:
        if report.retry_detected:
            return "retry"
        if report.effective_noop:
            return "noop"
        move = self._controlled_motion(report)
        structural = bool(
            report.transformed_objects
            or report.appeared_object_ids
            or report.disappeared_object_ids
        )
        if report.appeared_object_ids or report.disappeared_object_ids:
            return "transform"
        if move is not None:
            return "movement"
        if report.is_simple_translation or (report.controlled_candidate_id and report.moved_objects):
            return "movement"
        if structural:
            return "transform"
        if report.interaction_event:
            return "interaction"
        if report.counter_delta is not None or report.life_delta is not None:
            return "resource_delta"
        return "state_change"

    def _record_transition(self, pending: PendingAction, report: TransitionReport, current_scene: SceneSnapshot, *, cross_level: bool = False) -> None:
        level = self.memory.level
        before_state, after_state, action_key = pending.scene_before.state_hash, current_scene.state_hash, pending.action_key()
        outcome = "level_advanced" if cross_level else self._classify_outcome(report)
        progress_score = 1.0 if cross_level else self._transition_progress_score(pending, report, current_scene)
        if not cross_level and self._hidden_resource_transition_consumed(pending, report, outcome, current_scene):
            level.hidden_resource_used_this_attempt += 1
        if report.counter_delta == -1:
            level.counter_tick_streak += 1
            if level.counter_tick_streak >= 10 and not level.counter_is_step_tax:
                # ls20-style: the counter is a per-step tax every action pays. There is
                # no "saving" it; resource guards must stop vetoing ordinary actions.
                level.counter_is_step_tax = True
                # Purge resource_guard bans accumulated before detection (all written
                # from bankruptcy-poisoned evidence); they persist forever otherwise
                # because crisis rarely clears in a step-tax game.
                purged = 0
                for rules in level.contract_forbidden_by_state.values():
                    for rk in [k for k, v in rules.items() if str(v).startswith("resource_guard:")]:
                        del rules[rk]
                        purged += 1
                self.logger.log_event("counter_step_tax_detected_v20", {"level": level.level_index, "streak": level.counter_tick_streak, "counter": report.counter_ratio_after, "resource_guard_rules_purged": purged})
        elif report.counter_delta is not None and report.counter_delta <= 0:
            # A refill (>0) is the bankruptcy tick, not a break in the per-step tax.
            # Only a genuine non-decrement resets the streak, so the +42 refills in
            # ls20 no longer prevent/reset detection.
            level.counter_tick_streak = 0
        if not cross_level:
            level.tried_actions_by_state.setdefault(before_state, set()).add(action_key)
            level.state_action_outcomes.setdefault(before_state, {}).setdefault(action_key, Counter())[outcome] += 1
            level.action_outcomes.setdefault(pending.name, Counter())[outcome] += 1
            if report.effective_noop:
                level.known_noops_by_state.setdefault(before_state, set()).add(action_key)
                self._quarantine_noop_action(before_state, pending, action_key)
            else:
                level.transition_graph.setdefault(before_state, {})[action_key] = after_state
            self._update_action_meaning_from_transition(pending, report, current_scene)
            self._update_actor_and_terrain(pending, report, current_scene)
            self._update_plan_after_transition(pending, report, current_scene)
            self._update_click_memory(pending, report, current_scene)
            if pending.name != "RESET":
                level.action_progress_scores.setdefault(pending.name, deque(maxlen=8)).append(progress_score)

        if not cross_level:
            self._advance_chain_target_bindings(pending, report, current_scene)

        delta = self._transition_delta_dict(report)
        delta["progress_score"] = round(progress_score, 3)
        if level.hidden_resource_used_this_attempt:
            delta["hidden_resource_used_this_attempt"] = level.hidden_resource_used_this_attempt
        event = CompactEvent(self.memory.next_event_id, level.level_index, level.total_action_count, action_key, pending.source, before_state, after_state, outcome, ("level_advanced_by=" + action_key) if cross_level else report.summary[:700], delta)
        self.memory.next_event_id += 1
        level.recent_events.append(event)
        level.recent_action_keys.append(action_key)
        self._append_recent_state_hash(after_state)

        self._enforce_transition_evidence_contract(pending, report, current_scene, outcome)

        if report.transformed_objects:
            level.mechanism_evidence_object_ids.update(str(t.get("object_id") or "") for t in report.transformed_objects if t.get("object_id"))
        if outcome == "transform":
            level.transform_pressure += 1
            level.transform_state_visits[after_state] += 1
            if pending.source not in self._UNDIRECTED_ACTION_SOURCES:
                level.transform_state_directed_visits[after_state] += 1
            level.repeated_transform_streak = level.repeated_transform_streak + 1 if level.recent_action_keys and (len(level.recent_action_keys) < 2 or level.recent_action_keys[-2] == action_key) else 1
            if level.transform_state_visits[after_state] >= 3:
                level.bottleneck_reason = "transform_state_cycle"
                level.current_plan = []
                level.plan_cursor = 0
                # V2.5: ls20 regression - with the VLM stuck returning empty plans,
                # least_bad_nonreset/frontier_probe fallback wandering alone bounced
                # the actor back into the same merged-with-frame state 3x and
                # permanently escalated the mechanism hypothesis (see
                # _mechanism_hypothesis_locked, which never releases for the rest of
                # the level/attempts) even though nothing had actually probed a real
                # selector mechanism. Only escalate when at least one of the
                # repeated visits came from a directed plan/controller action.
                if level.transform_state_directed_visits.get(after_state, 0) > 0:
                    self._maybe_escalate_mechanism_hypothesis(current_scene, "transform_state_cycle")
            elif pending.source in {"deterministic_navigation", "transform_controller"} or (
                pending.source == "plan_executor" and not self._predicates_expect_structural_change(pending.expected_predicates)
            ):
                # V2.3: a plan_executor step from a long geometry action_sequence
                # (ls20: "move up 40 units") only declares a pure-movement
                # expectation (not_noop/no_retry/no_life_loss/controlled_motion) yet
                # still landed on a transform (actor merge, status-frame cycle, ...).
                # _update_plan_after_transition already marked it matched/done on
                # that predicate set alone, so without this the executor keeps
                # dispatching the rest of the stale route blind to the mechanism
                # change. Drop it and force a fresh bottleneck-driven VLM look.
                level.current_plan = []
                level.plan_cursor = 0
                level.bottleneck_reason = "transform_after_" + action_key
        elif outcome in {"movement", "movement_with_transform", "interaction", "state_change", "level_advanced"}:
            level.repeated_transform_streak = 0
        if outcome == "noop":
            level.repeated_noop_streak += 1
        else:
            level.repeated_noop_streak = 0
        if report.retry_detected or (report.life_delta is not None and report.life_delta < 0):
            level.bottleneck_reason = "retry_or_life_loss_after_" + action_key
            level.current_plan = []
            level.plan_cursor = 0
        if (report.interaction_event or report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids) and not (report.controlled_candidate_id and report.moved_objects):
            level.chunk_action_count = self.config.max_chunk_steps
        if (
            pending.source in {"plan_executor", "frontier_probe", "deterministic_click", "transform_controller", "deterministic_navigation", "nonreset_guard", "safe_probe_fallback", "absolute_nonreset", "escape_nonreset"}
            and (report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids or report.interaction_event)
            and progress_score <= 0.0
        ):
            # V2.2: forbid only after the same action produced nonprogress transforms
            # in >=2 distinct states. A single low-score transform is normal
            # exploration noise; first-sight bans exploded to ~1400/run and their only
            # surviving effect (state hashes drift every step) was a bottleneck storm.
            level.nonprogress_transform_actions[action_key] += 1
            level.nonprogress_transform_states.setdefault(action_key, set()).add(before_state)
            if level.nonprogress_transform_actions[action_key] >= 2 and len(level.nonprogress_transform_states[action_key]) >= 2:
                self._contract_forbid_action(
                    pending.scene_before,
                    pending.action_key(),
                    f"nonprogress_transform:{pending.source}:{progress_score:.2f}",
                )
                level.bottleneck_reason = level.bottleneck_reason or "nonprogress_transform_action"
        elif progress_score > 0.5 and level.nonprogress_transform_actions.get(action_key, 0):
            # Real progress rehabilitates the action (sp80's winning ACTION5 had been
            # nonprogress-banned twice before it advanced the level).
            level.nonprogress_transform_actions.pop(action_key, None)
            level.nonprogress_transform_states.pop(action_key, None)
        # V2.1 coordinate-level global fuse. A click that only churns decorative/cycling
        # pixels (ft09 38,38) keeps yielding "transform" with no real progress while the
        # state_hash drifts, so the state-scoped contract above never catches it. Track
        # such clicks per coordinate (survives state drift) and ban them outright.
        churn_transform = (
            outcome == "transform"
            and progress_score <= 0.0
            and not (report.controlled_candidate_id and report.moved_objects)
            and (report.counter_delta is None or report.counter_delta <= 0)
            and report.life_delta in (None, 0)
        )
        if churn_transform and action_key.startswith("ACTION6:") and action_key not in level.global_forbidden_click_coords:
            level.nonprogress_transform_coords[action_key] += 1
            if level.nonprogress_transform_coords[action_key] >= self.config.churn_click_fuse_after:
                level.global_forbidden_click_coords[action_key] = f"churn_transform_x{level.nonprogress_transform_coords[action_key]}"
                level.current_plan = []
                level.plan_cursor = 0
                level.bottleneck_reason = level.bottleneck_reason or "click_churn_fuse"
                level.notes_for_next_call = "A click coordinate only produced decorative/cycling pixel changes with no real progress and is now permanently banned this level. Re-check the true win condition and try a different mechanism, object, or coordinate."
                self.logger.log_event("click_churn_fuse_v20", {"level": level.level_index, "action": action_key, "count": level.nonprogress_transform_coords[action_key], "state": current_scene.state_hash[:12]})
        self.logger.log_event("transition_v20", event.as_prompt())

    def _advance_chain_target_bindings(self, pending: PendingAction, report: TransitionReport, current_scene: SceneSnapshot) -> None:
        """Advance target role bindings along chain levels (ls20: reaching glyph N
        consumes it and spawns glyph N+1). Previously the stale binding pointed at a
        vanished object, _navigation_target_id fell back to interest-scored junk
        (door frame / HUD, 38 wasted nav steps), and the target only refreshed after
        another VLM round-trip."""
        level = self.memory.level
        if not report.disappeared_object_ids:
            return
        disappeared = set(report.disappeared_object_ids)
        for role in ("target", "goal", "target_frame", "exit"):
            oid = level.local_bindings.get(role, "")
            if not oid or oid not in disappeared:
                continue
            old_obj = pending.scene_before.object_by_id(oid)
            replacement = ""
            candidates = [current_scene.object_by_id(a) for a in report.appeared_object_ids]
            candidates = [c for c in candidates if c is not None and c.track_id != level.controlled_object_id]
            if candidates:
                if old_obj is not None:
                    same_kind = [c for c in candidates if c.type_key == old_obj.type_key or c.shape_label == old_obj.shape_label]
                    pick = same_kind[0] if same_kind else candidates[0]
                else:
                    pick = candidates[0]
                replacement = pick.track_id
            if replacement:
                level.local_bindings[role] = replacement
                self.logger.log_event("target_chain_advanced_v20", {"level": level.level_index, "role": role, "old": oid, "new": replacement})
            else:
                level.local_bindings.pop(role, None)
                self.logger.log_event("target_chain_binding_cleared_v20", {"level": level.level_index, "role": role, "old": oid})

    def _actor_merged_into_container(self, pending: PendingAction, current_actor: str, *, new_candidate: ObjectObservation | None = None) -> bool:
        old_obj = pending.scene_before.object_by_id(current_actor)
        if old_obj is None:
            return False
        if old_obj.frame_color is not None:
            # V2.5: ls20 regression - once a first merge already rebound
            # controlled_object_id to the container frame itself, that frame's own
            # inner-pattern churn makes it look "disappeared" again on almost every
            # following step even though the actor never left it. Previously only
            # the very first merge was preserved and every step after that
            # cascaded into actor_disappeared/clear_all, racing through new object
            # ids (e.g. O2->O5->O8->...) for the rest of the level. As long as the
            # newly tracked candidate is itself another framed object near the
            # same spot, treat this as sustained containment instead.
            if new_candidate is not None and new_candidate.frame_color is not None:
                return self._bbox_intersects_with_margin(old_obj.bbox, new_candidate.bbox, 3)
            return False
        for cand in pending.scene_before.objects:
            if cand.track_id == current_actor or cand.frame_color is None:
                continue
            if self._bbox_intersects_with_margin(old_obj.bbox, cand.bbox, 3):
                return True
        return False

    def _enforce_transition_evidence_contract(self, pending: PendingAction, report: TransitionReport, current_scene: SceneSnapshot, outcome: str) -> None:
        level = self.memory.level
        current_actor = level.controlled_object_id
        disappeared = set(report.disappeared_object_ids)
        if current_actor and current_actor in disappeared:
            new_candidate_obj = current_scene.object_by_id(report.controlled_candidate_id) if report.controlled_candidate_id else None
            if new_candidate_obj is not None:
                level.controlled_object_id = report.controlled_candidate_id
                level.local_bindings["actor"] = report.controlled_candidate_id
            else:
                level.controlled_object_id = ""
                level.local_bindings.pop("actor", None)
            merged = (
                outcome not in {"retry"}
                and report.life_delta in (None, 0)
                and self._actor_merged_into_container(pending, current_actor, new_candidate=new_candidate_obj)
            )
            if self._should_preserve_vlm_route_plan_after_rebind(pending, report):
                level.notes_for_next_call = "Controlled object id changed during a progressing VLM route; preserve remaining route steps and re-bind actor."
                self.logger.log_event("plan_preserved_v20", {"level": level.level_index, "reason": "actor_disappeared_rebind_during_vlm_route", "old_actor": current_actor, "new_actor": report.controlled_candidate_id})
            elif merged:
                # ls20-style: the actor overlapped a framed container (lock) and the
                # tracker fused them into one object. Clearing the plan on every such
                # toggle causes a bottleneck/VLM storm; keep the plan and tell the VLM.
                level.notes_for_next_call = "Actor visually merged into a framed container; it is likely still there. Moving away (e.g. the opposite direction) should separate them. Do not re-plan from scratch."
                self.logger.log_event("plan_preserved_v20", {"level": level.level_index, "reason": "actor_merged_into_container", "old_actor": current_actor, "new_actor": report.controlled_candidate_id})
            else:
                self._invalidate_plan_steps("actor_disappeared_rebind", clear_all=True)
                level.notes_for_next_call = "Controlled object disappeared or changed; re-bind actor and goal before continuing."
            self.logger.log_event("actor_binding_invalidated_v20", {"level": level.level_index, "old_actor": current_actor, "new_candidate": report.controlled_candidate_id, "reason": "actor_merged_into_container" if merged else "actor_disappeared"})
            return
        if pending.source == "plan_executor" and pending.name != "ACTION6":
            if report.effective_noop:
                self._invalidate_plan_steps("plan_action_noop_evidence", actions={pending.name})
            elif (report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids or report.interaction_event) and not report.is_simple_translation and self._controlled_motion(report) is None:
                if self._has_pending_vlm_selector_sequence():
                    level.notes_for_next_call = "Preserved remaining VLM selector-pattern sequence after an expected transform."
                    self.logger.log_event("plan_preserved_v20", {"level": level.level_index, "reason": "selector_pattern_transform_sequence", "action": pending.name})
                else:
                    self._invalidate_plan_steps("transform_disproved_navigation_action", actions={pending.name})
        if outcome == "retry" or (report.life_delta is not None and report.life_delta < 0):
            self._invalidate_plan_steps("retry_or_life_loss_action", actions={pending.name})

    def _quarantine_noop_action(self, before_state: str, pending: PendingAction, action_key: str) -> None:
        level = self.memory.level
        step_until = level.total_action_count + self.config.quarantine_steps
        state_counts = level.state_action_outcomes.setdefault(before_state, {}).setdefault(action_key, Counter())
        q_state = level.quarantine_until_by_state.setdefault(before_state, {})
        if state_counts.get("noop", 0) >= self.config.state_noop_quarantine_after:
            q_state[action_key] = max(q_state.get(action_key, 0), step_until)
        action_counts = level.action_outcomes.setdefault(pending.name, Counter())
        if pending.name == "ACTION7" and action_counts.get("noop", 0) >= self.config.action7_noop_quarantine_after:
            level.global_quarantine_until[pending.name] = max(level.global_quarantine_until.get(pending.name, 0), level.total_action_count + 2 * self.config.quarantine_steps)
        elif pending.name != "ACTION6" and action_counts.get("noop", 0) >= self.config.global_noop_quarantine_after and action_counts.get("transform", 0) + action_counts.get("interaction", 0) == 0:
            level.global_quarantine_until[pending.name] = max(level.global_quarantine_until.get(pending.name, 0), step_until)

    def _has_pending_vlm_sequence(self, reasons: set[str] | None = None) -> bool:
        level = self.memory.level
        for step in level.current_plan[level.plan_cursor:]:
            if step.status != "pending" or step.step_type != "probe_action" or not step.action:
                continue
            if not isinstance(step.raw, dict) or "sequence_index" not in step.raw:
                continue
            route_reason = _short(step.raw.get("route_reason"), 80)
            if reasons and route_reason not in reasons:
                continue
            if self._trusted_plan_sequence(step, step.action):
                return True
        return False

    def _has_pending_vlm_route_sequence(self) -> bool:
        return self._has_pending_vlm_sequence({"geometry", "coupled_carrier_geometry"})

    def _has_pending_vlm_selector_sequence(self) -> bool:
        return self._has_pending_vlm_sequence({"selector_probe", "selector_calibration", "selector_cycle_learn", "selector_pattern_rule"})

    def _should_preserve_vlm_route_plan_after_rebind(self, pending: PendingAction, report: TransitionReport) -> bool:
        if pending.source != "plan_executor" or pending.name == "ACTION6":
            return False
        if report.retry_detected or report.life_delta not in (None, 0):
            return False
        if self._has_pending_vlm_selector_sequence():
            return bool(report.moved_objects or report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids)
        if not report.moved_objects:
            return False
        return self._has_pending_vlm_route_sequence()

    def _invalidate_plan_steps(self, reason: str, *, actions: set[str] | None = None, targets: set[str] | None = None, clear_all: bool = False) -> None:
        level = self.memory.level
        actions = {a.upper() for a in (actions or set()) if a}
        targets = {t.upper() for t in (targets or set()) if t}
        changed = False
        if clear_all:
            for step in level.current_plan[level.plan_cursor:]:
                if step.status == "pending":
                    step.status = "failed"
                    changed = True
        else:
            for step in level.current_plan[level.plan_cursor:]:
                if step.status != "pending":
                    continue
                resolved = self._resolve_step_target(step)
                if (actions and step.action.upper() in actions) or (targets and resolved.upper() in targets):
                    step.status = "failed"
                    changed = True
        if changed:
            level.bottleneck_reason = reason
            level.action_recovery_contract = True
            self.logger.log_event("plan_invalidated_v20", {"level": level.level_index, "reason": reason, "actions": sorted(actions), "targets": sorted(targets), "clear_all": clear_all})

    def _would_repeat_terminal_suffix(self, action_key: str) -> bool:
        level = self.memory.level
        if not level.bad_action_suffixes:
            return False
        candidate = tuple(list(level.recent_action_keys) + [action_key.upper()])
        for suffix in level.bad_action_suffixes[-12:]:
            n = min(len(suffix), len(candidate), 8)
            if n >= 4 and candidate[-n:] == suffix[-n:]:
                return True
        return False

    def _source_storm_active(self, source: str, *, outcomes: set[str] | None = None) -> bool:
        level = self.memory.level
        window = max(4, self.config.force_source_fuse_after)
        recent = list(level.recent_events)[-window:]
        if len(recent) < window:
            return False
        if any(e.source != source for e in recent):
            return False
        if outcomes is None:
            outcomes = {"noop", "retry"}
        return all(e.outcome in outcomes for e in recent)

    def _source_low_yield_active(self, source: str, *, outcomes: set[str], min_ratio: float = 0.85) -> bool:
        level = self.memory.level
        window = max(4, self.config.force_source_fuse_after)
        recent = list(level.recent_events)[-window:]
        if len(recent) < window:
            return False
        source_events = [e for e in recent if e.source == source]
        if len(source_events) < max(4, int(window * 0.75)):
            return False

        def low_yield_event(event: CompactEvent) -> bool:
            if event.outcome in outcomes:
                return True
            delta = event.transition_delta if isinstance(event.transition_delta, dict) else {}
            try:
                progress = float(delta.get("progress_score", 1.0))
            except Exception:
                progress = 1.0
            try:
                changed = int(delta.get("changed_cell_count", 999))
            except Exception:
                changed = 999
            structural = bool(delta.get("appeared") or delta.get("disappeared") or delta.get("controlled_candidate_id"))
            if source == "deterministic_click" and event.outcome in {"transform", "movement", "interaction"}:
                structural_churn = bool(delta.get("transformed_objects") or delta.get("appeared") or delta.get("disappeared")) and not bool(delta.get("simple_translation"))
                if structural_churn:
                    return True
            if progress <= 0.25:
                return True
            return event.outcome in {"transform", "movement", "interaction"} and changed <= 2 and not structural

        low_yield = sum(1 for e in source_events if low_yield_event(e))
        return low_yield / max(1, len(source_events)) >= min_ratio

    def _click_only_noop_flood_active(self, legal: Sequence[Any]) -> bool:
        if not self._click_only_legal(legal):
            return False
        level = self.memory.level
        window = min(8, max(4, self.config.force_source_fuse_after))
        recent = list(level.recent_events)[-window:]
        if len(recent) < window:
            return False
        click_events = [e for e in recent if e.source == "deterministic_click"]
        if len(click_events) < max(4, int(window * 0.75)):
            return False
        low_yield = sum(1 for e in click_events if e.outcome in {"noop", "state_change", "retry"})
        return low_yield / max(1, len(click_events)) >= 0.75

    def _normal_click_low_yield_flood_active(self) -> bool:
        level = self.memory.level
        window = max(4, self.config.force_source_fuse_after)
        recent = list(level.recent_events)[-window:]
        if len(recent) < window:
            return False
        normal_sources = {"deterministic_click", "frontier_probe"}
        click_events = [e for e in recent if e.action_key.startswith("ACTION6:") and e.source in normal_sources]
        if len(click_events) < max(4, int(window * 0.75)):
            return False

        def low_yield(event: CompactEvent) -> bool:
            if event.outcome in {"noop", "state_change", "retry"}:
                return True
            delta = event.transition_delta if isinstance(event.transition_delta, dict) else {}
            try:
                progress = float(delta.get("progress_score", 1.0))
            except Exception:
                progress = 1.0
            if progress <= 0.25:
                return True
            structural_churn = bool(delta.get("transformed_objects") or delta.get("appeared") or delta.get("disappeared")) and not bool(delta.get("simple_translation"))
            return event.outcome in {"transform", "movement", "interaction"} and structural_churn

        return sum(1 for e in click_events if low_yield(e)) / max(1, len(click_events)) >= 0.75

    def _normal_click_fuse_active(self, scene: SceneSnapshot, legal: Sequence[Any], source: str) -> bool:
        level = self.memory.level
        cooldown_active = level.click_fuse_block_until > level.total_action_count
        active = cooldown_active
        if not active:
            active = (
                self._source_storm_active("deterministic_click", outcomes={"noop", "state_change"})
                or self._source_low_yield_active("deterministic_click", outcomes={"noop", "state_change"})
                or self._click_only_noop_flood_active(legal)
                or self._normal_click_low_yield_flood_active()
            )
        if not active:
            return False
        level.bottleneck_reason = "deterministic_click_low_yield"
        level.action_recovery_contract = True
        level.fallback_guard_block_until = max(level.fallback_guard_block_until, level.total_action_count + 1)
        level.click_fuse_block_until = max(level.click_fuse_block_until, level.total_action_count + max(3, self.config.quarantine_steps))
        self._record_click_fuse_feedback(scene, source)
        if not cooldown_active:
            self.logger.log_event("source_fuse_open_v20", {"level": level.level_index, "source": source, "state": scene.state_hash[:12], "reason": "low_yield_click_flood"})
        return True

    def _record_click_fuse_feedback(self, scene: SceneSnapshot, source: str) -> None:
        level = self.memory.level
        recent = [
            {"action": e.action_key, "source": e.source, "outcome": e.outcome}
            for e in list(level.recent_events)[-8:]
            if e.action_key.upper().startswith("ACTION6")
        ]
        feedback = {
            "stage": "source_fuse",
            "reason": "click_low_yield_fuse",
            "source": source,
            "state": scene.state_hash[:12],
            "recent_click_outcomes": recent[-6:],
            "requirement": "Do not broad-remap or sweep fresh click coordinates; propose a state-conditioned click sequence or a different mechanism.",
        }
        last = level.rejected_vlm_plan_feedback[-1] if level.rejected_vlm_plan_feedback else {}
        if last.get("reason") == feedback["reason"] and last.get("source") == source and last.get("state") == feedback["state"]:
            return
        level.rejected_vlm_plan_feedback.append(feedback)
        level.rejected_vlm_plan_feedback = level.rejected_vlm_plan_feedback[-12:]

    def _focused_action6_target(self, scene: SceneSnapshot, blocked: set[str]) -> tuple[int, int, str] | None:
        state_click = self._state_conditioned_success_click(scene, blocked)
        if state_click is not None:
            return state_click[0], state_click[1], ""
        success_neighbor = self._successful_click_neighbor(scene, blocked)
        if success_neighbor is not None:
            return success_neighbor[0], success_neighbor[1], ""
        return None

    def _click_fuse_blocks_broad_click(self, scene: SceneSnapshot, legal: Sequence[Any], source: str, blocked: set[str]) -> bool:
        if not self._normal_click_fuse_active(scene, legal, source):
            return False
        return self._focused_action6_target(scene, blocked) is None

    def _transition_matches_expected(self, report: TransitionReport, step: PlanStep, visits_before: int | None = None) -> bool:
        preds = step.expected_predicates or []
        default_match = bool(
            not report.effective_noop
            and not report.retry_detected
            and report.life_delta in (None, 0)
            and not self._minor_structural_churn(report, visits_before)
        )
        if not preds:
            return default_match
        if self._minor_structural_churn(report, visits_before):
            return False
        outcome = self._classify_outcome(report)
        recognized = 0
        for pred in preds:
            # V2.2: unknown/free-form predicate kinds are ignored instead of failing
            # the whole step. The old hard-fail turned every VLM improvisation into a
            # "predicate mismatch" and (via the contract) into an action ban.
            if not isinstance(pred, dict):
                continue
            kind = _short(pred.get("type") or pred.get("predicate"), 80).lower()
            if not kind:
                continue
            if kind == "not_noop":
                recognized += 1
                if report.effective_noop:
                    return False
            elif kind == "no_life_loss":
                recognized += 1
                if report.life_delta is not None and report.life_delta < 0:
                    return False
            elif kind == "no_retry":
                recognized += 1
                if report.retry_detected:
                    return False
            elif kind == "controlled_motion":
                recognized += 1
                if self._controlled_motion(report) is None:
                    return False
            elif kind == "appeared_or_disappeared":
                recognized += 1
                if not (report.appeared_object_ids or report.disappeared_object_ids):
                    return False
            elif kind == "structural_change":
                recognized += 1
                if not (report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids or report.interaction_event):
                    return False
            elif kind == "outcome_in":
                values = pred.get("values")
                if not isinstance(values, list) or not values:
                    continue
                allowed = {_short(v, 80) for v in values if _short(v, 80)}
                recognized += 1
                if outcome not in allowed:
                    return False
        if recognized == 0:
            return default_match
        return True

    def _transition_progress_score(self, pending: PendingAction, report: TransitionReport, current_scene: SceneSnapshot) -> float:
        level = self.memory.level
        if report.retry_detected:
            return -10.0
        if report.life_delta is not None and report.life_delta < 0:
            return -8.0
        score = 0.0
        if report.effective_noop:
            score -= 2.0
        else:
            score += 0.25
        # Genuinely new states (never visited this level) are real progress; revisiting
        # a seen state is churn unless we got there by controlled motion (backtracking).
        # When nearly every transition lands on a never-seen state (ambient animation,
        # e.g. tr87 moving pieces), "new state" carries almost no information, so both
        # the bonus and the revisit penalty are dampened.
        total_visits = sum(level.all_state_visits.values())
        unique_states = len(level.all_state_visits)
        ambient_drift = total_visits >= 20 and unique_states / max(1, total_visits) >= 0.9
        visits_before = level.all_state_visits.get(current_scene.state_hash, 0)
        controlled = self._controlled_motion(report) is not None
        if visits_before == 0:
            score += 0.2 if ambient_drift else 0.8
        elif controlled:
            score -= 0.2
        else:
            score -= 0.3 if ambient_drift else 1.25
        if report.counter_delta is not None and report.counter_delta < 0 and report.effective_noop:
            score -= 1.5
        if report.interaction_event:
            score += 0.8
        if report.appeared_object_ids or report.disappeared_object_ids:
            score += 0.5
        if controlled:
            score += 0.4
        # Cycling transform: pixels change but we keep landing on already-seen states
        # without controlled motion or resource gain -> churn, not progress.
        if (
            (report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids)
            and visits_before >= 1
            and not controlled
            and (report.counter_delta is None or report.counter_delta <= 0)
        ):
            score -= 1.5
        if self._minor_structural_churn(report, visits_before):
            score -= 2.0
        step = level.active_step()
        if step is not None and pending.source == "plan_executor":
            if self._transition_matches_expected(report, step, visits_before):
                score += 1.5
            else:
                score -= 0.8
        return score

    def _update_plan_after_transition(self, pending: PendingAction, report: TransitionReport, current_scene: SceneSnapshot) -> None:
        level = self.memory.level
        step = level.active_step()
        if step is None:
            return
        step.attempts += 1
        if report.retry_detected or (report.life_delta is not None and report.life_delta < 0):
            step.status = "failed"
            level.bottleneck_reason = "plan_step_caused_retry_or_life_loss"
            level.current_plan = []
            level.plan_cursor = 0
            level.action_recovery_contract = True
            return
        if report.effective_noop and pending.source == "plan_executor":
            step.status = "failed"
            # Skip only the current step. If later VLM steps remain, keep executing
            # them instead of immediately replacing the plan with another VLM call.
            level.plan_cursor += 1
            if level.active_step() is None:
                level.bottleneck_reason = "plan_step_noop"
                level.action_recovery_contract = True
            else:
                level.bottleneck_reason = ""
                level.action_recovery_contract = False
            return
        target_id = self._resolve_step_target(step)
        if target_id and self._target_reached(target_id, current_scene):
            step.status = "done"
            level.plan_cursor += 1
            return
        visits_before = level.all_state_visits.get(current_scene.state_hash, 0)
        matched = self._transition_matches_expected(report, step, visits_before)
        if pending.source == "plan_executor":
            if matched:
                step.status = "done"
                level.plan_cursor += 1
                return
            if not report.effective_noop:
                # V2.2: a wrong VLM prediction is not evidence the action is harmful.
                # The old contract-forbid here banned ordinary actions ~500x/run and
                # fed a bottleneck->VLM->ban storm; keep the bookkeeping only.
                step.status = "failed"
                level.plan_cursor += 1
                level.bottleneck_reason = "plan_step_expected_predicate_mismatch"
                level.action_recovery_contract = True
                return
        if matched and step.step_type in {"probe_action", "probe_object", "click"}:
            step.status = "done"
            level.plan_cursor += 1

    def _update_action_meaning_from_transition(self, pending: PendingAction, report: TransitionReport, current_scene: SceneSnapshot) -> None:
        name = pending.name
        if name == "RESET":
            return
        level = self.memory.level
        meaning = self.memory.game.action_meanings.setdefault(name, ActionMeaning(action=name))
        meaning.attempts += 1
        changed_cells = max(0, int(report.changed_cell_count or 0))
        world_changed_cells = max(0, int(report.world_changed_cell_count or 0))
        meaning.changed_cells_total += changed_cells
        meaning.world_changed_cells_total += world_changed_cells
        structural_change = bool(report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids)
        if report.effective_noop:
            meaning.noops += 1
        # V2.2: in a confirmed step-tax game (counter drains 1/step, refills on
        # bankruptcy and costs a life) the death is the tax running out, NOT evidence
        # that whichever action ran at counter=0 is dangerous. Attributing retry/
        # life_loss to that action permanently poisoned it -> _resource_risky flagged
        # the whole action space -> resource_guard contracts stuck forever ->
        # absolute_nonreset lockup (ls20 absEx 3->16, all 4 actions blacklisted).
        if report.retry_detected and not level.counter_is_step_tax:
            meaning.retries += 1
        if report.life_delta is not None and report.life_delta < 0 and not level.counter_is_step_tax:
            meaning.life_losses += 1
        if structural_change:
            meaning.transforms += 1
            if world_changed_cells <= 2 and not report.moved_objects:
                # V2.6: sb26 regression - counting every small transform toward the
                # resource_wasting_noop signal below poisoned ACTION5 (a selector bar
                # that shrinks by 1-3 cells per press, i.e. the actual win mechanism)
                # after just 3 presses, permanently hard-blocking it in
                # _action_blocked for the rest of the level. A small pixel footprint
                # only signals a decorative ticker once it starts repeating an
                # already-seen state; landing on a brand-new state each time is real
                # progress, same distinction _minor_structural_churn makes.
                if level.all_state_visits.get(current_scene.state_hash, 0) > 0:
                    meaning.small_transforms += 1
        if report.interaction_event:
            meaning.interactions += 1
        event_id = self.memory.next_event_id
        if event_id not in meaning.evidence_events:
            meaning.evidence_events.append(event_id)
            meaning.evidence_events = meaning.evidence_events[-16:]
        summary_nl = self._sanitize_game_text(report.summary[:260], current_scene)
        recent_states_before = list(level.recent_state_hashes)
        if name == "ACTION7" and not report.effective_noop and len(recent_states_before) >= 2 and current_scene.state_hash == recent_states_before[-2]:
            meaning.kind = "undo"
            meaning.meaning_nl = "returns to a recently visited previous state"
            meaning.confidence = max(meaning.confidence, 0.82)
            meaning.vector = None
            return
        move = self._controlled_motion(report, level.controlled_object_id) if name != "ACTION6" else None
        # V2.2: never derive a movement vector from (a) a ticker object's drift or
        # (b) a retry/life-loss transition. tu93 learned ACTION2 = (-18,0) from the
        # death-scroll displacement of its controlled object, and genuine timer bars
        # occasionally get picked as the largest mover. Both poison navigation.
        if move is not None:
            moved_id = str(move.get("object_id") or "")
            if (moved_id in level.ticker_object_ids and moved_id != level.controlled_object_id) or report.retry_detected or (report.life_delta is not None and report.life_delta < 0):
                move = None
        if move is not None:
            vec = (int(move["dx"]), int(move["dy"]))
            side_effect = bool(report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids or report.interaction_event)
            meaning.vector_votes[_vector_key(vec)] += 1
            meaning.movements += 1
            if side_effect and not report.is_simple_translation:
                dominant = _dominant_vector(meaning.vector_votes)
                # V2.2: judge by displacement consistency, not by "pure" movements.
                # The old pure_movement_votes = movements - transforms - interactions
                # is permanently <=0 in games where every move has side effects
                # (re86/ar25 coupled frames), so vectors could never be confirmed and
                # the whole vector-exemption navigation channel stayed dead.
                dominant_majority = dominant is not None and dominant[1] >= 2 and dominant[1] * 2 > meaning.movements
                if dominant_majority and meaning.noop_ratio < 0.35:
                    meaning.vector = dominant[0]
                    meaning.kind = "movement"
                    meaning.meaning_nl = f"moves by {dominant[0]} with possible side effects"
                    meaning.confidence = max(meaning.confidence, 0.58)
                else:
                    meaning.vector = None
                    meaning.kind = "interact_or_transform"
                    meaning.meaning_nl = f"moves/changes scene with side effects: {summary_nl}"
                    meaning.confidence = max(meaning.confidence, 0.58)
                return
            learned_vec = vec
            dominant = _dominant_vector(meaning.vector_votes)
            if meaning.vector is not None and meaning.vector != vec:
                old_vec = (int(meaning.vector[0]), int(meaning.vector[1]))
                old_votes = int(meaning.vector_votes.get(_vector_key(old_vec), 0))
                new_votes = int(meaning.vector_votes.get(_vector_key(vec), 0))
                unstable_context = side_effect or not report.is_simple_translation or len(report.moved_objects) > 1
                axis_changed = _vector_axis_signature(old_vec) != _vector_axis_signature(vec)
                if old_votes >= 1 and axis_changed and unstable_context:
                    learned_vec = old_vec
                elif old_votes >= 2 and old_votes >= new_votes and axis_changed:
                    learned_vec = old_vec
                elif _axis_aligned_vector(old_vec) and not _axis_aligned_vector(vec) and old_votes >= new_votes:
                    learned_vec = old_vec
                elif dominant is not None and dominant[1] >= max(2, new_votes + 1):
                    learned_vec = dominant[0]
            elif dominant is not None and dominant[1] >= 2 and dominant[0] != vec:
                learned_vec = dominant[0]
            meaning.vector = learned_vec
            meaning.kind = "movement"
            if learned_vec == vec:
                meaning.meaning_nl = f"moves the controlled object by vector {vec}" + (" with side effects" if side_effect else "")
            else:
                meaning.meaning_nl = f"usually moves the controlled object by vector {learned_vec}; observed context-dependent vector {vec}"
            meaning.confidence = max(meaning.confidence, 0.68 if side_effect else 0.74)
            return
        if name != "ACTION6" and structural_change and not report.moved_objects and meaning.movements == 0 and meaning.small_transforms >= 3 and meaning.transforms >= 3:
            meaning.kind = "resource_wasting_noop"
            meaning.meaning_nl = "mostly changes tiny/status marker"
            meaning.confidence = max(meaning.confidence, 0.66)
            meaning.vector = None
            return
        if structural_change or (report.interaction_event and not report.is_simple_translation):
            meaning.kind = "click_or_select" if name == "ACTION6" else "interact_or_transform"
            meaning.meaning_nl = "coordinate click/select can change the scene" if name == "ACTION6" else f"causes structural transform or interaction: {summary_nl}"
            meaning.confidence = max(meaning.confidence, 0.62)
            meaning.vector = None
            return
        if report.counter_delta is not None and report.counter_delta != 0:
            meaning.resource_delta = report.counter_delta
            if report.counter_delta < 0 and report.effective_noop:
                meaning.kind = "resource_wasting_noop"
            elif not meaning.kind or meaning.kind == "unknown":
                meaning.kind = "resource"
            meaning.meaning_nl = meaning.meaning_nl or f"affects visible resource counter by {report.counter_delta}"
            meaning.confidence = max(meaning.confidence, 0.45)
        if report.effective_noop and meaning.attempts >= 2 and meaning.noops / max(1, meaning.attempts) >= 0.8:
            meaning.kind = "unknown_or_blocked"
            meaning.meaning_nl = "mostly no visible effect in tested states"
            meaning.confidence = max(meaning.confidence, 0.30)

    def _update_actor_and_terrain(self, pending: PendingAction, report: TransitionReport, current_scene: SceneSnapshot) -> None:
        level = self.memory.level
        candidate = report.controlled_candidate_id
        if not candidate or not report.moved_objects:
            return
        moved_ids = {str(m.get("object_id", "")) for m in report.moved_objects}
        old_actor = level.controlled_object_id
        if old_actor and old_actor in moved_ids and candidate != old_actor:
            candidate = old_actor
        if candidate not in moved_ids:
            return
        level.actor_votes[candidate] += 1
        best = level.actor_votes.most_common(1)[0][0]
        candidate_votes = level.actor_votes[candidate]
        controlled_move = self._controlled_motion(report, candidate)
        moved_count = len(moved_ids)
        structural_side_effect = bool(
            report.transformed_objects
            or report.appeared_object_ids
            or report.disappeared_object_ids
            or (report.interaction_event and not report.is_simple_translation)
        )
        adopt = controlled_move is not None and report.is_simple_translation
        adopt = adopt or (
            controlled_move is not None
            and not structural_side_effect
            and moved_count <= 2
            and (not old_actor or old_actor == candidate or old_actor not in moved_ids)
        )
        adopt = adopt or (
            candidate_votes >= 2
            and not structural_side_effect
            and (not old_actor or old_actor not in moved_ids or old_actor == candidate)
            and (moved_count <= 3 or old_actor == candidate)
        )
        if old_actor and candidate != old_actor and structural_side_effect and not report.is_simple_translation:
            adopt = False
        if adopt:
            chosen = candidate
            level.controlled_object_id = chosen
            level.local_bindings["actor"] = chosen
            level.tentative_controlled_object_id = ""
            if old_actor and old_actor != chosen:
                if self._should_preserve_vlm_route_plan_after_rebind(pending, report):
                    self.logger.log_event("plan_preserved_v20", {"level": level.level_index, "reason": "actor_rebound_during_vlm_route", "old_actor": old_actor, "new_actor": chosen})
                else:
                    self._invalidate_plan_steps("actor_rebound_from_transition", clear_all=True)
                self.logger.log_event("actor_rebound_v20", {"level": level.level_index, "old_actor": old_actor, "new_actor": chosen, "votes": candidate_votes, "simple_translation": report.is_simple_translation})
        else:
            level.tentative_controlled_object_id = best
        for observed_scene in (pending.scene_before, current_scene):
            obj = observed_scene.object_by_id(candidate)
            if obj is not None:
                level.actor_bbox_history.append(obj.bbox)
        if report.is_simple_translation and not report.transformed_objects:
            self._learn_walkable_colour(pending.scene_before, current_scene, candidate)

    def _learn_walkable_colour(self, before: SceneSnapshot, after: SceneSnapshot, oid: str) -> None:
        old_actor, new_actor = before.object_by_id(oid), after.object_by_id(oid)
        if old_actor is None or new_actor is None or old_actor.bbox == new_actor.bbox:
            return
        nx0, ny0, nx1, ny1 = new_actor.bbox
        for y in range(old_actor.bbox[1], old_actor.bbox[3] + 1):
            for x in range(old_actor.bbox[0], old_actor.bbox[2] + 1):
                if (nx0 <= x <= nx1 and ny0 <= y <= ny1) or (x, y) in after.volatile_cells:
                    continue
                color = after.grid[y][x]
                if color != after.background_candidate:
                    self.memory.level.walkable_color_votes[color] += 1

    def _update_click_memory(self, pending: PendingAction, report: TransitionReport, current_scene: SceneSnapshot) -> None:
        if pending.name != "ACTION6":
            return
        level = self.memory.level
        before_state = pending.scene_before.state_hash
        after_state = current_scene.state_hash
        key = pending.action_key()
        level.click_coord_counts[key] += 1
        if pending.x is not None and pending.y is not None:
            level.click_region_counts[self._click_region_key(pending.x, pending.y)] += 1
        oid = pending.target_object_id or ""
        meaningful_transform = bool(
            report.transformed_objects
            and (
                report.world_changed_cell_count > 2
                or report.appeared_object_ids
                or report.disappeared_object_ids
                or report.controlled_candidate_id
            )
        )
        unstable_deterministic_click = bool(
            pending.source == "deterministic_click"
            and not report.is_simple_translation
            and (
                len(report.moved_objects) > 1
                or report.appeared_object_ids
                or report.disappeared_object_ids
                or (report.transformed_objects and report.controlled_candidate_id)
            )
        )
        progress_like = bool(
            report.controlled_candidate_id
            or report.appeared_object_ids
            or report.disappeared_object_ids
            or report.is_simple_translation
            or (report.interaction_event and report.world_changed_cell_count > 2)
            or meaningful_transform
        )
        # V2.1: a click that only flips decorative/cycling pixels (lands on an already
        # seen state without controlled motion or counter change) is churn, not a
        # "successful" click. Recording it as success previously granted positive
        # evidence that bypassed the per-coordinate click limit (ft09 38,38 loop).
        cycling_click = bool(
            not report.controlled_candidate_id
            and not report.is_simple_translation
            and (report.counter_delta is None or report.counter_delta <= 0)
            and report.life_delta in (None, 0)
            and level.all_state_visits.get(after_state, 0) >= 1
        )
        if report.effective_noop or not progress_like or unstable_deterministic_click or cycling_click:
            level.click_noops_by_object[oid] += 1
            return
        level.click_success_by_object[oid] += 1
        level.successful_clicks_by_state.setdefault(before_state, Counter())[key] += 1
        level.click_edges_by_state.setdefault(before_state, {})[key] = after_state
        if pending.x is not None and pending.y is not None:
            level.click_success_coords[key] += 1
            level.click_success_regions[self._click_region_key(pending.x, pending.y)] += 1

    def _hidden_resource_transition_consumed(self, pending: PendingAction, report: TransitionReport, outcome: str, current_scene: SceneSnapshot) -> bool:
        if pending.name == "RESET":
            return False
        if pending.scene_before.counter_value is not None or current_scene.counter_value is not None:
            return False
        if pending.scene_before.counter_ratio is not None or current_scene.counter_ratio is not None:
            return False
        if outcome in {"retry", "level_advanced"}:
            return True
        if pending.name == "ACTION6":
            return bool(outcome not in {"noop", "state_change"} and not report.effective_noop)
        return True

    def _hidden_resource_remaining_ratio(self) -> float | None:
        cap = self.memory.game.resource_model.hidden_action_budget_capacity
        if cap is None or cap <= 0:
            return None
        used = max(0, self.memory.level.hidden_resource_used_this_attempt)
        return max(0.0, min(1.0, (cap - used) / cap))

    def _infer_hidden_resource_budget_from_game_over(self, scene: SceneSnapshot) -> None:
        level = self.memory.level
        used = int(level.hidden_resource_used_this_attempt)
        if used < 8:
            return
        if scene.counter_value is not None or scene.counter_ratio is not None or scene.life_count is not None:
            return
        res = self.memory.game.resource_model
        res.hidden_budget_observations.append(used)
        res.hidden_budget_observations = res.hidden_budget_observations[-8:]
        inferred = min(res.hidden_budget_observations)
        if res.hidden_action_budget_capacity is None or inferred < res.hidden_action_budget_capacity:
            res.hidden_action_budget_capacity = inferred
        res.hidden_budget_source = "game_over_effective_action_count"
        res.description_nl = res.description_nl or "A hidden action/effect budget is inferred from GAME_OVER timing; avoid low-information probes near the inferred limit."
        self.logger.log_event(
            "hidden_resource_budget_inferred_v20",
            {
                "level": level.level_index,
                "attempt": level.attempt_index,
                "used_this_attempt": used,
                "capacity": res.hidden_action_budget_capacity,
                "observations": list(res.hidden_budget_observations),
            },
        )

    def _update_resource_model_from_scene(self, scene: SceneSnapshot) -> None:
        res = self.memory.game.resource_model
        if scene.counter_value is not None:
            res.visible_bar, res.last_value, res.capacity = True, scene.counter_value, scene.counter_capacity
            res.description_nl = res.description_nl or "A visible resource/step counter exists; planning should avoid wasting actions and favor high-information probes."
        if scene.life_count is not None:
            res.last_lives = scene.life_count

    def _refresh_strategic_flags(self, scene: SceneSnapshot, transition: TransitionReport | None) -> None:
        level = self.memory.level
        previous_lives = self.memory.game.resource_model.last_lives
        counter_low = False
        if scene.counter_ratio is not None:
            if scene.counter_ratio <= self.config.low_resource_ratio:
                counter_low = True
                level.bottleneck_reason = level.bottleneck_reason or "low_resource_counter"
            if scene.counter_ratio <= self.config.critical_resource_ratio:
                counter_low = True
                level.current_plan = []
                level.plan_cursor = 0
                level.bottleneck_reason = "critical_resource_counter"
        if scene.counter_value is not None and scene.counter_value <= 1:
            counter_low = True
        hidden_ratio = self._hidden_resource_remaining_ratio()
        if hidden_ratio is not None:
            if hidden_ratio <= self.config.low_resource_ratio:
                counter_low = True
                level.bottleneck_reason = level.bottleneck_reason or "low_hidden_resource_budget"
            if hidden_ratio <= self.config.critical_resource_ratio:
                counter_low = True
                level.current_plan = []
                level.plan_cursor = 0
                level.bottleneck_reason = "critical_hidden_resource_budget"
        life_lost = bool(scene.life_count is not None and previous_lives is not None and scene.life_count < previous_lives)
        life_critical = bool(scene.life_count is not None and scene.life_count <= 1)
        if life_lost:
            level.current_plan = []
            level.plan_cursor = 0
            level.bottleneck_reason = "life_loss_detected"
        if counter_low:
            level.resource_crisis = True
        elif level.resource_crisis and self._resource_recovered_from_scene(scene):
            level.resource_crisis = False
            if level.bottleneck_reason in {"low_resource_counter", "critical_resource_counter", "resource_crisis_no_safe_fallback"}:
                level.bottleneck_reason = ""
            # V2.2: crisis-time resource_guard bans must not outlive the crisis. They
            # are state-keyed and previously stayed forever, locking the same states
            # into the exhausted-guard chain even after the counter refilled (ls20).
            cleared_rules = 0
            for rules in level.contract_forbidden_by_state.values():
                for rule_key in [k for k, v in rules.items() if str(v).startswith("resource_guard:")]:
                    del rules[rule_key]
                    cleared_rules += 1
            self.logger.log_event("resource_crisis_cleared_v20", {"level": level.level_index, "state": scene.state_hash[:12], "counter": scene.counter_value, "ratio": scene.counter_ratio, "lives": scene.life_count, "resource_guard_rules_cleared": cleared_rules})
        if life_critical and not counter_low:
            level.bottleneck_reason = level.bottleneck_reason or "life_critical"
        transform_mode_active = self._transform_bottleneck_active()
        if transform_mode_active:
            level.bottleneck_reason = level.bottleneck_reason or "transform_mode"
        elif level.bottleneck_reason == "transform_mode":
            level.bottleneck_reason = ""
        self._update_resource_model_from_scene(scene)

    def _record_game_over(self, scene: SceneSnapshot, transition: TransitionReport | None) -> None:
        level = self.memory.level
        level.awaiting_reset = True
        level.pending_action = None
        level.last_resolved_pending_action = None
        level.current_plan = []
        level.plan_cursor = 0
        level.bottleneck_reason = "game_over"
        suffix = list(level.recent_action_keys)[-8:]
        if suffix:
            level.bad_action_suffixes.append(tuple(str(x).upper() for x in suffix))
            level.bad_action_suffixes = level.bad_action_suffixes[-24:]
        if suffix:
            note = f"failed_attempt_{level.attempt_index}: GAME_OVER after suffix={suffix}"
            self.memory.game.advisory_notes = _short((self.memory.game.advisory_notes + "\n" + note).strip(), 900)
        self._infer_hidden_resource_budget_from_game_over(scene)
        self.logger.log_event("game_over_v20", {"level": level.level_index, "attempt": level.attempt_index, "steps": level.total_action_count, "counter": scene.counter_value, "lives": scene.life_count, "hidden_resource_used": level.hidden_resource_used_this_attempt, "hidden_resource_capacity": self.memory.game.resource_model.hidden_action_budget_capacity, "transition": transition.summary if transition else None, "known_noops_preserved": sum(len(v) for v in level.known_noops_by_state.values()), "suffix": suffix})
        self._flush_last_vlm_io("game_over")

    def _record_level_success(self, scene: SceneSnapshot, new_levels_completed: int) -> None:
        level = self.memory.level
        if level.initial_scene is None or level.success_reflected:
            return
        pending = level.last_resolved_pending_action or level.pending_action
        pre_success = pending.scene_before if pending else level.pre_success_scene
        outcome = {"level": level.level_index, "steps": level.total_action_count, "new_levels_completed": new_levels_completed, "win_action": pending.action_key() if pending else "unknown", "recent_events": [e.as_prompt() for e in list(level.recent_events)[-12:]]}
        if self._vlm_available() and level.vlm_calls_this_level < self.config.max_vlm_calls_per_level:
            self._request_vlm_success_reflect(scene, outcome, pre_success)
        else:
            self._deterministic_success_reflect(scene, outcome, pre_success)
        level.success_reflected = True
        level.pending_action = None
        level.current_plan = []
        self.logger.log_event("success_consolidated", {"level": level.level_index, "name": "level_success_v20", "confidence": 1.0, "source_level": level.level_index, "new_levels_completed": new_levels_completed, "win_action": outcome.get("win_action"), "causal_event_ids": [e.event_id for e in list(level.recent_events)[-12:]]})
        self.logger.log_event("level_success_v20", outcome)
        self._flush_last_vlm_io("level_success")

    def _deterministic_success_reflect(self, scene: SceneSnapshot, outcome: dict[str, Any], pre_success: SceneSnapshot | None) -> None:
        description = "Reach or interact with the success target after satisfying local preconditions."
        roles: dict[str, VisualDescriptor] = {}
        src = pre_success or scene
        if src.template_relations:
            description = "Use framed/status pattern relations as the likely goal cue, then reach or interact with the matching target."
            for rel in src.template_relations[:2]:
                for role, oid in (("frame_relation_a", str(rel.get("left", ""))), ("frame_relation_b", str(rel.get("right", "")))):
                    obj = src.object_by_id(oid)
                    if obj is not None:
                        roles[role] = self._visual_descriptor(obj, src)
        mem = self.memory.game.win_condition or WinConditionMemory()
        mem.description_nl = self._sanitize_game_text(description, src)
        mem.visual_roles.update(roles)
        level_index = int(outcome.get("level", -1))
        if level_index >= 0 and level_index not in mem.confirmed_levels:
            mem.confirmed_levels.append(level_index)
        mem.confidence = max(mem.confidence, 0.55)
        mem.evidence.append({"level": outcome.get("level"), "win_action": outcome.get("win_action"), "source": "deterministic_success_reflect"})
        mem.evidence = mem.evidence[-12:]
        self.memory.game.win_condition = mem
        self.memory.game.solved_level_summaries.append({"level": outcome.get("level"), "description_nl": mem.description_nl, "win_action": outcome.get("win_action")})
        self.memory.game.solved_level_summaries = self.memory.game.solved_level_summaries[-8:]

    def _request_vlm_success_reflect(self, scene: SceneSnapshot, outcome: dict[str, Any], pre_success: SceneSnapshot | None) -> None:
        level = self.memory.level
        prompt = self._build_vlm_prompt(scene, None, (), VLMMode.SUCCESS_REFLECT.value, extra={"success_outcome": outcome, "pre_success_scene_summary": pre_success.summary[:2400] if pre_success else "", "initial_scene_summary": level.initial_scene.summary[:2400] if level.initial_scene else ""})
        comparison = self._make_success_comparison_image(level.initial_scene, pre_success, scene)
        request = VLMRequest(prompt, pre_success.rgb if pre_success and pre_success.rgb else scene.rgb or render_grid(scene.grid, self.config.image_size), level.initial_scene.rgb if level.initial_scene and level.initial_scene.rgb else None, comparison or scene.annotated_rgb, self.config.vlm_max_new_tokens)
        level.vlm_calls_this_level += 1
        self.memory.game.vlm_calls_total += 1
        level.last_vlm_mode = VLMMode.SUCCESS_REFLECT.value
        call = level.vlm_calls_this_level
        self.logger.log_event("vlm_call_v20", {"mode": VLMMode.SUCCESS_REFLECT.value, "level": level.level_index, "call": call, "bottleneck": "success_reflect"})
        raw_response: Any = None
        result: V20VLMResult | None = None
        parse_status = "empty_or_unparseable"
        error = ""
        try:
            raw_response = self.backend.decide(request)
            result = parse_vlm_result(raw_response)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.logger.log_exception(exc)
        if result is not None and result.mode != "INVALID":
            result.mode = VLMMode.SUCCESS_REFLECT.value
            parse_status = "parsed"
            self._apply_vlm_result(result, scene)
            self.logger.log_event("vlm_result_v20", {"mode": VLMMode.SUCCESS_REFLECT.value, "plan_steps": len(result.plan), "role_bindings": result.role_bindings, "has_win_update": bool(result.win_condition_update)})
        else:
            if result is not None and result.raw_invalid_excerpt:
                parse_status = "invalid"
                self.logger.log_event("vlm_invalid", {"mode": VLMMode.SUCCESS_REFLECT.value, "raw_excerpt": result.raw_invalid_excerpt})
            self._deterministic_success_reflect(scene, outcome, pre_success)
        self._record_vlm_io(request, raw_response, result, mode=VLMMode.SUCCESS_REFLECT.value, call=call, parse_status=parse_status, error=error, context={"bottleneck": "success_reflect", "success_outcome": outcome})

    def _make_success_comparison_image(self, initial: SceneSnapshot | None, pre_success: SceneSnapshot | None, post_success: SceneSnapshot | None) -> Image.Image | None:
        scenes = [initial, pre_success, post_success]
        if not any(s is not None for s in scenes):
            return None
        size = self.config.image_size
        imgs = []
        for s in scenes:
            imgs.append(Image.new("RGB", (size, size), (0, 0, 0)) if s is None else (s.rgb or render_grid(s.grid, size)).convert("RGB").resize((size, size), Image.Resampling.NEAREST))
        out = Image.new("RGB", (size * 3, size), (0, 0, 0))
        for i, img in enumerate(imgs):
            out.paste(img, (i * size, 0))
        draw = ImageDraw.Draw(out)
        for i, label in enumerate(["INITIAL", "PRE_SUCCESS", "POST_SUCCESS_OR_NEXT"]):
            draw.rectangle((i * size, 0, i * size + 190, 22), fill=(0, 0, 0))
            draw.text((i * size + 4, 4), label, fill=(255, 255, 255))
        return out

    def _record_vlm_io(self, request: VLMRequest, raw_response: Any, result: V20VLMResult | None, *, mode: str, call: int, parse_status: str, error: str = "", context: dict[str, Any] | None = None) -> None:
        if not self.config.log_vlm_io:
            return
        level = self.memory.level
        payload = {
            "level": level.level_index,
            "call": call,
            "mode": mode,
            "game_id": self._game_id(),
            "parse_status": parse_status,
            "error": error,
            "request": _vlm_request_for_log(request),
            "raw_response_type": type(raw_response).__name__,
            "raw_response": _vlm_raw_for_log(raw_response),
            "parsed_output": _vlm_result_for_log(result),
            "context": _json_safe(context or {}),
        }
        level.last_vlm_io = payload
        level.last_vlm_io_call = call
        if call <= 3:
            self.logger.log_event("vlm_io_v20", {"capture": f"first_{call}", **payload})

    def _flush_last_vlm_io(self, reason: str) -> None:
        if not self.config.log_vlm_io:
            return
        level = self.memory.level
        if not level.last_vlm_io or level.last_vlm_io_call <= level.flushed_last_vlm_io_call:
            return
        self.logger.log_event("vlm_io_last_v20", {"capture": "last", "flush_reason": reason, **level.last_vlm_io})
        level.flushed_last_vlm_io_call = level.last_vlm_io_call

    def _vlm_available(self) -> bool:
        available = getattr(self.backend, "available", False)
        try:
            backend_available = available() if callable(available) else bool(available)
        except Exception:
            backend_available = False
        return bool(self.config.enable_vlm and backend_available)

    def _vlm_temperature(self) -> float:
        # 升温只用于打破“同 prompt 同确定性答案”的死循环，正常对局保持 temperature=0。
        level = self.memory.level
        # 空计划/彻底无法解析的输出在 temperature=0 下都会确定性复现（实测 r11l 连续 9
        # 次空计划、ls20/dc22 连续 5 次空计划；ls20 另有一次连续复读非 JSON 长文本）；
        # notes 反馈不足以改变确定性输出，必须靠采样随机性跳出，因此升温而非归零。
        if level.consecutive_empty_plans >= 1:
            return 0.9 if level.consecutive_empty_plans >= 2 else 0.7
        if level.vlm_issue_repeat_count >= 3:
            return 0.7
        last_reject = level.rejected_vlm_plan_feedback[-1] if level.rejected_vlm_plan_feedback else None
        if isinstance(last_reject, dict):
            try:
                if level.total_action_count - int(last_reject.get("at_action_count", -999)) <= 8:
                    return 0.7
            except Exception:
                pass
        return 0.0

    def _update_vlm_issue_state(self, scene: SceneSnapshot, transition: TransitionReport | None) -> None:
        level = self.memory.level
        outcome = ""
        if transition is not None:
            outcome = "retry" if transition.retry_detected else "noop" if transition.effective_noop else "transform" if (transition.transformed_objects or transition.appeared_object_ids or transition.disappeared_object_ids) else "move" if transition.moved_objects else "change"
        signature = f"{scene.state_hash}:{level.bottleneck_reason}:{outcome}"
        seen_in_window = sum(1 for s in level.vlm_issue_recent_signatures if s == signature)
        level.vlm_issue_recent_signatures.append(signature)
        if signature == level.vlm_issue_signature:
            level.vlm_issue_repeat_count += 1
        elif seen_in_window:
            # Alternating signatures (state ping-pong) still count as a repeating issue.
            level.vlm_issue_signature = signature
            level.vlm_issue_repeat_count = max(2, level.vlm_issue_repeat_count)
        else:
            level.vlm_issue_signature = signature
            level.vlm_issue_repeat_count = 1

    def _vlm_pacing_blocked(self) -> bool:
        # Spread the per-level call budget over the whole action budget. Without this,
        # bottleneck storms burn all calls in the first ~25 actions and the rest of the
        # episode runs on blind local fallbacks (observed on ls20/ft09/tr87/ar25).
        level = self.memory.level
        if not level.initial_vlm_done and not level.initial_vlm_attempted:
            return False
        # Measure progress within the current attempt: after a game over the call
        # budget resets, so pacing must not treat the whole-episode step count as
        # already-earned allowance (that would re-allow an instant burst).
        attempt_actions = max(0, level.total_action_count - level.attempt_started_at_action_count)
        remaining_budget = max(1, self.MAX_ACTIONS - level.attempt_started_at_action_count)
        progress = min(1.0, attempt_actions / remaining_budget)
        pace_cap = int(math.ceil(self.config.max_vlm_calls_per_level * progress)) + self.config.vlm_budget_burst
        return level.vlm_calls_this_level >= pace_cap

    def _should_call_vlm(self, scene: SceneSnapshot, transition: TransitionReport | None, legal: Sequence[Any] | None = None, *, force_bottleneck: bool = False) -> bool:
        level = self.memory.level
        if not self._vlm_available() or level.vlm_calls_this_level >= self.config.max_vlm_calls_per_level:
            return False
        if self._vlm_pacing_blocked():
            return False
        if force_bottleneck:
            return True

        transform_mode_active = self._transform_bottleneck_active()
        active_step = level.active_step()
        active_raw = active_step.raw if active_step is not None and isinstance(active_step.raw, dict) else {}
        route_reason = _short(active_raw.get("route_reason"), 80) if active_raw else ""
        continuing_sequence = bool(
            active_step is not None
            and active_step.step_type == "probe_action"
            and active_step.action
            and "sequence_index" in active_raw
            and self._trusted_plan_sequence(active_step, active_step.action)
        )
        if continuing_sequence and transition is not None and not transition.retry_detected and transition.life_delta in (None, 0):
            selector_sequence = route_reason.startswith("selector_")
            structural_progress = bool(transition.transformed_objects or transition.appeared_object_ids or transition.disappeared_object_ids or transition.interaction_event)
            progressed = bool(transition.moved_objects or (selector_sequence and not transition.effective_noop and structural_progress))
            if progressed:
                extra_gap = 8 if selector_sequence else 6 if route_reason == "coupled_carrier_geometry" else 4
                sequence = active_raw.get("action_sequence") if isinstance(active_raw, dict) else None
                sequence_len = len(sequence) if isinstance(sequence, list) else 0
                grace_floor = max(self.config.max_chunk_steps, self.config.vlm_min_action_gap + extra_gap)
                if sequence_len:
                    grace_floor = max(grace_floor, sequence_len + 1)
                sequence_grace = min(self.config.max_plan_steps, grace_floor)
                if level.actions_since_vlm < sequence_grace and level.repeated_noop_streak < 2:
                    return False

        if not level.initial_vlm_done and not level.initial_vlm_attempted:
            return True
        if legal is not None and level.active_step() is None and level.actions_since_vlm >= 1:
            legal_names = {action_name(a).upper() for a in legal}
            simple_legal = {name for name in legal_names if re.fullmatch(r"ACTION\d+", name)}
            click_only = bool(legal_names) and simple_legal <= {"ACTION6"} and any(name == "ACTION6" for name in legal_names)
            last = level.last_resolved_pending_action
            click_success_seen = bool(level.click_success_by_object or level.click_success_coords or level.click_success_regions)
            vlm_click_exhausted = bool(last is not None and last.name == "ACTION6" and last.source == "plan_executor")
            if click_only and (click_success_seen or vlm_click_exhausted or (transition is not None and not transition.effective_noop)):
                return True
        if level.bottleneck_reason:
            if level.action_recovery_contract and level.bottleneck_reason == "plan_step_noop" and level.actions_since_vlm >= 1:
                return True
            if level.vlm_issue_repeat_count >= 2:
                # V2.2: escalate the cooldown while the same issue keeps repeating.
                # A flat 3-step cooldown still allowed ~86% of all calls to be
                # bottleneck re-asks that returned the same answer (4s each).
                cooldown = min(12, self.config.vlm_repeat_bottleneck_cooldown * (level.vlm_issue_repeat_count - 1))
                if level.actions_since_vlm < cooldown:
                    return False
            return level.actions_since_vlm >= max(0, self.config.vlm_min_action_gap - 1)
        if level.resource_crisis and level.actions_since_vlm >= max(0, self.config.vlm_min_action_gap - 1):
            return True
        if transform_mode_active and level.actions_since_vlm >= self.config.vlm_min_action_gap:
            return True
        if transition is not None and (transition.retry_detected or transition.transformed_objects or transition.appeared_object_ids or transition.disappeared_object_ids or transition.life_delta not in (None, 0)) and level.actions_since_vlm >= self.config.vlm_min_action_gap:
            return True
        if level.repeated_noop_streak >= 2 and level.actions_since_vlm >= self.config.vlm_min_action_gap:
            return True
        if level.chunk_action_count >= self.config.max_chunk_steps and level.actions_since_vlm >= self.config.vlm_min_action_gap:
            return True
        if level.active_step() is None and level.actions_since_vlm >= self.config.vlm_min_action_gap:
            return level.initial_vlm_done or not level.initial_vlm_attempted
        loop_active, _, _ = self._unproductive_state_loop_signal(scene)
        if loop_active:
            return True
        return False

    def _maybe_call_vlm(self, scene: SceneSnapshot, transition: TransitionReport | None, legal: Sequence[Any]) -> None:
        if not self._should_call_vlm(scene, transition, legal):
            return
        level = self.memory.level
        transform_mode_active = self._transform_bottleneck_active()
        if not level.initial_vlm_done and not level.initial_vlm_attempted:
            mode = VLMMode.LEVEL_INIT
        elif level.bottleneck_reason or level.resource_crisis or transform_mode_active:
            mode = VLMMode.BOTTLENECK
        else:
            mode = VLMMode.EVALUATE_CHUNK
        self._request_vlm_once(scene, transition, legal, mode)

    def _request_vlm_once(self, scene: SceneSnapshot, transition: TransitionReport | None, legal: Sequence[Any], mode: VLMMode) -> V20VLMResult | None:
        level = self.memory.level
        if not self._vlm_available() or level.vlm_calls_this_level >= self.config.max_vlm_calls_per_level:
            return None
        if self._vlm_pacing_blocked():
            return None
        if mode == VLMMode.LEVEL_INIT:
            level.initial_vlm_attempted = True
        prompt = self._build_vlm_prompt(scene, transition, legal, mode.value)
        request = VLMRequest(prompt, scene.rgb or render_grid(scene.grid, self.config.image_size), None if transition is None else transition.previous_rgb, scene.annotated_rgb, self.config.vlm_max_new_tokens, temperature=self._vlm_temperature())
        level.vlm_calls_this_level += 1
        self.memory.game.vlm_calls_total += 1
        level.last_vlm_mode = mode.value
        call = level.vlm_calls_this_level
        call_context = {"bottleneck": level.bottleneck_reason, "transform_pressure": level.transform_pressure, "resource_crisis": level.resource_crisis}
        self.logger.log_event("vlm_call_v20", {"mode": mode.value, "level": level.level_index, "call": call, **call_context})
        raw_response: Any = None
        result: V20VLMResult | None = None
        parse_status = "empty_or_unparseable"
        error = ""
        try:
            raw_response = self.backend.decide(request)
            result = parse_vlm_result(raw_response)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.logger.log_exception(exc)
            self._record_vlm_io(request, raw_response, result, mode=mode.value, call=call, parse_status=parse_status, error=error, context=call_context)
            return None
        if result is None:
            # A totally unparseable response is the same "no usable plan" failure as
            # a parsed-but-empty one below; without counting it here it never feeds
            # consecutive_empty_plans, so _vlm_temperature() would keep retrying at
            # temperature=0 forever if the model deterministically repeats garbage.
            level.consecutive_empty_plans += 1
            self.logger.log_event("vlm_invalid", {"mode": mode.value, "raw_excerpt": "empty_or_unparseable", "consecutive_empty": level.consecutive_empty_plans})
            self._record_vlm_io(request, raw_response, result, mode=mode.value, call=call, parse_status=parse_status, context=call_context)
            return None
        if result.mode == "INVALID":
            schema_echo_invalid = str(result.raw_invalid_excerpt or "").startswith("schema_echo_only:")
            salvaged = None if schema_echo_invalid else self._salvage_vlm_text(result.raw_invalid_excerpt, mode.value)
            if salvaged is not None:
                salvaged.raw_invalid_excerpt = result.raw_invalid_excerpt
                result = salvaged
                parse_status = "salvaged"
                self.logger.log_event("vlm_invalid", {"mode": mode.value, "raw_excerpt": _short(result.raw_invalid_excerpt or "salvaged_invalid_json", 700), "salvaged": True})
            else:
                parse_status = "invalid_schema_echo" if schema_echo_invalid else "invalid"
                # Same rationale as the `result is None` branch above: an
                # unsalvageable INVALID response must count toward
                # consecutive_empty_plans or repeated non-JSON rambling at
                # temperature=0 (observed on ls20-9607627b) never escalates temperature.
                level.consecutive_empty_plans += 1
                level.notes_for_next_call = _short(f"Invalid VLM output excerpt: {result.raw_invalid_excerpt}", 700)
                self.logger.log_event("vlm_invalid", {"mode": mode.value, "raw_excerpt": result.raw_invalid_excerpt, "schema_echo": schema_echo_invalid, "consecutive_empty": level.consecutive_empty_plans})
                self._record_vlm_io(request, raw_response, result, mode=mode.value, call=call, parse_status=parse_status, context=call_context)
                return result
        else:
            parse_status = "parsed"
        level.actions_since_vlm = 0
        level.chunk_action_count = 0
        result.mode = mode.value
        self._apply_vlm_result(result, scene, legal)
        self._record_vlm_io(request, raw_response, result, mode=mode.value, call=call, parse_status=parse_status, context=call_context)
        self.logger.log_event("vlm_result_v20", {"mode": mode.value, "plan_steps": len(result.plan), "role_bindings": result.role_bindings, "has_win_update": bool(result.win_condition_update)})
        # 空计划纠正：活局中“可解析但无 plan”的回答（常见于误判已通关/建议重置的
        # 幻觉）浪费了一次预算。V2.0 契约禁止同回合二次调用，因此不当场重试，而是把
        # 强纠正反馈写进 rejected_vlm_plan_feedback + notes，让下一次按节奏的调用以
        # temperature=0 带着反驳重问（见 _vlm_temperature 对 empty_plan 的特判）。
        if mode != VLMMode.SUCCESS_REFLECT:
            if result.plan:
                level.consecutive_empty_plans = 0
            elif parse_status in {"parsed", "salvaged"}:
                level.consecutive_empty_plans += 1
                self._record_vlm_plan_rejections(
                    "empty_plan",
                    [{"reason": "empty_plan_for_live_level", "step": {"purpose": "response contained no executable plan"}}],
                    0,
                    0,
                )
                level.notes_for_next_call = _short(
                    "Your previous response contained NO executable plan. This level is LIVE and NOT complete (it did not advance); if a game over happened the game was already reset. An empty plan is invalid: return 1-6 concrete executable steps (probe_action/action_sequence/click with x,y) now. "
                    + (level.notes_for_next_call or ""),
                    700,
                )
                self.logger.log_event("vlm_empty_plan_feedback_v20", {"level": level.level_index, "mode": mode.value, "call": call, "consecutive_empty": level.consecutive_empty_plans})
                # Empty plans were a diagnostic blind spot: flush this call's raw IO
                # so we can tell "model returned no plan" from "JSON truncated".
                self._flush_last_vlm_io("empty_plan")
        return result

    def _salvage_vlm_text(self, text: str, mode: str) -> V20VLMResult | None:
        if not text:
            return None
        result = V20VLMResult(mode=mode, raw_invalid_excerpt=text)
        for action, meaning, kind, vx, vy in re.findall(r'"action"\s*:\s*"(ACTION\d+)".*?"meaning_nl"\s*:\s*"(.*?)".*?"kind"\s*:\s*"(.*?)"(?:.*?"vector"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\])?', text, flags=re.DOTALL):
            update = {"action": action, "meaning_nl": meaning, "kind": kind, "confidence": 0.45, "evidence": "salvaged from invalid VLM JSON"}
            if vx and vy:
                try:
                    update["vector"] = [int(vx), int(vy)]
                except Exception:
                    pass
            result.action_meaning_updates.append(update)
        roles = re.search(r'"role_bindings"\s*:\s*\{(.*?)\}', text, flags=re.DOTALL)
        if roles:
            for role, oid in re.findall(r'"([^"{}]+)"\s*:\s*"(O\d+)"', roles.group(1)):
                result.role_bindings[_short(role, 80)] = _short(oid, 24).upper()
        win_desc = re.search(r'"win_condition_update"\s*:\s*\{.*?"description_nl"\s*:\s*"(.*?)"', text, flags=re.DOTALL)
        if win_desc:
            result.win_condition_update = {"description_nl": win_desc.group(1), "confidence": 0.35, "evidence": "salvaged from invalid VLM JSON"}
        plan_goal = re.search(r'"plan_goal"\s*:\s*"(.*?)"', text, flags=re.DOTALL)
        if plan_goal:
            result.plan_goal = _short(plan_goal.group(1), 420)
        return result if (result.action_meaning_updates or result.role_bindings or result.win_condition_update or result.plan_goal) else None

    def _build_vlm_prompt(self, scene: SceneSnapshot, transition: TransitionReport | None, legal: Sequence[Any], mode: str, extra: dict[str, Any] | None = None) -> str:
        legal_names = [action_name(a) for a in legal]
        objs = [self._object_prompt(o, scene) for o in self._prompt_objects(scene)]
        transition_payload = None
        if transition is not None:
            transition_payload = {"action": transition.action_key, "source": transition.action_source, "summary": transition.summary, **self._transition_delta_dict(transition)}
        level = self.memory.level
        payload = {
            "task": "Return a short executable plan using state-action-outcome evidence. Prefer progress over conserving step count, but do not waste counter/lives on known no-ops.",
            "arc_action_reference": {
                **V20_ARC_ACTION_REFERENCE,
                "transform_warning": "If an action changes patterns/objects without simple actor translation, treat it as transform/control, not navigation movement.",
            },
            "mode": mode,
            "game_id": self._game_id(),
            "episode_state": {
                "state": "ACTIVE",
                "attempt_index": level.attempt_index,
                "note": (
                    "A previous attempt ended in GAME_OVER and the game was ALREADY reset; this is a live fresh attempt. Never request RESET and never return an empty plan."
                    if level.attempt_index > 0
                    else "Live game. Never request RESET and never return an empty plan."
                ),
            },
            "image_order": ["CURRENT_RAW", "CURRENT_ANNOTATED"] if transition is None else ["BEFORE_RAW", "CURRENT_RAW", "CURRENT_ANNOTATED"],
            "game_action_space": [f"ACTION{v}" for v in self.memory.game.game_action_space] or None,
            "legal_actions_now": legal_names,
            "observer": {
                "scene_summary": scene.summary[:5600],
                "objects": objs,
                "framed_template_relations": list(scene.template_relations),
                "controlled_object_id": level.controlled_object_id or None,
                "resource": {"counter": scene.counter_value, "capacity": scene.counter_capacity, "ratio": scene.counter_ratio, "lives": scene.life_count},
            },
            "last_transition": transition_payload,
            "game_memory": self.memory.game.as_prompt(),
            "level_memory": level.as_prompt(scene),
            # Only genuinely new information lives here. quarantine-by-state,
            # resource_crisis, transform_pressure, click_noops/success_by_object,
            # click_success_coords/regions and terminal_bad_action_suffixes are
            # already sent once inside level_memory (LevelMemoryV20.as_prompt); do
            # not repeat them here, it only inflates the prompt (this game's
            # text_prompt grew past 40k chars) without adding signal.
            "v20_policy_hints": {
                "quarantine_global": dict(level.global_quarantine_until),
                "avoid_action7": self.config.avoid_action7,
                "recent_action_keys": list(level.recent_action_keys),
                "banned_click_coords": self._banned_click_coords_for_prompt(),
            },
            "plan_contract": {
                "allowed_route_reasons": list(TRUSTED_ROUTE_REASONS),
                "preferred_shapes": [
                    "probe_action with one concrete ACTION name",
                    "probe_action with action_sequence plus enum route_reason",
                    "click with x,y or coordinate_sequence",
                    "click with target_object_id plus click_hint when exact coordinates are uncertain",
                ],
                "bad_shapes": [
                    "single repeated movement guess with no transition evidence",
                    "click with only a vague natural-language target",
                    "move_to/interact against status_or_template_cue_not_move_target",
                ],
            },
            "requirements": [
                "Return only JSON. Do not ask for RESET unless terminal.",
                "OUTPUT BUDGET: keep the whole JSON under 1200 tokens. Put plan first. The plan is the one part you must NEVER shorten to save tokens: when budget is tight, omit action_meaning_updates/object_effect_updates/mechanics_updates instead of dropping plan steps. Keep every meaning_nl under 40 characters. Never quote transition deltas or pattern strings verbatim.",
                "VLM calls are budgeted per level. Whenever evidence justifies more than one action, return the full 2-6 action route in ONE response (action_sequence with route_reason, or multiple plan steps, or coordinate_sequence). Do not return a single-action plan when a longer justified route exists. HARD LIMIT: never put more than 8 actions in one action_sequence; long repetition wastes the output budget and gets truncated.",
                "Never propose coordinates listed in v20_policy_hints.banned_click_coords; pick a different object/region instead of repeating a banned click.",
                "A level is complete ONLY when the level actually advances. If you believe the goal is met but the level did not advance, your win-condition hypothesis is WRONG: revise it and return a new executable plan. Never return an empty plan for a live level.",
            ] + ([
                "FUEL GAME: the step counter drops by 1 on EVERY action and hitting 0 costs a life and teleports the actor back to spawn. Single-action probe plans waste fuel. Movement vectors are already confirmed in game_memory.action_meanings: compute the route arithmetically (steps_x = |dx|/|vector_x|, steps_y = |dy|/|vector_y|) and return the COMPLETE action_sequence to the current target with route_reason=geometry.",
            ] if level.counter_is_step_tax else []) + ([
                "COUNTER CRITICALLY LOW: do NOT return multi-step geometry routes. Return ONE probe_action with route_reason=mechanism_probe or selector_probe and structural_change expected_predicates; plan_executor is blocked at counter<=1.",
            ] if self._critical_counter_blocks_plan_executor(scene) else []) + ([
                "COUNTER IS CRITICALLY LOW: an empty plan now wastes a life. Return the shortest concrete action_sequence toward the bound target immediately, even if uncertain.",
            ] if level.resource_crisis and not self._critical_counter_blocks_plan_executor(scene) else []) + ([
                "This game's declared action space is small and explicitly includes ACTION7, so here ACTION7 is likely a core mechanic (confirm/cycle/use) rather than undo. If it is still untested, one evidence probe of ACTION7 is worthwhile.",
            ] if self._action7_probe_allowed() else []) + ([
                V20_LEVEL_INIT_SIMPLE_ACTION_HINT,
            ] if mode == VLMMode.LEVEL_INIT.value else []) + [
                "Use current local O-ids only inside level role_bindings/plan; game-level memory must use visual descriptors.",
                "If ACTION7 has recent no-op/undo evidence, do not choose it as the default interaction.",
                "LEVEL_INIT: do not assert action meanings, movement vectors, or win conditions until transition evidence exists; prefer a short probe/action_sequence with explicit uncertainty.",
                "For click-only games, return exact scalar x,y or coordinate_sequence when possible; if not, give target_object_id plus click_hint=center|edge|corner|outside-adjacent so the agent can derive one coordinate.",
                "For transform-heavy games, plan a sequence that tests/uses transform outcomes; do not repeat a single transform action indefinitely.",
                "For action_sequence, route_reason must be one of plan_contract.allowed_route_reasons exactly; use selector_* only for cycling/status-pattern mechanics.",
                "Under low counter/lives, choose one high-information action rather than guard-spamming.",
                "If a target is marked status_or_template_cue_not_move_target, do not navigate directly to it; use a reachable playfield object/waypoint or give a new mechanism hypothesis.",
                "If a target is marked mechanism_candidate_not_move_target, it has measured transform evidence (its pattern changed) but is not currently reachable by move_to: probe/observe it via probe_action or explain how a coupled object reaches it, do not silently ignore it as decoration.",
                "If transition evidence rejects an action as noop/transform for navigation, do not repeat it as a movement step.",
                "Return plan as executable action contract. Prefer probe_action/action_sequence/click coordinates. Avoid move_to unless a measured movement vector exists in game_memory.",
                "Every plan step should include expected_predicates: not_noop, no_retry, no_life_loss, and one of controlled_motion/structural_change/appeared_or_disappeared when applicable.",
                "If recovery_contract.rejected_vlm_plan_feedback says blocked_by_transition_evidence, do not repeat the same action unless the step has allow_once_if_blocked=true and a concrete expected_predicates list.",
            ],
            "output_schema": {
                "mode": mode,
                "action_meaning_updates": [{"action": "<ACTION_NAME>", "meaning_nl": "<SHORT_OBSERVED_EFFECT>", "kind": "movement|interact_or_transform|click_or_select|undo|resource|unknown_or_blocked", "vector": ["<DX_INT>", "<DY_INT>"], "confidence": "<0_TO_1>", "evidence": "<OBSERVED_TRANSITION_EVIDENCE>"}],
                "role_bindings": {"<ROLE_NAME>": "<CURRENT_O_ID>"},
                "win_condition_update": {"description_nl": "<TRANSFERABLE_RULE_NO_BARE_O_IDS>", "visual_roles": {"<ROLE_NAME>": {"shape_label": "<SHAPE_OR_PATTERN_LABEL>"}}, "confidence": "<0_TO_1>", "evidence": "<OBSERVED_TRANSITION_EVIDENCE>"},
                "object_effect_updates": [{"local_object_id": "<CURRENT_O_ID>", "visual_descriptor": {}, "effect_nl": "<SHORT_OBSERVED_EFFECT>", "confidence": "<0_TO_1>", "evidence": "<OBSERVED_TRANSITION_EVIDENCE>"}],
                "mechanics_updates": ["<SHORT_REUSABLE_MECHANIC>"],
                "resource_update": {"description_nl": "<RESOURCE_LIFE_COUNTER_BEHAVIOR>", "confidence": "<0_TO_1>"},
                "plan_goal": "<CURRENT_LEVEL_GOAL>",
                "plan": [
                    {"type": "probe_action", "action": "<ACTION_NAME>", "purpose": "<WHY_THIS_ACTION_IS_NEXT>", "stop_condition": "<STOP_AFTER_OBSERVABLE_CHANGE>", "expected_predicates": [{"type": "not_noop"}, {"type": "no_retry"}, {"type": "no_life_loss"}], "max_attempts": 1},
                    {"type": "probe_action", "action_sequence": ["<ACTION_NAME>", "<ACTION_NAME>"], "purpose": "<WHY_THIS_SEQUENCE_IS_NEXT>", "route_reason": "<EVIDENCE_FOR_THIS_ORDER>", "expected_predicates": [{"type": "not_noop"}, {"type": "structural_change"}], "max_attempts": 1},
                    {"type": "click", "action": "<CLICK_ACTION_NAME>", "x": "<X_INT>", "y": "<Y_INT>", "purpose": "<WHY_THIS_CLICK_IS_NEXT>", "stop_condition": "<STOP_AFTER_OBSERVABLE_CHANGE>", "expected_predicates": [{"type": "not_noop"}, {"type": "no_retry"}], "max_attempts": 1},
                ],
                "steps": [
                    {
                        "type": "probe_action",
                        "action": "<ACTION_NAME>",
                        "purpose": "<WHY_THIS_ACTION_IS_NEXT>",
                        "route_reason": "geometry|selector_probe|selector_calibration|selector_cycle_learn|selector_pattern_rule|mechanism_probe|resource_recovery",
                        "expected_predicates": [{"type": "not_noop"}, {"type": "no_retry"}, {"type": "no_life_loss"}, {"type": "structural_change"}],
                        "allow_once_if_blocked": False,
                        "max_attempts": 1,
                    },
                    {
                        "type": "click",
                        "action": "<CLICK_ACTION_NAME>",
                        "x": "<X_INT>",
                        "y": "<Y_INT>",
                        "coordinate_sequence": [["<X_INT>", "<Y_INT>"]],
                        "purpose": "<WHY_THIS_CLICK_IS_NEXT>",
                        "expected_predicates": [{"type": "not_noop"}, {"type": "no_retry"}],
                        "max_attempts": 1,
                    },
                ],
                "bottleneck_analysis": "",
                "notes_for_next_call": "",
            },
        }
        state_visits = Counter(level.recent_state_hashes)
        recovery_active = bool(
            level.action_recovery_contract
            or (mode == VLMMode.BOTTLENECK.value and level.bottleneck_reason)
            or level.repeated_noop_streak >= 2
            or level.transform_pressure >= max(4, self.config.transform_mode_threshold * 2)
            or any(v > 0 for v in level.click_noops_by_object.values())
        )
        if recovery_active:
            recent_failed = [e.as_prompt() for e in list(level.recent_events)[-8:] if e.outcome in {"noop", "retry", "transform"}]
            payload["recovery_contract"] = {
                "active": True,
                "reason": level.bottleneck_reason or "repeated failed local plan",
                "recent_failed_events": recent_failed[-5:],
                "rejected_vlm_plan_feedback": _json_safe(level.rejected_vlm_plan_feedback[-6:]),
                "rules": [
                    "Read rejected_vlm_plan_feedback before planning; do not repeat rejected action shapes unless new transition evidence contradicts the rejection.",
                    "Prefer action-level plans: probe_action steps or action_sequence lists.",
                    "Do not return move_to/interact with only target_object_id after target rejection or noop.",
                    "For ACTION6, give explicit x,y or coordinate_sequence; avoid center-only repeats.",
                    "For transform-heavy states, return a short ordered action_sequence, not movement labels.",
                ],
                "valid_plan_shapes": [
                    {"type": "probe_action", "action": "<ACTION_NAME>", "purpose": "<WHY_THIS_ACTION_IS_NEXT>", "expected_predicates": [{"type": "not_noop"}, {"type": "no_retry"}]},
                    {"type": "probe_action", "action_sequence": ["<ACTION_NAME>", "<ACTION_NAME>"], "purpose": "<WHY_THIS_SEQUENCE_IS_NEXT>", "route_reason": "<EVIDENCE_FOR_THIS_ORDER>", "expected_predicates": [{"type": "structural_change"}]},
                    {"type": "click", "action": "<CLICK_ACTION_NAME>", "coordinate_sequence": [["<X_INT>", "<Y_INT>"], ["<X_INT>", "<Y_INT>"]], "purpose": "<WHY_THIS_CLICK_IS_NEXT>", "expected_predicates": [{"type": "not_noop"}]},
                ],
            }
            payload["requirements"].extend([
                "RECOVERY MODE: plan must be executable without target pathfinding guesses.",
                "REJECTED PLAN FEEDBACK: if recovery_contract.rejected_vlm_plan_feedback lists blocked_by_transition_evidence or untrusted_repeated_action_sequence, return a different action sequence or click coordinate plan, not the same action/repetition.",
                "Use action_sequence for repeated simple actions; repeated actions are allowed when intentional.",
                "Use coordinate_sequence or explicit x,y for click-only recovery; do not rely on center-only click_hint.",
            ])
            payload["output_schema"]["plan"] = payload["recovery_contract"]["valid_plan_shapes"]
        loop_active, _, loop_reason = self._unproductive_state_loop_signal(scene)
        if loop_active:
            payload["loop_warning"] = {
                "reason": loop_reason,
                "repeated_states": {k[:12]: v for k, v in state_visits.items() if v >= 2},
                "advice": "Recent transitions were mostly no-op/retry while revisiting the same states. Propose a genuinely different action/target/coordinate; do not repeat the same probe.",
            }
        if self._transform_cycle_bottleneck_active(level) or self._mechanism_hypothesis_locked():
            payload["transform_cycle_contract"] = {
                "active": True,
                "reason": level.bottleneck_reason,
                "locked_for_level": self._mechanism_hypothesis_locked(),
                "selector_pair_objects": sorted(level.selector_pair_object_ids),
                "advice": (
                    "Recent actions cycled transform states without progress. "
                    "Do NOT return route_reason=geometry or coupled_carrier_geometry. "
                    "Return a short selector_probe/mechanism_probe action_sequence that tests "
                    "how the selector frame changes the status frame inner_pattern relative to the goal glyph."
                ),
            }
            payload["requirements"].extend([
                "TRANSFORM STATE CYCLE: geometry/coupled_carrier_geometry routes are rejected. Use selector_probe, selector_cycle_learn, or mechanism_probe with expected_predicates including structural_change.",
                "Bind role_bindings when possible: selector_frame=small cycling frame, status_template_frame=corner pattern frame, actor=controlled block, target_glyph=goal glyph.",
            ] + ([
                "This ban persists for the rest of the level (it survives a game-over reset): do not retry geometry/coupled_carrier_geometry routes even in a fresh attempt.",
            ] if self._mechanism_hypothesis_locked() else []))
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=True, default=str)

    def _banned_click_coords_for_prompt(self) -> list[str]:
        level = self.memory.level
        banned: dict[str, bool] = {}
        for key in level.global_forbidden_click_coords:
            banned[key] = True
        hard_cap = self.config.max_clicks_per_coord_total * 3
        for key, count in level.click_coord_counts.most_common(24):
            if count >= hard_cap:
                banned[str(key)] = True
        return sorted(banned)[:16]

    def _prompt_objects(self, scene: SceneSnapshot) -> list[ObjectObservation]:
        limit = max(1, int(self.config.max_prompt_objects))
        ranked = sorted(scene.objects, key=lambda o: (-float(o.salience), o.track_id))
        if len(ranked) <= limit:
            return ranked
        selected: list[ObjectObservation] = []
        seen: set[str] = set()

        def add(obj: ObjectObservation) -> None:
            if obj.track_id not in seen and len(selected) < limit:
                seen.add(obj.track_id)
                selected.append(obj)

        seed_count = min(len(ranked), max(4, min(6, limit // 3 or 1)))
        for obj in ranked[:seed_count]:
            add(obj)
        while len(selected) < limit and len(seen) < len(ranked):
            best: tuple[float, float, str, ObjectObservation] | None = None
            for obj in ranked:
                if obj.track_id in seen:
                    continue
                cx, cy = obj.centroid
                min_dist = min((cx - so.centroid[0]) ** 2 + (cy - so.centroid[1]) ** 2 for so in selected) if selected else 0.0
                score = min_dist + 120.0 * float(obj.salience)
                if obj.near_edge:
                    score -= 25.0
                tie = float(cx + cy)
                item = (score, tie, obj.track_id, obj)
                if best is None or item > best:
                    best = item
            if best is None:
                break
            add(best[3])
        return selected

    def _object_prompt(self, obj: ObjectObservation, scene: SceneSnapshot) -> dict[str, Any]:
        points = self._click_candidates_for_object(scene, obj)[:6]
        actor_id = self.memory.level.controlled_object_id
        if actor_id and obj.track_id == actor_id:
            nav_candidate = True
            role_hint = "controlled_actor"
        else:
            if self._is_possible_selector_pair_object(scene, obj):
                # V2.4: weak pre-label from template_relations before first transform.
                nav_candidate = False
                role_hint = "mechanism_candidate_not_move_target"
            else:
                nav_candidate = self._is_navigable_target(scene, obj) if actor_id else self._looks_like_playfield_object(scene, obj)
                if nav_candidate:
                    role_hint = "playfield_navigation_candidate"
                elif obj.track_id in self.memory.level.mechanism_evidence_object_ids:
                    # V2.3: distinct from plain decoration - this object has measured
                    # transform evidence but is not (yet) reachable by move_to. Tell the
                    # VLM to test/observe it via probe_action instead of silently
                    # bucketing it with HUD/template cues it should never touch.
                    role_hint = "mechanism_candidate_not_move_target"
                else:
                    role_hint = "status_or_template_cue_not_move_target"
        return {"id": obj.track_id, "descriptor": _json_safe(self._visual_descriptor(obj, scene)), "bbox": obj.bbox, "centroid": obj.centroid, "area": obj.area, "salience": round(obj.salience, 3), "navigation_candidate": nav_candidate, "role_hint": role_hint, "click_points_sample": points}

    def _visual_descriptor(self, obj: ObjectObservation, scene: SceneSnapshot | None = None) -> VisualDescriptor:
        if obj.area <= 9:
            size = "tiny"
        elif obj.area <= 40:
            size = "small"
        elif obj.area <= 160:
            size = "medium"
        else:
            size = "large"
        tags = []
        if scene is not None:
            for rel in scene.template_relations:
                if obj.track_id in {rel.get("left"), rel.get("right")}:
                    if rel.get("edge_vs_world"):
                        tags.append("paired_with_edge_or_status_frame")
                    if rel.get("exact_inner_match"):
                        tags.append("exact_inner_match")
                    elif rel.get("same_shape_under_rotation"):
                        tags.append("same_shape_under_rotation")
                    elif rel.get("possible_selector_pair"):
                        tags.append("possible_selector_pair")
        return VisualDescriptor(obj.shape_label, list(obj.colors), {str(k): int(v) for k, v in obj.color_areas}, obj.pattern[:240], obj.inner_pattern[:240], obj.frame_color, size, sorted(set(tags)), obj.near_edge, obj.type_key)

    def _sanitize_game_text(self, text: str, scene: SceneSnapshot | None = None) -> str:
        text = _short(text, 900)
        if scene is None:
            return re.sub(r"\bO\d+\b", "a local object", text)
        def repl(match: re.Match[str]) -> str:
            oid = match.group(0).upper()
            obj = scene.object_by_id(oid)
            if obj is None:
                return "a local object"
            desc = self._visual_descriptor(obj, scene)
            s = f"{desc.shape_label} colors={desc.colors} size={desc.size_bucket}"
            if desc.inner_pattern:
                s += f" inner_pattern={desc.inner_pattern[:80]}"
            if desc.relation_tags:
                s += f" tags={','.join(desc.relation_tags[:3])}"
            return s
        return re.sub(r"\bO\d+\b", repl, text)

    def _legal_action_name_set(self, legal: Sequence[Any] | None) -> set[str]:
        names: set[str] = set()
        for action in legal or ():
            name = action_name(action).upper()
            if name and name != "RESET":
                names.add(name)
        return names

    def _click_only_legal(self, legal: Sequence[Any] | None) -> bool:
        return self._legal_action_name_set(legal) == {"ACTION6"}

    def _initial_grounding_simple_actions(self, legal_names: set[str]) -> set[str]:
        candidates: set[str] = set()
        for name in legal_names:
            action = _short(name, 40).upper()
            if action == "ACTION6" or not re.fullmatch(r"ACTION\d+", action):
                continue
            if action == "ACTION7" and self.config.avoid_action7 and not self._action7_probe_allowed():
                meaning = self.memory.game.action_meanings.get(action)
                if not (
                    meaning is not None
                    and meaning.positive_count > 0
                    and meaning.noop_ratio < 0.5
                    and meaning.kind != "undo"
                ):
                    continue
            candidates.add(action)
        return candidates

    def _raw_plan_action_names(self, raw: dict[str, Any]) -> list[str]:
        names: list[str] = []
        sequence = raw.get("actions") or raw.get("action_sequence") or raw.get("probe_actions")
        if isinstance(sequence, list):
            for item in sequence:
                name = item.get("action") or item.get("name") if isinstance(item, dict) else item
                action = _short(name, 40).upper()
                if re.fullmatch(r"ACTION\d+", action):
                    names.append(action)
        name = _short(raw.get("action") or raw.get("name"), 40).upper()
        if re.fullmatch(r"ACTION\d+", name) and name not in names:
            names.insert(0, name)
        return names

    def _raw_plan_has_execution_evidence(self, raw: dict[str, Any]) -> bool:
        if any(_short(raw.get(key), 160) for key in ("route_reason", "evidence", "expected_change", "stop_condition")):
            return True
        expected = raw.get("expected_predicates")
        return isinstance(expected, list) and bool(expected)

    def _infer_raw_plan_target_id(self, raw: dict[str, Any], scene: SceneSnapshot) -> str:
        valid_ids = {o.track_id for o in scene.objects}
        if not valid_ids:
            return ""
        fields: list[str] = []
        for key in ("target_object_id", "object_id", "local_object_id", "target", "target_id", "purpose", "why", "click_hint", "click_point", "expected_change", "stop_condition", "evidence"):
            value = raw.get(key)
            if isinstance(value, str):
                fields.append(value)
        for key in ("actions", "action_sequence", "probe_actions"):
            sequence = raw.get(key)
            if not isinstance(sequence, list):
                continue
            for item in sequence:
                if not isinstance(item, dict):
                    continue
                for subkey in ("target_object_id", "object_id", "local_object_id", "target", "purpose", "why", "click_hint", "expected_change", "stop_condition", "evidence"):
                    value = item.get(subkey)
                    if isinstance(value, str):
                        fields.append(value)
        for text in fields:
            for oid in re.findall(r"\bO\d+\b", text.upper()):
                if oid in valid_ids:
                    return oid
        return ""

    def _vlm_recovery_probe_override(self, step: PlanStep, scene: SceneSnapshot, action: str) -> bool:
        action = _short(action, 40).upper()
        if not re.fullmatch(r"ACTION\d+", action) or action in {"ACTION6", "RESET"}:
            return False
        if step.step_type != "probe_action":
            return False
        raw = step.raw if isinstance(step.raw, dict) else {}
        if not raw or "sequence_index" in raw:
            return False
        level = self.memory.level
        recovery_active = bool(raw.get("vlm_recovery_override") or level.action_recovery_contract or level.bottleneck_reason or level.repeated_noop_streak >= 2 or self._in_transform_mode(scene))
        if not recovery_active or not self._raw_plan_has_execution_evidence(raw):
            return False
        meaning = self.memory.game.action_meanings.get(action)
        if meaning is not None and meaning.transforms + meaning.interactions > 0 and meaning.life_losses == 0 and meaning.retries == 0:
            return True
        if meaning is not None and meaning.kind == "resource_wasting_noop" and meaning.positive_count == 0:
            return False
        if action == "ACTION7" and self.config.avoid_action7:
            return bool(meaning is not None and meaning.positive_count > 0 and meaning.noop_ratio < 0.5 and meaning.kind != "undo")
        return True

    def _raw_plan_quality_issue(self, raw: dict[str, Any], scene: SceneSnapshot, legal_names: set[str], *, recovery_active: bool, click_only: bool) -> str:
        step_type = _short(raw.get("type") or raw.get("step_type") or raw.get("intent"), 40).lower()
        if step_type in {"click_object", "test_object"}:
            step_type = "click"
        actions = self._raw_plan_action_names(raw)
        if legal_names:
            for action in actions:
                if action not in legal_names:
                    return "action_not_legal_now"

        purpose = _schema_text(raw.get("purpose") or raw.get("why"))
        purpose_key = _schema_key(raw.get("purpose") or raw.get("why"))
        coords = self._iter_raw_click_coordinates(raw)
        has_evidence = self._raw_plan_has_execution_evidence(raw)
        if recovery_active:
            if purpose_key in _RECOVERY_TEMPLATE_PURPOSE_KEYS:
                return "recovery_template_echo"
            if actions == ["ACTION2", "ACTION2", "ACTION4"] and not has_evidence:
                return "recovery_template_echo"
            if coords == [(10, 10), (12, 10)] and not has_evidence:
                return "recovery_template_echo"
            if step_type in {"move_to", "interact"} and raw.get("target_object_id") and not actions:
                return "recovery_requires_action_sequence"

        click_intent = step_type == "click" or "ACTION6" in actions
        simple_nonclick_legal = bool(self._initial_grounding_simple_actions(legal_names))
        click_evidence_seen = bool(
            self.memory.level.click_success_by_object
            or self.memory.level.click_success_coords
            or self.memory.level.click_success_regions
            or self.memory.level.action_outcomes.get("ACTION6", Counter())
            or self.memory.level.click_noops_by_object
        )
        if click_intent and not click_only and simple_nonclick_legal and len(scene.objects) > 1 and not self.memory.level.initial_vlm_done and self.memory.level.total_action_count == 0 and not click_evidence_seen:
            return "initial_click_deferred_until_simple_actions_grounded"
        if click_intent and click_only and not coords:
            target = _short(raw.get("target_object_id") or raw.get("object_id"), 24).upper()
            role = _short(raw.get("target_role") or raw.get("role"), 80)
            resolved = target if scene.object_by_id(target) is not None else ""
            if not resolved and role:
                resolved = self.memory.level.local_bindings.get(role, "")
            if not resolved:
                resolved = self._infer_raw_plan_target_id(raw, scene)
            if not resolved or scene.object_by_id(resolved) is None:
                return "click_requires_explicit_coordinates"
        return ""

    def _filter_vlm_raw_plan_steps(self, raw_steps: list[dict[str, Any]], scene: SceneSnapshot, legal: Sequence[Any] | None) -> tuple[list[dict[str, Any]], str]:
        legal_names = self._legal_action_name_set(legal)
        if not legal_names:
            return [dict(step) for step in raw_steps if isinstance(step, dict)], ""
        level = self.memory.level
        recovery_active = bool(level.action_recovery_contract or level.bottleneck_reason or level.repeated_noop_streak >= 2 or self._in_transform_mode(scene) or self._resource_crisis_active(scene))
        click_only = self._click_only_legal(legal)
        kept: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        first_reason = ""
        for raw in raw_steps:
            if not isinstance(raw, dict):
                continue
            issue = self._raw_plan_quality_issue(raw, scene, legal_names, recovery_active=recovery_active, click_only=click_only)
            if issue:
                first_reason = first_reason or issue
                dropped.append({"reason": issue, "step": _json_safe(raw)})
                continue
            kept.append(dict(raw))
        if dropped:
            self.logger.log_event("vlm_plan_steps_rejected_v20", {"level": level.level_index, "reason": "raw_plan_quality", "dropped": dropped[:8], "kept": len(kept), "input": len(raw_steps)})
            self._record_vlm_plan_rejections("raw_plan_quality", dropped, len(kept), len(raw_steps))
        return kept, first_reason


    def _record_vlm_plan_rejections(self, gate: str, dropped: list[dict[str, Any]], kept_count: int, input_count: int) -> None:
        if not dropped:
            return
        level = self.memory.level
        reasons: Counter[str] = Counter()
        actions: Counter[str] = Counter()
        examples: list[dict[str, Any]] = []
        for item in dropped:
            if not isinstance(item, dict):
                continue
            reason = _short(item.get("reason"), 80) or "rejected"
            reasons[reason] += 1
            raw_step = item.get("step") if isinstance(item.get("step"), dict) else item
            if not isinstance(raw_step, dict):
                continue
            sequence = raw_step.get("action_sequence") or raw_step.get("actions") or raw_step.get("probe_actions")
            counted_sequence = False
            if isinstance(sequence, list):
                for raw_action in sequence:
                    name = raw_action.get("action") or raw_action.get("name") if isinstance(raw_action, dict) else raw_action
                    action = _short(name, 40).upper()
                    if re.fullmatch(r"ACTION\d+", action):
                        actions[action] += 1
                        counted_sequence = True
            action = _short(raw_step.get("action") or raw_step.get("name"), 40).upper()
            if action and re.fullmatch(r"ACTION\d+", action) and not counted_sequence:
                actions[action] += 1
            if len(examples) < 4:
                examples.append({
                    "reason": reason,
                    "type": _short(raw_step.get("type") or raw_step.get("step_type") or raw_step.get("intent"), 40),
                    "action": action,
                    "target_object_id": _short(raw_step.get("target_object_id") or raw_step.get("object_id"), 24).upper(),
                    "purpose": _short(raw_step.get("purpose") or raw_step.get("why"), 180),
                })
        entry = {
            "gate": _short(gate, 60),
            "reasons": dict(reasons.most_common(6)),
            "top_rejected_actions": dict(actions.most_common(6)),
            "examples": examples,
            "kept": int(kept_count),
            "input": int(input_count),
            "at_action_count": int(level.total_action_count),
        }
        level.rejected_vlm_plan_feedback.append(_json_safe(entry))
        level.rejected_vlm_plan_feedback = level.rejected_vlm_plan_feedback[-8:]

    def _apply_vlm_result(self, result: V20VLMResult, scene: SceneSnapshot, legal: Sequence[Any] | None = None) -> None:
        level = self.memory.level
        valid_ids = {o.track_id for o in scene.objects}
        actor_role_names = {"actor", "player", "controlled_object"}
        result_has_plan = bool(result.plan)
        deferred_actor_bindings: list[tuple[str, str]] = []
        accepted_vlm_plan_steps = False
        for role, oid in result.role_bindings.items():
            if oid not in valid_ids:
                continue
            role_key = _short(role, 80)
            role_l = role_key.lower()
            if role_l in self._STALE_NAV_ROLE_NAMES and self._mechanism_hypothesis_locked():
                # V2.4: ls20 regression - once escalated, the VLM kept reflexively
                # re-proposing role_bindings={"target": "<glyph>"} even in the SAME
                # call whose plan got vetted-rejected for route_reason=geometry. The
                # rejected plan never reached the executor, but this binding still
                # got written and _navigation_target_id's role_order lookup handed
                # it straight back to deterministic_navigation/fuel_target_greedy.
                self.logger.log_event("role_binding_rejected_v20", {"level": level.level_index, "role": role_key, "proposed": oid, "reason": "role_locked_by_mechanism_hypothesis"})
                continue
            if role_l in actor_role_names:
                obj = scene.object_by_id(oid)
                if self._mechanism_hypothesis_locked() and obj is not None and (obj.frame_color is not None or self._is_possible_selector_pair_object(scene, obj)):
                    self.logger.log_event("role_binding_rejected_v20", {"level": level.level_index, "role": role_key, "proposed": oid, "current": level.controlled_object_id, "reason": "actor_role_points_to_mechanism_frame"})
                    continue
                if level.controlled_object_id:
                    current_votes = level.actor_votes.get(level.controlled_object_id, 0)
                    proposed_votes = level.actor_votes.get(oid, 0)
                    if oid != level.controlled_object_id and (
                        (current_votes >= 1 and proposed_votes <= current_votes)
                        or (current_votes >= 2 and proposed_votes < current_votes)
                    ):
                        self.logger.log_event("role_binding_rejected_v20", {"level": level.level_index, "role": role_key, "proposed": oid, "current": level.controlled_object_id, "reason": "actor_transition_evidence_stronger"})
                        continue
                if result_has_plan and oid != level.controlled_object_id:
                    deferred_actor_bindings.append((role_key, oid))
                    continue
            level.local_bindings[role_key] = oid
            if role_l in actor_role_names:
                level.controlled_object_id = oid
        for upd in result.action_meaning_updates:
            action = _short(upd.get("action"), 40).upper()
            if not re.fullmatch(r"ACTION\d+", action):
                continue
            meaning = self.memory.game.action_meanings.setdefault(action, ActionMeaning(action))
            text = self._sanitize_game_text(_short(upd.get("meaning_nl") or upd.get("meaning"), 500), scene)
            conf = _clamp01(upd.get("confidence", 0.0))
            kind = self._normalize_vlm_action_kind(_short(upd.get("kind"), 80).lower(), meaning)
            vec = upd.get("vector")
            vlm_vec: tuple[int, int] | None = None
            if isinstance(vec, list) and len(vec) == 2:
                try:
                    vlm_vec = (int(vec[0]), int(vec[1]))
                except Exception:
                    vlm_vec = None

            measured_vec = meaning.vector if meaning.movements > 0 and meaning.vector is not None else None
            if kind == "movement" and action == "ACTION7" and self.config.avoid_action7 and meaning.noop_ratio > 0.5:
                kind = meaning.kind or "unknown_or_blocked"

            if kind == "movement":
                if measured_vec is not None:
                    if vlm_vec is not None and vlm_vec != measured_vec:
                        self.logger.log_event("vlm_action_update_rejected", {"level": level.level_index, "action": action, "reason": "vector_conflicts_with_transition", "vlm_vector": list(vlm_vec), "measured_vector": list(measured_vec), "vlm_confidence": conf})
                    meaning.kind = "movement"
                    meaning.vector = measured_vec
                    meaning.confidence = max(meaning.confidence, min(conf, 0.86))
                    if text and conf >= meaning.confidence - 0.15:
                        meaning.meaning_nl = text
                    continue
                # V2.2: accept the VLM's movement claim when the local displacement
                # votes already point the same way. In side-effect-heavy games the
                # strict vector confirmation above never fires, and rejecting a claim
                # that 80%+ of local observations support (re86: 84 rejections, 93%
                # locally corroborated) kept the navigation vector channel dead.
                dominant = _dominant_vector(meaning.vector_votes)
                if (
                    vlm_vec is not None
                    and dominant is not None
                    and dominant[1] >= 2
                    and meaning.movements > 0
                    and meaning.noop_ratio < 0.5
                    and _vector_axis_signature(dominant[0]) == _vector_axis_signature(vlm_vec)
                ):
                    meaning.kind = "movement"
                    meaning.vector = dominant[0]
                    meaning.confidence = max(meaning.confidence, min(conf, 0.6))
                    if text and conf >= meaning.confidence - 0.15:
                        meaning.meaning_nl = text
                    self.logger.log_event("vlm_action_update_accepted_by_votes_v20", {"level": level.level_index, "action": action, "vlm_vector": list(vlm_vec), "adopted_vector": list(dominant[0]), "votes": dominant[1], "movements": meaning.movements})
                    continue
                # Treat unverified VLM vectors as hypotheses only. Transition evidence is the
                # source of truth; otherwise a single confident hallucination can overwrite a
                # known noop/transform action (e.g. ACTION5 in ar25 or ACTION2 in ls20).
                reason = "movement_conflicts_with_noop_evidence" if meaning.noops > 0 and meaning.movements == 0 else "unverified_movement_vector"
                self.logger.log_event("vlm_action_update_rejected", {"level": level.level_index, "action": action, "reason": reason, "vlm_vector": list(vlm_vec) if vlm_vec else None, "attempts": meaning.attempts, "vlm_confidence": conf})
                # Do not invalidate or filter the plan from the same VLM call.
                meaning.confidence = max(meaning.confidence, min(conf, 0.25 if meaning.attempts == 0 else 0.35))
                if text and not meaning.meaning_nl:
                    meaning.meaning_nl = text
                continue

            if kind and not (meaning.movements > 0 and kind in {"interact_or_transform", "undo", "unknown_or_blocked", "resource_wasting_noop"} and meaning.transforms == 0):
                meaning.kind = kind
                if kind in {"interact_or_transform", "undo", "unknown_or_blocked", "resource_wasting_noop"}:
                    meaning.vector = None
            if text and conf >= meaning.confidence - 0.05:
                meaning.meaning_nl = text
            meaning.confidence = max(meaning.confidence, conf)
        win = result.win_condition_update
        if win:
            desc = self._sanitize_game_text(_short(win.get("description_nl") or win.get("description") or win.get("claim"), 800), scene)
            if desc:
                mem = self.memory.game.win_condition or WinConditionMemory()
                conf = _clamp01(win.get("confidence", 0.0))
                if conf > mem.confidence or not mem.description_nl:
                    mem.description_nl = desc
                mem.confidence = max(mem.confidence, conf)
                visual_roles = win.get("visual_roles") if isinstance(win.get("visual_roles"), dict) else {}
                for role, raw_desc in visual_roles.items():
                    if isinstance(raw_desc, dict):
                        mem.visual_roles[_short(role, 80)] = self._coerce_visual_descriptor(raw_desc)
                if result.mode == VLMMode.SUCCESS_REFLECT.value and level.level_index not in mem.confirmed_levels:
                    mem.confirmed_levels.append(level.level_index)
                evidence = self._sanitize_game_text(_short(win.get("evidence"), 500), scene)
                if evidence:
                    mem.evidence.append({"level": level.level_index, "evidence": evidence, "mode": result.mode})
                    mem.evidence = mem.evidence[-12:]
                self.memory.game.win_condition = mem
        for upd in result.object_effect_updates:
            effect = self._sanitize_game_text(_short(upd.get("effect_nl") or upd.get("effect"), 700), scene)
            if not effect:
                continue
            local_id = _short(upd.get("local_object_id") or upd.get("object_id"), 24).upper()
            raw_desc = upd.get("visual_descriptor") if isinstance(upd.get("visual_descriptor"), dict) else {}
            obj = scene.object_by_id(local_id)
            desc = self._visual_descriptor(obj, scene) if obj is not None and not raw_desc else self._coerce_visual_descriptor(raw_desc)
            entry = ObjectEffectMemory(desc, effect, _clamp01(upd.get("confidence", 0.0)), [{"level": level.level_index, "evidence": self._sanitize_game_text(_short(upd.get("evidence"), 400), scene)}])
            self._merge_object_effect(entry)
        for claim in result.mechanics_updates:
            claim = self._sanitize_game_text(claim, scene)
            if not claim:
                continue
            # Near-duplicate guard: the VLM restates the same mechanic with slightly
            # different wording every few calls (ls20 prompt carried ACTION1/2
            # semantics 4-5x), bloating every subsequent prompt.
            norm = re.sub(r"[^a-z0-9]+", "", claim.lower())[:80]
            existing = {re.sub(r"[^a-z0-9]+", "", c.lower())[:80] for c in self.memory.game.mechanics_nl}
            if norm not in existing:
                self.memory.game.mechanics_nl.append(claim)
                self.memory.game.mechanics_nl = self.memory.game.mechanics_nl[-12:]
        if result.resource_update:
            desc = self._sanitize_game_text(_short(result.resource_update.get("description_nl") or result.resource_update.get("claim"), 600), scene)
            if desc:
                self.memory.game.resource_model.description_nl = desc
        if result.plan_goal:
            level.plan_goal = _short(result.plan_goal, 500)
        if result.plan:
            old_active = level.active_step()
            old_plan = list(level.current_plan)
            old_cursor = level.plan_cursor
            raw_plan_steps, raw_reject_reason = self._filter_vlm_raw_plan_steps(result.plan, scene, legal)
            steps = self._plan_steps_from_vlm(raw_plan_steps, scene)
            steps = self._vet_vlm_plan_steps(steps, scene)
            accepted_vlm_plan_steps = bool(steps)
            preserve_existing_plan = bool(
                not steps
                and old_active is not None
                and level.fuel_nav_delegated_at != level.total_action_count
                and level.bottleneck_reason not in {
                    "plan_step_noop",
                    "plan_step_caused_retry_or_life_loss",
                    "life_loss_detected",
                    "critical_resource_counter",
                    "game_over",
                }
            )
            if preserve_existing_plan:
                level.current_plan = old_plan
                level.plan_cursor = old_cursor
                if raw_reject_reason == "initial_click_deferred_until_simple_actions_grounded":
                    level.bottleneck_reason = ""
                    level.action_recovery_contract = False
                    self.logger.log_event("vlm_plan_preserved_seed_v20", {"level": level.level_index, "reason": raw_reject_reason, "seed_steps": len(level.current_plan)})
                else:
                    level.bottleneck_reason = raw_reject_reason or level.bottleneck_reason or "vlm_zero_step_preserved_existing_plan"
                    level.action_recovery_contract = True
                    self.logger.log_event(
                        "vlm_plan_preserved_existing_v20",
                        {
                            "level": level.level_index,
                            "reason": raw_reject_reason,
                            "existing_remaining": len(level.current_plan[level.plan_cursor:]),
                        },
                    )
            else:
                level.current_plan = steps
                level.plan_cursor = 0
                if steps:
                    level.bottleneck_reason = ""
                    level.action_recovery_contract = False
                    level.fallback_guard_block_until = 0
                elif level.fuel_nav_delegated_at == level.total_action_count:
                    # The lone movement step was handed to the local navigator on
                    # purpose; this is a successful handoff, not a planning failure.
                    level.bottleneck_reason = ""
                    level.action_recovery_contract = False
                else:
                    level.bottleneck_reason = raw_reject_reason or level.bottleneck_reason or "vlm_unusable_plan"
                    level.action_recovery_contract = True
                    level.fallback_guard_block_until = max(level.fallback_guard_block_until, level.total_action_count + 1)
                    if not steps and self._transform_cycle_bottleneck_active(level):
                        self._maybe_escalate_mechanism_hypothesis(scene, "vlm_empty_plan_in_cycle")
        elif result.mode in {VLMMode.BOTTLENECK.value, VLMMode.EVALUATE_CHUNK.value}:
            old_active = level.active_step()
            hard_clear_reasons = {
                "plan_step_noop",
                "plan_step_caused_retry_or_life_loss",
                "life_loss_detected",
                "critical_resource_counter",
                "game_over",
            }
            if old_active is None or level.bottleneck_reason in hard_clear_reasons:
                level.current_plan = []
                level.plan_cursor = 0
            level.bottleneck_reason = level.bottleneck_reason or "vlm_empty_plan"
            level.action_recovery_contract = True
            level.fallback_guard_block_until = max(level.fallback_guard_block_until, level.total_action_count + 1)
        if deferred_actor_bindings:
            if accepted_vlm_plan_steps:
                for role_key, oid in deferred_actor_bindings:
                    level.local_bindings[role_key] = oid
                    level.controlled_object_id = oid
            else:
                for role_key, oid in deferred_actor_bindings:
                    self.logger.log_event("role_binding_rejected_v20", {"level": level.level_index, "role": role_key, "proposed": oid, "current": level.controlled_object_id, "reason": "actor_binding_deferred_until_executable_plan"})
        if result.bottleneck_analysis:
            level.notes_for_next_call = _short(result.bottleneck_analysis, 700)
        if result.notes_for_next_call:
            level.notes_for_next_call = _short(result.notes_for_next_call, 700)
        if result.mode == VLMMode.LEVEL_INIT.value:
            level.initial_vlm_done = True

    def _trusted_plan_sequence(self, step: PlanStep, action_name_raw: str = "") -> bool:
        if not isinstance(step.raw, dict):
            return False
        route_reason = _short(step.raw.get("route_reason"), 80)
        if (self._transform_cycle_bottleneck_active() or self._mechanism_hypothesis_locked()) and route_reason in {"geometry", "coupled_carrier_geometry"}:
            return False
        trusted_route = route_reason in TRUSTED_ROUTE_REASONS
        action = (action_name_raw or step.action).upper()
        if route_reason == "mechanism_probe":
            meaning = self.memory.game.action_meanings.get(action)
            if meaning is None or meaning.life_losses > 0 or meaning.retries > 0 or not step.expected_predicates:
                return False
            measured_movement = (
                meaning.kind == "movement"
                and meaning.vector is not None
                and meaning.movements >= 1
                and meaning.noop_ratio < 0.35
            )
            observed_mechanism = meaning.transforms + meaning.interactions > 0
            return bool(measured_movement or observed_mechanism)
        if route_reason == "resource_recovery":
            return bool(step.expected_predicates and self.memory.level.resource_crisis)
        if "sequence_index" not in step.raw:
            return bool(trusted_route and step.step_type == "probe_action" and (action_name_raw or step.action))
        if trusted_route:
            return True
        meaning = self.memory.game.action_meanings.get(action)
        if not meaning:
            return False
        if meaning.kind == "movement" and meaning.vector is not None and meaning.movements >= 1 and meaning.noop_ratio < 0.35:
            return True
        return bool(meaning.attempts > 0 and meaning.movements + meaning.transforms + meaning.interactions > 0 and meaning.noop_ratio < 0.35 and not self.memory.level.resource_crisis)

    def _plan_repetition_key(self, step: PlanStep, scene: SceneSnapshot) -> str:
        action = step.action.upper()
        if action == "ACTION6":
            coords = self._iter_raw_click_coordinates(step.raw if isinstance(step.raw, dict) else {})
            if coords:
                x, y = coords[0]
                return f"ACTION6:{x},{y}"
            target = self._resolve_step_target(step)
            if target:
                return f"ACTION6OBJ:{target}"
            return "ACTION6"
        if action:
            return action
        target = self._resolve_step_target(step)
        return f"{step.step_type}:{target}" if target else step.step_type

    def _click_step_has_executable_candidate(self, step: PlanStep, scene: SceneSnapshot) -> bool:
        raw = step.raw if isinstance(step.raw, dict) else {}
        coords = self._iter_raw_click_coordinates(raw)
        if coords:
            if any(
                0 <= x < scene.width
                and 0 <= y < scene.height
                and not self._action_blocked(
                    scene,
                    f"ACTION6:{x},{y}",
                    ignore_recent=self._trusted_plan_sequence(step, "ACTION6"),
                    ignore_click_exhaustion=self._click_key_has_positive_evidence(f"ACTION6:{x},{y}"),
                )
                for x, y in coords
            ):
                return True
            # All explicit coordinates are exhausted/banned. Only remap to a
            # coordinate backed by positive click evidence; broad local remaps make
            # VLM failures look executable and caused click-only games to sweep the grid.
            blocked = set(self.memory.level.known_noops_by_state.get(scene.state_hash, set())) | set(self.memory.level.tried_actions_by_state.get(scene.state_hash, set()))
            if self._focused_action6_target(scene, blocked) is not None:
                if isinstance(step.raw, dict):
                    step.raw = {**step.raw, "click_coords_exhausted_remap": True}
                return True
            return False
        target_id = self._resolve_step_target(step)
        target = scene.object_by_id(target_id) if target_id else None
        if target is None:
            return True
        level = self.memory.level
        if level.click_noops_by_object.get(target.track_id, 0) >= self.config.max_object_click_noops and level.click_success_by_object.get(target.track_id, 0) == 0:
            return False
        return self._next_click_point_for_object(scene, target, hint=_short(raw.get("click_hint") or raw.get("click_point") or "", 40).lower()) is not None

    def _geometry_plan_chases_selector_pair_without_waypoint(self, step: PlanStep, scene: SceneSnapshot, route_reason: str) -> bool:
        if route_reason != "geometry":
            return False
        raw = step.raw if isinstance(step.raw, dict) else {}
        waypoint_keys = ("route_waypoint", "route_waypoints", "waypoint", "waypoint_object_id", "via", "via_object_id")
        if any(_short(raw.get(key), 80) for key in waypoint_keys):
            return False

        texts = [step.purpose, step.stop_condition]
        for key in ("purpose", "why", "route_to", "target", "target_id", "target_object_id", "object_id", "expected_change", "stop_condition", "evidence"):
            value = raw.get(key)
            if isinstance(value, str):
                texts.append(value)
        for key in ("actions", "action_sequence", "probe_actions"):
            sequence = raw.get(key)
            if not isinstance(sequence, list):
                continue
            for item in sequence:
                if not isinstance(item, dict):
                    continue
                for subkey in ("purpose", "why", "target", "target_id", "target_object_id", "object_id", "expected_change", "stop_condition", "evidence"):
                    value = item.get(subkey)
                    if isinstance(value, str):
                        texts.append(value)

        mentioned_ids = {oid for text in texts for oid in re.findall(r"\bO\d+\b", _short(text, 1000).upper())}
        for oid in mentioned_ids:
            obj = scene.object_by_id(oid)
            if obj is not None and self._is_possible_selector_pair_object(scene, obj):
                return True
        return False

    def _plan_step_rejected_reason(self, step: PlanStep, scene: SceneSnapshot) -> str:
        action = step.action.upper()
        level = self.memory.level
        route_reason = _short(step.raw.get("route_reason"), 80) if isinstance(step.raw, dict) else ""
        if (self._transform_cycle_bottleneck_active(level) or self._mechanism_hypothesis_locked()) and route_reason in {"geometry", "coupled_carrier_geometry"}:
            return "transform_state_cycle_rejects_geometry_plan"
        if self._geometry_plan_chases_selector_pair_without_waypoint(step, scene, route_reason):
            return "geometry_to_selector_pair_mechanism_without_waypoint"
        if step.step_type in {"move_to", "interact"} and self._in_transform_mode(scene):
            return "transform_mode_rejects_navigation_plan"
        if action == "RESET":
            return "reset_not_allowed_in_vlm_plan"
        if action == "ACTION6":
            if not self._click_step_has_executable_candidate(step, scene):
                return "click_target_or_coordinate_exhausted"
            return ""
        if action:
            trusted = self._trusted_plan_sequence(step, action)
            seq_index = -1
            if isinstance(step.raw, dict):
                try:
                    seq_index = int(step.raw.get("sequence_index", -1))
                except Exception:
                    seq_index = -1
            recovery_override = self._vlm_recovery_probe_override(step, scene, action)
            if self._action_blocked(
                scene,
                action,
                ignore_recent=trusted or recovery_override,
                ignore_loop_quarantine=trusted or recovery_override,
                ignore_noop_evidence=recovery_override,
                ignore_contract=recovery_override,
            ):
                if not recovery_override and not (trusted and seq_index > 0):
                    # V2.6: only persist a permanent per-state contract when the block
                    # comes from durable evidence (noop/contract/suffix/quarantine). A
                    # block caused solely by the 2-step repeat cooldown is transient:
                    # in sb26 the transform controller pressed ACTION5 constantly, so
                    # 20 VLM proposals of ACTION5 (the win-trigger action) landed in
                    # the cooldown window and each wrote a permanent state ban.
                    # Recheck ignores transient blocks only (repeat cooldown + loop
                    # quarantine). Durable noop/contract/suffix evidence still writes
                    # the permanent per-state ban. Previously ignore_loop_quarantine
                    # was omitted here, so a temporary quarantine alone could stamp
                    # vlm_vet forever.
                    if self._action_blocked(scene, action, ignore_recent=True, ignore_loop_quarantine=True):
                        self._contract_forbid_action(scene, action, "vlm_vet:blocked_by_transition_evidence")
                        return "blocked_by_transition_evidence"
                    return "action_repeat_cooldown"
            if not trusted and self._resource_crisis_active(scene) and self._resource_risky_nonreset_action(scene, action):
                return "resource_risky_without_positive_evidence"
            meaning = self.memory.game.action_meanings.get(action)
            if self._in_transform_mode(scene) and step.step_type == "move_to" and meaning is not None and meaning.transforms > meaning.movements:
                return "transform_action_used_as_navigation"
        target_id = self._resolve_step_target(step)
        target = scene.object_by_id(target_id) if target_id else None
        if target is not None and step.step_type in {"move_to", "interact"} and not self._is_navigable_target(scene, target):
            return "non_navigable_target"
        return ""

    def _vet_vlm_plan_steps(self, steps: list[PlanStep], scene: SceneSnapshot) -> list[PlanStep]:
        if not steps:
            return []
        level = self.memory.level
        # V2.2 fuel-game delegation: in a step-tax game a lone "move once toward the
        # target" plan burns a whole VLM round per counter tick and ping-pongs
        # (ls20: 33/42 calls returned a single movement step, UP/DOWN alternation
        # drained the fuel; the VLM also mixes up axes). Once movement vectors are
        # confirmed and a navigable target is bound, the local navigator computes
        # routes with correct geometry - drop the lone step and let it drive. Multi-
        # step routes and non-movement steps are untouched.
        if level.counter_is_step_tax and len(steps) == 1:
            lone = steps[0]
            lone_action = lone.action.upper()
            lone_meaning = self.memory.game.action_meanings.get(lone_action)
            learned_vectors = sum(1 for m in self.memory.game.action_meanings.values() if m.kind == "movement" and m.vector is not None and m.confidence >= 0.4)
            nav_target = self._navigation_target_id(scene) if learned_vectors >= 2 else ""
            if (
                lone.step_type == "probe_action"
                and lone_meaning is not None
                and lone_meaning.kind == "movement"
                and lone_meaning.vector is not None
                and nav_target
                and not self._trusted_plan_sequence(lone, lone_action)
            ):
                self.logger.log_event("fuel_single_move_delegated_v20", {"level": level.level_index, "action": lone_action, "nav_target": nav_target})
                note = "Single-step movement plans are delegated to the local navigator in this step-counter game. Return either a FULL multi-step route (action_sequence) or just updated role_bindings/win_condition; a lone 1-step move wastes the call."
                if note[:50] not in (level.notes_for_next_call or ""):
                    level.notes_for_next_call = _short(note + " " + (level.notes_for_next_call or ""), 700)
                level.fuel_nav_delegated_at = level.total_action_count
                return []
        kept: list[PlanStep] = []
        dropped: list[dict[str, Any]] = []
        consecutive_key = ""
        consecutive_count = 0
        click_targets: Counter[str] = Counter()
        resource_active = self._resource_crisis_active(scene) or scene.counter_value is not None or scene.life_count is not None
        # V2.3: counter at/below critical_resource_ratio means the very next noop/
        # retry can end the attempt. TRUSTED_ROUTE_REASONS (geometry/selector_*)
        # steps are exempt from resource_plan_max_steps below by design (routes with
        # confirmed vectors are normally safe to commit to), but that exemption must
        # not extend to critical fuel - ls20 call#18 kept a full 10-action trusted
        # "geometry" route at counter=0/42 with zero chance to react mid-route.
        counter_critical = scene.counter_ratio is not None and scene.counter_ratio <= self.config.critical_resource_ratio
        for step in steps:
            reason = self._plan_step_rejected_reason(step, scene)
            rep_key = self._plan_repetition_key(step, scene)
            if rep_key == consecutive_key:
                consecutive_count += 1
            else:
                consecutive_key = rep_key
                consecutive_count = 1
            trusted_step = self._trusted_plan_sequence(step, step.action)
            if not reason and not trusted_step:
                cap = 2 if (resource_active or self._in_transform_mode(scene)) else self.config.max_vlm_consecutive_same_action
                if consecutive_count > cap:
                    reason = "untrusted_repeated_action_sequence"
            if not reason and step.action.upper() == "ACTION6":
                target = self._resolve_step_target(step)
                if target:
                    click_targets[target] += 1
                    if level.click_success_by_object.get(target, 0) == 0 and click_targets[target] > self.config.max_vlm_same_click_target:
                        reason = "too_many_clicks_for_unproven_target"
            if reason:
                dropped.append({"reason": reason, "step": step.as_prompt()})
                continue
            if self._vlm_recovery_probe_override(step, scene, step.action) and isinstance(step.raw, dict):
                step.raw = {**step.raw, "vlm_recovery_override": True}
            kept.append(step)
            if counter_critical and len(kept) >= 1:
                remaining = len(steps) - (len(kept) + len(dropped))
                if remaining > 0:
                    dropped.append({"reason": "critical_resource_plan_truncated", "remaining": remaining})
                break
            if resource_active and not trusted_step and len(kept) >= self.config.resource_plan_max_steps:
                remaining = len(steps) - (len(kept) + len(dropped))
                if remaining > 0:
                    dropped.append({"reason": "resource_plan_truncated", "remaining": remaining})
                break
        if dropped:
            self.logger.log_event("vlm_plan_steps_rejected_v20", {"level": level.level_index, "reason": "plan_vetting", "dropped": dropped[:8], "kept": len(kept), "input": len(steps)})
            self._record_vlm_plan_rejections("plan_vetting", dropped, len(kept), len(steps))
            if not kept and any(d.get("reason") == "geometry_to_selector_pair_mechanism_without_waypoint" for d in dropped):
                self._maybe_escalate_mechanism_hypothesis(scene, "geometry_to_selector_pair_rejected")
            elif not kept and any(d.get("reason") == "transform_state_cycle_rejects_geometry_plan" for d in dropped):
                self._maybe_escalate_mechanism_hypothesis(scene, "geometry_plan_rejected_in_cycle")
            elif not kept and self._transform_cycle_bottleneck_active(level) and level.vlm_issue_repeat_count >= 2:
                self._maybe_escalate_mechanism_hypothesis(scene, "repeated_vlm_rejection")
        return kept

    def _normalize_vlm_action_kind(self, raw_kind: str, meaning: ActionMeaning) -> str:
        if not raw_kind:
            return ""
        parts = [p.strip() for p in re.split(r"[|,/]+", raw_kind) if p.strip()]
        known = {"movement", "interact_or_transform", "click_or_select", "undo", "resource", "unknown_or_blocked", "resource_wasting_noop"}
        parts = [p for p in parts if p in known]
        if not parts:
            return raw_kind if raw_kind in known else ""
        if meaning.movements > 0 and "movement" in parts:
            return "movement"
        if meaning.transforms > 0 and "interact_or_transform" in parts:
            return "interact_or_transform"
        if meaning.interactions > 0:
            return "click_or_select" if "click_or_select" in parts else "interact_or_transform" if "interact_or_transform" in parts else parts[0]
        return parts[0]

    def _coerce_visual_descriptor(self, raw: dict[str, Any]) -> VisualDescriptor:
        colors_raw = raw.get("colors")
        colors = []
        if isinstance(colors_raw, list):
            for c in colors_raw:
                try:
                    colors.append(int(c))
                except Exception:
                    pass
        try:
            frame_color = int(raw["frame_color"]) if raw.get("frame_color") is not None else None
        except Exception:
            frame_color = None
        tags = raw.get("relation_tags")
        return VisualDescriptor(_short(raw.get("shape_label") or raw.get("shape"), 80), colors, {str(k): int(v) for k, v in raw.get("color_areas", {}).items()} if isinstance(raw.get("color_areas"), dict) else {}, _short(raw.get("pattern"), 240), _short(raw.get("inner_pattern"), 240), frame_color, _short(raw.get("size_bucket") or raw.get("size"), 40), [_short(t, 80) for t in tags] if isinstance(tags, list) else [], bool(raw.get("near_edge")) if raw.get("near_edge") is not None else None, _short(raw.get("type_key"), 80))

    def _merge_object_effect(self, entry: ObjectEffectMemory) -> None:
        key = json.dumps(_json_safe(entry.visual_descriptor), sort_keys=True, default=str)
        for existing in self.memory.game.object_effects:
            existing_key = json.dumps(_json_safe(existing.visual_descriptor), sort_keys=True, default=str)
            if existing_key == key or (existing.visual_descriptor.inner_pattern and existing.visual_descriptor.inner_pattern == entry.visual_descriptor.inner_pattern and existing.visual_descriptor.shape_label == entry.visual_descriptor.shape_label):
                if entry.confidence >= existing.confidence:
                    existing.effect_nl = entry.effect_nl
                existing.confidence = max(existing.confidence, entry.confidence)
                existing.evidence = (existing.evidence + entry.evidence)[-12:]
                return
        self.memory.game.object_effects.append(entry)
        self.memory.game.object_effects = self.memory.game.object_effects[-20:]

    def _plan_steps_from_vlm(self, raw_steps: list[dict[str, Any]], scene: SceneSnapshot) -> list[PlanStep]:
        valid_ids = {o.track_id for o in scene.objects}
        steps: list[PlanStep] = []
        for raw in raw_steps[:self.config.max_plan_steps]:
            if not isinstance(raw, dict):
                continue
            sequence = raw.get("actions") or raw.get("action_sequence") or raw.get("probe_actions")
            if isinstance(sequence, list):
                # 防御退化复读：截断后的超长 action_sequence（如 ACTION2 x60）只取前段，
                # 否则 trusted route_reason 会绕过重复上限,把整条退化序列执行完。
                for idx, item in enumerate(sequence[:12]):
                    if len(steps) >= self.config.max_plan_steps:
                        break
                    item_raw = dict(item) if isinstance(item, dict) else {}
                    name = (item_raw.get("action") or item_raw.get("name")) if item_raw else item
                    action_name_raw = _short(name, 40).upper()
                    if not (action_name_raw.startswith("ACTION") and action_name_raw[6:].isdigit()):
                        continue
                    seq_raw = {**raw, **item_raw, "sequence_index": idx}
                    step_type = _short(item_raw.get("type") or item_raw.get("step_type") or raw.get("type"), 40).lower()
                    if action_name_raw == "ACTION6":
                        step_type = "click"
                    elif step_type not in {"probe_action", "interact", "move_to"}:
                        step_type = "probe_action"
                    target_role = _short(item_raw.get("target_role") or item_raw.get("role") or raw.get("target_role") or raw.get("role"), 80)
                    target = _short(item_raw.get("target_object_id") or item_raw.get("object_id") or raw.get("target_object_id") or raw.get("object_id"), 24).upper()
                    if target and target not in valid_ids:
                        target = ""
                    if not target:
                        target = self._infer_raw_plan_target_id(seq_raw, scene)
                    purpose = _short(item_raw.get("purpose") or item_raw.get("why") or raw.get("purpose") or raw.get("why") or "execute VLM action sequence", 320)
                    stop = _short(item_raw.get("stop_condition") or item_raw.get("expected_change") or raw.get("stop_condition") or raw.get("expected_change") or "observe sequence progress", 320)
                    expected = item_raw.get("expected_predicates") if isinstance(item_raw.get("expected_predicates"), list) else raw.get("expected_predicates") if isinstance(raw.get("expected_predicates"), list) else []
                    target_obj = scene.object_by_id(target)
                    if target_obj is not None and step_type in {"move_to", "interact"} and not self._is_navigable_target(scene, target_obj):
                        self.logger.log_event("vlm_plan_target_rejected", {"level": self.memory.level.level_index, "target": target, "step_type": step_type, "reason": "not_navigation_candidate_sequence"})
                        replacement = self._navigation_target_id(scene, exclude={target})
                        if replacement and step_type == "move_to":
                            seq_raw = {**seq_raw, "target_rejected_reason": "not_navigation_candidate", "original_target_object_id": target, "target_repaired_to": replacement}
                            target = replacement
                            target_role = ""
                        else:
                            step_type = "probe_action"
                            target = ""
                            target_role = ""
                    steps.append(PlanStep(step_type, action_name_raw, target_role, target, purpose, stop, expected, max_attempts=1, raw=seq_raw))
                continue
            step_type = _short(raw.get("type") or raw.get("step_type") or raw.get("intent"), 40).lower()
            if step_type in {"navigate_to_object", "move"}:
                step_type = "move_to"
            if step_type in {"click_object", "test_object"}:
                step_type = "click" if _short(raw.get("action"), 40).upper() == "ACTION6" else "probe_object"
            if step_type not in {"probe_action", "probe_object", "move_to", "interact", "click", "wait"}:
                step_type = "probe_action" if raw.get("action") else "probe_object" if raw.get("target_object_id") else ""
            if not step_type:
                continue
            target_role = _short(raw.get("target_role") or raw.get("role"), 80)
            action = _short(raw.get("action") or raw.get("name"), 40).upper()
            if action in {"RESET", "NAVIGATE", "CLICK", "MOVE", "PATHFIND", "GO"}:
                action = ""
            target = _short(raw.get("target_object_id") or raw.get("object_id"), 24).upper()
            if target and target not in valid_ids:
                target = ""
            if not target:
                target = self._infer_raw_plan_target_id(raw, scene)
            target_obj = scene.object_by_id(target)
            if target_obj is not None and step_type in {"move_to", "interact"} and not self._is_navigable_target(scene, target_obj):
                rejected_step_type = step_type
                self.logger.log_event("vlm_plan_target_rejected", {"level": self.memory.level.level_index, "target": target, "step_type": step_type, "reason": "not_navigation_candidate"})
                replacement = self._navigation_target_id(scene, exclude={target})
                if replacement and step_type == "move_to" and not action:
                    raw = {**raw, "target_rejected_reason": "not_navigation_candidate", "original_target_object_id": target, "original_step_type": rejected_step_type, "target_repaired_to": replacement}
                    self.logger.log_event("vlm_plan_target_repaired_v20", {"level": self.memory.level.level_index, "target": target, "replacement": replacement, "step_type": rejected_step_type})
                    target = replacement
                    target_role = ""
                elif action and action != "ACTION6" and action.startswith("ACTION") and action[6:].isdigit():
                    step_type = "probe_action"
                    raw = {**raw, "target_rejected_reason": "not_navigation_candidate", "original_step_type": rejected_step_type}
                    target = ""
                    target_role = ""
                else:
                    step_type = "probe_action"
                    raw = {**raw, "target_rejected_reason": "not_navigation_candidate", "original_target_object_id": target, "original_step_type": rejected_step_type, "action_level_repair": True}
                    target = ""
                    target_role = ""
                    self.logger.log_event("vlm_plan_target_repaired_v20", {"level": self.memory.level.level_index, "target": raw.get("original_target_object_id", ""), "replacement": "action_level_probe", "step_type": rejected_step_type})
            purpose = _short(raw.get("purpose") or raw.get("why") or "execute VLM plan step", 320)
            stop = _short(raw.get("stop_condition") or raw.get("expected_change") or "observe plan progress", 320)
            expected = raw.get("expected_predicates") if isinstance(raw.get("expected_predicates"), list) else []
            steps.append(PlanStep(step_type, action, target_role, target, purpose, stop, expected, max_attempts=1, raw=raw))
        return steps[:self.config.max_plan_steps]

    def _seed_initial_plan(self, scene: SceneSnapshot, legal: Sequence[Any]) -> None:
        steps: list[PlanStep] = []
        legal_names = {action_name(a).upper() for a in legal}
        simple_names = simple_action_names(legal)
        special_objects = [o for o in sorted(scene.objects, key=lambda x: -self._object_interest_score(scene, x)) if not o.near_edge][:4]
        for name in simple_names:
            if name == "ACTION7" and self.config.avoid_action7 and not self._action7_probe_allowed():
                continue
            m = self.memory.game.action_meanings.get(name)
            if m is None or m.confidence < 0.4:
                steps.append(PlanStep("probe_action", action=name, purpose=f"learn what {name} does", stop_condition="stop after visible movement/change/noop", max_attempts=1))
            if len([s for s in steps if s.step_type == "probe_action"]) >= 3:
                break
        if "ACTION6" in legal_names:
            click_seed_count = 3 if not simple_names else 1
            for obj in special_objects[:click_seed_count]:
                steps.append(PlanStep("click", action="ACTION6", target_object_id=obj.track_id, purpose=f"test salient object {obj.track_id} with structured click", stop_condition="stop after structural change or no-op", max_attempts=1, raw={"click_hint": "center"}))
        else:
            for obj in special_objects[:2]:
                steps.append(PlanStep("probe_object", target_object_id=obj.track_id, purpose=f"navigate to or test salient object {obj.track_id}", stop_condition="stop after structural change or no-op", max_attempts=1))
        self.memory.level.current_plan = steps[:self.config.max_plan_steps]
        self.memory.level.plan_goal = "initial exploration: test salient/special patterns, ground actions, avoid known undo/no-op actions"

    def _execute_next_plan_action(self, scene: SceneSnapshot, legal: Sequence[Any]) -> tuple[Any, dict[str, Any], str] | None:
        level = self.memory.level
        if self._critical_counter_blocks_plan_executor(scene) and level.active_step() is not None:
            # V2.4: even a 1-step trusted geometry route burns the last counter tick
            # (ls20 step 132: critical_resource_plan_truncated kept 1 step, plan_executor
            # ran ACTION1, counter 1->0 game over). Block executor; fall through to
            # nonreset_guard / safe single probes instead.
            level.current_plan = []
            level.plan_cursor = 0
            level.bottleneck_reason = level.bottleneck_reason or "critical_resource_counter"
            level.action_recovery_contract = True
            self.logger.log_event("plan_executor_blocked_critical_counter_v20", {"level": level.level_index, "counter": scene.counter_value, "ratio": scene.counter_ratio})
            return None
        active = level.active_step()
        active_raw = active.raw if active is not None and isinstance(active.raw, dict) else {}
        trusted_sequence = bool(active is not None and active.action and "sequence_index" in active_raw and self._trusted_plan_sequence(active, active.action))
        if not trusted_sequence and (
            self._source_storm_active("plan_executor", outcomes={"noop", "state_change", "transform", "resource_delta"})
            or self._source_low_yield_active("plan_executor", outcomes={"noop", "state_change", "transform", "resource_delta"})
        ):
            dropped = {"reason": "plan_executor_low_yield", "step": active.as_prompt() if active is not None else {}}
            level.current_plan = []
            level.plan_cursor = 0
            level.bottleneck_reason = "plan_executor_low_yield"
            level.action_recovery_contract = True
            level.fallback_guard_block_until = max(level.fallback_guard_block_until, level.total_action_count + 1)
            self._record_vlm_plan_rejections("source_fuse", [dropped], 0, 1 if active is not None else 0)
            self.logger.log_event("source_fuse_open_v20", {"level": level.level_index, "source": "plan_executor", "state": scene.state_hash[:12], "reason": "low_yield_plan_executor"})
            return None
        while True:
            step = level.active_step()
            if step is None:
                return None
            proposal = self._proposal_for_step(step, scene, legal)
            if proposal is None:
                if step.status in {"done", "skipped"}:
                    continue
                # 当前 step 无法生成提案（target 缺失/类型不支持），跳到下一步而非
                # 立即 fall through 到局部控制器，让 plan 的其他步骤仍有机会执行。
                step.status = "skipped"
                level.bottleneck_reason = f"cannot_execute_step:{step.step_type}:{step.target_role or step.target_object_id or step.action}"
                level.action_recovery_contract = True
                level.plan_cursor += 1
                continue
            action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if action is None:
                step.attempts += 1
                if step.attempts >= step.max_attempts:
                    step.status = "failed"
                    level.bottleneck_reason = f"invalid_action_for_step:{step.step_type}"
                    level.action_recovery_contract = True
                    level.plan_cursor += 1
                    continue
                return None
            return action, proposal, "plan_executor"

    def _proposal_for_step(self, step: PlanStep, scene: SceneSnapshot, legal: Sequence[Any]) -> dict[str, Any] | None:
        legal_by = {action_name(a).upper(): a for a in legal}
        target_id = self._resolve_step_target(step)
        target = scene.object_by_id(target_id)
        if target is not None and step.step_type in {"move_to", "interact", "probe_object"} and not self._is_navigable_target(scene, target):
            replacement = self._navigation_target_id(scene, exclude={target.track_id})
            if replacement:
                target_id = replacement
                target = scene.object_by_id(target_id)
            elif step.step_type == "move_to":
                self.logger.log_event("plan_target_rejected", {"level": self.memory.level.level_index, "target": target.track_id, "reason": "outside_actor_play_region", "step_type": step.step_type})
                if step.action:
                    return {"name": step.action, "target_object_id": target.track_id, "purpose": step.purpose or f"directional fallback for rejected target {target.track_id}", "expected_change": step.stop_condition or "test VLM-specified direction after target rejection", "nav_fallback": "rejected_target_direction"}
                return None
        if step.step_type == "wait":
            step.status = "skipped"
            self.memory.level.plan_cursor += 1
            return None
        if step.step_type == "probe_action":
            name = step.action if step.action in legal_by else self._next_unknown_simple_action(legal, scene)
            if not name:
                return None
            trusted_sequence = self._trusted_plan_sequence(step, name)
            recovery_override = self._vlm_recovery_probe_override(step, scene, name)
            ignore_recent = trusted_sequence or recovery_override
            ignore_loop_quarantine = trusted_sequence or recovery_override
            if self._action_blocked(
                scene,
                name,
                ignore_recent=ignore_recent,
                ignore_loop_quarantine=ignore_loop_quarantine,
                ignore_noop_evidence=recovery_override,
                ignore_contract=recovery_override or trusted_sequence,
            ):
                return None
            return {"name": name, "purpose": step.purpose or f"probe {name}", "expected_change": step.stop_condition or "learn action effect", "expected_predicates": step.expected_predicates, "ignore_recent": ignore_recent, "ignore_loop_quarantine": ignore_loop_quarantine, "ignore_noop_evidence": recovery_override, "ignore_contract": recovery_override or trusted_sequence, "trusted_plan_sequence": trusted_sequence, "vlm_recovery_override": recovery_override}
        if step.step_type in {"click", "probe_object"}:
            if "ACTION6" in legal_by:
                for explicit_x, explicit_y in self._iter_raw_click_coordinates(step.raw):
                    if 0 <= explicit_x < scene.width and 0 <= explicit_y < scene.height:
                        key = f"ACTION6:{explicit_x},{explicit_y}"
                        positive_click_evidence = self._click_key_has_positive_evidence(key)
                        explicit_click_sequence = bool(self._iter_raw_click_coordinates(step.raw))
                        ignore_contract = positive_click_evidence or explicit_click_sequence
                        if self._action_blocked(
                            scene,
                            key,
                            ignore_recent=bool(step.action),
                            ignore_click_exhaustion=positive_click_evidence,
                            ignore_contract=ignore_contract,
                        ):
                            continue
                        return {
                            "name": "ACTION6",
                            "x": explicit_x,
                            "y": explicit_y,
                            "target_object_id": target_id,
                            "purpose": step.purpose or "click/test explicit VLM coordinate",
                            "expected_change": step.stop_condition or "observe coordinate response",
                            "expected_predicates": step.expected_predicates,
                            "ignore_recent": bool(step.action),
                            "ignore_click_exhaustion": positive_click_evidence,
                            "ignore_contract": ignore_contract,
                        }
                    self.logger.log_event(
                        "vlm_click_coordinate_rejected",
                        {
                            "level": self.memory.level.level_index,
                            "x": explicit_x,
                            "y": explicit_y,
                            "width": scene.width,
                            "height": scene.height,
                            "reason": "outside_scene",
                        },
                    )
            if target is None:
                target = self._choose_special_object(scene)
                if target is None:
                    return None
                target_id = target.track_id
            if "ACTION6" in legal_by:
                hint = _short(step.raw.get("click_hint") or step.raw.get("click_point") or "", 40).lower()
                point = self._next_click_point_for_object(scene, target, hint=hint)
                if point is None and step.raw.get("click_coords_exhausted_remap"):
                    blocked = set(self.memory.level.known_noops_by_state.get(scene.state_hash, set())) | set(self.memory.level.tried_actions_by_state.get(scene.state_hash, set()))
                    picked = self._focused_action6_target(scene, blocked)
                    if picked is not None:
                        point = (picked[0], picked[1])
                if point is None:
                    return None
                x, y = point
                key = f"ACTION6:{x},{y}"
                if step.raw.get("click_coords_exhausted_remap"):
                    self.logger.log_event("vlm_click_remapped_v20", {"level": self.memory.level.level_index, "x": x, "y": y, "target": target_id, "reason": "explicit_coordinates_exhausted"})
                return {"name": "ACTION6", "x": x, "y": y, "target_object_id": target_id, "purpose": step.purpose or f"click/test {target_id}", "expected_change": step.stop_condition or "observe object response", "expected_predicates": step.expected_predicates, "ignore_contract": self._click_key_has_positive_evidence(key)}
            if self._target_reached(target_id, scene):
                interact = self._best_interaction_action(scene, legal, target_id)
                if interact:
                    return {"name": interact, "target_object_id": target_id, "purpose": step.purpose or f"interact with {target_id}", "expected_change": step.stop_condition or "observe object interaction"}
            return self._navigation_proposal_to_target(target_id, scene, legal, step)
        if step.step_type == "move_to":
            if target is None:
                return None
            if self._target_reached(target_id, scene):
                step.status = "done"
                self.memory.level.plan_cursor += 1
                return None
            return self._navigation_proposal_to_target(target_id, scene, legal, step)
        if step.step_type == "interact":
            if target is not None and not self._target_reached(target_id, scene):
                return self._navigation_proposal_to_target(target_id, scene, legal, step)
            name = step.action if step.action in legal_by and not self._action_blocked(scene, step.action, ignore_contract=self._trusted_plan_sequence(step, step.action) or self._vlm_recovery_probe_override(step, scene, step.action)) else self._best_interaction_action(scene, legal, target_id)
            if name:
                return {"name": name, "target_object_id": target_id, "purpose": step.purpose or f"interact with {target_id}", "expected_change": step.stop_condition or "interaction/change"}
        return None

    def _resolve_step_target(self, step: PlanStep) -> str:
        if step.target_object_id:
            return step.target_object_id
        return self.memory.level.local_bindings.get(step.target_role, "") if step.target_role else ""

    def _iter_raw_click_coordinates(self, raw: dict[str, Any]) -> list[tuple[int, int]]:
        candidates: list[Any] = []
        for key in ("coords", "coordinates", "coordinate_sequence", "points", "click_points", "xy_sequence"):
            value = raw.get(key)
            if isinstance(value, list):
                candidates.extend(value)
        if raw.get("x") is not None or raw.get("y") is not None:
            candidates.insert(0, {"x": raw.get("x"), "y": raw.get("y")})
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for item in candidates:
            x = y = None
            if isinstance(item, dict):
                x = self._coerce_int(item.get("x"))
                y = self._coerce_int(item.get("y"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                x = self._coerce_int(item[0])
                y = self._coerce_int(item[1])
            if x is None or y is None:
                continue
            point = (x, y)
            if point not in seen:
                seen.add(point)
                out.append(point)
        return out

    @staticmethod
    def _click_region_key(x: int, y: int) -> str:
        return f"ACTION6REG:{int(x) // 8},{int(y) // 8}"

    @classmethod
    def _click_region_key_from_action_key(cls, key: str) -> str:
        m = re.fullmatch(r"ACTION6:(-?\d+),(-?\d+)", key.upper())
        if not m:
            return ""
        return cls._click_region_key(int(m.group(1)), int(m.group(2)))

    def _click_key_has_positive_evidence(self, key: str) -> bool:
        key = key.upper()
        level = self.memory.level
        if level.click_success_coords.get(key, 0) > 0:
            return True
        region_key = self._click_region_key_from_action_key(key)
        return bool(region_key and level.click_success_regions.get(region_key, 0) > 0)

    def _contract_forbid_action(self, scene: SceneSnapshot, action_key: str, reason: str) -> None:
        key = _short(action_key, 80).upper()
        if not key or key == "RESET":
            return
        base = key.split(":", 1)[0]
        rules = self.memory.level.contract_forbidden_by_state.setdefault(scene.state_hash, {})
        rules[key] = _short(reason, 160)
        if base != "ACTION6":
            rules[base] = _short(reason, 160)
        self.logger.log_event(
            "contract_action_forbidden_v20",
            {
                "level": self.memory.level.level_index,
                "state": scene.state_hash[:12],
                "action": key,
                "base": base,
                "reason": _short(reason, 160),
            },
        )

    def _contract_blocks_action(self, scene: SceneSnapshot, action_key: str) -> bool:
        key = _short(action_key, 80).upper()
        if not key:
            return False
        base = key.split(":", 1)[0]
        rules = self.memory.level.contract_forbidden_by_state.get(scene.state_hash, {})
        return key in rules or base in rules

    def _action_blocked(self, scene: SceneSnapshot, name_or_key: str, *, ignore_recent: bool = False, ignore_loop_quarantine: bool = False, ignore_noop_evidence: bool = False, ignore_click_exhaustion: bool = False, ignore_contract: bool = False, ignore_defer: bool = False) -> bool:
        level = self.memory.level
        now = level.total_action_count
        key = name_or_key.upper()
        base_name = key.split(":", 1)[0]
        if base_name == "RESET":
            return True
        # V2.1 hard, state-independent ban for churn-fused click coordinates. Must NOT be
        # bypassable by ignore_contract / ignore_click_exhaustion (which VLM explicit
        # coordinate plans set), otherwise ft09-style click loops immediately re-open.
        if base_name == "ACTION6" and ":" in key and key in level.global_forbidden_click_coords:
            return True
        if not ignore_contract and self._contract_blocks_action(scene, key):
            return True
        if base_name == "ACTION6" and ":" in key:
            coord_count = level.click_coord_counts.get(key, 0)
            region_key = self._click_region_key_from_action_key(key)
            region_count = level.click_region_counts.get(region_key, 0) if region_key else 0
            if not ignore_click_exhaustion:
                if coord_count >= self.config.max_clicks_per_coord_total:
                    return True
                if region_key and region_count >= self.config.max_clicks_per_region_total:
                    return True
            # Hard ceiling that positive-evidence / explicit-VLM coords cannot bypass.
            # Games like ft09 recolor neighbours so every click yields a fresh state
            # hash; revisit-based churn checks then never fire, letting a single
            # coordinate be spammed. Cap absolute repeats per coordinate/region anyway.
            if coord_count >= self.config.max_clicks_per_coord_total * 3:
                return True
            if region_key and region_count >= self.config.max_clicks_per_region_total * 2:
                return True
        if base_name == "ACTION6" and not ignore_defer and self._defer_unproven_click_action(scene):
            return True
        if base_name != "ACTION6":
            meaning = self.memory.game.action_meanings.get(base_name)
            # V2.7: resource_wasting_noop is a strong prior, but if recent presses of
            # this action still open never-seen states, keep it available for
            # confirm/transform probes (sb26 ACTION5 was permanently hard-blocked
            # after early pre-selection presses looked like a tiny status ticker).
            if (
                meaning is not None
                and meaning.kind == "resource_wasting_noop"
                and meaning.movements == 0
                and self._action_recent_novel_state_yield(base_name) == 0
            ):
                return True
            if not ignore_defer and self._defer_unproven_nonmovement_action(scene, base_name):
                return True
        if self._would_repeat_terminal_suffix(key):
            return True
        # V2.2 hard, non-bypassable block: an action whose back-to-back spam ended
        # >=2 attempts (and >=half of all deaths) is lethal regardless of state
        # hash. bp35 died 10x to ACTION3 runs because plan_executor's
        # ignore_contract kept re-opening it in fresh states. Requiring the last
        # TWO presses to match keeps coincidental "last key before a timer death"
        # (tu93 dies every 50 steps no matter what) from banning movement keys.
        if base_name != "ACTION6" and len(level.bad_action_suffixes) >= 2:
            terminal_hits = sum(
                1
                for s in level.bad_action_suffixes
                if len(s) >= 2 and s[-1].split(":", 1)[0] == base_name and s[-2].split(":", 1)[0] == base_name
            )
            if terminal_hits >= 2 and terminal_hits * 2 >= len(level.bad_action_suffixes):
                return True
        if not ignore_noop_evidence and key in level.known_noops_by_state.get(scene.state_hash, set()):
            return True
        if not ignore_loop_quarantine:
            if level.quarantine_until_by_state.get(scene.state_hash, {}).get(key, 0) > now:
                return True
            if base_name != "ACTION6" and level.global_quarantine_until.get(base_name, 0) > now:
                return True
        if self.config.avoid_action7 and base_name == "ACTION7":
            meaning = self.memory.game.action_meanings.get(base_name)
            counts = level.action_outcomes.get(base_name, Counter())
            if counts.get("noop", 0) >= self.config.action7_noop_quarantine_after and counts.get("interaction", 0) + counts.get("transform", 0) == 0:
                return True
            if (meaning is None or meaning.kind not in {"interact_or_transform", "resource"} or meaning.noop_ratio >= 0.5 or meaning.kind == "undo") and not self._action7_probe_allowed():
                return True
        if not ignore_recent and self.config.action_repeat_cooldown > 0:
            recent = list(level.recent_action_keys)[-self.config.action_repeat_cooldown:]
            if key in recent and base_name not in {"ACTION6"}:
                return True
        return False

    def _next_unknown_simple_action(self, legal: Sequence[Any], scene: SceneSnapshot) -> str:
        for name in simple_action_names(legal):
            if self._action_blocked(scene, name, ignore_recent=True):
                continue
            m = self.memory.game.action_meanings.get(name)
            if m is None or m.kind in {"unknown", "unknown_or_blocked"} or m.confidence < 0.55:
                return name
            if m.kind in {"interact_or_transform", "click_or_select", "resource"} and m.positive_count < 2 and m.noop_ratio < 0.5:
                return name
        for name in simple_action_names(legal):
            if self._action_blocked(scene, name) or name in self.memory.level.tried_actions_by_state.get(scene.state_hash, set()):
                continue
            m = self.memory.game.action_meanings.get(name)
            if m is not None and m.kind == "movement" and m.confidence >= 0.55:
                continue
            return name
        return ""

    def _best_interaction_action(self, scene: SceneSnapshot, legal: Sequence[Any], target_id: str = "") -> str:
        scored: list[tuple[float, str]] = []
        state_outcomes = self.memory.level.state_action_outcomes.get(scene.state_hash, {})
        for name in simple_action_names(legal):
            if self._action_blocked(scene, name):
                continue
            m = self.memory.game.action_meanings.get(name)
            counts = self.memory.level.action_outcomes.get(name, Counter())
            score = 0.35 if name == "ACTION5" else 0.0
            if name == "ACTION7":
                score -= 3.5 if self.config.avoid_action7 else 0.75
            if m:
                if m.kind in {"interact_or_transform", "resource"}:
                    score += 2.2 + m.confidence + 0.35 * m.positive_count
                elif m.kind == "click_or_select":
                    score += 0.8 + m.confidence
                elif m.kind == "undo":
                    score -= 4.0
                elif m.kind in {"unknown_or_blocked", "resource_wasting_noop"}:
                    score -= 1.5 + 2.0 * m.noop_ratio
                if m.vector and m.kind == "movement":
                    score -= 1.1
                if m.noop_ratio >= 0.6:
                    score -= 2.0 * m.noop_ratio
                if m.life_losses or m.retries:
                    score -= 2.0 * (m.life_losses + m.retries)
            score += 0.5 * counts.get("interaction", 0) + 0.35 * counts.get("transform", 0) - 0.45 * counts.get("noop", 0)
            local = state_outcomes.get(name, Counter())
            score += 1.0 * (local.get("interaction", 0) + local.get("transform", 0)) - 2.0 * local.get("noop", 0)
            scored.append((score, name))
        if not scored:
            return ""
        scored.sort(key=lambda x: (-x[0], int(re.search(r"\d+", x[1]).group(0)) if re.search(r"\d+", x[1]) else 99))
        best_score, best_name = scored[0]
        # Do not return a merely least-bad action; this prevents ar25-style ACTION7 no-op loops.
        return best_name if best_score >= 0.25 else ""

    def _object_interest_score(self, scene: SceneSnapshot, obj: ObjectObservation) -> float:
        actor_id = self.memory.level.controlled_object_id
        s = obj.salience + (10 if obj.frame_color is not None else 0) + (6 if obj.inner_pattern else 0) + (4 if obj.area <= 120 and obj.frame_color is None else 0)
        if obj.track_id == actor_id:
            s -= 8
        if obj.near_edge:
            s -= 1.5
        s -= 1.8 * self.memory.level.click_noops_by_object.get(obj.track_id, 0)
        s += 3.0 * self.memory.level.click_success_by_object.get(obj.track_id, 0)
        for effect in self.memory.game.object_effects:
            if self._descriptor_matches(effect.visual_descriptor, obj):
                s += 8 * effect.confidence
        return s

    def _choose_special_object(self, scene: SceneSnapshot) -> ObjectObservation | None:
        actor_id = self.memory.level.controlled_object_id
        candidates = [o for o in scene.objects if o.track_id != actor_id and not o.near_edge] or [o for o in scene.objects if o.track_id != actor_id]
        return max(candidates, key=lambda o: self._object_interest_score(scene, o), default=None)

    def _descriptor_matches(self, desc: VisualDescriptor, obj: ObjectObservation) -> bool:
        score = 0
        if desc.type_key and desc.type_key == obj.type_key:
            score += 4
        if desc.shape_label and desc.shape_label == obj.shape_label:
            score += 2
        if desc.inner_pattern and obj.inner_pattern.startswith(desc.inner_pattern[:48]):
            score += 2
        if desc.frame_color is not None and desc.frame_color == obj.frame_color:
            score += 1
        if desc.colors and len(set(desc.colors).intersection(obj.colors)) >= max(1, len(desc.colors) // 2):
            score += 1
        return score >= 3

    def _navigation_proposal_to_target(self, target_id: str, scene: SceneSnapshot, legal: Sequence[Any], step: PlanStep) -> dict[str, Any] | None:
        if self._in_transform_mode(scene):
            return None
        target = scene.object_by_id(target_id)
        if target is None or not self._is_navigable_target(scene, target):
            return None
        level = self.memory.level
        path = self._plan_path_to_object(scene, target_id, legal)
        if path:
            after_state = level.transition_graph.get(scene.state_hash, {}).get(path[0])
            state_visits = Counter(level.recent_state_hashes)
            if not (after_state and state_visits.get(after_state, 0) >= 2):
                return {"name": path[0], "target_object_id": target_id, "purpose": step.purpose or f"move toward {target_id}", "expected_change": step.stop_condition or "movement toward target", "nav_path_len": len(path), "ignore_recent": True}
        greedy = self._greedy_move_toward_target(scene, target_id, legal)
        if greedy:
            return {"name": greedy, "target_object_id": target_id, "purpose": step.purpose or f"try moving toward {target_id}", "expected_change": step.stop_condition or "greedy movement toward target", "nav_path_len": 0, "nav_fallback": "greedy", "ignore_recent": True}
        return None

    def _target_reached(self, target_id: str, scene: SceneSnapshot) -> bool:
        actor, target = scene.object_by_id(self.memory.level.controlled_object_id), scene.object_by_id(target_id)
        if actor is None or target is None:
            return False
        tx, ty = int(round(target.centroid[0])), int(round(target.centroid[1]))
        if actor.bbox[0] <= tx <= actor.bbox[2] and actor.bbox[1] <= ty <= actor.bbox[3]:
            return True
        # Also consider adjacency as reached for interact-style games; this avoids overshooting targets.
        gap_x, gap_y = _bbox_gap(actor.bbox, target.bbox)
        return gap_x == 0 and gap_y <= 1 or gap_y == 0 and gap_x <= 1

    def _recent_transform_pressure_active(self) -> bool:
        level = self.memory.level
        recent = list(level.recent_events)[-8:]
        transforms = sum(1 for e in recent if e.outcome in {"transform", "movement_with_transform"})
        movements = sum(1 for e in recent if e.outcome in {"movement", "movement_with_transform"})
        return transforms >= self.config.transform_mode_threshold and transforms >= movements

    def _transform_bottleneck_active(self) -> bool:
        level = self.memory.level
        return level.repeated_transform_streak >= self.config.transform_mode_threshold or self._recent_transform_pressure_active()

    def _in_transform_mode(self, scene: SceneSnapshot) -> bool:
        level = self.memory.level
        if level.repeated_transform_streak >= self.config.transform_mode_threshold:
            return True
        if self._recent_transform_pressure_active():
            return True
        simple = [name for name, m in self.memory.game.action_meanings.items() if m.kind == "interact_or_transform" and m.transforms >= 1 and m.movements == 0]
        return len(simple) >= 1 and level.repeated_transform_streak >= 1

    def _action_vectors(self, actor: ObjectObservation, legal: Sequence[Any]) -> dict[str, tuple[int, int]]:
        if self._in_transform_mode(self.memory.level.current_scene or SceneSnapshot((), (), 0, 0, "", "", 0, (), (), (), frozenset(), None, None, None, None, None, None, (), "")):
            return {}
        legal_names = {action_name(a).upper() for a in legal}
        vectors: dict[str, tuple[int, int]] = {}
        for name, m in self.memory.game.action_meanings.items():
            if (
                name in legal_names
                and m.vector is not None
                and m.kind == "movement"
                and m.movements >= 2
                and m.confidence >= 0.55
                and m.noop_ratio < 0.35
                and m.transforms <= m.movements
            ):
                vectors[name] = m.vector
        weak = {"ACTION1": (0, -actor.height), "ACTION2": (0, actor.height), "ACTION3": (-actor.width, 0), "ACTION4": (actor.width, 0)}
        for name, vec in weak.items():
            m = self.memory.game.action_meanings.get(name)
            if name in legal_names and name not in vectors and (
                m is None
                or (
                    m.kind not in {"interact_or_transform", "undo", "resource_wasting_noop"}
                    and m.transforms == 0
                    and m.interactions == 0
                    and m.noop_ratio < 0.35
                )
            ):
                vectors[name] = vec
        return vectors

    def _walkable_colors(self, scene: SceneSnapshot, actor: ObjectObservation) -> set[int]:
        votes = self.memory.level.walkable_color_votes
        if votes:
            best = max(votes.values())
            floor = {c for c, n in votes.items() if n >= max(1, best // 2)}
            floor.add(scene.background_candidate)
            return floor
        counts: Counter[int] = Counter()
        x0, y0, x1, y1 = actor.bbox
        for y in range(max(0, y0 - 1), min(scene.height - 1, y1 + 1) + 1):
            for x in range(max(0, x0 - 1), min(scene.width - 1, x1 + 1) + 1):
                if x0 <= x <= x1 and y0 <= y <= y1:
                    continue
                color = scene.grid[y][x]
                if color != scene.background_candidate and color not in actor.colors:
                    counts[color] += 1
        floor = {scene.background_candidate}
        if counts:
            floor.add(counts.most_common(1)[0][0])
        return floor

    def _plan_path_to_object(self, scene: SceneSnapshot, target_id: str, legal: Sequence[Any]) -> list[str] | None:
        actor, target = scene.object_by_id(self.memory.level.controlled_object_id), scene.object_by_id(target_id)
        if actor is None or target is None:
            return None
        vectors = self._action_vectors(actor, legal)
        floor = self._walkable_colors(scene, actor)
        if not vectors or not floor:
            return None
        start = (actor.bbox[0], actor.bbox[1])
        width, height = actor.width, actor.height
        q: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
        seen = {start}
        while q:
            pos, path = q.popleft()
            test = (pos[0], pos[1], pos[0] + width - 1, pos[1] + height - 1)
            tx, ty = int(round(target.centroid[0])), int(round(target.centroid[1]))
            if test[0] <= tx <= test[2] and test[1] <= ty <= test[3]:
                return path
            if len(path) >= self.config.navigation_max_depth:
                continue
            for name in simple_action_names(legal):
                if name not in vectors or self._action_blocked(scene, name, ignore_recent=True):
                    continue
                dx, dy = vectors[name]
                nxt = (pos[0] + dx, pos[1] + dy)
                if nxt in seen or not self._position_passable(scene, nxt, width, height, floor, actor, target):
                    continue
                seen.add(nxt)
                q.append((nxt, [*path, name]))
        return None

    def _position_passable(self, scene: SceneSnapshot, pos: tuple[int, int], width: int, height: int, floor: set[int], actor: ObjectObservation, target: ObjectObservation) -> bool:
        x0, y0 = pos
        x1, y1 = x0 + width - 1, y0 + height - 1
        if x0 < 0 or y0 < 0 or x1 >= scene.width or y1 >= scene.height:
            return False
        blockers = [o for o in scene.objects if o.track_id != actor.track_id and not o.near_edge and o.area <= 180 and o.frame_color is None]
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if actor.bbox[0] <= x <= actor.bbox[2] and actor.bbox[1] <= y <= actor.bbox[3]:
                    continue
                if target.bbox[0] <= x <= target.bbox[2] and target.bbox[1] <= y <= target.bbox[3]:
                    continue
                if any(o.bbox[0] <= x <= o.bbox[2] and o.bbox[1] <= y <= o.bbox[3] for o in blockers):
                    continue
                if scene.grid[y][x] in floor:
                    continue
                return False
        return True

    def _deterministic_navigation_action(self, scene: SceneSnapshot, legal: Sequence[Any]) -> tuple[Any, dict[str, Any], str] | None:
        level = self.memory.level
        if self._in_transform_mode(scene):
            return None
        actor = scene.object_by_id(level.controlled_object_id)
        legal_names = {action_name(a).upper() for a in legal}
        learned_vectors = [name for name, meaning in self.memory.game.action_meanings.items() if name in legal_names and meaning.vector is not None and meaning.kind == "movement" and meaning.confidence >= 0.4]
        if actor is None or len(learned_vectors) < 2:
            return None
        target_id = self._navigation_target_id(scene)
        if not target_id:
            return None
        if self._target_reached(target_id, scene):
            interact = self._best_interaction_action(scene, legal, target_id)
            if interact:
                proposal = {"name": interact, "target_object_id": target_id, "purpose": f"interact after reaching {target_id}", "expected_change": "finish or reveal goal effect"}
                action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
                if action is not None:
                    return action, proposal, "deterministic_navigation"
            level.bottleneck_reason = level.bottleneck_reason or "target_reached_no_safe_interaction"
            return None
        step = PlanStep("move_to", target_object_id=target_id, purpose=f"deterministically move controlled object toward {target_id}", stop_condition="stop when target is reached", max_attempts=1)
        proposal = self._navigation_proposal_to_target(target_id, scene, legal, step)
        if proposal is None:
            return None
        action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
        if action is None:
            return None
        return action, proposal, "deterministic_navigation"

    def _looks_like_playfield_object(self, scene: SceneSnapshot, obj: ObjectObservation) -> bool:
        if obj.near_edge:
            return False
        if scene.hud_panel_bbox is not None:
            hx0, hy0, hx1, hy1 = scene.hud_panel_bbox
            ox0, oy0, ox1, oy1 = obj.bbox
            if not (ox1 < hx0 or ox0 > hx1 or oy1 < hy0 or oy0 > hy1):
                return False
        if scene.counter_bbox is not None:
            cx0, cy0, cx1, cy1 = scene.counter_bbox
            ox0, oy0, ox1, oy1 = obj.bbox
            if not (ox1 < cx0 or ox0 > cx1 or oy1 < cy0 or oy0 > cy1):
                return False
        if self._looks_like_nonplayfield_status_cue(scene, obj):
            return False
        return True

    def _actor_floor_region_bbox(self, scene: SceneSnapshot, actor: ObjectObservation) -> tuple[int, int, int, int] | None:
        votes = self.memory.level.walkable_color_votes
        if votes:
            best = max(votes.values())
            floor = {c for c, n in votes.items() if n >= max(1, best // 2)}
            floor.add(scene.background_candidate)
        else:
            floor = {scene.background_candidate}
        starts: list[tuple[int, int]] = []
        x0, y0, x1, y1 = actor.bbox
        for x in range(max(0, x0 - 1), min(scene.width - 1, x1 + 1) + 1):
            for y in (y0 - 1, y1 + 1):
                if 0 <= y < scene.height and scene.grid[y][x] in floor:
                    starts.append((x, y))
        for y in range(max(0, y0 - 1), min(scene.height - 1, y1 + 1) + 1):
            for x in (x0 - 1, x1 + 1):
                if 0 <= x < scene.width and scene.grid[y][x] in floor:
                    starts.append((x, y))
        if not starts:
            return None
        q: deque[tuple[int, int]] = deque(starts)
        seen = set(starts)
        min_x = min_y = 10**9
        max_x = max_y = -1
        while q:
            x, y = q.popleft()
            min_x, min_y, max_x, max_y = min(min_x, x), min(min_y, y), max(max_x, x), max(max_y, y)
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not (0 <= nx < scene.width and 0 <= ny < scene.height) or (nx, ny) in seen:
                    continue
                if scene.grid[ny][nx] not in floor:
                    continue
                seen.add((nx, ny))
                q.append((nx, ny))
        return (min_x, min_y, max_x, max_y) if max_x >= 0 else None

    def _is_navigable_target(self, scene: SceneSnapshot, target: ObjectObservation) -> bool:
        actor = scene.object_by_id(self.memory.level.controlled_object_id)
        if actor is None:
            return False
        if target.track_id == actor.track_id:
            return True
        if self._is_possible_selector_pair_object(scene, target):
            self._note_navigation_target_rejected(target, "selector_pair_mechanism_object", {"level": self.memory.level.level_index, "target": target.track_id, "reason": "selector_pair_mechanism_object"})
            return False
        motion_region = self._actor_motion_region_bbox(scene, actor)
        if self._looks_like_nonplayfield_status_cue(scene, target, actor=actor):
            if motion_region is None or not self._bbox_intersects_with_margin(target.bbox, motion_region, 3):
                self._note_navigation_target_rejected(target, "status_or_template_cue", {"level": self.memory.level.level_index, "target": target.track_id, "target_bbox": target.bbox, "actor": actor.track_id, "actor_motion_region": motion_region, "reason": "status_or_template_cue"})
                return False
        region = self._actor_floor_region_bbox(scene, actor)
        if region is None:
            return not target.near_edge
        intersects = self._bbox_intersects_with_margin(target.bbox, region, 1)
        if not intersects and self._looks_like_playfield_object(scene, target) and self._has_learned_vector_toward(actor, target):
            self.logger.log_event("navigation_target_allowed_by_vector", {"level": self.memory.level.level_index, "target": target.track_id, "target_bbox": target.bbox, "actor": actor.track_id, "actor_floor_region": region})
            return True
        if not intersects:
            self._note_navigation_target_rejected(target, "outside_actor_floor_region", {"level": self.memory.level.level_index, "target": target.track_id, "target_bbox": target.bbox, "actor": actor.track_id, "actor_floor_region": region, "reason": "outside_actor_floor_region"})
        return intersects

    def _note_navigation_target_rejected(self, target: ObjectObservation, reason: str, payload: dict[str, Any]) -> None:
        """Dedupe navigation-rejection logging and feed repeated rejections back to
        the VLM. ar25 previously logged the same (target, reason) 250x per episode
        while the VLM kept re-binding the exact same unreachable target because the
        rejection never reached its prompt."""
        level = self.memory.level
        key = f"{target.track_id}:{reason}"
        level.nav_reject_counts[key] += 1
        count = level.nav_reject_counts[key]
        if count in (1, 5, 25, 100):
            self.logger.log_event("navigation_target_rejected", {**payload, "repeat_count": count})
        if count == 3:
            note = (
                f"Navigation target {target.track_id} keeps being rejected ({reason}): the controlled actor cannot reach it. "
                "Do NOT re-bind it as a move_to target. Either pick a target reachable by the actor, or explain the mechanism "
                "(e.g. a coupled/secondary object is what must reach it) via probe_action/action_sequence steps instead of move_to."
            )
            if note[:60] not in (level.notes_for_next_call or ""):
                level.notes_for_next_call = _short(note + " " + (level.notes_for_next_call or ""), 700)

    def _has_learned_vector_toward(self, actor: ObjectObservation, target: ObjectObservation) -> bool:
        dx = float(target.centroid[0] - actor.centroid[0])
        dy = float(target.centroid[1] - actor.centroid[1])
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return True
        horizontal = abs(dx) >= abs(dy)
        for meaning in self.memory.game.action_meanings.values():
            if meaning.kind != "movement" or meaning.vector is None or meaning.confidence < 0.4 or meaning.noop_ratio >= 0.7:
                continue
            vx, vy = meaning.vector
            if horizontal and dx * vx > 0:
                return True
            if not horizontal and dy * vy > 0:
                return True
        return False

    @staticmethod
    def _bbox_intersects_with_margin(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (ax1 < bx0 - margin or ax0 > bx1 + margin or ay1 < by0 - margin or ay0 > by1 + margin)

    def _seed_selector_pair_hints(self, scene: SceneSnapshot) -> None:
        level = self.memory.level
        ids: set[str] = set()
        for rel in scene.template_relations:
            if rel.get("possible_selector_pair"):
                for key in ("left", "right"):
                    oid = _short(rel.get(key), 24).upper()
                    if oid:
                        ids.add(oid)
        level.selector_pair_object_ids = ids

    def _is_possible_selector_pair_object(self, scene: SceneSnapshot, obj: ObjectObservation) -> bool:
        if obj.track_id in self.memory.level.selector_pair_object_ids:
            return True
        for rel in scene.template_relations:
            if rel.get("possible_selector_pair") and obj.track_id in {rel.get("left"), rel.get("right")}:
                return True
        return False

    def _transform_cycle_bottleneck_active(self, level: LevelMemoryV20 | None = None) -> bool:
        level = level or self.memory.level
        if level.bottleneck_reason == "transform_state_cycle":
            return True
        if level.transform_state_visits and max(level.transform_state_visits.values()) >= 3:
            return True
        return False

    def _critical_counter_blocks_plan_executor(self, scene: SceneSnapshot) -> bool:
        if scene.counter_value is not None and scene.counter_value <= 1:
            return True
        if scene.counter_ratio is not None and scene.counter_ratio <= self.config.critical_resource_ratio:
            return True
        return False

    # V2.4: once escalated, this must stay true for the rest of the LEVEL, including
    # after a game-over reset (_start_new_attempt carries hypothesis_escalation_count
    # over). ls20 regression: attempt 2's LEVEL_INIT immediately re-issued a
    # route_reason=geometry plan because the momentary transform_state_cycle signal
    # (which resets with fresh per-attempt state) had gone quiet, even though the
    # selector-pair hypothesis from attempt 1 was still correct.
    def _mechanism_hypothesis_locked(self) -> bool:
        return self.memory.level.hypothesis_escalation_count >= 1

    _STALE_NAV_ROLE_NAMES = {
        "target", "goal", "target_frame", "exit", "target_glyph", "glyph", "navigation_target",
    }

    # V2.5: action sources that pick a move without any specific hypothesis about
    # game mechanics (pure exhaustive/least-bad/frontier probing while the VLM or
    # the plan executor has nothing to offer). A transform-state revisit reached
    # only through these should not be trusted as evidence of a genuine cycling
    # mechanism - see the transform_state_cycle escalation gate in _record_transition
    # and _maybe_escalate_mechanism_hypothesis.
    _UNDIRECTED_ACTION_SOURCES = {
        "least_bad_nonreset", "frontier_probe", "safe_probe_fallback", "nonreset_guard",
        "escape_nonreset", "evidence_exhausted_nonreset",
    }

    def _bind_selector_pair_roles(self, scene: SceneSnapshot) -> dict[str, VisualDescriptor]:
        level = self.memory.level
        pair_rels = [r for r in scene.template_relations if r.get("possible_selector_pair")]
        if not pair_rels:
            return {}
        rel = pair_rels[0]
        left = _short(rel.get("left"), 24).upper()
        right = _short(rel.get("right"), 24).upper()
        # Note: left is always treated as the template/status frame and right as the
        # selector, matching the existing possible_selector_pair convention (left is
        # the first-seen/lower-id framed object in _template_relations' i<j pairing);
        # this mirrors pre-V2.4 behavior exactly, no reordering by position.
        status_id, selector_id = left, right
        level.local_bindings["status_template_frame"] = status_id
        level.local_bindings["selector_frame"] = selector_id
        # V2.7: do NOT copy selector_pair_object_ids into mechanism_evidence_object_ids.
        # Binding is a visual hypothesis; evidence must come from observed transforms
        # in _record_transition. Mixing them made the V2.6 evidence gate a no-op on
        # the second escalate (pair ids were already "evidence" after the first bind).
        visual_roles: dict[str, VisualDescriptor] = {}
        for role_name, oid in (("status_template_frame", status_id), ("selector_frame", selector_id)):
            obj = scene.object_by_id(oid)
            if obj is not None:
                visual_roles[role_name] = self._visual_descriptor(obj, scene)
        return visual_roles

    def _maybe_escalate_mechanism_hypothesis(self, scene: SceneSnapshot, trigger: str) -> None:
        level = self.memory.level
        # V2.7: one lock is enough. A second rewrite used to re-fire from vetting
        # feedback after _bind_selector_pair_roles had already polluted evidence.
        if level.hypothesis_escalation_count >= 1:
            return
        should_escalate = trigger in {
            "transform_state_cycle",
            "geometry_plan_rejected_in_cycle",
            "geometry_to_selector_pair_rejected",
            "vlm_empty_plan_in_cycle",
            "repeated_vlm_rejection",
        }
        if level.bottleneck_reason.startswith("transform_after_") and level.transform_pressure >= 3:
            should_escalate = True
        recent = level.rejected_vlm_plan_feedback[-4:]
        geo_rejects = 0
        for entry in recent:
            if not isinstance(entry, dict):
                continue
            reasons = entry.get("reasons") if isinstance(entry.get("reasons"), dict) else {}
            geo_rejects += int(reasons.get("transform_state_cycle_rejects_geometry_plan", 0))
        if geo_rejects >= 2:
            should_escalate = True
        if level.vlm_issue_repeat_count >= 3 and self._transform_cycle_bottleneck_active(level):
            should_escalate = True
        if not should_escalate:
            return
        attempt_actions = max(0, level.total_action_count - level.attempt_started_at_action_count)
        if attempt_actions < self.config.min_hypothesis_escalation_actions:
            # V2.7: permanently banning geometry mid-attempt is irreversible for the
            # rest of the level (including after game-over reset). Wait until the
            # attempt has explored enough that a short early cycle is unlikely to be
            # ordinary navigation/transform noise (ls20 locked ~step 30).
            self.logger.log_event(
                "hypothesis_escalation_deferred_v20",
                {
                    "level": level.level_index,
                    "trigger": trigger,
                    "reason": "attempt_budget",
                    "attempt_actions": attempt_actions,
                    "min_actions": self.config.min_hypothesis_escalation_actions,
                    "evidence_object_count": len(level.mechanism_evidence_object_ids),
                },
            )
            return
        if trigger != "transform_state_cycle":
            # V2.6: vetting-side triggers (plan rejections, empty plans, repeat
            # bottlenecks) must be corroborated by gameplay evidence before locking
            # geometry navigation out for the rest of the level: at least one
            # selector-pair candidate object must actually have been observed
            # transforming. Without this gate the vetting layer escalates off its
            # own feedback loop - ls20 locked at ~step 15 from
            # geometry_to_selector_pair_rejected with ZERO transforms all run, then
            # rejected 84 VLM geometry steps + 100 navigation targets and spent
            # 149/240 steps ping-ponging in least_bad_nonreset until fuel bankruptcy.
            # transform_state_cycle stays exempt: it is already gated on directed
            # visits in _record_transition and implies observed transforms.
            pair_ids = set(level.selector_pair_object_ids)
            for rel in scene.template_relations:
                if rel.get("possible_selector_pair"):
                    for key in ("left", "right"):
                        oid = _short(rel.get(key), 24).upper()
                        if oid:
                            pair_ids.add(oid)
            if not pair_ids or not (pair_ids & level.mechanism_evidence_object_ids):
                self.logger.log_event(
                    "hypothesis_escalation_deferred_v20",
                    {
                        "level": level.level_index,
                        "trigger": trigger,
                        "selector_pair": sorted(pair_ids),
                        "evidence_object_count": len(level.mechanism_evidence_object_ids),
                    },
                )
                return

        cleared_roles: list[str] = []
        for role in list(level.local_bindings.keys()):
            if role.lower() in self._STALE_NAV_ROLE_NAMES or role in self._STALE_NAV_ROLE_NAMES:
                cleared_roles.append(role)
                del level.local_bindings[role]

        desc_parts = [
            "Mechanism hypothesis (escalated): framed inner-pattern objects may form a selector pair.",
            "Actions likely cycle the selector frame's inner pattern and update the status/template frame pattern.",
            "Win by aligning the status frame inner_pattern with the goal glyph, not by navigating the actor to corner-frame coordinates.",
        ]
        visual_roles = self._bind_selector_pair_roles(scene)
        if visual_roles:
            status_id = level.local_bindings.get("status_template_frame", "")
            selector_id = level.local_bindings.get("selector_frame", "")
            desc_parts.append(f"status_template_frame={status_id} (template/status frame); selector_frame={selector_id} (cycling selector).")

        mem = self.memory.game.win_condition or WinConditionMemory()
        mem.description_nl = self._sanitize_game_text(" ".join(desc_parts), scene)
        mem.visual_roles.update(visual_roles)
        mem.confidence = max(mem.confidence, 0.45)
        mem.evidence.append({"level": level.level_index, "trigger": trigger, "source": "hypothesis_escalation_v20"})
        mem.evidence = mem.evidence[-12:]
        self.memory.game.win_condition = mem

        note = (
            "HYPOTHESIS ESCALATION: geometry/coupled_carrier_geometry routes are banned for the rest of this "
            "level (including after a reset/new attempt). Return a short selector_probe/mechanism_probe "
            "action_sequence testing how the selector frame changes the status frame inner_pattern relative "
            "to the goal glyph."
        )
        if note[:48] not in (level.notes_for_next_call or ""):
            level.notes_for_next_call = _short(note + " " + (level.notes_for_next_call or ""), 700)
        level.current_plan = []
        level.plan_cursor = 0
        level.action_recovery_contract = True
        level.hypothesis_escalation_count += 1
        self.logger.log_event(
            "hypothesis_escalated_v20",
            {
                "level": level.level_index,
                "trigger": trigger,
                "cleared_roles": cleared_roles,
                "selector_pair": sorted(level.selector_pair_object_ids),
                "count": level.hypothesis_escalation_count,
            },
        )

    def _looks_like_nonplayfield_status_cue(self, scene: SceneSnapshot, obj: ObjectObservation, actor: ObjectObservation | None = None) -> bool:
        # V2.3: this is a pure bbox-position heuristic (corner + far from actor ==
        # decorative HUD/template). ls20's O1 sits in that corner yet is a real
        # mechanism object - its inner pattern cycles on a subset of actions. Once
        # persistent-id transform evidence exists for this object, position alone
        # must not keep vetoing it as non-navigable/non-interactable forever.
        if obj.track_id in self.memory.level.mechanism_evidence_object_ids:
            return False
        if self._is_possible_selector_pair_object(scene, obj):
            return False
        x0, y0, x1, y1 = obj.bbox
        bottom_band = y0 >= max(0, scene.height - max(10, scene.height // 5)) and obj.area >= 25
        side_bottom = bottom_band and (x1 <= max(10, scene.width // 5) or x0 >= scene.width - max(10, scene.width // 5))
        if not side_bottom:
            return False
        if actor is None:
            return True
        ax0, ay0, ax1, ay1 = actor.bbox
        far_side = x1 < ax0 - max(6, actor.width * 2) or x0 > ax1 + max(6, actor.width * 2)
        below_actor = y0 > ay1 + max(2, actor.height // 2)
        return far_side or below_actor

    def _actor_motion_region_bbox(self, scene: SceneSnapshot, actor: ObjectObservation) -> tuple[int, int, int, int] | None:
        motion = self.memory.level.actor_motion_bbox()
        if motion is None:
            return None
        mx0, my0, mx1, my1 = motion
        ax0, ay0, ax1, ay1 = actor.bbox
        mx0, my0, mx1, my1 = min(mx0, ax0), min(my0, ay0), max(mx1, ax1), max(my1, ay1)
        width = max(1, mx1 - mx0 + 1)
        height = max(1, my1 - my0 + 1)
        if len(self.memory.level.actor_bbox_history) < 2 or (width <= actor.width + 1 and height <= actor.height + 1):
            return None
        pad_x = max(actor.width * 2, 6)
        pad_y = max(actor.height * 2, 6)
        return (max(0, mx0 - pad_x), max(0, my0 - pad_y), min(scene.width - 1, mx1 + pad_x), min(scene.height - 1, my1 + pad_y))

    def _navigation_target_id(self, scene: SceneSnapshot, exclude: set[str] | None = None) -> str:
        actor_id = self.memory.level.controlled_object_id
        exclude = exclude or set()
        # V2.5: ls20 regression - the VLM consistently named the chained maze
        # target "target_glyph" (already recognized by _STALE_NAV_ROLE_NAMES for
        # *rejection* purposes once the mechanism hypothesis is locked), but this
        # lookup order never recognized it for actual navigation, so even a
        # legitimately accepted binding could never drive
        # deterministic_navigation/fuel_target_greedy.
        role_order = ("target_frame", "goal", "target", "target_glyph", "glyph", "navigation_target", "exit", "transformer", "status_pattern")
        if self._in_transform_mode(scene):
            role_order = ("transformer", "status_pattern", "target_frame", "goal", "target", "target_glyph", "glyph", "navigation_target", "exit")
        for role in role_order:
            oid = self.memory.level.local_bindings.get(role, "")
            obj = scene.object_by_id(oid)
            if oid and oid != actor_id and oid not in exclude and obj is not None and self._is_navigable_target(scene, obj):
                return oid
        win = self.memory.game.win_condition
        if win is not None:
            for desc in win.visual_roles.values():
                for obj in self._descriptor_target_candidates(scene, desc, actor_id):
                    if obj.track_id not in exclude and self._is_navigable_target(scene, obj):
                        return obj.track_id
        if self._mechanism_hypothesis_locked():
            # V2.4: ls20 regression - once escalated, an explicit role/descriptor
            # binding is still honored above, but the "pick whatever looks most
            # interesting" fallback below kept re-selecting the goal glyph (not a
            # selector-pair object, so not filtered by _is_navigable_target) and fed
            # it straight back into deterministic_navigation/fuel_target_greedy,
            # silently reintroducing the exact geometry-chase the escalation banned.
            return ""
        for obj in sorted((o for o in scene.objects if o.track_id != actor_id and o.track_id not in exclude), key=lambda o: -self._object_interest_score(scene, o)):
            if self._is_navigable_target(scene, obj):
                return obj.track_id
        return ""

    def _descriptor_target_candidates(self, scene: SceneSnapshot, desc: VisualDescriptor, actor_id: str) -> list[ObjectObservation]:
        candidates = [o for o in scene.objects if o.track_id != actor_id and self._descriptor_matches(desc, o)]
        return sorted(candidates, key=lambda o: -self._object_interest_score(scene, o))

    def _greedy_move_toward_target(self, scene: SceneSnapshot, target_id: str, legal: Sequence[Any]) -> str:
        actor = scene.object_by_id(self.memory.level.controlled_object_id)
        target = scene.object_by_id(target_id)
        if actor is None or target is None:
            return ""
        vectors = self._action_vectors(actor, legal)
        if not vectors:
            return ""
        level = self.memory.level
        state_visits = Counter(level.recent_state_hashes)
        trans_graph = level.transition_graph.get(scene.state_hash, {})
        ax, ay = actor.centroid
        tx, ty = target.centroid
        base_dist = abs(ax - tx) + abs(ay - ty)
        floor = self._walkable_colors(scene, actor)
        scored: list[tuple[float, str]] = []
        for name, (dx, dy) in vectors.items():
            if self._action_blocked(scene, name, ignore_recent=True):
                continue
            nxt = (actor.bbox[0] + dx, actor.bbox[1] + dy)
            dist = abs((ax + dx) - tx) + abs((ay + dy) - ty)
            score = base_dist - dist
            if self._position_passable(scene, nxt, actor.width, actor.height, floor, actor, target):
                score += 0.75
            if name in set(level.recent_action_keys):
                score -= 1.25
            after_state = trans_graph.get(name)
            if after_state:
                visits = state_visits.get(after_state, 0)
                if visits >= 2:
                    score -= 2.5 * visits
            m = self.memory.game.action_meanings.get(name)
            if m is not None:
                score -= 1.5 * m.noop_ratio
                if m.kind != "movement" and m.transforms > 0:
                    score -= 2.0
            scored.append((score, name))
        if not scored:
            return ""
        scored.sort(key=lambda item: (-item[0], int(re.search(r"\d+", item[1]).group(0)) if re.search(r"\d+", item[1]) else 0))
        return scored[0][1] if scored[0][0] > -0.75 else ""

    def _recent_loop_actions(self, outcomes: set[str], *, min_len: int = 8, max_unique: int = 2, require_state_cycle: bool = False) -> set[str]:
        level = self.memory.level
        recent = list(level.recent_events)[-min_len:]
        if len(recent) < min_len or any(e.outcome not in outcomes for e in recent):
            return set()
        actions = [e.action_key for e in recent]
        if any(a.startswith("ACTION6:") for a in actions):
            return set()
        unique = set(actions)
        if not unique or len(unique) > max_unique:
            return set()
        if require_state_cycle:
            states = list(level.recent_state_hashes)[-(min_len + 1):]
            if len(states) >= min_len and len(set(states)) > max_unique + 2:
                return set()
        return unique

    def _quarantine_loop_actions(self, scene: SceneSnapshot, actions: set[str], reason: str) -> None:
        if not actions:
            return
        level = self.memory.level
        until = level.total_action_count + self.config.quarantine_steps
        q_state = level.quarantine_until_by_state.setdefault(scene.state_hash, {})
        changed = False
        for key in actions:
            old_state_until = q_state.get(key, 0)
            q_state[key] = max(old_state_until, until)
            changed = changed or q_state[key] != old_state_until
            base = key.split(":", 1)[0]
            if base != "ACTION6":
                old_global_until = level.global_quarantine_until.get(base, 0)
                level.global_quarantine_until[base] = max(old_global_until, until)
                changed = changed or level.global_quarantine_until[base] != old_global_until
        level.bottleneck_reason = reason
        if changed:
            self.logger.log_event("action_loop_quarantine_v20", {"level": level.level_index, "state": scene.state_hash[:12], "actions": sorted(actions), "reason": reason, "until": until})

    def _transform_controller_action(self, scene: SceneSnapshot, legal: Sequence[Any]) -> tuple[Any, dict[str, Any], str] | None:
        level = self.memory.level
        if not self._in_transform_mode(scene):
            return None
        if self._critical_counter_blocks_plan_executor(scene):
            level.bottleneck_reason = "critical_resource_counter"
            level.action_recovery_contract = True
            self.logger.log_event("transform_controller_stand_down_v20", {"level": level.level_index, "state": scene.state_hash[:12], "counter_value": scene.counter_value, "counter_ratio": scene.counter_ratio})
            return None
        if (
            self._source_storm_active("transform_controller", outcomes={"noop", "state_change", "transform", "movement_with_transform"})
            or self._source_low_yield_active("transform_controller", outcomes={"noop", "state_change", "transform", "movement_with_transform"})
        ):
            feedback = {
                "stage": "source_fuse",
                "reason": "transform_controller_low_yield",
                "source": "transform_controller",
                "recent_events": [e.as_prompt() for e in list(level.recent_events)[-6:]],
            }
            level.bottleneck_reason = "transform_controller_low_yield"
            level.action_recovery_contract = True
            level.fallback_guard_block_until = max(level.fallback_guard_block_until, level.total_action_count + 1)
            level.rejected_vlm_plan_feedback.append(feedback)
            level.rejected_vlm_plan_feedback = level.rejected_vlm_plan_feedback[-12:]
            self.logger.log_event("source_fuse_open_v20", {"level": level.level_index, "source": "transform_controller", "state": scene.state_hash[:12], "reason": "low_yield_transform_controller"})
            return None
        loop_actions = self._recent_loop_actions({"transform", "movement_with_transform"}, min_len=8)
        mixed_loop_actions = self._recent_loop_actions({"transform", "movement", "movement_with_transform"}, min_len=8)
        if loop_actions or mixed_loop_actions:
            self._quarantine_loop_actions(scene, loop_actions | mixed_loop_actions, "transform_action_loop")
        state_visits = Counter(level.recent_state_hashes)
        trans_graph = level.transition_graph.get(scene.state_hash, {})
        scored: list[tuple[float, str]] = []
        for name in simple_action_names(legal):
            if self._action_blocked(scene, name, ignore_recent=True):
                continue
            m = self.memory.game.action_meanings.get(name)
            counts = level.action_outcomes.get(name, Counter())
            local = level.state_action_outcomes.get(scene.state_hash, {}).get(name, Counter())
            score = 0.0
            if m:
                # V2.6: cap the interaction bonus like the transform bonus. m.interactions
                # grows by 1 on every press of a frequently-used interactive action, so an
                # uncapped 1.8x term reached +50 and up in sb26 (140 ACTION5 presses) and
                # drowned out every anti-repeat penalty below (-5.0/-6.0), producing 4-8x
                # ACTION5 runs that burned the hidden energy budget to game over. The cap
                # keeps early evidence meaningful while letting the run/revisit penalties
                # actually bite once the action stops yielding new states.
                score += 0.2 * min(m.transforms, 2) + 1.8 * min(m.interactions, 3) - 2.2 * m.noop_ratio - 2.0 * (m.kind == "undo")
                if level.transform_pressure >= 8 and m.transforms > m.interactions + 2:
                    score -= 0.35 * (m.transforms - m.interactions)
            progress_samples = list(level.action_progress_scores.get(name, ()))
            if progress_samples:
                recent_progress = progress_samples[-4:]
                avg_progress = sum(recent_progress) / max(1, len(recent_progress))
                # V2.7: when recent presses of this action still land on never-seen
                # after-states, treat the avg_progress penalty as exploration noise
                # rather than proof the action is dead. sb26 ACTION5 often scored
                # negative (tiny bar shrink) while every press was a novel state;
                # the -3.0 branch then permanently starved confirm/transform picks.
                novel_yield = self._action_recent_novel_state_yield(name)
                if novel_yield >= 2:
                    score += 1.5 + 0.4 * min(novel_yield, 4)
                else:
                    score += 1.2 * avg_progress
                    if avg_progress <= 0.0 and counts.get("transform", 0) >= 2:
                        score -= 3.0
            score += 0.1 * min(counts.get("transform", 0), 2) - 0.18 * max(0, counts.get("transform", 0) - 2) + 0.9 * min(counts.get("interaction", 0), 3) - 1.0 * counts.get("noop", 0)
            score += 1.25 * local.get("interaction", 0) - 1.4 * local.get("noop", 0) - 0.45 * local.get("transform", 0)
            after_state = trans_graph.get(name)
            if after_state:
                visits = state_visits.get(after_state, 0) + level.transform_state_visits.get(after_state, 0)
                if visits >= 2:
                    score -= 2.75 * visits
            recent_same = 0
            for k in reversed(level.recent_action_keys):
                if k == name:
                    recent_same += 1
                else:
                    break
            if recent_same >= self.config.max_same_transform_action:
                score -= 5.0
            recent_window = list(level.recent_action_keys)[-8:]
            recent_transform_events = [e for e in list(level.recent_events)[-8:] if e.outcome == "transform"]
            if len(recent_window) >= 6 and len(set(recent_window)) <= 2 and len(recent_transform_events) >= 6 and name in set(recent_window):
                score -= 6.0
            if name in set(level.recent_action_keys):
                score -= 0.5
            if name == "ACTION7" and self.config.avoid_action7:
                score -= 4.0
            if (loop_actions or mixed_loop_actions) and name not in (loop_actions | mixed_loop_actions):
                score += 0.35
            scored.append((score, name))
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[0], int(re.search(r"\d+", x[1]).group(0)) if re.search(r"\d+", x[1]) else 99))
        if scored[0][0] < 0.2:
            return None
        name = scored[0][1]
        proposal = {"name": name, "purpose": "transform-mode controller: test next non-blocked transform action instead of treating it as navigation", "expected_change": "structural transform, object appearance/disappearance, or goal progress", "ignore_recent": True}
        action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
        return (action, proposal, "transform_controller") if action is not None else None

    def _action_recent_novel_state_yield(self, action_name_key: str, *, lookback: int = 8) -> int:
        """Count recent presses of `action_name_key` that landed on a then-novel state.

        Used to protect expensive-but-progressive interact actions (selector bars,
        confirm triggers) from being starved by raw avg_progress alone.
        """
        level = self.memory.level
        base = action_name_key.upper().split(":", 1)[0]
        novel = 0
        for event in list(level.recent_events)[-lookback:]:
            if event.action_key.split(":", 1)[0] != base:
                continue
            if event.outcome not in {"transform", "interaction", "movement_with_transform", "state_change"}:
                continue
            # all_state_visits includes the landing itself; ==1 means first visit.
            if level.all_state_visits.get(event.after_state, 0) <= 1:
                novel += 1
        return novel

    def _click_disfavored_by_simple_evidence(self, legal: Sequence[Any]) -> bool:
        legal_names = {action_name(a).upper() for a in legal}
        simple_legal = [a for a in legal_names if re.fullmatch(r"ACTION\d+", a) and a not in {"ACTION6", "ACTION7"}]
        if not simple_legal or "ACTION6" not in legal_names:
            return False
        level = self.memory.level
        if sum(level.click_success_by_object.values()) > 0 or sum(level.click_success_coords.values()) > 0:
            return False
        meaning = self.memory.game.action_meanings.get("ACTION6")
        if meaning is not None and meaning.attempts >= 2 and meaning.noop_ratio >= 0.75:
            return True
        if sum(level.click_noops_by_object.values()) >= 4:
            return True
        if level.action_outcomes.get("ACTION6", Counter()).get("noop", 0) >= 3:
            return True
        return False

    def _deterministic_click_action(self, scene: SceneSnapshot, legal: Sequence[Any]) -> tuple[Any, dict[str, Any], str] | None:
        legal_names = {action_name(a).upper() for a in legal}
        if "ACTION6" not in legal_names:
            return None
        simple_legal = [a for a in legal_names if re.fullmatch(r"ACTION\d+", a) and a != "ACTION6"]
        if self._click_disfavored_by_simple_evidence(legal):
            return None
        level = self.memory.level
        if self._normal_click_fuse_active(scene, legal, "deterministic_click"):
            return None
        # Use deterministic click when there are no grounded movement vectors or when VLM has a click target.
        vector_count = sum(1 for m in self.memory.game.action_meanings.values() if m.kind == "movement" and m.vector is not None and m.confidence >= 0.4)
        if simple_legal and vector_count >= 2 and not level.resource_crisis and not self._in_transform_mode(scene):
            return None
        target_id = self._navigation_target_id(scene)
        candidates: list[ObjectObservation] = []
        if target_id and scene.object_by_id(target_id) is not None:
            candidates.append(scene.object_by_id(target_id))  # type: ignore[arg-type]
        candidates.extend(o for o in sorted(scene.objects, key=lambda o: -self._object_interest_score(scene, o)) if o not in candidates)
        blocked = set(level.known_noops_by_state.get(scene.state_hash, set())) | set(k for k, until in level.quarantine_until_by_state.get(scene.state_hash, {}).items() if until > level.total_action_count)
        tried = set(level.tried_actions_by_state.get(scene.state_hash, set()))
        point = self._state_conditioned_success_click(scene, blocked | tried)
        if point is not None:
            x, y = point
            proposal = {
                "name": "ACTION6",
                "x": x,
                "y": y,
                "target_object_id": "",
                "purpose": "replay state-conditioned successful click for this exact state",
                "expected_change": "continue known click-state sequence rather than global neighbor expansion",
                "ignore_click_exhaustion": True,
            }
            action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if action is not None:
                self.logger.log_event("deterministic_click_state_memory", {"level": level.level_index, "state": scene.state_hash[:12], "action": f"ACTION6:{x},{y}"})
                return action, proposal, "deterministic_click_state_memory"
        if level.click_success_coords or level.click_success_regions:
            point = self._successful_click_neighbor(scene, blocked | tried)
            if point is not None:
                x, y = point
                proposal = {"name": "ACTION6", "x": x, "y": y, "target_object_id": "", "purpose": "expand local frontier around successful click", "expected_change": "test neighboring grid-aligned click after prior coordinate caused progress"}
                action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
                if action is not None:
                    return action, proposal, "deterministic_click"
        for obj in candidates:
            if level.click_noops_by_object.get(obj.track_id, 0) >= self.config.max_object_click_noops and level.click_success_by_object.get(obj.track_id, 0) == 0:
                continue
            point = self._next_click_point_for_object(scene, obj, blocked=blocked | tried)
            if point is None:
                continue
            x, y = point
            key = f"ACTION6:{x},{y}"
            if key in blocked:
                continue
            # 防止同一坐标连续点击过多次：ft09 曾对 (38,54) 连点 18 次，每次产生
            # transform 但状态随机漂移（transform 非 noop 不计入 click_noops，无衰减）。
            same_run = 0
            for k in reversed(level.recent_action_keys):
                if k == key:
                    same_run += 1
                else:
                    break
            if same_run >= self.config.max_same_click_run:
                continue
            proposal = {"name": "ACTION6", "x": x, "y": y, "target_object_id": obj.track_id, "purpose": f"structured deterministic click on {obj.track_id}", "expected_change": "trigger target response; try center/edge/corner/outside-adjacent points instead of center-only"}
            action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if action is not None:
                return action, proposal, "deterministic_click"
        picked = self._first_untried_action6_target(scene, blocked | tried)
        if picked is not None:
            x, y, tid = picked
            proposal = {"name": "ACTION6", "x": x, "y": y, "target_object_id": tid, "purpose": f"frontier click probe {tid or 'grid'}", "expected_change": "explore fresh coordinate after object click candidates exhausted"}
            action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if action is not None:
                return action, proposal, "deterministic_click"
        return None

    def _click_candidates_for_object(self, scene: SceneSnapshot, obj: ObjectObservation) -> list[tuple[int, int]]:
        x0, y0, x1, y1 = obj.bbox
        raw: list[tuple[int, int]] = []
        cx, cy = int(round(obj.centroid[0])), int(round(obj.centroid[1]))
        raw.append((cx, cy))
        raw.append(((x0 + x1) // 2, (y0 + y1) // 2))
        raw.extend([(x0, y0), (x1, y1), (x0, y1), (x1, y0)])
        raw.extend([((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1), (x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2)])
        if obj.cells:
            cx_f, cy_f = float(obj.centroid[0]), float(obj.centroid[1])
            cells = sorted(((x, y) for x, y, _ in obj.cells), key=lambda p: ((p[0] - cx_f) ** 2 + (p[1] - cy_f) ** 2, p[1], p[0]))
            raw.extend(cells[:6])
            raw.extend(cells[-4:])
        # outside-adjacent probes for hitboxes that require border/neighbor cells
        for x in range(x0, x1 + 1):
            raw.append((x, y0 - 1))
            raw.append((x, y1 + 1))
        for y in range(y0, y1 + 1):
            raw.append((x0 - 1, y))
            raw.append((x1 + 1, y))
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for x, y in raw:
            x, y = max(0, min(scene.width - 1, int(x))), max(0, min(scene.height - 1, int(y)))
            if (x, y) not in seen:
                seen.add((x, y))
                out.append((x, y))
            if len(out) >= self.config.max_click_points_per_object:
                break
        return out

    def _next_click_point_for_object(self, scene: SceneSnapshot, obj: ObjectObservation, *, hint: str = "", blocked: set[str] | None = None) -> tuple[int, int] | None:
        blocked = blocked or (set(self.memory.level.known_noops_by_state.get(scene.state_hash, set())) | set(self.memory.level.tried_actions_by_state.get(scene.state_hash, set())))
        points = self._click_candidates_for_object(scene, obj)
        if hint:
            def pref(p: tuple[int, int]) -> int:
                x, y = p
                x0, y0, x1, y1 = obj.bbox
                if hint.startswith("corner"):
                    return 0 if (x in {x0, x1} and y in {y0, y1}) else 1
                if hint.startswith("edge"):
                    return 0 if (x in {x0, x1} or y in {y0, y1}) else 1
                if hint.startswith("outside"):
                    return 0 if not (x0 <= x <= x1 and y0 <= y <= y1) else 1
                if hint.startswith("center"):
                    cx, cy = obj.centroid
                    return int((x - cx) ** 2 + (y - cy) ** 2)
                return 0
            points = sorted(points, key=pref)
        for x, y in points:
            key = f"ACTION6:{x},{y}"
            if key not in blocked and not self._action_blocked(scene, key):
                return x, y
        return None

    def _state_conditioned_success_click(self, scene: SceneSnapshot, blocked: set[str]) -> tuple[int, int] | None:
        level = self.memory.level
        counter = level.successful_clicks_by_state.get(scene.state_hash, Counter())
        if not counter:
            return None
        blocked_upper = {str(k).upper() for k in blocked}
        recent_states = set(list(level.recent_state_hashes)[-8:])
        for key, _count in counter.most_common():
            key = str(key).upper()
            m = re.fullmatch(r"ACTION6:(-?\d+),(-?\d+)", key)
            if not m:
                continue
            x, y = int(m.group(1)), int(m.group(2))
            if key in blocked_upper:
                continue
            after_state = level.click_edges_by_state.get(scene.state_hash, {}).get(key, "")
            if after_state and after_state in recent_states and len(counter) > 1:
                continue
            if not self._action_blocked(scene, key, ignore_click_exhaustion=True):
                return x, y
        return None

    def _successful_click_neighbor(self, scene: SceneSnapshot, blocked: set[str]) -> tuple[int, int] | None:
        level = self.memory.level
        if not level.click_success_coords and not level.click_success_regions:
            return None
        candidates: list[tuple[float, int, int]] = []

        def add(x: int, y: int, score: float) -> None:
            x = max(0, min(scene.width - 1, int(x)))
            y = max(0, min(scene.height - 1, int(y)))
            key = f"ACTION6:{x},{y}"
            if key in blocked or self._action_blocked(scene, key):
                return
            if level.click_coord_counts.get(key, 0) >= 2:
                return
            candidates.append((score - 0.35 * level.click_coord_counts.get(key, 0), x, y))

        offsets = [(0, 8), (8, 0), (0, -8), (-8, 0), (8, 8), (8, -8), (-8, 8), (-8, -8), (0, 16), (16, 0), (0, -16), (-16, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (3, 0), (-3, 0), (0, 3), (0, -3), (0, 0)]
        for key, count in level.click_success_coords.most_common(8):
            m = re.fullmatch(r"ACTION6:(-?\d+),(-?\d+)", str(key).upper())
            if not m:
                continue
            sx, sy = int(m.group(1)), int(m.group(2))
            for rank, (dx, dy) in enumerate(offsets):
                add(sx + dx, sy + dy, 10.0 * count - 0.2 * rank)
        for region, count in level.click_success_regions.most_common(8):
            m = re.search(r":(-?\d+),(-?\d+)$", str(region).upper())
            if not m:
                continue
            rx, ry = int(m.group(1)), int(m.group(2))
            for drx in (-1, 0, 1):
                for dry in (-1, 0, 1):
                    add((rx + drx) * 8 + 4, (ry + dry) * 8 + 4, 7.5 * count - abs(drx) - abs(dry))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
        return candidates[0][1], candidates[0][2]

    def _first_untried_action6_target(self, scene: SceneSnapshot, noops: set[str]) -> tuple[int, int, str] | None:
        actor_id = self.memory.level.controlled_object_id
        state_click = self._state_conditioned_success_click(scene, noops)
        if state_click is not None:
            x, y = state_click
            return x, y, ""
        success_neighbor = self._successful_click_neighbor(scene, noops)
        if success_neighbor is not None:
            x, y = success_neighbor
            return x, y, ""
        for obj in sorted(scene.objects, key=lambda o: -self._object_interest_score(scene, o)):
            if obj.track_id == actor_id:
                continue
            if self.memory.level.click_noops_by_object.get(obj.track_id, 0) >= self.config.max_object_click_noops and self.memory.level.click_success_by_object.get(obj.track_id, 0) == 0:
                continue
            for x, y in self._click_candidates_for_object(scene, obj):
                key = f"ACTION6:{x},{y}"
                if key not in noops and not self._action_blocked(scene, key):
                    return x, y, obj.track_id
        bg = scene.background_candidate
        structural = set(scene.structural_colors)
        coords = [(xx, yy) for yy in range(scene.height) for xx in range(scene.width)]
        # deterministic frontier order: salient non-background cells first, then sparse grid.
        coords.sort(key=lambda p: (scene.grid[p[1]][p[0]] == bg or scene.grid[p[1]][p[0]] in structural, (p[0] + 3 * p[1]) % 7, p[1], p[0]))
        for xx, yy in coords:
            c = scene.grid[yy][xx]
            if c == bg or c in structural:
                continue
            key = f"ACTION6:{xx},{yy}"
            if key not in noops and not self._action_blocked(scene, key):
                return xx, yy, ""
        return None

    def _confirm_probe_action(self, scene: SceneSnapshot, legal: Sequence[Any]) -> tuple[Any, dict[str, Any], str] | None:
        """点击-选择类游戏（词表=ACTION6+至多2个简单动作，如 sb26 的 5/6[/7]）：
        一次点击刚产生可见变化时，立刻在“新状态”里测一次简单动作（确认/提交）。
        证据显示 sb26 的 ACTION5 在选择前的状态里全是 noop 并被全局隔离，导致
        “点击选中 -> 确认”组合从未被测试过。该探测按状态去重（每状态每动作一次），
        且不覆盖 retry/掉命/合约等硬证据。

        V2.7: also fire when the last click landed on a brand-new state (even if
        progress_score was negative from a tiny pixel delta). That is the common
        "select then confirm" signature across click+simple-action games.
        """
        space = self.memory.game.game_action_space
        if not space or 6 not in space:
            return None
        simple_space = [v for v in space if v not in (6, 7)]
        if self.config.action7_space_probe and 7 in space:
            simple_space = [v for v in space if v != 6]
        if not 1 <= len(simple_space) <= 2:
            return None
        level = self.memory.level
        if not level.recent_events:
            return None
        last = level.recent_events[-1]
        if not last.action_key.startswith("ACTION6:") or last.outcome in {"noop", "retry"}:
            return None
        delta = last.transition_delta if isinstance(last.transition_delta, dict) else {}
        if delta.get("life_delta") not in (None, 0):
            return None
        # Prefer probing after a click that actually opened a new state; otherwise
        # keep the old "any non-noop click" behaviour for first-contact learning.
        after_visits = level.all_state_visits.get(last.after_state, 0)
        if after_visits > 2 and last.outcome not in {"transform", "interaction", "movement_with_transform"}:
            return None
        legal_names = {action_name(a).upper() for a in legal}
        tried_here = level.tried_actions_by_state.get(scene.state_hash, set())
        # Prefer simple actions that previously yielded novel states / interactions
        # over never-tried ones only when both are available.
        ranked = sorted(
            simple_space,
            key=lambda value: (
                -self._action_recent_novel_state_yield(f"ACTION{value}"),
                -int((self.memory.game.action_meanings.get(f"ACTION{value}") or ActionMeaning(f"ACTION{value}")).interactions),
                value,
            ),
        )
        for value in ranked:
            name = f"ACTION{value}"
            if name == "ACTION7" and not self._action7_probe_allowed():
                continue
            if name not in legal_names or name in tried_here:
                continue
            state_counts = level.state_action_outcomes.get(scene.state_hash, {}).get(name, Counter())
            if state_counts.get("noop", 0) or state_counts.get("retry", 0):
                continue
            meaning = self.memory.game.action_meanings.get(name)
            if meaning is not None and (meaning.retries or meaning.life_losses):
                continue
            # Do not hard-skip resource_wasting_noop here when the action still
            # produces novel states elsewhere - that label can be stale from early
            # pre-selection presses.
            if (
                meaning is not None
                and meaning.kind == "resource_wasting_noop"
                and self._action_recent_novel_state_yield(name) == 0
                and meaning.positive_count == 0
            ):
                continue
            proposal = {
                "name": name,
                "purpose": f"confirm-probe: test {name} immediately after a click changed the scene",
                "expected_change": "commit/confirm the click selection or reveal the click-action pairing",
                "ignore_recent": True,
                "ignore_loop_quarantine": True,
            }
            made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if made is not None:
                self.logger.log_event(
                    "confirm_probe_v20",
                    {
                        "level": level.level_index,
                        "state": scene.state_hash[:12],
                        "action": name,
                        "after_click": last.action_key,
                        "after_visits": after_visits,
                    },
                )
                return made, proposal, "confirm_probe"
        return None

    def _frontier_probe_action(self, scene: SceneSnapshot, legal: Sequence[Any]) -> tuple[Any, dict[str, Any], str] | None:
        level = self.memory.level
        name = self._next_unknown_simple_action(legal, scene)
        if name:
            proposal = {"name": name, "purpose": "frontier probe of least-tested non-blocked simple action", "expected_change": "learn action effect or identify no-op"}
            action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if action is not None:
                return action, proposal, "frontier_probe"
        if "ACTION6" in {action_name(a).upper() for a in legal} and not self._click_disfavored_by_simple_evidence(legal) and not self._normal_click_fuse_active(scene, legal, "frontier_probe"):
            blocked = set(level.known_noops_by_state.get(scene.state_hash, set())) | set(level.tried_actions_by_state.get(scene.state_hash, set()))
            picked = self._first_untried_action6_target(scene, blocked)
            if picked is not None:
                x, y, target_id = picked
                proposal = {"name": "ACTION6", "x": x, "y": y, "target_object_id": target_id, "purpose": "frontier click probe on a fresh coordinate", "expected_change": "observe object or coordinate response"}
                action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
                if action is not None:
                    return action, proposal, "frontier_probe"
        return None

    def _reject_resource_risky_action(self, scene: SceneSnapshot, action: str, reason: str, *, purpose: str = "") -> None:
        self.logger.log_event(
            "resource_guard_action_rejected",
            {
                "level": self.memory.level.level_index,
                "state": scene.state_hash[:12],
                "action": action,
                "reason": reason,
                "purpose": _short(purpose, 160),
            },
        )
        self._contract_forbid_action(scene, action, f"resource_guard:{reason}")

    def _resource_recovered_from_scene(self, scene: SceneSnapshot) -> bool:
        if scene.counter_ratio is None:
            return False
        if scene.counter_value is not None and scene.counter_value <= 1:
            return False
        recovery_ratio = max(0.5, min(0.95, self.config.low_resource_ratio * 2.0))
        return scene.counter_ratio >= recovery_ratio

    def _resource_crisis_active(self, scene: SceneSnapshot) -> bool:
        if self.memory.level.resource_crisis and not self._resource_recovered_from_scene(scene):
            return True
        if scene.counter_ratio is not None and scene.counter_ratio <= self.config.low_resource_ratio:
            return True
        if scene.counter_value is not None and scene.counter_value <= 1:
            return True
        hidden_ratio = self._hidden_resource_remaining_ratio()
        if hidden_ratio is not None and hidden_ratio <= self.config.low_resource_ratio:
            return True
        return False

    def _resource_or_status_signal_active(self, scene: SceneSnapshot) -> bool:
        if self._resource_crisis_active(scene):
            return True
        res = self.memory.game.resource_model
        if res.visible_bar or scene.counter_value is not None or scene.life_count is not None or res.hidden_action_budget_capacity is not None:
            return True
        for obj in scene.objects:
            if obj.near_edge and (obj.width <= 2 or obj.height <= 2 or obj.area <= 80):
                return True
        return False

    def _grounded_movement_directions(self) -> set[str]:
        directions: set[str] = set()
        for _name, meaning in self.memory.game.action_meanings.items():
            if meaning.kind != "movement" or meaning.movements <= 0 or meaning.vector is None or meaning.confidence < 0.4:
                continue
            dx, dy = meaning.vector
            if abs(dx) >= abs(dy) and dx != 0:
                directions.add("right" if dx > 0 else "left")
            elif dy != 0:
                directions.add("down" if dy > 0 else "up")
        return directions

    def _defer_unproven_nonmovement_action(self, scene: SceneSnapshot, name: str) -> bool:
        name = name.split(":", 1)[0].upper()
        meaning = self.memory.game.action_meanings.get(name)
        if meaning is not None and (meaning.kind == "movement" or meaning.movements > 0):
            return False
        if self._in_transform_mode(scene):
            return False
        if len(self._grounded_movement_directions()) < 4:
            return False
        if not self._resource_or_status_signal_active(scene):
            return False
        if meaning is None:
            return True
        if meaning.kind == "interact_or_transform" and meaning.transforms + meaning.interactions > 0:
            return False
        if meaning.kind in {"unknown", "unknown_or_blocked", "resource_wasting_noop"}:
            return True
        if meaning.kind == "interact_or_transform" and meaning.movements == 0 and (meaning.positive_count <= 0 or meaning.small_transforms > 0):
            return True
        if meaning.noop_ratio >= 0.5 and meaning.movements == 0:
            return True
        return False

    def _defer_unproven_click_action(self, scene: SceneSnapshot) -> bool:
        level = self.memory.level
        if sum(level.click_success_by_object.values()) > 0 or sum(level.click_success_coords.values()) > 0:
            return False
        if self._in_transform_mode(scene):
            return False
        if len(self._grounded_movement_directions()) < 4:
            return False
        return self._resource_or_status_signal_active(scene)

    def _action_leads_to_recent_state(self, scene: SceneSnapshot, name: str, *, window: int = 10) -> bool:
        after_state = self.memory.level.transition_graph.get(scene.state_hash, {}).get(name.upper())
        return bool(after_state and after_state in set(list(self.memory.level.recent_state_hashes)[-window:]))

    def _resource_risky_nonreset_action(self, scene: SceneSnapshot, name: str) -> bool:
        name = name.split(":", 1)[0].upper()
        if name == "ACTION6":
            return False
        level = self.memory.level
        # V2.2: when the counter is a per-step tax every action pays equally (ls20),
        # NO action is more resource-risky than another. The retry/life_loss signals
        # below are all counter-exhaustion bankruptcies, not per-action danger, and
        # firing them here blacklisted the whole action space (absolute_nonreset
        # lockup). Must short-circuit BEFORE the retry/life checks, since local
        # state_action retry counts are also poisoned by the bankruptcy step.
        if level.counter_is_step_tax:
            return False
        local = level.state_action_outcomes.get(scene.state_hash, {}).get(name, Counter())
        meaning = self.memory.game.action_meanings.get(name)
        if local.get("retry", 0):
            return True
        if meaning is not None and (meaning.life_losses > 0 or meaning.retries > 0):
            return True
        if local.get("noop", 0) and local.get("transform", 0) + local.get("interaction", 0) + local.get("movement", 0) + local.get("movement_with_transform", 0) == 0:
            return True
        if self._action_leads_to_recent_state(scene, name):
            return True
        if meaning is None:
            return True
        if meaning.transforms + meaning.interactions > 0:
            return False
        if meaning.kind == "resource_wasting_noop" or meaning.noop_ratio >= 0.5:
            return True
        if meaning.positive_count <= 0:
            return True
        return False

    def _committed_guard_action(self, scene: SceneSnapshot, legal: Sequence[Any]) -> str:
        """When the previous guard/escape action made visible progress, keep going in
        that direction instead of alternating opposite moves that cancel each other
        (the ls20 ACTION1/ACTION2 ping-pong that drained the whole counter)."""
        level = self.memory.level
        if not level.recent_events:
            return ""
        last = level.recent_events[-1]
        guard_sources = {"nonreset_guard", "escape_nonreset", "evidence_exhausted_nonreset", "safe_probe_fallback", "least_bad_nonreset"}
        if last.source not in guard_sources or last.outcome in {"noop", "retry"}:
            return ""
        delta = last.transition_delta if isinstance(last.transition_delta, dict) else {}
        if delta.get("life_delta") not in (None, 0):
            return ""
        try:
            if float(delta.get("progress_score", 0.0)) <= 0.0:
                return ""
        except Exception:
            return ""
        key = last.action_key.upper()
        if key.startswith("ACTION6"):
            return ""
        run = 0
        for k in reversed(level.recent_action_keys):
            if k == key:
                run += 1
            else:
                break
        if run >= self.config.escape_commit_max_run:
            return ""
        legal_names = {action_name(a).upper() for a in legal}
        return key if key in legal_names else ""

    def _reliable_nonmovement_action(self, scene: SceneSnapshot, legal: Sequence[Any], blocked: set[str]) -> str:
        """Pick a non-ACTION6 action with a confirmed interact_or_transform track
        record (e.g. sb26's ACTION5 selector cycle) that safe_probe_fallback would
        otherwise ignore once its meaning is no longer "unverified". Capped at
        escape_commit_max_run consecutive picks so this cannot itself become an
        unbounded single-action loop; the VLM cadence still runs on its own
        schedule independent of this cap."""
        level = self.memory.level
        best_name, best_transforms = "", 0
        for name in simple_action_names(legal):
            if name in {"ACTION6", "RESET"} or name in blocked:
                continue
            if self._action_blocked(scene, name, ignore_recent=True):
                continue
            meaning = self.memory.game.action_meanings.get(name)
            if meaning is None or meaning.confidence < 0.55 or meaning.kind != "interact_or_transform":
                continue
            if meaning.noop_ratio >= 0.5 or meaning.transforms < 2:
                continue
            run = 0
            for key in reversed(level.recent_action_keys):
                if key != name:
                    break
                run += 1
            if run >= self.config.escape_commit_max_run:
                continue
            if meaning.transforms > best_transforms:
                best_transforms, best_name = meaning.transforms, name
        return best_name

    def _fallback_nonreset_action(self, scene: SceneSnapshot | None, legal: Sequence[Any], purpose: str, *, allow_guard: bool = True) -> tuple[Any, dict[str, Any], str] | None:
        if scene is None:
            return None
        level = self.memory.level
        noops = set(level.known_noops_by_state.get(scene.state_hash, set()))
        tried = set(level.tried_actions_by_state.get(scene.state_hash, set()))
        blocked = noops | tried
        recent = set(level.recent_action_keys)
        resource_crisis = self._resource_crisis_active(scene)
        for name in simple_action_names(legal):
            if name in blocked or name in recent or self._action_blocked(scene, name):
                continue
            if resource_crisis and self._resource_risky_nonreset_action(scene, name):
                self._reject_resource_risky_action(scene, name, "low_resource_risky_fallback")
                continue
            meaning = self.memory.game.action_meanings.get(name)
            if meaning is not None and meaning.confidence >= 0.55 and meaning.kind not in {"unknown", "unknown_or_blocked"}:
                continue
            proposal = {"name": name, "purpose": purpose, "expected_change": "one-step safe probe of an unverified action"}
            made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if made is not None:
                return made, proposal, "safe_probe_fallback"
        # V2.6: sb26 regression - once a non-movement action's meaning is confidently
        # learned (e.g. a selector/palette cycle), the unverified-action probe loop
        # above skips it on purpose, and this fallback previously fell straight
        # through to blind ACTION6 coordinate scanning - which never runs out of
        # fresh coordinates to try and so never gave the known-productive action a
        # turn again - even in games whose only real lever is repeating that one
        # action. Prefer it over an undirected click probe when it is not itself
        # flagged resource-risky. Checked before the ACTION6 branch below so an
        # always-available "untried coordinate" cannot starve it forever.
        reliable = self._reliable_nonmovement_action(scene, legal, blocked)
        if reliable and not (resource_crisis and self._resource_risky_nonreset_action(scene, reliable)):
            proposal = {"name": reliable, "purpose": f"repeat confirmed interactive action {reliable} instead of an undirected click probe", "expected_change": "continue the only known non-movement action with a track record of real transforms", "ignore_recent": True}
            made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if made is not None:
                return made, proposal, "known_interaction_repeat"
        if "ACTION6" in {action_name(a).upper() for a in legal} and not self._click_fuse_blocks_broad_click(scene, legal, "safe_probe_fallback", blocked):
            picked = self._first_untried_action6_target(scene, blocked)
            if picked is not None:
                x, y, target_id = picked
                proposal = {"name": "ACTION6", "x": x, "y": y, "target_object_id": target_id, "purpose": purpose, "expected_change": "one-step safe click probe on a fresh coordinate"}
                made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
                if made is not None:
                    return made, proposal, "safe_probe_fallback"
        # V2.2 fuel-game directed fallback: every step costs counter, so an undirected
        # least-bad probe is strictly worse than a greedy step toward the bound target
        # (ls20 burned 52 fallback steps at ~0 progress ping-ponging between 2-3
        # states while a valid target existed). Greedy scoring already penalizes
        # noop-prone moves and recently revisited states.
        if level.counter_is_step_tax:
            nav_target = self._navigation_target_id(scene)
            greedy = self._greedy_move_toward_target(scene, nav_target, legal) if nav_target else ""
            if greedy:
                proposal = {"name": greedy, "target_object_id": nav_target, "purpose": f"fuel-aware greedy step toward {nav_target}", "expected_change": "reduce distance to target instead of undirected probing", "ignore_recent": True}
                made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
                if made is not None:
                    return made, proposal, "fuel_target_greedy"
        if resource_crisis:
            level.bottleneck_reason = level.bottleneck_reason or "resource_crisis_no_safe_fallback"
            return None
        if not allow_guard:
            level.bottleneck_reason = level.bottleneck_reason or "fallback_waiting_for_vlm"
            return None
        movement_loop = self._recent_loop_actions({"movement", "movement_with_transform"}, min_len=8, require_state_cycle=True)
        transform_loop = self._recent_loop_actions({"transform", "movement_with_transform"}, min_len=8)
        if movement_loop or transform_loop:
            self._quarantine_loop_actions(scene, movement_loop | transform_loop, "nonreset_guard_action_loop")
        committed = self._committed_guard_action(scene, legal)
        if committed and committed != "ACTION6" and not self._action_blocked(scene, committed, ignore_recent=True):
            proposal = {"name": committed, "purpose": purpose, "expected_change": "continue the guard direction that just made progress", "ignore_recent": True}
            made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if made is not None:
                return made, proposal, "nonreset_guard"
        click_fuse_blocks_broad = self._click_fuse_blocks_broad_click(scene, legal, "nonreset_guard", blocked)
        for action in legal:
            name = action_name(action).upper()
            if name == "RESET" or self._action_blocked(scene, name, ignore_recent=True):
                continue
            if name == "ACTION6":
                if click_fuse_blocks_broad:
                    continue
                picked = self._first_untried_action6_target(scene, set(noops) | set(tried))
                if picked is None:
                    continue
                x, y, target_id = picked
                proposal = {"name": "ACTION6", "x": x, "y": y, "target_object_id": target_id, "purpose": purpose, "expected_change": "non-reset guard with fresh click coordinate"}
                made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
                if made is not None:
                    return made, proposal, "nonreset_guard"
                continue
            proposal = {"name": name, "purpose": purpose, "expected_change": "non-reset guard after VLM/probe exhaustion"}
            made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if made is not None:
                return made, proposal, "nonreset_guard"
        return None


    def _absolute_nonreset_action(self, scene: SceneSnapshot, legal: Sequence[Any], purpose: str) -> tuple[Any, dict[str, Any], str] | None:
        level = self.memory.level
        resource_crisis = self._resource_crisis_active(scene)
        if resource_crisis:
            self.logger.log_event(
                "absolute_nonreset_resource_escape_v20",
                {
                    "level": level.level_index,
                    "state": scene.state_hash[:12],
                    "counter": scene.counter_value,
                    "ratio": scene.counter_ratio,
                    "resource_crisis": level.resource_crisis,
                    "purpose": _short(purpose, 160),
                },
            )

        recent = list(level.recent_action_keys)
        state_noops = set(level.known_noops_by_state.get(scene.state_hash, set()))
        state_tried = set(level.tried_actions_by_state.get(scene.state_hash, set()))
        contract_rules = level.contract_forbidden_by_state.get(scene.state_hash, {})
        now = level.total_action_count

        def contract_reason_for(name: str) -> str:
            if name == "ACTION6":
                return ""
            return _short(contract_rules.get(name, ""), 160)

        def hard_danger(name: str, reason: str) -> bool:
            lowered = reason.lower()
            if "retry" in lowered or "life_loss" in lowered or "life-loss" in lowered:
                return True
            outcomes = level.state_action_outcomes.get(scene.state_hash, {}).get(name, Counter())
            if outcomes.get("retry", 0) > 0:
                return True
            meaning = self.memory.game.action_meanings.get(name)
            return bool(meaning is not None and (meaning.retries > 0 or meaning.life_losses > 0))

        critical_resource = bool(
            (scene.counter_ratio is not None and scene.counter_ratio <= self.config.critical_resource_ratio)
            or (scene.counter_value is not None and scene.counter_value <= 1)
        )
        candidates: list[tuple[tuple[float, ...], str, Any, str]] = []
        hard_blocked: list[dict[str, Any]] = []
        for action in legal:
            name = action_name(action).upper()
            if name == "RESET" or not re.fullmatch(r"ACTION\d+", name):
                continue
            reason = contract_reason_for(name)
            if hard_danger(name, reason):
                hard_blocked.append({"action": name, "reason": reason or "retry_or_life_loss_evidence"})
                continue
            if reason:
                hard_blocked.append({"action": name, "reason": reason})
                continue
            value = float(action_value(action))
            if name == "ACTION6":
                score = (3.0, 1.0 if reason else 0.0, 0.0, 0.0, float(recent.count(name)), value)
            else:
                meaning = self.memory.game.action_meanings.get(name)
                positive = float(meaning.positive_count if meaning is not None else 0)
                risky = self._resource_risky_nonreset_action(scene, name)
                known_noop = name in state_noops
                tried = name in state_tried
                local_outcomes = level.state_action_outcomes.get(scene.state_hash, {}).get(name, Counter())
                local_progress = local_outcomes.get("movement", 0) + local_outcomes.get("transform", 0) + local_outcomes.get("interaction", 0) + local_outcomes.get("level_advanced", 0)
                if known_noop and local_progress == 0:
                    hard_blocked.append({"action": name, "reason": "known_noop_here"})
                    continue
                if meaning is not None and meaning.kind == "resource_wasting_noop" and positive <= 0:
                    hard_blocked.append({"action": name, "reason": "resource_wasting_noop"})
                    continue
                critical_noop = critical_resource and (known_noop or (local_outcomes.get("noop", 0) > 0 and local_progress == 0))
                action7_penalty = 8.0 if name == "ACTION7" and self.config.avoid_action7 else 0.0
                score = (
                    action7_penalty,
                    1.0 if reason else 0.0,
                    3.0 if critical_noop else 0.0,
                    1.0 if known_noop else 0.0,
                    1.0 if risky else 0.0,
                    1.0 if tried else 0.0,
                    float(recent.count(name)),
                    -positive,
                    value,
                )
            candidates.append((score, name, action, reason))

        def attempt_candidate(name: str, action: Any, reason: str) -> tuple[Any, dict[str, Any], str] | None:
            if name == "ACTION6":
                picked = self._first_untried_action6_target(scene, set())
                if picked is None:
                    target = self._choose_special_object(scene)
                    if target is not None:
                        point = next((pt for pt in self._click_candidates_for_object(scene, target) if 0 <= pt[0] < scene.width and 0 <= pt[1] < scene.height), None)
                        picked = (point[0], point[1], target.track_id) if point is not None else None
                if picked is None and scene.width > 0 and scene.height > 0:
                    picked = (scene.width // 2, scene.height // 2, "")
                if picked is None:
                    return None
                x, y, target_id = picked
                proposal = {
                    "name": "ACTION6",
                    "x": x,
                    "y": y,
                    "target_object_id": target_id,
                    "purpose": purpose,
                    "expected_change": "escape non-reset click after all guarded planners were exhausted",
                    "ignore_recent": True,
                    "ignore_loop_quarantine": True,
                    "ignore_defer": True,
                    "ignore_resource_risk": True,
                }
                made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            else:
                proposal = {
                    "name": name,
                    "purpose": purpose,
                    "expected_change": "escape non-reset probe after all guarded planners were exhausted",
                    "ignore_recent": True,
                    "ignore_loop_quarantine": True,
                    "ignore_defer": True,
                    "ignore_resource_risk": True,
                }
                made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if made is None:
                return None
            self.logger.log_event(
                "absolute_nonreset_v20",
                {
                    "level": level.level_index,
                    "state": scene.state_hash[:12],
                    "action": name,
                    "source": "escape_nonreset",
                    "resource_crisis": resource_crisis,
                    "ignored_contract": False,
                    "contract_reason": reason,
                    "purpose": _short(purpose, 160),
                },
            )
            return made, proposal, "escape_nonreset"

        ordered = sorted(candidates, key=lambda item: item[0])
        committed = self._committed_guard_action(scene, legal)
        if committed:
            for _score, name, action, reason in ordered:
                if name == committed and not reason:
                    selected = attempt_candidate(name, action, reason)
                    if selected is not None:
                        return selected
                    break
        for _score, name, action, reason in ordered:
            if reason:
                continue
            selected = attempt_candidate(name, action, reason)
            if selected is not None:
                return selected
        self.logger.log_event(
            "absolute_nonreset_exhausted_v20",
            {
                "level": level.level_index,
                "state": scene.state_hash[:12],
                "resource_crisis": resource_crisis,
                "candidate_count": len(candidates),
                "hard_blocked": hard_blocked[:8],
                "purpose": _short(purpose, 160),
            },
        )
        if hard_blocked and not candidates:
            level.bottleneck_reason = "all_nonreset_actions_blocked_by_evidence"
            level.action_recovery_contract = True
        return None

    def _evidence_exhausted_nonreset_action(self, scene: SceneSnapshot, legal: Sequence[Any], purpose: str) -> tuple[Any, dict[str, Any], str] | None:
        level = self.memory.level
        legal_nonreset = [a for a in legal if action_name(a).upper() != "RESET" and re.fullmatch(r"ACTION\d+", action_name(a).upper())]
        if not legal_nonreset:
            return None
        state_noops = set(level.known_noops_by_state.get(scene.state_hash, set()))
        state_outcomes = level.state_action_outcomes.get(scene.state_hash, {})
        contract_rules = level.contract_forbidden_by_state.get(scene.state_hash, {})

        def reason_for(name: str) -> str:
            if name in contract_rules:
                return _short(contract_rules[name], 120) or "contract_forbidden"
            outcomes = state_outcomes.get(name, Counter())
            meaning = self.memory.game.action_meanings.get(name)
            if outcomes.get("retry", 0) or (meaning is not None and (meaning.retries or meaning.life_losses)):
                return "retry_or_life_loss_evidence"
            if name in state_noops:
                return "known_noop_here"
            if name != "ACTION6" and self._resource_risky_nonreset_action(scene, name):
                return "resource_risky"
            if self._would_repeat_terminal_suffix(name):
                return "terminal_suffix_risk"
            if name == "ACTION7" and self.config.avoid_action7:
                return "action7_guard"
            return "guard_exhausted"

        def hard_bad_reason(reason: str) -> bool:
            if reason == "guard_exhausted":
                return False
            lower_reason = reason.lower()
            return any(
                marker in lower_reason
                for marker in ("noop", "no-op", "retry", "life", "resource", "terminal", "suffix", "contract", "blocked", "forbidden", "action7")
            )

        immediate_counter_crisis = scene.counter_value is not None and scene.counter_value <= 1
        if immediate_counter_crisis:
            reasons = [(action_name(a).upper(), reason_for(action_name(a).upper())) for a in legal_nonreset]
            if reasons and all(hard_bad_reason(reason) for _, reason in reasons):
                level.bottleneck_reason = "critical_resource_no_safe_sentinel"
                level.action_recovery_contract = True
                self.logger.log_event(
                    "evidence_exhausted_nonreset_stand_down_v20",
                    {
                        "level": level.level_index,
                        "state": scene.state_hash[:12],
                        "counter_value": scene.counter_value,
                        "counter_ratio": scene.counter_ratio,
                        "reasons": {name: reason for name, reason in reasons[:8]},
                        "purpose": _short(purpose, 160),
                    },
                )
                return None

        recent_usage = Counter(list(level.recent_action_keys)[-8:])
        state_tried = level.tried_actions_by_state.get(scene.state_hash, set())

        def rank(item: tuple[Any, str]) -> tuple[float, float, float, float]:
            action, name = item
            reason = reason_for(name)
            danger_order = {
                "guard_exhausted": 0.0,
                "resource_risky": 1.0,
                "known_noop_here": 2.0,
                "contract_forbidden": 3.0,
                "terminal_suffix_risk": 4.0,
                "retry_or_life_loss_evidence": 5.0,
                "action7_guard": 6.0,
            }
            reason_key = reason
            if reason_key not in danger_order:
                lower_reason = reason.lower()
                if "retry" in lower_reason or "life_loss" in lower_reason or "life-loss" in lower_reason:
                    reason_key = "retry_or_life_loss_evidence"
                elif "resource" in lower_reason:
                    reason_key = "resource_risky"
                elif "noop" in lower_reason or "no-op" in lower_reason:
                    reason_key = "known_noop_here"
                elif "terminal" in lower_reason or "suffix" in lower_reason:
                    reason_key = "terminal_suffix_risk"
                else:
                    reason_key = "contract_forbidden"
            danger = danger_order.get(reason_key, 3.0)
            # V2.2: since the sentinel executes a blacklisted action anyway, at least
            # prefer an untried (state, action) pair; 81% of ls20 sentinel actions
            # were repeats of already-observed pairs, i.e. paid resources for zero
            # information.
            tried_here = 1.0 if name in state_tried else 0.0
            return (danger, tried_here, float(recent_usage.get(name, 0)), float(action_value(action)))

        action, name = sorted(((a, action_name(a).upper()) for a in legal_nonreset), key=rank)[0]
        reason = reason_for(name)
        proposal: dict[str, Any] = {
            "name": name,
            "purpose": purpose,
            "expected_change": f"non-reset sentinel despite exhausted guards ({reason})",
            "target_object_id": "",
        }
        if name == "ACTION6":
            picked = self._first_untried_action6_target(scene, set())
            if picked is None:
                target = self._choose_special_object(scene)
                point = self._click_candidates_for_object(scene, target)[0] if target is not None else (scene.width // 2, scene.height // 2)
                picked = (point[0], point[1], target.track_id if target is not None else "")
            x, y, target_id = picked
            proposal.update({"x": x, "y": y, "target_object_id": target_id})
            made = self._make_action6(action, x, y, proposal)
        else:
            made = self._attach_reasoning(action, proposal)
        self.logger.log_event(
            "evidence_exhausted_nonreset_v20",
            {
                "level": level.level_index,
                "state": scene.state_hash[:12],
                "action": name,
                "reason": reason,
                "legal_actions": [action_name(a).upper() for a in legal],
                "purpose": _short(purpose, 160),
            },
        )
        return made, proposal, "evidence_exhausted_nonreset"

    def _least_bad_nonreset_action(self, scene: SceneSnapshot, legal: Sequence[Any]) -> tuple[Any, dict[str, Any], str] | None:
        legal_by_value = {action_value(a): a for a in legal}
        level = self.memory.level
        resource_crisis = self._resource_crisis_active(scene)
        committed = self._committed_guard_action(scene, legal)
        if committed and committed != "ACTION6" and not (committed == "ACTION7" and self.config.avoid_action7):
            if not self._action_blocked(scene, committed, ignore_recent=True) and not (resource_crisis and self._resource_risky_nonreset_action(scene, committed)):
                proposal = {"name": committed, "purpose": "least-bad non-reset probe (continue progressing direction)", "expected_change": "keep the direction that just made progress", "ignore_recent": True}
                made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
                if made is not None:
                    return made, proposal, "least_bad_nonreset"
        for action in legal:
            name = action_name(action).upper()
            if name == "RESET" or (name == "ACTION7" and self.config.avoid_action7):
                continue
            if name != "ACTION6" and self._action_blocked(scene, name, ignore_recent=True):
                continue
            if resource_crisis and name != "ACTION6" and self._resource_risky_nonreset_action(scene, name):
                self._reject_resource_risky_action(scene, name, "low_resource_risky_least_bad")
                continue
            if name == "ACTION6":
                blocked = set(level.known_noops_by_state.get(scene.state_hash, set())) | set(level.tried_actions_by_state.get(scene.state_hash, set()))
                if self._click_fuse_blocks_broad_click(scene, legal, "least_bad_nonreset", blocked):
                    continue
                picked = self._first_untried_action6_target(scene, blocked)
                if picked is None:
                    continue
                x, y, target_id = picked
                proposal = {"name": "ACTION6", "x": x, "y": y, "target_object_id": target_id, "purpose": "least-bad non-reset probe", "expected_change": "last safe non-reset probe with fresh click coordinate"}
                made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
                if made is not None:
                    return made, proposal, "least_bad_nonreset"
                continue
            action_obj = get_action_by_name(name)
            if action_obj is None or action_value(action_obj) not in legal_by_value:
                continue
            proposal = {"name": name, "purpose": "least-bad non-reset probe", "expected_change": "avoid reset only if action is not known noop/risky"}
            made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if made is not None:
                return made, proposal, "least_bad_nonreset"
        self.logger.log_event("least_bad_nonreset_exhausted_v20", {"level": level.level_index, "state": scene.state_hash[:12], "resource_crisis": resource_crisis})
        return None

    def _make_valid_action(self, proposal: dict[str, Any] | None, scene: SceneSnapshot, legal: Sequence[Any], *, allow_reset: bool) -> Any | None:
        if not proposal:
            return None
        name = _short(proposal.get("name") or proposal.get("action"), 40).upper()
        if name in {"CLICK", "MOVE", "GO", "PATHFIND", "NAVIGATE"} or (name == "RESET" and not allow_reset):
            return None
        action = get_action_by_name(name)
        if action is None:
            return None
        legal_by_value = {action_value(a): a for a in legal}
        if name != "RESET" and action_value(action) not in legal_by_value:
            return None
        if name == "ACTION6":
            x, y = self._coerce_int(proposal.get("x")), self._coerce_int(proposal.get("y"))
            if x is None or y is None:
                target = scene.object_by_id(_short(proposal.get("target_object_id"), 24).upper()) or self._choose_special_object(scene)
                if target is None:
                    return None
                point = self._next_click_point_for_object(scene, target)
                if point is None:
                    return None
                x, y = point
            if not (0 <= x < scene.width and 0 <= y < scene.height):
                return None
            key = f"ACTION6:{x},{y}"
            same_run = 0
            for recent_key in reversed(self.memory.level.recent_action_keys):
                if recent_key == key:
                    same_run += 1
                else:
                    break
            if same_run >= self.config.max_same_click_run:
                return None
            if self._action_blocked(
                scene,
                key,
                ignore_recent=bool(proposal.get("ignore_recent")),
                ignore_loop_quarantine=bool(proposal.get("ignore_loop_quarantine")),
                ignore_noop_evidence=bool(proposal.get("ignore_noop_evidence")),
                ignore_click_exhaustion=bool(proposal.get("ignore_click_exhaustion")),
                ignore_contract=bool(proposal.get("ignore_contract")),
                ignore_defer=bool(proposal.get("ignore_defer")),
            ):
                return None
            return self._make_action6(legal_by_value.get(action_value(action), action), x, y, {**proposal, "name": "ACTION6", "x": x, "y": y})
        if self._action_blocked(
            scene,
            name,
            ignore_recent=bool(proposal.get("ignore_recent")),
            ignore_loop_quarantine=bool(proposal.get("ignore_loop_quarantine")),
            ignore_noop_evidence=bool(proposal.get("ignore_noop_evidence")),
            ignore_contract=bool(proposal.get("ignore_contract")),
            ignore_defer=bool(proposal.get("ignore_defer")),
        ):
            return None
        if not allow_reset and not bool(proposal.get("trusted_plan_sequence")) and not bool(proposal.get("ignore_resource_risk")) and self._resource_crisis_active(scene) and self._resource_risky_nonreset_action(scene, name):
            self._reject_resource_risky_action(scene, name, "low_resource_risky_proposal", purpose=_short(proposal.get("purpose"), 160))
            return None
        return self._attach_reasoning(legal_by_value.get(action_value(action), action), {**proposal, "name": name})

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
            return int(value.strip())
        return None

    def _make_action6(self, action: Any, x: int, y: int, proposal: dict[str, Any]) -> Any:
        data = {"x": int(x), "y": int(y)}
        if hasattr(action, "validate_data"):
            try:
                if action.validate_data(data) is False:
                    raise ObservationError(f"ACTION6 data rejected: {data}")
            except ObservationError:
                raise
            except Exception as exc:
                self.logger.log_event("action6_validate_error", {"reason": _short(str(exc), 200), "data": data})
        made = None
        if hasattr(action, "set_data"):
            try:
                made = action.set_data(data)
            except Exception:
                made = None
        made = made or action
        if action6_data(made) != data:
            for attr in ("action_data", "data"):
                try:
                    setattr(made, attr, dict(data))
                    if action6_data(made) == data:
                        break
                except Exception:
                    pass
        if action6_data(made) != data:
            made = ActionWithData(get_action_by_name("ACTION6") or action, dict(data))
        return self._attach_reasoning(made, {**proposal, "name": "ACTION6", "x": data["x"], "y": data["y"]})

    def _attach_reasoning(self, action: Any, proposal: dict[str, Any]) -> Any:
        reasoning = {"agent": self.config.agent_version, "purpose": _short(proposal.get("purpose"), 260), "expected_change": _short(proposal.get("expected_change"), 260), "target_object_id": _short(proposal.get("target_object_id"), 24).upper(), "name": proposal.get("name")}
        if proposal.get("x") is not None or proposal.get("y") is not None:
            reasoning["x"], reasoning["y"] = proposal.get("x"), proposal.get("y")
        try:
            setattr(action, "reasoning", reasoning)
            return action
        except Exception:
            data = action6_data(action) if action_name(action).upper() == "ACTION6" else None
            if data is not None:
                return ActionWithData(getattr(action, "id", None) or get_action_by_name("ACTION6") or action, data, reasoning)
        return action

    def _action_key_for_action(self, action: Any) -> str:
        name = action_name(action).upper()
        if name != "ACTION6":
            return name
        data = action6_data(action)
        if data is None:
            r = getattr(action, "reasoning", None)
            if isinstance(r, dict) and r.get("x") is not None and r.get("y") is not None:
                data = {"x": int(r["x"]), "y": int(r["y"])}
        return f"ACTION6:{data['x']},{data['y']}" if data is not None else "ACTION6:None,None"

    def _record_returned_action(self, action: Any, proposal: dict[str, Any], scene: SceneSnapshot, latest_frame: Any, *, source: str) -> None:
        name, key, level = action_name(action).upper(), self._action_key_for_action(action), self.memory.level
        if name == "RESET":
            level.pending_action = None
            return
        level.consecutive_nonterminal_resets = 0
        level.tried_actions_by_state.setdefault(scene.state_hash, set()).add(key)
        level.total_action_count += 1
        level.actions_since_vlm += 1
        level.chunk_action_count += 1
        x = y = None
        if name == "ACTION6" and ":" in key:
            try:
                x, y = [int(v) for v in key.split(":", 1)[1].split(",", 1)]
            except Exception:
                x = y = None
        level.pre_success_scene = scene
        level.pending_action = PendingAction(name, x, y, _short(proposal.get("purpose"), 260), _short(proposal.get("expected_change"), 260), _short(proposal.get("target_object_id"), 24).upper(), scene, id(latest_frame), self._call_index, source, proposal.get("expected_predicates") if isinstance(proposal.get("expected_predicates"), list) else [], latest_frame, level.last_vlm_mode)
        self.logger.log_event("action_v20", {"level": level.level_index, "step": level.total_action_count, "state": scene.state_hash[:12], "action": key, "source": source, "purpose": proposal.get("purpose"), "counter": scene.counter_value, "lives": scene.life_count, "transform_pressure": level.transform_pressure, "resource_crisis": level.resource_crisis})

    def _emergency_raw_nonreset_action(self, legal: Sequence[Any], purpose: str) -> Any | None:
        candidates: list[tuple[int, int, int, Any, str]] = []
        for action in legal:
            name = action_name(action).upper()
            if name in {"", "RESET", "ACTION6"} or not re.fullmatch(r"ACTION\d+", name):
                continue
            try:
                number = int(name.replace("ACTION", "", 1))
            except Exception:
                number = 999
            action7_penalty = 1 if name == "ACTION7" and self.config.avoid_action7 else 0
            recent_count = list(self.memory.level.recent_action_keys).count(name)
            candidates.append((action7_penalty, recent_count, number, action, name))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        _, _, _, action, name = candidates[0]
        proposal = {
            "action": name,
            "purpose": purpose,
            "expected_change": "Use a legal simple action because the current observation cannot be parsed.",
            "confidence": 0.05,
            "target_object_id": "",
            "expected_predicates": [],
        }
        level = self.memory.level
        level.consecutive_nonterminal_resets = 0
        level.total_action_count += 1
        level.actions_since_vlm += 1
        level.chunk_action_count += 1
        level.recent_action_keys.append(name)
        self.logger.log_event("emergency_raw_nonreset_v20", {"action": name, "legal_actions": [action_name(a).upper() for a in legal], "purpose": purpose})
        return self._attach_reasoning(action, proposal)

    def _reset_action(self, *, reason: str = "unspecified", state: str | None = None, scene: SceneSnapshot | None = None, legal: Sequence[Any] = ()) -> Any:
        legal_names = [action_name(a).upper() for a in legal]
        level = self.memory.level
        if state not in {"WIN", "GAME_OVER", "NOT_PLAYED", None}:
            level.consecutive_nonterminal_resets += 1
            level.current_plan = []
            level.plan_cursor = 0
            level.bottleneck_reason = level.bottleneck_reason or "nonterminal_reset_requested"
        else:
            level.consecutive_nonterminal_resets = 0
        if reason in {"game_over", "win", "max_actions"}:
            self._flush_last_vlm_io(reason)
        self.logger.log_event("reset_v20", {"reason": reason, "state_name": state or "", "level": level.level_index, "step": level.total_action_count, "state": scene.state_hash[:12] if scene is not None else "", "legal_actions": legal_names, "consecutive_nonterminal_resets": level.consecutive_nonterminal_resets})
        return get_action_by_name("RESET") or getattr(GameAction, "RESET")

    def _emergency_action(self, latest_frame: Any, legal: Sequence[Any]) -> Any:
        state = state_name(getattr(latest_frame, "state", None))
        if state in {"NOT_PLAYED", "GAME_OVER", "WIN"}:
            return self._reset_action(reason=state.lower(), state=state, legal=legal)
        try:
            scene = self.observer.scene_from_frame(latest_frame)
            selected = self._frontier_probe_action(scene, legal)
            if selected is not None:
                action, proposal, source = selected
                self._record_returned_action(action, proposal, scene, latest_frame, source=f"emergency_{source}")
                return action
        except Exception:
            pass
        try:
            scene = self.observer.scene_from_frame(latest_frame)
            selected = self._fallback_nonreset_action(scene, legal, "emergency safe non-reset fallback")
            if selected is None:
                selected = self._absolute_nonreset_action(scene, legal, "emergency final non-reset fallback")
            if selected is not None:
                action, proposal, source = selected
                self._record_returned_action(action, proposal, scene, latest_frame, source=f"emergency_{source}")
                return action
        except Exception:
            pass
        raw = self._emergency_raw_nonreset_action(legal, "emergency observation failure non-reset fallback")
        if raw is not None:
            return raw
        return self._reset_action(reason="emergency_no_nonreset", state=state, legal=legal)


__all__ = [
    "ActionMeaning", "ActionWithData", "AgentConfig", "CompactEvent", "ComponentObservation",
    "DecisionLogger", "GameAction", "GameState", "GameMemoryV20", "LevelMemoryV20",
    "MyAgent", "ObjectEffectMemory", "ObjectObservation", "ObservationError", "Observer",
    "OpenAICompatibleBackend", "PendingAction", "PlanStep", "Qwen35Backend", "ResourceModel",
    "RuntimeMemoryV20", "SceneSnapshot", "TransitionReport", "V20VLMResult", "VLMMode",
    "VLMRequest", "VisualDescriptor", "WinConditionMemory", "action6_data", "action_name",
    "action_value", "get_action_by_id", "get_action_by_name", "make_vlm_backend",
    "normalize_legal_actions", "normalize_one_action", "parse_vlm_result", "render_grid",
    "simple_action_names", "state_name", "vlm_uses_remote_api",
]
