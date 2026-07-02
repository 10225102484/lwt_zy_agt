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
    agent_version: str = "v1.9-lean-frontier-vlm-memory"
    enable_vlm: bool = True
    model_path: str | None = None
    image_size: int = 512
    max_actions: int = 240
    max_vlm_calls_per_level: int = 24
    vlm_min_action_gap: int = 2
    vlm_max_new_tokens: int = 1800
    max_chunk_steps: int = 4
    max_plan_steps: int = 12
    max_prompt_objects: int = 16
    max_recent_events: int = 12
    navigation_max_depth: int = 96
    nonterminal_reset_allowed: bool = False
    log_dir: str | None = "./runs/agent_v1_9"
    debug: bool = False
    vlm_backend: str = "local"
    vlm_api_base_url: str | None = None
    vlm_api_key: str | None = None
    vlm_api_model: str | None = None
    vlm_api_timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            enable_vlm=_bool_env(("ARC_V19_ENABLE_VLM", "ARC_V18_ENABLE_VLM", "ARC_V15_ENABLE_VLM"), True),
            model_path=_str_env(("ARC_V19_MODEL_PATH", "ARC_V18_MODEL_PATH", "ARC_V15_MODEL_PATH", "QWEN_MODEL_PATH")),
            image_size=_int_env(("ARC_V19_IMAGE_SIZE", "ARC_V18_IMAGE_SIZE", "ARC_V15_IMAGE_SIZE"), 512, 64, 1024),
            max_actions=_int_env(("ARC_V19_MAX_ACTIONS", "ARC_V18_MAX_ACTIONS", "ARC_V15_MAX_ACTIONS"), 240, 1, 10000),
            max_vlm_calls_per_level=_int_env(("ARC_V19_VLM_CALLS_PER_LEVEL", "ARC_V18_VLM_CALLS_PER_LEVEL", "ARC_V15_VLM_CALLS_PER_LEVEL"), 24, 0, 1000),
            vlm_min_action_gap=_int_env(("ARC_V19_VLM_MIN_ACTION_GAP", "ARC_V18_VLM_MIN_ACTION_GAP", "ARC_V15_VLM_MIN_ACTION_GAP"), 2, 0, 32),
            vlm_max_new_tokens=_int_env(("ARC_V19_VLM_MAX_NEW_TOKENS", "ARC_V18_VLM_MAX_NEW_TOKENS", "ARC_V15_VLM_MAX_NEW_TOKENS"), 1800, 256, 4096),
            max_chunk_steps=_int_env(("ARC_V19_MAX_CHUNK_STEPS", "ARC_V18_MAX_CHUNK_STEPS"), 4, 1, 12),
            max_plan_steps=_int_env(("ARC_V19_MAX_PLAN_STEPS", "ARC_V18_MAX_PLAN_STEPS"), 12, 1, 32),
            max_prompt_objects=_int_env(("ARC_V19_MAX_PROMPT_OBJECTS", "ARC_V18_MAX_PROMPT_OBJECTS"), 16, 4, 32),
            max_recent_events=_int_env(("ARC_V19_MAX_RECENT_EVENTS", "ARC_V18_MAX_RECENT_EVENTS"), 12, 4, 48),
            navigation_max_depth=_int_env(("ARC_V19_NAVIGATION_MAX_DEPTH", "ARC_V18_NAVIGATION_MAX_DEPTH", "ARC_V15_NAVIGATION_MAX_DEPTH"), 96, 8, 512),
            nonterminal_reset_allowed=_bool_env(("ARC_V19_ALLOW_NONTERMINAL_RESET", "ARC_V18_ALLOW_NONTERMINAL_RESET"), False),
            log_dir=_str_env(("ARC_V19_LOG_DIR", "ARC_V18_LOG_DIR", "ARC_V15_LOG_DIR"), "./runs/agent_v1_9"),
            debug=_bool_env(("ARC_V19_DEBUG", "ARC_V18_DEBUG", "ARC_V15_DEBUG"), False),
            vlm_backend=_str_env(("ARC_V19_VLM_BACKEND", "ARC_V18_VLM_BACKEND", "ARC_V15_VLM_BACKEND"), "local") or "local",
            vlm_api_base_url=_str_env(("ARC_V19_VLM_API_BASE_URL", "ARC_V18_VLM_API_BASE_URL", "ARC_V15_VLM_API_BASE_URL")),
            vlm_api_key=_str_env(("ARC_V19_VLM_API_KEY", "ARC_V18_VLM_API_KEY", "ARC_V15_VLM_API_KEY", "INF_API_KEY")),
            vlm_api_model=_str_env(("ARC_V19_VLM_API_MODEL", "ARC_V18_VLM_API_MODEL", "ARC_V15_VLM_API_MODEL")),
            vlm_api_timeout_s=_float_env(("ARC_V19_VLM_API_TIMEOUT_S", "ARC_V18_VLM_API_TIMEOUT_S", "ARC_V15_VLM_API_TIMEOUT_S"), 60.0, 1.0, 600.0),
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
    if isinstance(value, (list, tuple, set, deque)):
        return [_json_safe(v) for v in list(value)]
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        return {str(k): _json_safe(getattr(value, k)) for k in value.__dataclass_fields__}
    return value



_LOCAL_OBJECT_ID_RE = re.compile(r"\bO\d+\b")


def _strip_local_object_ids(value: Any) -> Any:
    if isinstance(value, str):
        return _LOCAL_OBJECT_ID_RE.sub("a local object", value)
    if isinstance(value, dict):
        return {(_strip_local_object_ids(k) if isinstance(k, str) else k): _strip_local_object_ids(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, deque)):
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
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)


def _bbox_contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    return max(0, max(a[0], b[0]) - min(a[2], b[2]) - 1), max(0, max(a[1], b[1]) - min(a[3], b[3]) - 1)


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
    def __init__(self, image_size: int = 512):
        self.image_size = image_size
        self._last_grid: tuple[tuple[int, ...], ...] | None = None
        self._previous_objects: tuple[ObjectObservation, ...] = ()
        self._next_track_id = 1
        self._hud_panel_bbox: tuple[int, int, int, int] | None = None
        self._counter_bbox: tuple[int, int, int, int] | None = None
        self._counter_fill_color: int | None = None
        self._counter_capacity: int | None = None
        self._life_slots: list[tuple[tuple[int, int, int, int], int]] = []
        self._structural_colors: set[int] = set()

    def reset_level(self) -> None:
        self._previous_objects = ()
        self._next_track_id = 1
        self._hud_panel_bbox = None
        self._counter_bbox = None
        self._counter_fill_color = None
        self._counter_capacity = None
        self._life_slots = []
        self._structural_colors = set()

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

    def analyze_grid(self, grid: tuple[tuple[int, ...], ...], rgb: Image.Image | None = None) -> SceneSnapshot:
        h, w = len(grid), len(grid[0])
        counts = Counter(v for row in grid for v in row)
        background = min(c for c, n in counts.items() if n == max(counts.values()))
        raw_components = self._components(grid, background)
        self._update_hud_model(grid, raw_components, background)
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
        scene = SceneSnapshot(grid, world_grid, w, h, _stable_hash(world_grid), _stable_hash(grid), background, tuple(c for c, _ in structural_colors.most_common()), comps, objects, frozenset(volatile), self._hud_panel_bbox, self._counter_bbox, counter, capacity, ratio, lives, self._template_relations(objects), "", rgb or render_grid(grid, self.image_size))
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
            colors = Counter(c for _, _, c in cells)
            norm = tuple((x - box[0], y - box[1], c) for x, y, c in cells)
            sig = hashlib.blake2b(repr(norm).encode(), digest_size=12).hexdigest()
            pattern = self._object_pattern(cells, box)
            frame_color = self._frame_color(parts, box)
            inner = self._inner_pattern(cells, box, frame_color)
            label = self._shape_label(cells, box, len(colors), frame_color)
            near_edge = box[0] <= 2 or box[1] <= 2 or box[2] >= len(grid[0]) - 3 or box[3] >= len(grid) - 3
            salience = len(colors) * 3 + len(parts) * 1.5 + min(25, len(cells) / 4) + (4 if frame_color is not None else 0) - (3 if near_edge else 0)
            objs.append(ObjectObservation("", box, (sum(xs) / len(xs), sum(ys) / len(ys)), len(cells), tuple(sorted(colors)), tuple(sorted(colors.items())), len(parts), cells, sig, label, pattern, inner, frame_color, near_edge, salience))
        return tuple(sorted(objs, key=lambda o: (-o.salience, o.bbox[1], o.bbox[0])))

    @staticmethod
    def _object_pattern(cells: Sequence[tuple[int, int, int]], box: tuple[int, int, int, int]) -> str:
        x0, y0, x1, y1 = box
        if x1 - x0 + 1 > 14 or y1 - y0 + 1 > 14:
            return f"<{x1-x0+1}x{y1-y0+1} pattern omitted>"
        mapping = {(x, y): c for x, y, c in cells}
        return "/".join("".join(_hex(mapping[(x, y)]) if (x, y) in mapping else "." for x in range(x0, x1 + 1)) for y in range(y0, y1 + 1))

    @staticmethod
    def _frame_color(comps: Sequence[ComponentObservation], box: tuple[int, int, int, int]) -> int | None:
        x0, y0, x1, y1 = box
        perim = {(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1) if x in {x0, x1} or y in {y0, y1}}
        if len(perim) < 12:
            return None
        for comp in sorted(comps, key=lambda c: c.area, reverse=True):
            if len(perim.intersection(comp.cells)) / max(1, len(perim)) >= 0.62:
                return comp.color
        return None

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
                score = sig_bonus + 2.0 * inter / max(1, union) + color_overlap
                if score > best_score:
                    best_i, best_score = i, score
            if best_i is not None and best_score >= 1.25:
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
        for i, a in enumerate(framed):
            a_rows = [r for r in a.inner_pattern.split("/") if r]
            a_bin = tuple("".join("#" if ch != "." else "." for ch in row) for row in a_rows)
            for b in framed[i + 1:]:
                b_rows = [r for r in b.inner_pattern.split("/") if r]
                b_bin = tuple("".join("#" if ch != "." else "." for ch in row) for row in b_rows)
                if a_bin and b_bin and a_bin == b_bin:
                    rels.append({"left": a.track_id, "right": b.track_id, "same_shape_under_rotation": True, "quarter_turns_left_to_right": 0, "same_inner_colors": a.inner_pattern == b.inner_pattern, "exact_inner_match": a.inner_pattern == b.inner_pattern, "edge_vs_world": a.near_edge != b.near_edge})
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
        candidates = [m for m in moved if not after_by[m["object_id"]].near_edge and after_by[m["object_id"]].shape_label != "frame_with_inner_pattern" and after_by[m["object_id"]].area <= 160]
        if not candidates and moved:
            # 严格过滤无结果时退回到所有移动对象中位移最大的（排除 near_edge），
            # 避免把真正的受控对象（大尺寸或带框图案的 actor）误排除导致 actor 无法绑定。
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
        for obj in sorted(scene.objects, key=lambda o: -o.salience)[:16]:
            lines.append(f"{obj.track_id}: type={obj.shape_label} bbox={obj.bbox} colors={dict(obj.color_areas)} area={obj.area} near_edge={obj.near_edge} pattern={obj.pattern}" + (f" inner={obj.inner_pattern}" if obj.inner_pattern else ""))
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
    resource_delta: int | None = None
    confidence: float = 0.0
    evidence_events: list[int] = field(default_factory=list)
    attempts: int = 0
    noops: int = 0

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
    refill_descriptors: list[VisualDescriptor] = field(default_factory=list)
    hazard_descriptors: list[VisualDescriptor] = field(default_factory=list)


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
            "summary": self.summary[:360],
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
class GameMemoryV18:
    action_meanings: dict[str, ActionMeaning] = field(default_factory=dict)
    win_condition: WinConditionMemory | None = None
    object_effects: list[ObjectEffectMemory] = field(default_factory=list)
    mechanics_nl: list[str] = field(default_factory=list)
    resource_model: ResourceModel = field(default_factory=ResourceModel)
    solved_level_summaries: list[dict[str, Any]] = field(default_factory=list)
    advisory_notes: str = ""
    vlm_calls_total: int = 0

    def as_prompt(self) -> dict[str, Any]:
        prompt = {
            "action_meanings": {k: v.as_prompt() for k, v in sorted(self.action_meanings.items())},
            "win_condition": _json_safe(self.win_condition),
            "object_effects": [_json_safe(x) for x in self.object_effects[-12:]],
            "mechanics_nl": self.mechanics_nl[-10:],
            "resource_model": _json_safe(self.resource_model),
            "solved_level_summaries": self.solved_level_summaries[-6:],
            "advisory_notes": self.advisory_notes[:700],
        }
        return _strip_local_object_ids(prompt)


@dataclass
class LevelMemoryV18:
    level_index: int = 0
    levels_completed_at_start: int = 0
    initial_scene: SceneSnapshot | None = None
    current_scene: SceneSnapshot | None = None
    pre_success_scene: SceneSnapshot | None = None
    local_bindings: dict[str, str] = field(default_factory=dict)
    current_plan: list[PlanStep] = field(default_factory=list)
    plan_cursor: int = 0
    plan_goal: str = ""
    recent_events: deque[CompactEvent] = field(default_factory=lambda: deque(maxlen=16))
    known_noops_by_state: dict[str, set[str]] = field(default_factory=dict)
    tried_actions_by_state: dict[str, set[str]] = field(default_factory=dict)
    transition_graph: dict[str, dict[str, str]] = field(default_factory=dict)
    controlled_object_id: str = ""
    tentative_controlled_object_id: str = ""
    actor_votes: Counter[str] = field(default_factory=Counter)
    walkable_color_votes: Counter[int] = field(default_factory=Counter)
    wall_color_votes: Counter[int] = field(default_factory=Counter)
    resource_state: dict[str, Any] = field(default_factory=dict)
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
    recent_state_hashes: deque[str] = field(default_factory=lambda: deque(maxlen=8))
    recent_action_keys: deque[str] = field(default_factory=lambda: deque(maxlen=8))
    initial_vlm_done: bool = False
    success_reflected: bool = False
    consecutive_nonterminal_resets: int = 0

    def active_step(self) -> PlanStep | None:
        while self.plan_cursor < len(self.current_plan) and self.current_plan[self.plan_cursor].status in {"done", "skipped", "failed"}:
            self.plan_cursor += 1
        return self.current_plan[self.plan_cursor] if self.plan_cursor < len(self.current_plan) else None

    def as_prompt(self, scene: SceneSnapshot | None = None) -> dict[str, Any]:
        state = scene.state_hash if scene else ""
        return {
            "level_index": self.level_index,
            "local_bindings": dict(self.local_bindings),
            "controlled_object_id": self.controlled_object_id,
            "tentative_controlled_object_id": self.tentative_controlled_object_id,
            "plan_goal": self.plan_goal[:320],
            "plan_cursor": self.plan_cursor,
            "current_plan": [s.as_prompt() for s in self.current_plan[:12]],
            "recent_events": [e.as_prompt() for e in list(self.recent_events)[-10:]],
            "known_noops_here": sorted(self.known_noops_by_state.get(state, set())),
            "tried_here": sorted(self.tried_actions_by_state.get(state, set())),
            "terrain_model": {"walkable_color_votes": dict(self.walkable_color_votes), "wall_color_votes": dict(self.wall_color_votes)},
            "resource_state": dict(self.resource_state),
            "actions_since_vlm": self.actions_since_vlm,
            "chunk_action_count": self.chunk_action_count,
            "vlm_calls_this_level": self.vlm_calls_this_level,
            "bottleneck_reason": self.bottleneck_reason,
            "notes_for_next_call": self.notes_for_next_call[:500],
        }


@dataclass
class RuntimeMemoryV18:
    game: GameMemoryV18 = field(default_factory=GameMemoryV18)
    level: LevelMemoryV18 = field(default_factory=LevelMemoryV18)
    next_event_id: int = 1


@dataclass
class VLMRequest:
    text_prompt: str
    current_rgb: Image.Image
    previous_rgb: Image.Image | None = None
    analysis_rgb: Image.Image | None = None
    max_new_tokens: int = 1100


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
                self.path = root / f"agent_v1_9_{int(time.time())}_{os.getpid()}.jsonl"
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
        self.log_event("exception", {"type": type(exc).__name__, "message": str(exc)[:500], "trace": traceback.format_exc()[-2000:]})


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
                if not torch.cuda.is_available() and not _bool_env(("ARC_V19_ALLOW_CPU_VLM", "ARC_V18_ALLOW_CPU_VLM", "ARC_V15_ALLOW_CPU_VLM"), False):
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
            messages = [{"role": "system", "content": [{"type": "text", "text": V18_SYSTEM_PROMPT}]}, {"role": "user", "content": content}]
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
        missing = [name for name, value in (("ARC_V19_VLM_API_BASE_URL", config.vlm_api_base_url), ("ARC_V19_VLM_API_KEY/INF_API_KEY", config.vlm_api_key), ("ARC_V19_VLM_API_MODEL", config.vlm_api_model)) if not value]
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
                messages=[{"role": "system", "content": V18_SYSTEM_PROMPT}, {"role": "user", "content": content}],
                max_tokens=request.max_new_tokens,
                temperature=0,
                timeout=self.config.vlm_api_timeout_s,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
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
            self.logger.log_event("vlm_error", {"reason": str(exc)[:500]})
            return None


def make_vlm_backend(config: AgentConfig, logger: DecisionLogger) -> VLMBackend:
    return OpenAICompatibleBackend(config, logger) if vlm_uses_remote_api(config) else Qwen35Backend(config, logger)


V18_SYSTEM_PROMPT = """You control an abstract ARC-AGI-3 grid game.
Use the raw image, annotated image, object table, and evidence memory together.
Object ids like O1/O2 are LOCAL to the current level. Game-level memory must not store bare O-ids; convert them to visual descriptors and roles.
At level start you may guess the goal, but prioritize testing special symbols/patterns and validating hypotheses.
Return exactly one JSON object. No markdown or prose outside JSON. Do not propose RESET unless terminal.
Plans must be short and include stop conditions."""


@dataclass
class V18VLMResult:
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


def _parse_payload(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    candidates = re.findall(r"```json\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE) + _balanced_objects(cleaned)
    if not candidates:
        candidates = [cleaned]
    for cand in sorted(candidates, key=len, reverse=True):
        for text in (cand.strip(), re.sub(r",\s*([}\]])", r"\1", cand.strip())):
            try:
                val = json.loads(text)
                if isinstance(val, dict):
                    return val
            except Exception:
                pass
    return None


def parse_vlm_result(raw: Any) -> V18VLMResult | None:
    payload = _parse_payload(raw)
    if payload is None:
        return V18VLMResult(raw_invalid_excerpt=_short(raw, 1500), mode="INVALID") if isinstance(raw, str) and _short(raw, 1) else None

    def dict_list(key: str, limit: int) -> list[dict[str, Any]]:
        value = payload.get(key)
        return [dict(x) for x in value[:limit] if isinstance(x, dict)] if isinstance(value, list) else []

    def str_list(key: str, limit: int) -> list[str]:
        value = payload.get(key)
        return [_short(x, 300) for x in value[:limit] if _short(x, 300)] if isinstance(value, list) else []

    rb = payload.get("role_bindings") if isinstance(payload.get("role_bindings"), dict) else {}
    return V18VLMResult(
        mode=_short(payload.get("mode"), 60),
        action_meaning_updates=dict_list("action_meaning_updates", 12),
        role_bindings={_short(k, 80): _short(v, 24).upper() for k, v in rb.items() if _short(k, 80) and _short(v, 24)},
        win_condition_update=dict(payload.get("win_condition_update")) if isinstance(payload.get("win_condition_update"), dict) else {},
        object_effect_updates=dict_list("object_effect_updates", 12),
        mechanics_updates=str_list("mechanics_updates", 12),
        resource_update=dict(payload.get("resource_update")) if isinstance(payload.get("resource_update"), dict) else {},
        plan_goal=_short(payload.get("plan_goal") or payload.get("goal"), 360),
        plan=dict_list("plan", 20),
        bottleneck_analysis=_short(payload.get("bottleneck_analysis"), 700),
        notes_for_next_call=_short(payload.get("notes_for_next_call"), 700),
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
    MAX_ACTIONS = _int_env(("ARC_V19_MAX_ACTIONS", "ARC_V18_MAX_ACTIONS", "ARC_V15_MAX_ACTIONS"), 240, 1, 10000)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        backend = kwargs.pop("backend", None)
        config = kwargs.pop("config", None)
        super().__init__(*args, **kwargs)
        self.config = config or AgentConfig.from_env()
        self.MAX_ACTIONS = self.config.max_actions
        self.observer = Observer(self.config.image_size)
        self.logger = DecisionLogger(self.config)
        self.backend = backend if backend is not None else make_vlm_backend(self.config, self.logger)
        self.memory = RuntimeMemoryV18()
        self._call_index = 0
        self._done = False

    @property
    def name(self) -> str:
        try:
            base = super().name
        except Exception:
            base = type(self).__name__
        return f"{base}.{self.config.agent_version}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return state_name(getattr(latest_frame, "state", None)) == "WIN" or self._done or self.memory.level.total_action_count >= self.MAX_ACTIONS or self.memory.level.consecutive_nonterminal_resets >= 50

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        self._call_index += 1
        state = state_name(getattr(latest_frame, "state", None))
        levels_completed = int(getattr(latest_frame, "levels_completed", 0) or 0)
        legal = normalize_legal_actions(getattr(latest_frame, "available_actions", None), self._env_action_space(), allow_env_fallback=(state == "NOT_PLAYED"))
        try:
            if self.memory.level.total_action_count >= self.MAX_ACTIONS:
                self._done = True
                scene = self.memory.level.current_scene
                if scene is None and state != "NOT_PLAYED":
                    scene = self.observer.scene_from_frame(latest_frame)
                if scene is not None:
                    selected = self._fallback_nonreset_action(scene, legal, "max actions reached; return a non-reset sentinel action")
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
            self._maybe_break_loop(scene)
            vlm_calls_at_turn_start = self.memory.level.vlm_calls_this_level
            selected = None
            if self.memory.level.bottleneck_reason:
                self._maybe_call_vlm(scene, transition, legal)
                selected = self._execute_next_plan_action(scene, legal)
            if selected is None:
                selected = self._deterministic_navigation_action(scene, legal)
            if selected is None:
                self._maybe_call_vlm(scene, transition, legal)
                selected = self._execute_next_plan_action(scene, legal)
            called_vlm_this_turn = self.memory.level.vlm_calls_this_level > vlm_calls_at_turn_start
            if selected is None and not called_vlm_this_turn and self.memory.level.actions_since_vlm >= 1 and self._should_call_vlm(scene, transition, force_bottleneck=True):
                self.memory.level.bottleneck_reason = self.memory.level.bottleneck_reason or "no_executable_plan_step"
                self._request_vlm_once(scene, transition, legal, VLMMode.BOTTLENECK)
                selected = self._execute_next_plan_action(scene, legal)
            if selected is None:
                selected = self._minimal_information_probe(scene, legal)
            called_vlm_this_turn = self.memory.level.vlm_calls_this_level > vlm_calls_at_turn_start
            if selected is None and not called_vlm_this_turn and self.memory.level.bottleneck_reason and self._should_call_vlm(scene, transition, force_bottleneck=True):
                self._request_vlm_once(scene, transition, legal, VLMMode.BOTTLENECK)
                selected = self._execute_next_plan_action(scene, legal)
            if selected is None:
                called_vlm_this_turn = self.memory.level.vlm_calls_this_level > vlm_calls_at_turn_start
                allow_guard = called_vlm_this_turn or not self._vlm_available() or self.memory.level.vlm_calls_this_level >= self.config.max_vlm_calls_per_level
                selected = self._fallback_nonreset_action(scene, legal, "safe non-reset fallback after planner/probe exhaustion", allow_guard=allow_guard)
            if selected is None and state not in {"WIN", "GAME_OVER", "NOT_PLAYED"} and self.memory.level.consecutive_nonterminal_resets >= 3:
                selected = self._force_nonreset_action(scene, legal)
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

    def _game_id(self) -> str:
        return str(getattr(self, "game_id", "unknown"))

    def _start_new_level(self, levels_completed: int, scene: SceneSnapshot, legal: Sequence[Any]) -> None:
        self.memory.level = LevelMemoryV18(level_index=levels_completed, levels_completed_at_start=levels_completed, initial_scene=scene, current_scene=scene, resource_state=self._resource_state_from_scene(scene))
        self.memory.level.recent_state_hashes.append(scene.state_hash)
        self._update_resource_model_from_scene(scene)
        self._seed_initial_plan(scene, legal)
        self.logger.log_event("new_level_v19", {"level": levels_completed, "state": scene.state_hash[:12], "objects": [o.track_id for o in scene.objects], "counter": scene.counter_value, "lives": scene.life_count})

    def _resource_state_from_scene(self, scene: SceneSnapshot) -> dict[str, Any]:
        return {"counter": scene.counter_value, "capacity": scene.counter_capacity, "ratio": scene.counter_ratio, "lives": scene.life_count}

    def _remember_current_state(self, scene: SceneSnapshot) -> None:
        level = self.memory.level
        level.current_scene = scene
        if not level.recent_state_hashes or level.recent_state_hashes[-1] != scene.state_hash:
            level.recent_state_hashes.append(scene.state_hash)
        level.resource_state = self._resource_state_from_scene(scene)

    def _maybe_break_loop(self, scene: SceneSnapshot) -> None:
        # 检测状态循环：近期同一状态高频出现时，若 VLM 不可用/耗尽，清空当前状态 known_noops
        # 给 navigation 重试被锁死动作的机会，打破 known_noops 固化导致的震荡。
        level = self.memory.level
        hashes = level.recent_state_hashes
        if len(hashes) < 6:
            return
        state_visits = Counter(hashes)
        max_visits = max(state_visits.values())
        if max_visits < 3:
            return
        if scene.state_hash in level.known_noops_by_state:
            cleared = level.known_noops_by_state.pop(scene.state_hash)
            self.logger.log_event("loop_break_v19", {"state": scene.state_hash[:12], "cleared_noops": sorted(cleared), "max_state_visits": max_visits, "vlm_available": self._vlm_available()})
            level.bottleneck_reason = "state_loop_break"

    def _process_pending_transition(self, current_scene: SceneSnapshot, latest_frame: Any, *, cross_level: bool = False) -> TransitionReport | None:
        pending = self.memory.level.pending_action
        if pending is None:
            return None
        # id() 相同时若状态也已变化，说明环境复用了帧对象但游戏已推进，仍需解析转移
        if id(latest_frame) == pending.source_frame_id and current_scene.state_hash == pending.scene_before.state_hash:
            return None
        report = self.observer.compare(pending.scene_before, current_scene)
        report.previous_rgb = pending.scene_before.rgb
        report.annotated_rgb = current_scene.annotated_rgb
        report.action_key = pending.action_key()
        report.action_source = pending.source
        self.memory.level.last_resolved_pending_action = pending
        self.memory.level.pending_action = None
        self._record_transition(pending, report, current_scene, cross_level=cross_level)
        return report

    def _transition_delta_dict(self, report: TransitionReport) -> dict[str, Any]:
        return {
            "changed_cell_count": report.changed_cell_count,
            "world_changed_cell_count": report.world_changed_cell_count,
            "effective_noop": report.effective_noop,
            "counter_delta": report.counter_delta,
            "life_delta": report.life_delta,
            "retry_detected": report.retry_detected,
            "moved_objects": report.moved_objects[:6],
            "transformed_objects": report.transformed_objects[:6],
            "appeared": report.appeared_object_ids[:6],
            "disappeared": report.disappeared_object_ids[:6],
            "controlled_candidate_id": report.controlled_candidate_id,
            "interaction_event": report.interaction_event,
        }

    def _classify_outcome(self, report: TransitionReport) -> str:
        if report.retry_detected:
            return "retry"
        if report.effective_noop:
            return "noop"
        if report.is_simple_translation or (report.controlled_candidate_id and report.moved_objects):
            return "movement"
        if report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids:
            return "transform"
        if report.counter_delta is not None or report.life_delta is not None:
            return "resource_delta"
        if report.interaction_event:
            return "interaction"
        return "state_change"

    def _record_transition(self, pending: PendingAction, report: TransitionReport, current_scene: SceneSnapshot, *, cross_level: bool = False) -> None:
        level = self.memory.level
        before_state, after_state, action_key = pending.scene_before.state_hash, current_scene.state_hash, pending.action_key()
        if not cross_level:
            level.tried_actions_by_state.setdefault(before_state, set()).add(action_key)
            if report.effective_noop:
                level.known_noops_by_state.setdefault(before_state, set()).add(action_key)
            else:
                level.transition_graph.setdefault(before_state, {})[action_key] = after_state
            self._update_action_meaning_from_transition(pending, report, current_scene)
            self._update_actor_and_terrain(pending, report, current_scene)
            self._update_plan_after_transition(pending, report, current_scene)
        event = CompactEvent(self.memory.next_event_id, level.level_index, level.total_action_count, action_key, pending.source, before_state, after_state, "level_advanced" if cross_level else self._classify_outcome(report), ("level_advanced_by=" + action_key) if cross_level else report.summary[:700], self._transition_delta_dict(report))
        self.memory.next_event_id += 1
        level.recent_events.append(event)
        level.recent_action_keys.append(action_key)
        level.recent_state_hashes.append(after_state)
        if report.retry_detected or (report.life_delta is not None and report.life_delta < 0):
            level.bottleneck_reason = "retry_or_life_loss_after_" + action_key
            level.current_plan = []
            level.plan_cursor = 0
        if (report.interaction_event or report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids) and not (report.controlled_candidate_id and report.moved_objects):
            level.chunk_action_count = self.config.max_chunk_steps
        self.logger.log_event("transition_v19", event.as_prompt())

    def _update_plan_after_transition(self, pending: PendingAction, report: TransitionReport, current_scene: SceneSnapshot) -> None:
        level = self.memory.level
        step = level.active_step()
        if step is None:
            return
        step.attempts += 1
        if report.retry_detected:
            step.status = "failed"
            level.bottleneck_reason = "plan_step_caused_retry"
            return
        if report.effective_noop and pending.source == "plan_executor":
            step.status = "failed"
            level.bottleneck_reason = "plan_step_noop"
            level.current_plan = []
            level.plan_cursor = 0
            return
        target_id = self._resolve_step_target(step)
        if target_id and self._target_reached(target_id, current_scene):
            step.status = "done"
            level.plan_cursor += 1
            return
        if report.interaction_event or report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids:
            step.status = "done"
            level.plan_cursor += 1
            return
        if step.step_type in {"probe_action", "probe_object", "click"} and not report.effective_noop:
            step.status = "done"
            level.plan_cursor += 1

    def _update_action_meaning_from_transition(self, pending: PendingAction, report: TransitionReport, current_scene: SceneSnapshot) -> None:
        name = pending.name
        if name == "RESET":
            return
        level = self.memory.level
        meaning = self.memory.game.action_meanings.setdefault(name, ActionMeaning(action=name))
        meaning.attempts += 1
        if report.effective_noop:
            meaning.noops += 1
        event_id = self.memory.next_event_id
        if event_id not in meaning.evidence_events:
            meaning.evidence_events.append(event_id)
            meaning.evidence_events = meaning.evidence_events[-12:]
        summary_nl = self._sanitize_game_text(report.summary[:220], current_scene)
        recent_states_before = list(level.recent_state_hashes)
        if name == "ACTION7" and not report.effective_noop and len(recent_states_before) >= 2 and current_scene.state_hash == recent_states_before[-2]:
            meaning.kind = "undo"
            meaning.meaning_nl = "returns to the previous recently visited state"
            meaning.confidence = max(meaning.confidence, 0.7)
            return
        if report.controlled_candidate_id and name != "ACTION6":
            move = next((m for m in report.moved_objects if m["object_id"] == report.controlled_candidate_id), None)
            if move is not None:
                vec = (int(move["dx"]), int(move["dy"]))
                meaning.vector = vec
                meaning.kind = "movement"
                if report.is_simple_translation:
                    meaning.meaning_nl = f"moves the controlled object by vector {vec}"
                    meaning.confidence = max(meaning.confidence, 0.72)
                else:
                    meaning.meaning_nl = f"moves the controlled object by vector {vec} with simultaneous scene changes"
                    meaning.confidence = max(meaning.confidence, 0.62)
        elif report.interaction_event or report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids:
            meaning.kind = "click_or_select" if name == "ACTION6" else "interact_or_transform"
            meaning.meaning_nl = "coordinate click/select can change the scene" if name == "ACTION6" else f"causes interaction or structural change: {summary_nl}"
            meaning.confidence = max(meaning.confidence, 0.58)
        elif report.counter_delta is not None and report.counter_delta != 0:
            meaning.resource_delta = report.counter_delta
            meaning.meaning_nl = meaning.meaning_nl or f"affects visible resource counter by {report.counter_delta}"
            meaning.confidence = max(meaning.confidence, 0.45)
        elif report.effective_noop and meaning.attempts >= 2 and meaning.noops >= meaning.attempts:
            meaning.kind = "unknown_or_blocked"
            meaning.meaning_nl = "no visible effect in tested states so far"
            meaning.confidence = max(meaning.confidence, 0.25)

    def _update_actor_and_terrain(self, pending: PendingAction, report: TransitionReport, current_scene: SceneSnapshot) -> None:
        level = self.memory.level
        if report.controlled_candidate_id:
            level.actor_votes[report.controlled_candidate_id] += 1
            best, votes = level.actor_votes.most_common(1)[0]
            if votes >= 2 or report.is_simple_translation:
                level.controlled_object_id = best
                level.local_bindings.setdefault("actor", best)
            else:
                level.tentative_controlled_object_id = best
            self._learn_walkable_colour(pending.scene_before, current_scene, report.controlled_candidate_id)

    def _learn_walkable_colour(self, before: SceneSnapshot, after: SceneSnapshot, oid: str) -> None:
        old_actor, new_actor = before.object_by_id(oid), after.object_by_id(oid)
        if old_actor is None or new_actor is None or old_actor.bbox == new_actor.bbox:
            return
        nx0, ny0, nx1, ny1 = new_actor.bbox
        for y in range(old_actor.bbox[1], old_actor.bbox[3] + 1):
            for x in range(old_actor.bbox[0], old_actor.bbox[2] + 1):
                if nx0 <= x <= nx1 and ny0 <= y <= ny1 or (x, y) in after.volatile_cells:
                    continue
                color = after.grid[y][x]
                if color != after.background_candidate:
                    self.memory.level.walkable_color_votes[color] += 1

    def _update_resource_model_from_scene(self, scene: SceneSnapshot) -> None:
        res = self.memory.game.resource_model
        if scene.counter_value is not None:
            res.visible_bar, res.last_value, res.capacity = True, scene.counter_value, scene.counter_capacity
            res.description_nl = res.description_nl or "A visible resource/step counter exists; planning should avoid wasting actions."
        if scene.life_count is not None:
            res.last_lives = scene.life_count

    def _record_game_over(self, scene: SceneSnapshot, transition: TransitionReport | None) -> None:
        level = self.memory.level
        level.awaiting_reset = True
        level.current_plan = []
        level.plan_cursor = 0
        level.bottleneck_reason = "game_over"
        # 清空本关 known_noops，重开后给 navigation 重新探索机会，避免旧固化记忆导致重陷循环
        level.known_noops_by_state.clear()
        self.logger.log_event("game_over_v19", {"level": level.level_index, "steps": level.total_action_count, "counter": scene.counter_value, "lives": scene.life_count, "transition": transition.summary if transition else None})
        self.logger.log_event("failure_model_updated", {"level": level.level_index, "reason": "game_over", "steps": level.total_action_count})

    def _record_level_success(self, scene: SceneSnapshot, new_levels_completed: int) -> None:
        level = self.memory.level
        if level.initial_scene is None or level.success_reflected:
            return
        pending = level.last_resolved_pending_action or level.pending_action
        pre_success = pending.scene_before if pending else level.pre_success_scene
        outcome = {"level": level.level_index, "steps": level.total_action_count, "new_levels_completed": new_levels_completed, "win_action": pending.action_key() if pending else "unknown", "recent_events": [e.as_prompt() for e in list(level.recent_events)[-10:]]}
        if self._vlm_available() and level.vlm_calls_this_level < self.config.max_vlm_calls_per_level:
            self._request_vlm_success_reflect(scene, outcome, pre_success)
        else:
            self._deterministic_success_reflect(scene, outcome, pre_success)
        level.success_reflected = True
        level.pending_action = None
        level.current_plan = []
        self.logger.log_event("level_success_v19", outcome)
        self.logger.log_event("success_consolidated", {"level": level.level_index, "new_levels_completed": new_levels_completed, "win_action": outcome.get("win_action")})

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
        try:
            result = parse_vlm_result(self.backend.decide(request))
        except Exception as exc:
            self.logger.log_exception(exc)
            result = None
        if result is not None:
            result.mode = VLMMode.SUCCESS_REFLECT.value
            self._apply_vlm_result(result, scene)
        else:
            self._deterministic_success_reflect(scene, outcome, pre_success)

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
    def _vlm_available(self) -> bool:
        return bool(self.config.enable_vlm and getattr(self.backend, "available", False))

    def _should_call_vlm(self, scene: SceneSnapshot, transition: TransitionReport | None, *, force_bottleneck: bool = False) -> bool:
        level = self.memory.level
        if not self._vlm_available() or level.vlm_calls_this_level >= self.config.max_vlm_calls_per_level:
            return False
        if force_bottleneck:
            return True
        if not level.initial_vlm_done:
            return True
        if level.bottleneck_reason:
            return level.actions_since_vlm >= max(0, self.config.vlm_min_action_gap - 1)
        if transition is not None and (transition.interaction_event or transition.retry_detected or transition.transformed_objects or transition.appeared_object_ids or transition.disappeared_object_ids) and level.actions_since_vlm >= self.config.vlm_min_action_gap:
            return True
        if level.chunk_action_count >= self.config.max_chunk_steps and level.actions_since_vlm >= self.config.vlm_min_action_gap:
            return True
        if level.active_step() is None and level.actions_since_vlm >= self.config.vlm_min_action_gap:
            return True
        # 状态循环检测：最近 8 步内同一状态出现 >=3 次视为陷入循环，主动求救
        if len(level.recent_state_hashes) >= 6:
            state_counts = Counter(level.recent_state_hashes)
            if any(n >= 3 for n in state_counts.values()):
                return True
        return False

    def _maybe_call_vlm(self, scene: SceneSnapshot, transition: TransitionReport | None, legal: Sequence[Any]) -> None:
        if not self._should_call_vlm(scene, transition):
            return
        level = self.memory.level
        if not level.initial_vlm_done:
            mode = VLMMode.LEVEL_INIT
        elif level.bottleneck_reason:
            mode = VLMMode.BOTTLENECK
        else:
            mode = VLMMode.EVALUATE_CHUNK
        self._request_vlm_once(scene, transition, legal, mode)

    def _request_vlm_once(self, scene: SceneSnapshot, transition: TransitionReport | None, legal: Sequence[Any], mode: VLMMode) -> V18VLMResult | None:
        level = self.memory.level
        if not self._vlm_available() or level.vlm_calls_this_level >= self.config.max_vlm_calls_per_level:
            return None
        prompt = self._build_vlm_prompt(scene, transition, legal, mode.value)
        request = VLMRequest(prompt, scene.rgb or render_grid(scene.grid, self.config.image_size), None if transition is None else transition.previous_rgb, scene.annotated_rgb, self.config.vlm_max_new_tokens)
        level.vlm_calls_this_level += 1
        self.memory.game.vlm_calls_total += 1
        level.last_vlm_mode = mode.value
        self.logger.log_event("vlm_call_v19", {"mode": mode.value, "level": level.level_index, "call": level.vlm_calls_this_level, "bottleneck": level.bottleneck_reason})
        try:
            result = parse_vlm_result(self.backend.decide(request))
        except Exception as exc:
            self.logger.log_exception(exc)
            return None
        if result is None:
            return None
        if result.mode == "INVALID":
            level.notes_for_next_call = _short(f"Invalid VLM output excerpt: {result.raw_invalid_excerpt}", 700)
            return result
        # 仅在拿到有效结果后才重置间隔计数，避免无效调用白白清零 actions_since_vlm 拖慢重试
        level.actions_since_vlm = 0
        level.chunk_action_count = 0
        result.mode = mode.value
        self._apply_vlm_result(result, scene)
        self.logger.log_event("vlm_result_v19", {"mode": mode.value, "plan_steps": len(result.plan), "role_bindings": result.role_bindings, "has_win_update": bool(result.win_condition_update)})
        return result

    def _build_vlm_prompt(self, scene: SceneSnapshot, transition: TransitionReport | None, legal: Sequence[Any], mode: str, extra: dict[str, Any] | None = None) -> str:
        legal_names = [action_name(a) for a in legal]
        objs = [self._object_prompt(o, scene) for o in sorted(scene.objects, key=lambda x: -x.salience)[:self.config.max_prompt_objects]]
        transition_payload = None
        if transition is not None:
            transition_payload = {"action": transition.action_key, "source": transition.action_source, "summary": transition.summary, **self._transition_delta_dict(transition)}
        payload = {
            "task": "Maintain lean evidence-backed memory and return a short executable plan.",
            "arc_action_reference": {
                "RESET": "Start/restart. v1.9 must not use it unless terminal or no non-reset legal action exists.",
                "ACTION1-ACTION5-ACTION7": "Simple actions; meaning varies per game and must be inferred.",
                "ACTION6": "Coordinate action requiring x,y. Prefer target_object_id/click_object; controller chooses click point.",
                "principle": "At level start, infer possible goal from image but first test special symbols/patterns.",
            },
            "mode": mode,
            "game_id": self._game_id(),
            "image_order": ["CURRENT_RAW", "CURRENT_ANNOTATED"] if transition is None else ["BEFORE_RAW", "CURRENT_RAW", "CURRENT_ANNOTATED"],
            "legal_actions_now": legal_names,
            "observer": {
                "scene_summary": scene.summary[:5200],
                "objects": objs,
                "framed_template_relations": list(scene.template_relations),
                "controlled_object_id": self.memory.level.controlled_object_id or None,
                "resource": {"counter": scene.counter_value, "capacity": scene.counter_capacity, "ratio": scene.counter_ratio, "lives": scene.life_count},
            },
            "last_transition": transition_payload,
            "game_memory": self.memory.game.as_prompt(),
            "level_memory": self.memory.level.as_prompt(scene),
            "requirements": [
                "Game-level memory must use natural language plus visual descriptors, never bare O1/O2/O3 ids.",
                "Level role_bindings may use current local object ids.",
                "At LEVEL_INIT: guess possible goals, prioritize special patterns/symbols and validation probes.",
                "At EVALUATE_CHUNK: update action meanings/object effects, then continue or revise a short plan.",
                "At BOTTLENECK: explain failure and produce a new short plan; do not ask for RESET unless terminal.",
                "At SUCCESS_REFLECT: compare initial and pre-success frames; write transferable win condition without O-ids.",
                "plan should contain 2-4 connected steps when a path to the goal is identifiable; return a single step only when genuinely uncertain. Avoid actions that return to recently visited states (see loop_warning if present).",
            ],
            "output_schema": {
                "mode": mode,
                "action_meaning_updates": [{"action": "ACTION1", "meaning_nl": "moves actor up", "kind": "movement|interact|click_or_select|undo|resource|unknown", "vector": [0, -1], "confidence": 0.0, "evidence": "cite event/observation"}],
                "role_bindings": {"actor": "O3", "target_frame": "O1", "status_pattern": "O2", "transformer": "O4"},
                "win_condition_update": {"description_nl": "transferable rule with no O-ids", "visual_roles": {"target_frame": {"shape_label": "frame_with_inner_pattern", "inner_pattern": "..."}}, "confidence": 0.0, "evidence": "why"},
                "object_effect_updates": [{"local_object_id": "O4", "visual_descriptor": {}, "effect_nl": "changes target pattern", "confidence": 0.0, "evidence": "event or visual clue"}],
                "mechanics_updates": ["short natural-language mechanic claim"],
                "resource_update": {"description_nl": "resource/life behavior", "confidence": 0.0},
                "plan_goal": "short current-level goal",
                "plan": [
                    {"type": "move_to", "action": "ACTION1", "target_role": "target_frame", "target_object_id": "O1", "purpose": "approach the target frame", "stop_condition": "stop when adjacent to target", "max_attempts": 2},
                    {"type": "click", "action": "ACTION6", "target_role": "transformer", "target_object_id": "O4", "purpose": "activate the transformer", "stop_condition": "stop after transform", "max_attempts": 1},
                    {"type": "interact", "action": "ACTION5", "target_role": "target_frame", "target_object_id": "O1", "purpose": "complete the goal", "stop_condition": "stop after win or major change", "max_attempts": 1}
                ],
                "bottleneck_analysis": "",
                "notes_for_next_call": "",
            },
        }
        state_visits = Counter(self.memory.level.recent_state_hashes)
        if len(self.memory.level.recent_state_hashes) >= 6 and max(state_visits.values()) >= 3:
            payload["loop_warning"] = {"repeated_states": {k[:12]: v for k, v in state_visits.items() if v >= 3}, "advice": "You are stuck in a state loop; propose a different action, target, or plan to break it."}
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=True, default=str)

    def _object_prompt(self, obj: ObjectObservation, scene: SceneSnapshot) -> dict[str, Any]:
        return {"id": obj.track_id, "descriptor": _json_safe(self._visual_descriptor(obj, scene)), "bbox": obj.bbox, "centroid": obj.centroid, "area": obj.area, "salience": round(obj.salience, 3)}

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

    def _apply_vlm_result(self, result: V18VLMResult, scene: SceneSnapshot) -> None:
        level = self.memory.level
        valid_ids = {o.track_id for o in scene.objects}
        for role, oid in result.role_bindings.items():
            if oid in valid_ids:
                level.local_bindings[_short(role, 80)] = oid
                if role.lower() in {"actor", "player", "controlled_object"}:
                    level.controlled_object_id = oid
        for upd in result.action_meaning_updates:
            action = _short(upd.get("action"), 40).upper()
            if not re.fullmatch(r"ACTION\d+", action):
                continue
            meaning = self.memory.game.action_meanings.setdefault(action, ActionMeaning(action))
            text = self._sanitize_game_text(_short(upd.get("meaning_nl") or upd.get("meaning"), 400), scene)
            conf = _clamp01(upd.get("confidence", 0.0))
            if text and conf >= meaning.confidence - 0.05:
                meaning.meaning_nl = text
            kind = _short(upd.get("kind"), 40).lower()
            if kind:
                meaning.kind = kind
            vec = upd.get("vector")
            if isinstance(vec, list) and len(vec) == 2:
                try:
                    meaning.vector = (int(vec[0]), int(vec[1]))
                except Exception:
                    pass
            meaning.confidence = max(meaning.confidence, conf)
        win = result.win_condition_update
        if win:
            desc = self._sanitize_game_text(_short(win.get("description_nl") or win.get("description") or win.get("claim"), 700), scene)
            if desc:
                mem = self.memory.game.win_condition or WinConditionMemory()
                conf = _clamp01(win.get("confidence", 0.0))
                if conf >= mem.confidence - 0.08 or not mem.description_nl:
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
            effect = self._sanitize_game_text(_short(upd.get("effect_nl") or upd.get("effect"), 600), scene)
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
            if claim and claim not in self.memory.game.mechanics_nl:
                self.memory.game.mechanics_nl.append(claim)
                self.memory.game.mechanics_nl = self.memory.game.mechanics_nl[-16:]
        if result.resource_update:
            desc = self._sanitize_game_text(_short(result.resource_update.get("description_nl") or result.resource_update.get("claim"), 500), scene)
            if desc:
                self.memory.game.resource_model.description_nl = desc
        if result.plan_goal:
            level.plan_goal = _short(result.plan_goal, 400)
        if result.plan:
            level.current_plan = self._plan_steps_from_vlm(result.plan, scene)
            level.plan_cursor = 0
            level.bottleneck_reason = ""
        if result.bottleneck_analysis:
            level.notes_for_next_call = _short(result.bottleneck_analysis, 700)
        if result.notes_for_next_call:
            level.notes_for_next_call = _short(result.notes_for_next_call, 700)
        if result.mode == VLMMode.LEVEL_INIT.value:
            level.initial_vlm_done = True

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
        self.memory.game.object_effects = self.memory.game.object_effects[-16:]

    def _plan_steps_from_vlm(self, raw_steps: list[dict[str, Any]], scene: SceneSnapshot) -> list[PlanStep]:
        valid_ids = {o.track_id for o in scene.objects}
        steps: list[PlanStep] = []
        for raw in raw_steps[:self.config.max_plan_steps]:
            step_type = _short(raw.get("type") or raw.get("step_type") or raw.get("intent"), 40).lower()
            if step_type in {"navigate_to_object", "move"}:
                step_type = "move_to"
            if step_type in {"click_object", "test_object"}:
                step_type = "click" if _short(raw.get("action"), 40).upper() == "ACTION6" else "probe_object"
            if step_type not in {"probe_action", "probe_object", "move_to", "interact", "click", "wait"}:
                if raw.get("action"):
                    step_type = "probe_action"
                else:
                    continue
            target = _short(raw.get("target_object_id") or raw.get("object_id"), 24).upper()
            if target and target not in valid_ids:
                target = ""
            action = _short(raw.get("action") or raw.get("name"), 40).upper()
            if action in {"RESET", "NAVIGATE", "CLICK", "MOVE", "PATHFIND", "GO"}:
                action = ""
            max_attempts = self._coerce_int(raw.get("max_attempts")) or 2
            steps.append(PlanStep(step_type, action, _short(raw.get("target_role") or raw.get("role"), 80), target, _short(raw.get("purpose") or raw.get("why"), 300), _short(raw.get("stop_condition") or raw.get("expected_change"), 300), raw.get("expected_predicates") if isinstance(raw.get("expected_predicates"), list) else [], "pending", 0, max(1, min(4, max_attempts)), dict(raw)))
        return steps

    def _seed_initial_plan(self, scene: SceneSnapshot, legal: Sequence[Any]) -> None:
        steps: list[PlanStep] = []
        legal_names = {action_name(a).upper() for a in legal}
        special_objects = [o for o in sorted(scene.objects, key=lambda x: -x.salience) if not o.near_edge][:3]
        if "ACTION6" in legal_names:
            for obj in special_objects:
                steps.append(PlanStep("probe_object", target_object_id=obj.track_id, purpose=f"test special visible object {obj.track_id}", stop_condition="stop after structural change or no-op", max_attempts=1))
        for name in simple_action_names(legal):
            m = self.memory.game.action_meanings.get(name)
            if m is None or m.confidence < 0.4:
                steps.append(PlanStep("probe_action", action=name, purpose=f"learn what {name} does", stop_condition="stop after visible movement/change/noop", max_attempts=1))
            if len([s for s in steps if s.step_type == "probe_action"]) >= 3:
                break
        if "ACTION6" not in legal_names:
            for obj in special_objects:
                steps.append(PlanStep("probe_object", target_object_id=obj.track_id, purpose=f"test special visible object {obj.track_id}", stop_condition="stop after structural change or no-op", max_attempts=1))
        self.memory.level.current_plan = steps[:self.config.max_plan_steps]
        self.memory.level.plan_goal = "initial exploration: test salient/special patterns, then ground actions"

    def _execute_next_plan_action(self, scene: SceneSnapshot, legal: Sequence[Any]) -> tuple[Any, dict[str, Any], str] | None:
        level = self.memory.level
        # 跳过被 _proposal_for_step 主动标记为 done/skipped 的步骤（如 move_to 已抵达目标），
        # 继续尝试下一个 plan 步，避免误触 bottleneck VLM 与无谓的二次 VLM 调用。
        while True:
            step = level.active_step()
            if step is None:
                return None
            proposal = self._proposal_for_step(step, scene, legal)
            if proposal is None:
                if step.status in {"done", "skipped"}:
                    continue
                if step.attempts >= step.max_attempts:
                    step.status = "failed"
                    level.bottleneck_reason = f"cannot_execute_step:{step.step_type}:{step.target_role or step.target_object_id or step.action}"
                return None
            action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if action is None:
                step.attempts += 1
                if step.attempts >= step.max_attempts:
                    step.status = "failed"
                    level.bottleneck_reason = f"invalid_action_for_step:{step.step_type}"
                return None
            return action, proposal, "plan_executor"

    def _proposal_for_step(self, step: PlanStep, scene: SceneSnapshot, legal: Sequence[Any]) -> dict[str, Any] | None:
        legal_by = {action_name(a).upper(): a for a in legal}
        target_id = self._resolve_step_target(step)
        target = scene.object_by_id(target_id)
        if step.step_type == "wait":
            return None
        if step.step_type == "probe_action":
            name = step.action if step.action in legal_by else self._next_unknown_simple_action(legal, scene)
            if not name or name in self.memory.level.known_noops_by_state.get(scene.state_hash, set()):
                return None
            return {"name": name, "purpose": step.purpose or f"probe {name}", "expected_change": step.stop_condition or "learn action effect", "expected_predicates": step.expected_predicates}
        if step.step_type in {"click", "probe_object"}:
            if target is None:
                target = self._choose_special_object(scene)
                if target is None:
                    return None
                target_id = target.track_id
            if "ACTION6" in legal_by:
                x, y = self._object_click_point(scene, target)
                return {"name": "ACTION6", "x": x, "y": y, "target_object_id": target_id, "purpose": step.purpose or f"click/test {target_id}", "expected_change": step.stop_condition or "observe object response", "expected_predicates": step.expected_predicates}
            if self._target_reached(target_id, scene):
                interact = self._best_interaction_action(legal)
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
            name = step.action if step.action in legal_by else self._best_interaction_action(legal)
            if name:
                return {"name": name, "target_object_id": target_id, "purpose": step.purpose or f"interact with {target_id}", "expected_change": step.stop_condition or "interaction/change"}
        return None

    def _resolve_step_target(self, step: PlanStep) -> str:
        if step.target_object_id:
            return step.target_object_id
        return self.memory.level.local_bindings.get(step.target_role, "") if step.target_role else ""

    def _next_unknown_simple_action(self, legal: Sequence[Any], scene: SceneSnapshot) -> str:
        noops = self.memory.level.known_noops_by_state.get(scene.state_hash, set())
        tried = self.memory.level.tried_actions_by_state.get(scene.state_hash, set())
        for name in simple_action_names(legal):
            if name in noops:
                continue
            m = self.memory.game.action_meanings.get(name)
            if m is None or m.confidence < 0.55:
                return name
        for name in simple_action_names(legal):
            if name not in noops and name not in tried:
                return name
        return ""

    def _best_interaction_action(self, legal: Sequence[Any]) -> str:
        scored: list[tuple[float, str]] = []
        for name in simple_action_names(legal):
            m = self.memory.game.action_meanings.get(name)
            score = 0.35 if name == "ACTION5" else 0.0
            if name == "ACTION7":
                score -= 0.75
            if m:
                if m.kind in {"interact_or_transform", "resource"}:
                    score += 2.0 + m.confidence
                elif m.kind == "click_or_select":
                    score += 0.8 + m.confidence
                elif m.kind == "undo":
                    score -= 2.5
                elif m.kind == "unknown_or_blocked":
                    score -= 1.0
                if m.vector:
                    score -= 0.8
                if m.attempts >= 2 and m.noops >= m.attempts:
                    score -= 1.0
            scored.append((score, name))
        if not scored:
            return ""
        scored.sort(reverse=True)
        return scored[0][1]

    def _choose_special_object(self, scene: SceneSnapshot) -> ObjectObservation | None:
        actor = scene.object_by_id(self.memory.level.controlled_object_id)
        candidates = [o for o in scene.objects if o.track_id != (actor.track_id if actor else "") and not o.near_edge] or [o for o in scene.objects if o.track_id != (actor.track_id if actor else "")]
        def score(o: ObjectObservation) -> float:
            s = o.salience + (10 if o.frame_color is not None else 0) + (6 if o.inner_pattern else 0) + (4 if o.area <= 120 and o.frame_color is None else 0)
            for effect in self.memory.game.object_effects:
                if self._descriptor_matches(effect.visual_descriptor, o):
                    s += 8 * effect.confidence
            return s
        return max(candidates, key=score, default=None)

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
        level = self.memory.level
        path = self._plan_path_to_object(scene, target_id, legal)
        if path:
            after_state = level.transition_graph.get(scene.state_hash, {}).get(path[0])
            state_visits = Counter(level.recent_state_hashes)
            # 路径第一步若会回到近期高频状态，弃用路径转 greedy（带回访惩罚），避免两状态震荡
            if not (after_state and state_visits.get(after_state, 0) >= 2):
                return {"name": path[0], "target_object_id": target_id, "purpose": step.purpose or f"move toward {target_id}", "expected_change": step.stop_condition or "movement toward target", "nav_path_len": len(path)}
        greedy = self._greedy_move_toward_target(scene, target_id, legal)
        if greedy:
            return {"name": greedy, "target_object_id": target_id, "purpose": step.purpose or f"try moving toward {target_id}", "expected_change": step.stop_condition or "greedy movement toward target", "nav_path_len": 0, "nav_fallback": "greedy"}
        return None

    def _target_reached(self, target_id: str, scene: SceneSnapshot) -> bool:
        actor, target = scene.object_by_id(self.memory.level.controlled_object_id), scene.object_by_id(target_id)
        if actor is None or target is None:
            return False
        tx, ty = int(round(target.centroid[0])), int(round(target.centroid[1]))
        return actor.bbox[0] <= tx <= actor.bbox[2] and actor.bbox[1] <= ty <= actor.bbox[3]

    def _action_vectors(self, actor: ObjectObservation, legal: Sequence[Any]) -> dict[str, tuple[int, int]]:
        legal_names = {action_name(a).upper() for a in legal}
        vectors = {name: m.vector for name, m in self.memory.game.action_meanings.items() if name in legal_names and m.vector is not None and m.confidence >= 0.4}
        weak = {"ACTION1": (0, -actor.height), "ACTION2": (0, actor.height), "ACTION3": (-actor.width, 0), "ACTION4": (actor.width, 0)}
        for name, vec in weak.items():
            if name in legal_names and name not in vectors:
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
                if name not in vectors:
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
        actor = scene.object_by_id(level.controlled_object_id)
        legal_names = {action_name(a).upper() for a in legal}
        learned_vectors = [
            name
            for name, meaning in self.memory.game.action_meanings.items()
            if name in legal_names and meaning.vector is not None and meaning.confidence >= 0.4
        ]
        # checkpoint/loop 失败后优先回到 VLM，不让确定性导航继续消耗动作。
        if level.bottleneck_reason and self._vlm_available() and level.vlm_calls_this_level < self.config.max_vlm_calls_per_level:
            return None
        # ACTION6 点击导航：无移动动作但有 ACTION6 时（常见于点击类关卡），直接点击目标对象，
        # 覆盖 ACTION6-only 游戏（无移动向量导致传统 navigation 失效的场景）
        if level.initial_vlm_done and "ACTION6" in legal_names and len(learned_vectors) < 2:
            target_id = self._navigation_target_id(scene)
            blocked = set(level.known_noops_by_state.get(scene.state_hash, set())) | set(level.tried_actions_by_state.get(scene.state_hash, set()))
            picked = None
            if target_id:
                target = scene.object_by_id(target_id)
                if target is not None:
                    x, y = self._object_click_point(scene, target)
                    if f"ACTION6:{x},{y}" not in blocked:
                        picked = (x, y, target_id)
            if picked is None:
                picked = self._first_untried_action6_target(scene, blocked)
            if picked is not None:
                x, y, tid = picked
                proposal = {"name": "ACTION6", "x": x, "y": y, "target_object_id": tid, "purpose": f"deterministically click target {tid or target_id}", "expected_change": "trigger target response or explore fresh coordinate"}
                action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
                if action is not None:
                    return action, proposal, "deterministic_click"
        if actor is None or len(learned_vectors) < 2:
            return None
        target_id = self._navigation_target_id(scene)
        if not target_id:
            return None
        if self._target_reached(target_id, scene):
            interact = self._best_interaction_action(legal)
            if interact:
                proposal = {"name": interact, "target_object_id": target_id, "purpose": f"interact after reaching {target_id}", "expected_change": "finish or reveal goal effect"}
                action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
                if action is not None:
                    return action, proposal, "deterministic_navigation"
            return None
        step = PlanStep("move_to", target_object_id=target_id, purpose=f"deterministically move controlled object toward {target_id}", stop_condition="stop when target is reached", max_attempts=1)
        proposal = self._navigation_proposal_to_target(target_id, scene, legal, step)
        if proposal is None:
            return None
        action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
        if action is None:
            return None
        return action, proposal, "deterministic_navigation"

    def _navigation_target_id(self, scene: SceneSnapshot) -> str:
        actor_id = self.memory.level.controlled_object_id
        for role in ("target_frame", "goal", "target", "exit", "transformer", "status_pattern"):
            oid = self.memory.level.local_bindings.get(role, "")
            if oid and oid != actor_id and scene.object_by_id(oid) is not None:
                return oid
        win = self.memory.game.win_condition
        if win is not None:
            for desc in win.visual_roles.values():
                obj = self._descriptor_target_candidate(scene, desc, actor_id)
                if obj is not None:
                    return obj.track_id
        obj = self._choose_special_object(scene)
        return obj.track_id if obj is not None and obj.track_id != actor_id else ""

    def _descriptor_target_candidate(self, scene: SceneSnapshot, desc: VisualDescriptor, actor_id: str) -> ObjectObservation | None:
        candidates = [o for o in scene.objects if o.track_id != actor_id and self._descriptor_matches(desc, o)]
        return max(candidates, key=lambda o: o.salience, default=None)

    def _greedy_move_toward_target(self, scene: SceneSnapshot, target_id: str, legal: Sequence[Any]) -> str:
        actor = scene.object_by_id(self.memory.level.controlled_object_id)
        target = scene.object_by_id(target_id)
        if actor is None or target is None:
            return ""
        vectors = self._action_vectors(actor, legal)
        if not vectors:
            return ""
        level = self.memory.level
        noops = level.known_noops_by_state.get(scene.state_hash, set())
        recent = set(level.recent_action_keys)
        state_visits = Counter(level.recent_state_hashes)
        trans_graph = level.transition_graph.get(scene.state_hash, {})
        ax, ay = actor.centroid
        tx, ty = target.centroid
        base_dist = abs(ax - tx) + abs(ay - ty)
        floor = self._walkable_colors(scene, actor)
        scored: list[tuple[float, str]] = []
        for name, (dx, dy) in vectors.items():
            if name in noops:
                continue
            nxt = (actor.bbox[0] + dx, actor.bbox[1] + dy)
            dist = abs((ax + dx) - tx) + abs((ay + dy) - ty)
            score = base_dist - dist
            if self._position_passable(scene, nxt, actor.width, actor.height, floor, actor, target):
                score += 0.75
            if name in recent:
                score -= 1.25
            after_state = trans_graph.get(name)
            if after_state:
                visits = state_visits.get(after_state, 0)
                if visits >= 2:
                    score -= 2.5 * visits
            scored.append((score, name))
        if not scored:
            return ""
        def _action_num(name: str) -> int:
            m = re.search(r"\d+", name)
            return int(m.group(0)) if m else 0
        scored.sort(key=lambda item: (-item[0], _action_num(item[1])))
        return scored[0][1] if scored[0][0] > -0.75 else ""

    def _first_untried_action6_target(self, scene: SceneSnapshot, noops: set[str]) -> tuple[int, int, str] | None:
        # 先按显著度遍历对象取其点击点；全部试过则退回扫描网格非背景/非结构色格子，
        # 保证 ACTION6 唯一合法时仍能找到新坐标，避免单坐标 noop 死锁。
        for obj in sorted(scene.objects, key=lambda o: -o.salience):
            x, y = self._object_click_point(scene, obj)
            if f"ACTION6:{x},{y}" not in noops:
                return x, y, obj.track_id
        bg = scene.background_candidate
        structural = set(scene.structural_colors)
        for yy in range(scene.height):
            for xx in range(scene.width):
                c = scene.grid[yy][xx]
                if c == bg or c in structural:
                    continue
                if f"ACTION6:{xx},{yy}" not in noops:
                    return xx, yy, ""
        return None

    def _fallback_nonreset_action(self, scene: SceneSnapshot, legal: Sequence[Any], purpose: str, *, allow_guard: bool = True) -> tuple[Any, dict[str, Any], str] | None:
        level = self.memory.level
        noops = set(level.known_noops_by_state.get(scene.state_hash, set()))
        tried = set(level.tried_actions_by_state.get(scene.state_hash, set()))
        blocked = noops | tried
        recent = set(level.recent_action_keys)
        for name in simple_action_names(legal):
            if name in blocked or name in recent:
                continue
            meaning = self.memory.game.action_meanings.get(name)
            if meaning is not None and meaning.confidence >= 0.55 and meaning.kind not in {"unknown", "unknown_or_blocked"}:
                continue
            proposal = {"name": name, "purpose": purpose, "expected_change": "one-step safe probe of an unverified action"}
            made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if made is not None:
                return made, proposal, "safe_probe_fallback"
        if "ACTION6" in {action_name(a).upper() for a in legal}:
            picked = self._first_untried_action6_target(scene, blocked)
            if picked is not None:
                x, y, target_id = picked
                proposal = {"name": "ACTION6", "x": x, "y": y, "target_object_id": target_id, "purpose": purpose, "expected_change": "one-step safe click probe on a fresh coordinate"}
                made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
                if made is not None:
                    return made, proposal, "safe_probe_fallback"
        if not allow_guard:
            level.bottleneck_reason = level.bottleneck_reason or "fallback_waiting_for_vlm"
            return None
        for action in legal:
            name = action_name(action).upper()
            if name == "RESET" or name in noops:
                continue
            if name == "ACTION6":
                picked = self._first_untried_action6_target(scene, set(noops) | set(tried))
                if picked is None:
                    continue
                x, y, target_id = picked
                proposal = {"name": "ACTION6", "x": x, "y": y, "target_object_id": target_id, "purpose": purpose, "expected_change": "non-reset guard after VLM/probe exhaustion"}
                made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
                if made is not None:
                    return made, proposal, "nonreset_guard"
                continue
            proposal = {"name": name, "purpose": purpose, "expected_change": "non-reset guard after VLM/probe exhaustion"}
            made = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if made is not None:
                return made, proposal, "nonreset_guard"
        return None

    def _force_nonreset_action(self, scene: SceneSnapshot, legal: Sequence[Any]) -> tuple[Any, dict[str, Any], str] | None:
        # 连续非终端 reset 后强制打破循环：忽略 known_noops 返回任意非 RESET 合法动作，
        # 让 total_action_count 推进以触发 is_done，并给环境一次状态变化的机会。
        legal_by_value = {action_value(a): a for a in legal}
        for action in legal:
            name = action_name(action).upper()
            if name == "RESET":
                continue
            if name == "ACTION6":
                target = self._choose_special_object(scene)
                if target is None:
                    continue
                x, y = self._object_click_point(scene, target)
                if not (0 <= x < scene.width and 0 <= y < scene.height):
                    continue
                proposal = {"name": "ACTION6", "x": x, "y": y, "target_object_id": target.track_id, "purpose": "force break non-terminal reset loop", "expected_change": "break reset storm"}
                try:
                    made = self._make_action6(legal_by_value.get(action_value(action), action), x, y, proposal)
                except Exception:
                    continue
                if made is not None:
                    return made, proposal, "force_break_reset_loop"
                continue
            action_obj = get_action_by_name(name)
            if action_obj is None or action_value(action_obj) not in legal_by_value:
                continue
            proposal = {"name": name, "purpose": "force break non-terminal reset loop", "expected_change": "break reset storm"}
            return self._attach_reasoning(legal_by_value.get(action_value(action_obj), action_obj), proposal), proposal, "force_break_reset_loop"
        return None

    def _minimal_information_probe(self, scene: SceneSnapshot, legal: Sequence[Any]) -> tuple[Any, dict[str, Any], str] | None:
        level = self.memory.level
        if level.chunk_action_count >= max(2, self.config.max_chunk_steps):
            level.bottleneck_reason = "probe_budget_reached"
            return None
        noops = level.known_noops_by_state.get(scene.state_hash, set())
        tried = level.tried_actions_by_state.get(scene.state_hash, set())
        name = self._next_unknown_simple_action(legal, scene)
        if name and name not in noops:
            proposal = {"name": name, "purpose": "minimal safe probe to gather evidence", "expected_change": "learn action meaning or no-op"}
            action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
            if action is not None:
                return action, proposal, "minimal_probe"
        if "ACTION6" in {action_name(a).upper() for a in legal}:
            picked = self._first_untried_action6_target(scene, set(noops) | set(tried))
            if picked is not None:
                x, y, target_id = picked
                proposal = {"name": "ACTION6", "x": x, "y": y, "target_object_id": target_id, "purpose": "minimal click probe on a fresh target coordinate", "expected_change": "observe object response"}
                action = self._make_valid_action(proposal, scene, legal, allow_reset=False)
                if action is not None:
                    return action, proposal, "minimal_probe"
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
                x, y = self._object_click_point(scene, target)
            if not (0 <= x < scene.width and 0 <= y < scene.height):
                return None
            if f"ACTION6:{x},{y}" in self.memory.level.known_noops_by_state.get(scene.state_hash, set()):
                return None
            return self._make_action6(legal_by_value.get(action_value(action), action), x, y, {**proposal, "name": "ACTION6", "x": x, "y": y})
        if name in self.memory.level.known_noops_by_state.get(scene.state_hash, set()):
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

    def _object_click_point(self, scene: SceneSnapshot, obj: ObjectObservation) -> tuple[int, int]:
        cx, cy = float(obj.centroid[0]), float(obj.centroid[1])
        if obj.cells:
            x, y, *_ = min(obj.cells, key=lambda c: ((float(c[0]) - cx) ** 2 + (float(c[1]) - cy) ** 2, int(c[1]), int(c[0])))
        else:
            x, y = int(round(cx)), int(round(cy))
        return max(0, min(scene.width - 1, int(x))), max(0, min(scene.height - 1, int(y)))

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
        reasoning = {"agent": self.config.agent_version, "purpose": _short(proposal.get("purpose"), 240), "expected_change": _short(proposal.get("expected_change"), 240), "target_object_id": _short(proposal.get("target_object_id"), 24).upper(), "name": proposal.get("name")}
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
        level.recent_action_keys.append(key)
        x = y = None
        if name == "ACTION6" and ":" in key:
            try:
                x, y = [int(v) for v in key.split(":", 1)[1].split(",", 1)]
            except Exception:
                x = y = None
        level.pre_success_scene = scene
        level.pending_action = PendingAction(name, x, y, _short(proposal.get("purpose"), 240), _short(proposal.get("expected_change"), 240), _short(proposal.get("target_object_id"), 24).upper(), scene, id(latest_frame), self._call_index, source, proposal.get("expected_predicates") if isinstance(proposal.get("expected_predicates"), list) else [], latest_frame, level.last_vlm_mode)
        self.logger.log_event("action_v19", {"level": level.level_index, "step": level.total_action_count, "state": scene.state_hash[:12], "action": key, "source": source, "purpose": proposal.get("purpose"), "counter": scene.counter_value, "lives": scene.life_count})

    def _reset_action(self, *, reason: str = "unspecified", state: str | None = None, scene: SceneSnapshot | None = None, legal: Sequence[Any] = ()) -> Any:
        legal_names = [action_name(a).upper() for a in legal]
        level = self.memory.level
        if state not in {"WIN", "GAME_OVER", "NOT_PLAYED", None}:
            level.consecutive_nonterminal_resets += 1
        else:
            level.consecutive_nonterminal_resets = 0
        self.logger.log_event("reset_v19", {"reason": reason, "state_name": state or "", "level": level.level_index, "step": level.total_action_count, "state": scene.state_hash[:12] if scene is not None else "", "legal_actions": legal_names, "consecutive_nonterminal_resets": level.consecutive_nonterminal_resets})
        return get_action_by_name("RESET") or getattr(GameAction, "RESET")

    def _emergency_action(self, latest_frame: Any, legal: Sequence[Any]) -> Any:
        state = state_name(getattr(latest_frame, "state", None))
        if state in {"NOT_PLAYED", "GAME_OVER", "WIN"}:
            return self._reset_action(reason=state.lower(), state=state, legal=legal)
        try:
            scene = self.observer.scene_from_frame(latest_frame)
            selected = self._minimal_information_probe(scene, legal)
            if selected is not None:
                return selected[0]
        except Exception:
            pass
        try:
            scene = self.observer.scene_from_frame(latest_frame)
            selected = self._fallback_nonreset_action(scene, legal, "emergency safe non-reset fallback")
            if selected is not None:
                return selected[0]
        except Exception:
            pass
        return self._reset_action(reason="emergency_no_nonreset", state=state, legal=legal)


__all__ = [
    "ActionMeaning", "ActionWithData", "AgentConfig", "CompactEvent", "ComponentObservation",
    "DecisionLogger", "GameAction", "GameMemoryV18", "GameState", "LevelMemoryV18",
    "MyAgent", "ObjectEffectMemory", "ObjectObservation", "ObservationError", "Observer",
    "OpenAICompatibleBackend", "PendingAction", "PlanStep", "Qwen35Backend", "ResourceModel",
    "RuntimeMemoryV18", "SceneSnapshot", "TransitionReport", "V18VLMResult", "VLMMode",
    "VLMRequest", "VisualDescriptor", "WinConditionMemory", "action6_data", "action_name",
    "action_value", "get_action_by_id", "get_action_by_name", "make_vlm_backend",
    "normalize_legal_actions", "normalize_one_action", "parse_vlm_result", "render_grid",
    "simple_action_names", "state_name", "vlm_uses_remote_api",
]
