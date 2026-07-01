from __future__ import annotations

import ast
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

try:  # pragma: no cover - official runtime.
    from arcengine import GameAction, GameState
except Exception:  # pragma: no cover - local tests.
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

        def set_data(self, data: dict[str, Any]) -> "ComplexAction":
            return ComplexAction(self, dict(data))

    @dataclass(frozen=True)
    class ComplexAction:
        id: GameAction
        action_data: dict[str, Any]


@dataclass
class ActionWithData:
    id: Any
    action_data: dict[str, int]
    reasoning: dict[str, Any] = field(default_factory=dict)


# The Kaggle starter exposes agents.agent.Agent. Keep the older import as a
# compatibility fallback for local harnesses.
try:  # pragma: no cover - official Kaggle starter.
    from agents.agent import Agent as _BaseAgent  # type: ignore
except Exception:  # pragma: no cover
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


PALETTE_16 = np.asarray(
    [
        (0, 0, 0),
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
        (230, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 128, 128),
    ],
    dtype=np.uint8,
)


def _bool_env_any(names: Sequence[str], default: bool) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _int_env_any(
    names: Sequence[str], default: int, low: int | None = None, high: int | None = None
) -> int:
    value = default
    for name in names:
        raw = os.getenv(name)
        if raw not in (None, ""):
            try:
                value = int(raw)
            except ValueError:
                value = default
            break
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _float_env_any(
    names: Sequence[str], default: float, low: float | None = None, high: float | None = None
) -> float:
    value = default
    for name in names:
        raw = os.getenv(name)
        if raw not in (None, ""):
            try:
                value = float(raw)
            except ValueError:
                value = default
            break
    if not math.isfinite(value):
        value = default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _str_env_any(names: Sequence[str], default: str | None = None) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw not in (None, ""):
            return raw
    return default



@dataclass(frozen=True)
class AgentConfig:
    agent_version: str = "v1.7-option-repair-explore"
    enable_vlm: bool = True
    model_path: str | None = None
    image_size: int = 512
    max_actions: int = 240
    max_vlm_calls_per_level: int = 48
    vlm_first_life_soft_limit: int = 18
    vlm_retry_reserve: int = 10
    vlm_min_action_gap: int = 4
    vlm_max_new_tokens: int = 2048
    plan_horizon: int = 20
    max_recent_events: int = 36
    max_event_log: int = 3000
    max_prompt_events: int = 12
    max_prompt_objects: int = 14
    max_action6_candidates: int = 32
    action6_noop_radius: int = 2
    action6_cooldown_steps: int = 18
    loop_window: int = 12
    target_failure_limit: int = 2
    navigation_max_depth: int = 96
    vlm_success_consolidation_reserve: int = 2
    enable_success_consolidation_vlm: bool = True
    enable_transfer_bootstrap: bool = True
    log_dir: str | None = "./runs/agent_v1_7"
    debug: bool = False
    vlm_backend: str = "local"
    vlm_api_base_url: str | None = None
    vlm_api_key: str | None = None
    vlm_api_model: str | None = None
    vlm_api_timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            enable_vlm=_bool_env_any(
                ("ARC_V15_ENABLE_VLM", "ARC_V13_ENABLE_VLM", "ARC_V12_ENABLE_VLM", "ARC_V3_ENABLE_VLM", "ARC_V2_ENABLE_VLM", "ARC_V1_ENABLE_VLM"),
                True,
            ),
            model_path=_str_env_any(
                ("ARC_V15_MODEL_PATH", "ARC_V13_MODEL_PATH", "ARC_V12_MODEL_PATH", "ARC_V3_MODEL_PATH", "ARC_V2_MODEL_PATH", "QWEN_MODEL_PATH")
            ),
            image_size=_int_env_any(
                ("ARC_V15_IMAGE_SIZE", "ARC_V13_IMAGE_SIZE", "ARC_V12_IMAGE_SIZE", "ARC_V3_IMAGE_SIZE", "ARC_V2_IMAGE_SIZE"),
                512,
                64,
                1024,
            ),
            max_actions=_int_env_any(
                ("ARC_V15_MAX_ACTIONS", "ARC_V13_MAX_ACTIONS", "ARC_V12_MAX_ACTIONS", "ARC_V3_MAX_ACTIONS", "ARC_V2_MAX_ACTIONS"),
                240,
                1,
                10_000,
            ),
            max_vlm_calls_per_level=_int_env_any(
                ("ARC_V15_VLM_CALLS_PER_LEVEL", "ARC_V13_VLM_CALLS_PER_LEVEL", "ARC_V12_VLM_CALLS_PER_LEVEL", "ARC_V1_VLM_CALLS_PER_LEVEL"),
                48,
                0,
                1000,
            ),
            vlm_first_life_soft_limit=_int_env_any(
                ("ARC_V15_VLM_FIRST_LIFE_SOFT_LIMIT", "ARC_V13_VLM_FIRST_LIFE_SOFT_LIMIT"), 18, 1, 1000
            ),
            vlm_retry_reserve=_int_env_any(
                ("ARC_V15_VLM_RETRY_RESERVE", "ARC_V13_VLM_RETRY_RESERVE", "ARC_V12_VLM_RETRY_RESERVE"), 10, 0, 64
            ),
            vlm_min_action_gap=_int_env_any(
                ("ARC_V15_VLM_MIN_ACTION_GAP", "ARC_V13_VLM_MIN_ACTION_GAP", "ARC_V12_VLM_MIN_ACTION_GAP"), 4, 0, 32
            ),
            vlm_max_new_tokens=_int_env_any(
                ("ARC_V15_VLM_MAX_NEW_TOKENS", "ARC_V13_VLM_MAX_NEW_TOKENS", "ARC_V12_VLM_MAX_NEW_TOKENS", "ARC_V3_VLM_MAX_NEW_TOKENS"),
                2048,
                128,
                4096,
            ),
            plan_horizon=_int_env_any(
                ("ARC_V15_PLAN_HORIZON", "ARC_V13_PLAN_HORIZON", "ARC_V12_PLAN_HORIZON"), 20, 2, 48
            ),
            max_prompt_objects=_int_env_any(
                ("ARC_V15_MAX_PROMPT_OBJECTS", "ARC_V13_MAX_PROMPT_OBJECTS"), 14, 4, 32
            ),
            max_action6_candidates=_int_env_any(
                ("ARC_V15_MAX_ACTION6_CANDIDATES", "ARC_V13_MAX_ACTION6_CANDIDATES", "ARC_V12_MAX_ACTION6_CANDIDATES"),
                32,
                1,
                128,
            ),
            max_event_log=_int_env_any(("ARC_V15_MAX_EVENT_LOG",), 3000, 64, 8000),
            max_prompt_events=_int_env_any(
                ("ARC_V15_MAX_PROMPT_EVENTS", "ARC_V13_MAX_PROMPT_EVENTS"), 12, 4, 64
            ),
            action6_noop_radius=_int_env_any(("ARC_V15_ACTION6_NOOP_RADIUS",), 2, 0, 8),
            action6_cooldown_steps=_int_env_any(("ARC_V15_ACTION6_COOLDOWN",), 18, 1, 200),
            loop_window=_int_env_any(("ARC_V15_LOOP_WINDOW",), 12, 4, 64),
            target_failure_limit=_int_env_any(
                ("ARC_V15_TARGET_FAILURE_LIMIT", "ARC_V13_TARGET_FAILURE_LIMIT"), 2, 1, 8
            ),
            navigation_max_depth=_int_env_any(
                ("ARC_V15_NAVIGATION_MAX_DEPTH", "ARC_V13_NAVIGATION_MAX_DEPTH"), 96, 8, 512
            ),
            log_dir=_str_env_any(
                ("ARC_V15_LOG_DIR", "ARC_V13_LOG_DIR", "ARC_V12_LOG_DIR", "ARC_V3_LOG_DIR"),
                "./runs/agent_v1_7",
            ),
            enable_success_consolidation_vlm=_bool_env_any(("ARC_V15_SUCCESS_VLM",), True),
            enable_transfer_bootstrap=_bool_env_any(("ARC_V15_TRANSFER_BOOTSTRAP",), True),
            debug=_bool_env_any(
                ("ARC_V15_DEBUG", "ARC_V13_DEBUG", "ARC_V12_DEBUG", "ARC_V3_DEBUG"), False
            ),
            vlm_backend=_str_env_any(
                ("ARC_V15_VLM_BACKEND", "ARC_V13_VLM_BACKEND", "ARC_V12_VLM_BACKEND", "ARC_V3_VLM_BACKEND"),
                "local",
            )
            or "local",
            vlm_api_base_url=_str_env_any(
                ("ARC_V15_VLM_API_BASE_URL", "ARC_V13_VLM_API_BASE_URL", "ARC_V12_VLM_API_BASE_URL", "ARC_V3_VLM_API_BASE_URL")
            ),
            vlm_api_key=_str_env_any(
                ("ARC_V15_VLM_API_KEY", "ARC_V13_VLM_API_KEY", "ARC_V12_VLM_API_KEY", "ARC_V3_VLM_API_KEY", "INF_API_KEY")
            ),
            vlm_api_model=_str_env_any(
                ("ARC_V15_VLM_API_MODEL", "ARC_V13_VLM_API_MODEL", "ARC_V12_VLM_API_MODEL", "ARC_V3_VLM_API_MODEL")
            ),
            vlm_api_timeout_s=_float_env_any(
                ("ARC_V15_VLM_API_TIMEOUT_S", "ARC_V13_VLM_API_TIMEOUT_S", "ARC_V12_VLM_API_TIMEOUT_S", "ARC_V3_VLM_API_TIMEOUT_S"),
                60.0,
                1.0,
                600.0,
            ),
        )


class ObservationError(RuntimeError):
    pass


class V1Phase(str, Enum):
    INIT = "INIT"
    TRANSFER_BOOTSTRAP = "TRANSFER_BOOTSTRAP"
    ACTION_GROUNDING = "ACTION_GROUNDING"
    MECHANIC_EXPLORATION = "MECHANIC_EXPLORATION"
    GOAL_HYPOTHESIS = "GOAL_HYPOTHESIS"
    GOAL_VALIDATION = "GOAL_VALIDATION"
    PLAN_SYNTHESIS = "PLAN_SYNTHESIS"
    EXECUTE_PLAN = "EXECUTE_PLAN"
    MODEL_REPAIR_LIGHT = "MODEL_REPAIR_LIGHT"
    FAILURE_RECOVERY = "FAILURE_RECOVERY"
    SUCCESS_CONSOLIDATE = "SUCCESS_CONSOLIDATE"

    # Backward-compatible aliases for older tests/log references.
    INITIALIZE = "INIT"
    ACTION_EXPLORATION = "ACTION_GROUNDING"
    OBJECT_EXPLORATION = "MECHANIC_EXPLORATION"
    SOLVE = "EXECUTE_PLAN"
    RECOVER = "FAILURE_RECOVERY"


class VLMMode(str, Enum):
    INIT_ANALYSIS = "INIT_ANALYSIS"
    TRANSFER_INSTANTIATION = "TRANSFER_INSTANTIATION"
    TRANSITION_EXPLANATION = "TRANSITION_EXPLANATION"
    EXPERIMENT_DESIGN = "EXPERIMENT_DESIGN"
    PLAN_SYNTHESIS = "PLAN_SYNTHESIS"
    SUCCESS_CONSOLIDATION = "SUCCESS_CONSOLIDATION"
    FAILURE_REPAIR = "FAILURE_REPAIR"
    REPAIR_DECISION = "REPAIR_DECISION"


class RejectReason(str, Enum):
    ILLEGAL_ACTION = "illegal_action"
    RESET_NOT_ALLOWED = "reset_not_allowed"
    ACTION7_UNDO_ONLY = "action7_undo_only"
    ACTION6_BAD_COORD = "action6_bad_coord"
    ACTION6_DUPLICATE_OR_COOLDOWN = "action6_duplicate_or_cooldown"
    KNOWN_NOOP = "known_noop"
    LOOP_RISK = "loop_risk"
    FAILURE_SUFFIX = "failure_suffix"
    TARGET_BLOCKED = "target_blocked"
    BAD_NAVIGATE_TARGET = "bad_navigate_target"
    NAV_ACTOR_UNKNOWN = "nav_actor_unknown"
    NAV_ACTION_VECTORS_UNKNOWN = "nav_action_vectors_unknown"
    NAV_WALKABLE_UNKNOWN = "nav_walkable_unknown"
    NAV_NO_PATH_KNOWN = "nav_no_path_known"
    CONTRACT_REPAIR_FAILED = "contract_repair_failed"
    UNSUPPORTED_ACTION_TOKEN = "unsupported_action_token"
    PREDICATE_UNSUPPORTED = "predicate_unsupported"


class IntentType(str, Enum):
    PRIMITIVE_ACTION = "primitive_action"
    NAVIGATE_TO_OBJECT = "navigate_to_object"
    CLICK_CANDIDATE = "click_candidate"
    CLICK_OBJECT = "click_object"
    TEST_OBJECT = "test_object"
    VALIDATE_SCHEMA_SLOT = "validate_schema_slot"


class CompileStatus(str, Enum):
    OK = "ok"
    REPAIRED = "repaired"
    ILLEGAL_ACTION = "illegal_action"
    ACTION6_BAD_COORD = "action6_bad_coord"
    ACTION6_NO_CANDIDATE = "action6_no_candidate"
    TARGET_MISSING = "target_missing"
    TARGET_NOT_VISIBLE = "target_not_visible"
    ACTOR_UNKNOWN = "actor_unknown"
    ACTION_VECTORS_UNKNOWN = "action_vectors_unknown"
    WALKABLE_UNKNOWN = "walkable_unknown"
    NO_PATH_KNOWN = "no_path_known"
    TARGET_BLOCKED = "target_blocked"
    UNSUPPORTED_INTENT = "unsupported_intent"


@dataclass
class ActionIntent:
    source: str
    intent_type: str
    action_name: str = ""
    target_object_id: str = ""
    action6_candidate_id: str = ""
    x: int | None = None
    y: int | None = None
    purpose: str = ""
    expected_predicates: list[dict[str, Any]] = field(default_factory=list)
    risk: str = "low"
    reversible: bool = True
    information_gain: float = 0.0
    goal_progress: float = 0.0
    novelty: float = 0.0
    priority: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionProposal:
    source: str
    action_name: str
    x: int | None = None
    y: int | None = None
    target_object_id: str = ""
    purpose: str = ""
    phase: V1Phase = V1Phase.MECHANIC_EXPLORATION
    hypothesis_ids: list[str] = field(default_factory=list)
    discriminates: list[str] = field(default_factory=list)
    expected_predicates: list[dict[str, Any]] = field(default_factory=list)
    risk: str = "low"
    reversible: bool = False
    information_gain: float = 0.0
    goal_progress: float = 0.0
    novelty: float = 0.0
    cost: float = 1.0
    priority: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def action_key(self) -> str:
        if self.action_name == "ACTION6":
            return f"ACTION6:{self.x},{self.y}"
        return self.action_name

    def to_plan_step(self) -> dict[str, Any]:
        expected_change = ""
        if self.expected_predicates:
            expected_change = _short_string(self.expected_predicates[0].get("summary"), 240)
        step = {
            "name": self.action_name,
            "x": self.x,
            "y": self.y,
            "target_object_id": self.target_object_id,
            "purpose": self.purpose,
            "expected_change": expected_change,
            "expected_predicates": self.expected_predicates,
            "source": self.source,
        }
        for extra_key in (
            "candidate_id",
            "candidate_kind",
            "repair_candidate_id",
            "vlm_repair_selected",
            "relaxed_frontier",
            "only_action6_frontier",
        ):
            if extra_key in self.raw:
                step[extra_key] = self.raw[extra_key]
        if self.raw.get("relaxed_frontier"):
            step["relaxed_recovery"] = True
        return step


@dataclass
class ActivePlanStep:
    step_id: str
    intent: ActionIntent
    expected_predicates: list[dict[str, Any]] = field(default_factory=list)
    preconditions: list[dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    attempts: int = 0
    max_attempts: int = 1
    last_failure: str = ""


@dataclass
class ActivePlan:
    plan_id: str
    source: str
    goal: str = ""
    theory_id: str = ""
    steps: list[ActivePlanStep] = field(default_factory=list)
    cursor: int = 0
    confidence: float = 0.0
    evidence_level: int = 0
    status: str = "candidate"
    failures: int = 0

    def current_step(self) -> ActivePlanStep | None:
        while self.cursor < len(self.steps) and self.steps[self.cursor].status in {"done", "skipped"}:
            self.cursor += 1
        if self.cursor >= len(self.steps):
            return None
        return self.steps[self.cursor]

    def as_prompt(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "source": self.source,
            "goal": self.goal[:300],
            "theory_id": self.theory_id,
            "cursor": self.cursor,
            "confidence": round(self.confidence, 3),
            "evidence_level": self.evidence_level,
            "status": self.status,
            "failures": self.failures,
            "steps": [
                {
                    "step_id": step.step_id,
                    "status": step.status,
                    "attempts": step.attempts,
                    "intent": _json_safe(step.intent),
                    "expected_predicates": step.expected_predicates[:6],
                    "last_failure": step.last_failure,
                }
                for step in self.steps[:12]
            ],
        }


@dataclass(frozen=True)
class Action6Signature:
    state_hash: str
    x: int
    y: int
    object_id: str = ""
    object_type_key: str = ""
    candidate_kind: str = ""
    local_patch_signature: str = ""

    @property
    def xy_key(self) -> str:
        return f"ACTION6:{self.x},{self.y}"

    @property
    def exact_key(self) -> str:
        seed = f"{self.state_hash}:{self.x},{self.y}:{self.object_id}:{self.candidate_kind}:{self.local_patch_signature}"
        return "A6E:" + hashlib.sha1(seed.encode()).hexdigest()[:12]

    @property
    def local_key(self) -> str:
        seed = f"{self.state_hash}:{self.object_type_key}:{self.candidate_kind}:{self.local_patch_signature}"
        return "A6L:" + hashlib.sha1(seed.encode()).hexdigest()[:12]


@dataclass
class CompileResult:
    ok: bool
    intent: ActionIntent
    proposal: ActionProposal | None = None
    status: CompileStatus = CompileStatus.OK
    detail: str = ""
    severity: str = "hard"


@dataclass
class ValidationResult:
    ok: bool
    proposal: ActionProposal
    reason: RejectReason | None = None
    detail: str = ""
    score: float = 0.0


@dataclass
class Action6Candidate:
    x: int
    y: int
    target_object_id: str
    target_type_key: str
    candidate_kind: str
    local_patch_signature: str
    salience: float
    expected_role: str = ""
    prior_score: float = 0.0
    candidate_id: str = ""

    def key(self) -> str:
        return f"{self.target_object_id}:{self.candidate_kind}:{self.x},{self.y}:{self.local_patch_signature[:12]}"


@dataclass
class Action6ProbeRecord:
    state_hash: str
    abstract_state_hash: str
    x: int
    y: int
    target_object_id: str
    target_type_key: str
    candidate_kind: str
    local_patch_signature: str
    outcome: str
    event_id: int
    cooldown_until_step: int


@dataclass
class Action6Memory:
    records: list[Action6ProbeRecord] = field(default_factory=list)
    duplicate_suppressed: int = 0

    def remember(self, record: Action6ProbeRecord, max_records: int = 360) -> None:
        self.records.append(record)
        if len(self.records) > max_records:
            del self.records[:-max_records]


@dataclass
class EventRecord:
    event_id: int
    level_index: int
    attempt_index: int
    life_index: int
    step_index: int
    action_name: str
    action_key: str
    action_params: dict[str, Any]
    source: str
    before_state_hash: str
    after_state_hash: str
    before_full_hash: str
    after_full_hash: str
    before_summary: str
    after_summary: str
    transition_summary: str
    transition_delta: dict[str, Any]
    available_actions_before: list[str]
    available_actions_after: list[str]
    expected_predicates: list[dict[str, Any]] = field(default_factory=list)
    predicate_check: dict[str, Any] = field(default_factory=dict)
    outcome: str = "unknown"
    vlm_mode: str = ""


@dataclass
class Hypothesis:
    hypothesis_id: str
    scope: str
    kind: str
    statement: str
    status: str = "candidate"
    confidence: float = 0.0
    evidence_level: int = 0
    predictions: list[dict[str, Any]] = field(default_factory=list)
    supporting_event_ids: list[int] = field(default_factory=list)
    contradicting_event_ids: list[int] = field(default_factory=list)
    cheapest_falsification_test: dict[str, Any] | None = None


@dataclass
class GameGoalSchema:
    schema_id: str
    name: str
    statement: str
    role_slots: dict[str, dict[str, Any]] = field(default_factory=dict)
    success_predicates: list[dict[str, Any]] = field(default_factory=list)
    trigger_action_patterns: list[str] = field(default_factory=list)
    required_mechanics: list[str] = field(default_factory=list)
    known_variations: list[str] = field(default_factory=list)
    source_levels: list[int] = field(default_factory=list)
    evidence_event_ids: list[int] = field(default_factory=list)
    confidence: float = 0.0
    evidence_level: int = 0


@dataclass
class LevelGoalInstantiation:
    level_index: int
    schema_id: str = ""
    role_bindings: dict[str, str] = field(default_factory=dict)
    concrete_values: dict[str, Any] = field(default_factory=dict)
    unknown_slots: list[str] = field(default_factory=list)
    confidence: float = 0.0
    next_disambiguating_tests: list[dict[str, Any]] = field(default_factory=list)
    current_recipe: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LevelTheory:
    theory_id: str
    level_index: int
    created_step: int
    source: str = "vlm_init"
    confidence: float = 0.0
    evidence_level: int = 0
    status: str = "candidate"
    win_condition_hypothesis: str = ""
    mechanism_hypothesis: str = ""
    critical_objects: list[dict[str, Any]] = field(default_factory=list)
    expected_progress_signals: list[dict[str, Any]] = field(default_factory=list)
    solve_sketch: list[dict[str, Any]] = field(default_factory=list)
    discriminating_tests: list[dict[str, Any]] = field(default_factory=list)
    invalidating_evidence: list[str] = field(default_factory=list)
    supporting_event_ids: list[int] = field(default_factory=list)
    contradicting_event_ids: list[int] = field(default_factory=list)

    def as_prompt(self) -> dict[str, Any]:
        return {
            "theory_id": self.theory_id,
            "level_index": self.level_index,
            "status": self.status,
            "confidence": round(self.confidence, 3),
            "evidence_level": self.evidence_level,
            "win_condition_hypothesis": self.win_condition_hypothesis[:400],
            "mechanism_hypothesis": self.mechanism_hypothesis[:400],
            "critical_objects": self.critical_objects[:8],
            "expected_progress_signals": self.expected_progress_signals[:8],
            "solve_sketch": self.solve_sketch[:10],
            "discriminating_tests": self.discriminating_tests[:8],
            "supporting_event_ids": self.supporting_event_ids[-12:],
            "contradicting_event_ids": self.contradicting_event_ids[-12:],
        }


@dataclass
class LevelOutcomeMemory:
    level_index: int
    initial_state_hash: str
    pre_success_state_hash: str
    post_success_state_hash: str
    post_success_is_next_level_start: bool
    success_action_key: str
    action_trace: list[str]
    causal_event_ids: list[int]
    initial_summary: str
    pre_success_summary: str
    post_success_summary: str
    start_to_pre_success_diff: dict[str, Any] = field(default_factory=dict)
    success_transition_summary: str = ""
    inferred_mechanism: str = ""
    reusable_schema_id: str = ""
    confidence: float = 0.0

    def as_prompt(self) -> dict[str, Any]:
        return {
            "level_index": self.level_index,
            "initial_state": self.initial_state_hash[:12],
            "pre_success_state": self.pre_success_state_hash[:12],
            "post_success_state": self.post_success_state_hash[:12],
            "post_success_is_next_level_start": self.post_success_is_next_level_start,
            "success_action_key": self.success_action_key,
            "action_trace_tail": self.action_trace[-40:],
            "causal_event_ids": self.causal_event_ids[-20:],
            "initial_summary": self.initial_summary[:500],
            "pre_success_summary": self.pre_success_summary[:500],
            "post_success_summary": self.post_success_summary[:500],
            "start_to_pre_success_diff": self.start_to_pre_success_diff,
            "success_transition_summary": self.success_transition_summary[:500],
            "inferred_mechanism": self.inferred_mechanism[:500],
            "reusable_schema_id": self.reusable_schema_id,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class FailureModel:
    forbidden_action_suffixes: list[dict[str, Any]] = field(default_factory=list)
    dangerous_objects: dict[str, dict[str, Any]] = field(default_factory=dict)
    dangerous_regions: list[dict[str, Any]] = field(default_factory=list)
    resource_thresholds: list[dict[str, Any]] = field(default_factory=list)
    failed_goal_bindings: list[dict[str, Any]] = field(default_factory=list)

    def add_suffix(self, suffix: Sequence[str], reason: str, step: int, level: int, attempt: int) -> None:
        if not suffix:
            return
        trimmed = list(suffix)[-8:]
        signature = ">".join(trimmed)
        for item in self.forbidden_action_suffixes:
            if item.get("signature") == signature:
                item["count"] = int(item.get("count", 1)) + 1
                item["last_step"] = step
                return
        self.forbidden_action_suffixes.append({"signature": signature, "suffix": trimmed, "reason": _short_string(reason, 320), "level": level, "attempt": attempt, "count": 1, "last_step": step})
        self.forbidden_action_suffixes = self.forbidden_action_suffixes[-24:]

    def forbids(self, recent_actions: Sequence[str], proposal_key: str) -> bool:
        candidate = [*list(recent_actions)[-7:], proposal_key]
        candidate_sig = ">".join(candidate)
        for item in self.forbidden_action_suffixes:
            sig = str(item.get("signature", ""))
            # A single-action suffix would forbid an entire primitive globally
            # (every sequence ending in that action), over-generalizing across
            # states and levels and potentially banning the only usable action.
            # Single-step lethal actions belong in state-bound memory, not here,
            # so only 2+ action patterns are allowed to hard-forbid.
            if len(item.get("suffix", []) or []) < 2:
                continue
            if sig and candidate_sig.endswith(sig):
                return True
        return False


@dataclass
class UndoContext:
    active: bool = False
    reason: str = ""
    started_event_id: int = -1
    expires_step: int = 0
    rollback_state_hash: str = ""


@dataclass
class UndoManager:
    context: UndoContext = field(default_factory=UndoContext)

    def allow(self, reason: str, event_id: int, current_step: int, state_hash: str, ttl: int = 2) -> None:
        self.context = UndoContext(True, reason, event_id, current_step + ttl, state_hash)

    def can_emit(self, current_step: int) -> bool:
        return self.context.active and current_step <= self.context.expires_step

    def clear(self) -> None:
        self.context = UndoContext()


@dataclass
class ControllerStats:
    proposal_counts: Counter[str] = field(default_factory=Counter)
    reject_counts: Counter[str] = field(default_factory=Counter)
    selected_counts: Counter[str] = field(default_factory=Counter)
    vlm_mode_counts: Counter[str] = field(default_factory=Counter)
    contract_repairs: Counter[str] = field(default_factory=Counter)
    compile_failures: Counter[str] = field(default_factory=Counter)
    recovery_counts: Counter[str] = field(default_factory=Counter)
    level_theory_updates: Counter[str] = field(default_factory=Counter)
    action7_forbidden_count: int = 0
    action6_duplicate_suppressed: int = 0
    unstructured_fallback_count: int = 0
    loop_soft_penalties: int = 0


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
    if hasattr(action, "value"):
        return int(action.value)
    if hasattr(action, "id") and hasattr(action.id, "value"):
        return int(action.id.value)
    if hasattr(action, "x") and hasattr(action, "y"):
        return 6
    try:
        return int(action)
    except Exception:
        return 999


def action6_data(action: Any) -> dict[str, int] | None:
    x = getattr(action, "x", None)
    y = getattr(action, "y", None)
    if x is not None and y is not None:
        try:
            return {"x": int(x), "y": int(y)}
        except Exception:
            return None
    for attr in ("action_data", "data"):
        data = getattr(action, attr, None)
        if callable(data):
            continue
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
    for action in GameAction:
        if action_value(action) == int(action_id):
            return action
    return None


def get_action_by_name(name: str) -> Any | None:
    clean = str(name).strip().split(".")[-1].upper()
    if clean.isdigit():
        return get_action_by_id(int(clean))
    try:
        return getattr(GameAction, clean)
    except Exception:
        return None


def normalize_one_action(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if "id" in value:
            return normalize_one_action(value["id"])
        if "name" in value:
            return normalize_one_action(value["name"])
        return None
    if isinstance(value, str):
        return get_action_by_name(value)
    if isinstance(value, int):
        return get_action_by_id(value)
    if hasattr(value, "value") and action_name(value).startswith(("ACTION", "RESET")):
        return value
    if hasattr(value, "id"):
        return normalize_one_action(value.id)
    return None


def normalize_legal_actions(
    frame_actions: Any,
    env_action_space: Iterable[Any] | None = None,
    *,
    allow_env_fallback: bool = True,
) -> tuple[Any, ...]:
    empty_frame_actions = frame_actions is None or (
        isinstance(frame_actions, (list, tuple, set)) and len(frame_actions) == 0
    )
    source = env_action_space if (empty_frame_actions and allow_env_fallback and frame_actions is None) else frame_actions
    if source is None:
        return ()
    if isinstance(source, (str, bytes)) or not isinstance(source, Iterable):
        raw_values = [source]
    else:
        raw_values = list(source)
    parsed: dict[int, Any] = {}
    for raw in raw_values:
        action = normalize_one_action(raw)
        if action is not None:
            parsed[action_value(action)] = action
    return tuple(parsed[k] for k in sorted(parsed))


def render_grid(grid: Sequence[Sequence[int]], size: int = 512) -> Image.Image:
    arr = np.asarray(grid, dtype=np.uint8)
    rgb = PALETTE_16[arr % 16]
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    return image.resize((size, size), resample=Image.Resampling.NEAREST)


def _stable_grid_hash(grid: tuple[tuple[int, ...], ...]) -> str:
    height = len(grid)
    width = len(grid[0]) if height else 0
    h = hashlib.blake2b(digest_size=16)
    h.update(bytes([height, width]))
    for row in grid:
        h.update(bytes(row))
    return h.hexdigest()


@dataclass(frozen=True)
class ComponentObservation:
    color: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    touches_border: bool
    shape_signature: tuple[tuple[int, int], ...]
    cells: tuple[tuple[int, int], ...]

    @property
    def key(self) -> str:
        x0, y0, x1, y1 = self.bbox
        return f"c{self.color}:a{self.area}:b{x0},{y0},{x1},{y1}"


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
    binary_signature: tuple[tuple[int, int], ...]
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
    source: str = "emergency"
    plan_id: int | None = None
    source_frame_ref: Any | None = None
    expected_predicates: list[dict[str, Any]] = field(default_factory=list)
    available_actions_before: list[str] = field(default_factory=list)
    vlm_mode: str = ""

    def action_key(self) -> str:
        if self.name == "ACTION6":
            return f"{self.name}:{self.x},{self.y}"
        return self.name


@dataclass
class ActionKnowledge:
    status: str = "unknown"
    attempts: int = 0
    successes: int = 0
    blocked: int = 0
    structural_successes: int = 0
    global_changes: int = 0
    retry_failures: int = 0
    movement_vectors: dict[str, int] = field(default_factory=dict)
    controlled_object_votes: dict[str, int] = field(default_factory=dict)
    recent_effect_scores: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    last_evidence: str = ""

    def best_vector(self) -> tuple[int, int] | None:
        if not self.movement_vectors:
            return None
        key, count = max(self.movement_vectors.items(), key=lambda item: item[1])
        if count < 1:
            return None
        try:
            dx, dy = key.split(",", 1)
            return int(dx), int(dy)
        except Exception:
            return None

    def effect_score(self) -> float:
        if self.recent_effect_scores:
            recent = sum(self.recent_effect_scores) / len(self.recent_effect_scores)
        else:
            recent = 0.0
        attempts = max(1, self.attempts)
        return round(
            recent
            + 0.45 * (self.successes / attempts)
            + 0.35 * (self.structural_successes / attempts)
            - 0.55 * (self.blocked / attempts)
            - 0.75 * (self.retry_failures / attempts),
            3,
        )


MECHANISM_MODES = {
    "unknown",
    "click_selection",
    "direct_navigation",
    "global_transform",
    "frame_gate",
    "resource",
    "hybrid",
}


@dataclass
class LevelMechanismState:
    mode: str = "unknown"
    scores: dict[str, float] = field(
        default_factory=lambda: {
            "click_selection": 0.0,
            "direct_navigation": 0.0,
            "global_transform": 0.0,
            "frame_gate": 0.0,
            "resource": 0.0,
        }
    )
    evidence: list[str] = field(default_factory=list)
    last_switch_step: int = 0

    def note(self, key: str, amount: float, evidence: str) -> None:
        if key not in self.scores:
            return
        self.scores[key] = round(max(0.0, self.scores.get(key, 0.0) + amount), 3)
        evidence = _short_string(evidence, 180)
        if evidence:
            self.evidence.append(f"{key}:{evidence}")
            self.evidence = self.evidence[-12:]

    def choose_mode(self) -> str:
        ranked = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] < 2.0:
            return "unknown"
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if ranked[0][1] - second < 1.0:
            return "hybrid"
        return ranked[0][0] if ranked[0][0] in MECHANISM_MODES else "unknown"

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "scores": dict(sorted(self.scores.items())),
            "evidence": self.evidence[-8:],
            "last_switch_step": self.last_switch_step,
        }


@dataclass
class ExecutableStrategyRule:
    rule_id: str
    kind: str
    trigger_features: dict[str, Any] = field(default_factory=dict)
    preconditions: dict[str, Any] = field(default_factory=dict)
    policy: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    target_descriptor: dict[str, Any] = field(default_factory=dict)
    actor_required: bool = False
    support: int = 1
    failures: int = 0
    confidence: float = 0.0
    source_level: int = 0

    def as_prompt(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "kind": self.kind,
            "trigger_features": self.trigger_features,
            "preconditions": self.preconditions,
            "policy": self.policy,
            "steps": self.steps[:12],
            "target_descriptor": self.target_descriptor,
            "actor_required": self.actor_required,
            "support": self.support,
            "failures": self.failures,
            "confidence": round(self.confidence, 3),
            "source_level": self.source_level,
        }


@dataclass
class NearSuccessRoute:
    route_id: str = ""
    target_id: str = ""
    safe_prefix: list[str] = field(default_factory=list)
    last_bad_action: str = ""
    counter_before: int | None = None
    counter_after: int | None = None
    lives_after: int | None = None
    state_before_last: str = ""
    life: int = 0
    attempt: int = 0
    progress: str = ""
    uses: int = 0
    failures: int = 0
    cooldown_until: int = 0
    last_resumed_state: str = ""
    last_resumed_life: int = -1
    last_resumed_attempt: int = -1

    def stable_id(self) -> str:
        if self.route_id:
            return self.route_id
        seed = json.dumps(
            {
                "target": self.target_id,
                "safe": self.safe_prefix[-24:],
                "bad": self.last_bad_action,
                "state": self.state_before_last[:12],
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha1(seed.encode()).hexdigest()[:12]

    def failure_signature(self) -> str:
        return f"{self.target_id}:{self.last_bad_action}:{self.state_before_last[:12]}:{self.stable_id()}"

    def as_prompt(self) -> dict[str, Any]:
        return {
            "route_id": self.stable_id(),
            "target_id": self.target_id,
            "safe_prefix": self.safe_prefix[-24:],
            "last_bad_action": self.last_bad_action,
            "counter_before": self.counter_before,
            "counter_after": self.counter_after,
            "lives_after": self.lives_after,
            "state_before_last": self.state_before_last[:12],
            "life": self.life,
            "attempt": self.attempt,
            "progress": self.progress[:240],
            "uses": self.uses,
            "failures": self.failures,
            "cooldown_until": self.cooldown_until,
        }


@dataclass
class TargetFailureRecord:
    target_id: str
    reason: str
    life: int
    attempt: int
    state_signature: str
    count: int = 1

    def as_prompt(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "reason": self.reason,
            "life": self.life,
            "attempt": self.attempt,
            "state_signature": self.state_signature[:12],
            "count": self.count,
        }


@dataclass
class GameMemory:
    goal_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    mechanics: list[dict[str, Any]] = field(default_factory=list)
    action_knowledge: dict[str, ActionKnowledge] = field(default_factory=dict)
    object_type_roles: dict[str, dict[str, Any]] = field(default_factory=dict)
    successful_levels: list[dict[str, Any]] = field(default_factory=list)
    level_outcomes: list[LevelOutcomeMemory] = field(default_factory=list)
    mechanism_library: list[dict[str, Any]] = field(default_factory=list)
    strategy_rules: list[ExecutableStrategyRule] = field(default_factory=list)
    goal_schemas: list[GameGoalSchema] = field(default_factory=list)
    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    failure_model: FailureModel = field(default_factory=FailureModel)
    controller_stats: ControllerStats = field(default_factory=ControllerStats)
    advisory_summary: str = ""
    levels_seen: int = 0
    vlm_calls_total: int = 0
    total_vlm_errors: int = 0


@dataclass
class LevelMemory:
    level_index: int = 0
    levels_completed_at_start: int = 0
    attempt_index: int = 0
    life_index: int = 0
    stage: V1Phase = V1Phase.INIT
    initial_scene_summary: str = ""
    initial_state_hash: str = ""
    local_goal: str = ""
    advisory_summary: str = ""
    object_beliefs: dict[str, dict[str, Any]] = field(default_factory=dict)
    object_visit_counts: dict[str, int] = field(default_factory=dict)
    object_effect_counts: dict[str, int] = field(default_factory=dict)
    recent_events: deque[str] = field(default_factory=lambda: deque(maxlen=36))
    tried_actions_by_state: dict[str, set[str]] = field(default_factory=dict)
    action_attempt_counts_by_state: dict[str, dict[str, int]] = field(default_factory=dict)
    noop_actions_by_state: dict[str, set[str]] = field(default_factory=dict)
    transition_graph: dict[str, dict[str, str]] = field(default_factory=dict)
    plan_goal: str = ""
    plan_target_id: str = ""
    plan_id: int = 0
    active_plan: ActivePlan | None = None
    candidate_plans: deque[ActivePlan] = field(default_factory=lambda: deque(maxlen=6))
    disabled_action_families_until: dict[str, int] = field(default_factory=dict)
    nonprogress_action_records: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=48))
    suspicious_action_counts: Counter[str] = field(default_factory=Counter)
    last_nonprogress_repair_step: int = -999
    fallback_action_records: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=48))
    fallback_overuse_repair_step: int = -999
    repair_action_candidates: list[dict[str, Any]] = field(default_factory=list)
    vlm_repair_choice: dict[str, Any] = field(default_factory=dict)
    last_repair_vlm_reason: str = ""
    last_repair_vlm_step: int = -999
    anti_loop_escape_failures: int = 0
    last_anti_loop_escape_failure_step: int = -999
    decision_frames: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=120))
    last_compile_failures: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=40))
    active_strategy_rule_id: str = ""
    active_near_success_route_id: str = ""
    target_failure_counts: dict[str, int] = field(default_factory=dict)
    target_failure_records: list[TargetFailureRecord] = field(default_factory=list)
    near_success_routes: deque[NearSuccessRoute] = field(default_factory=lambda: deque(maxlen=6))
    failed_route_signatures: set[str] = field(default_factory=set)
    mechanism: LevelMechanismState = field(default_factory=LevelMechanismState)
    plan_quality_rejections: int = 0
    force_vlm_reason: str = "initial_scene"
    actions_since_vlm: int = 999
    last_vlm_action_count: int = -999
    vlm_calls_this_level: int = 0
    vlm_invalid_streak: int = 0
    total_action_count: int = 0
    controller_tick_count: int = 0
    life_action_count: int = 0
    action_trace: deque[str] = field(default_factory=lambda: deque(maxlen=360))
    failed_lives: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=8))
    pending_action: PendingAction | None = None
    last_resolved_pending_action: PendingAction | None = None
    awaiting_reset: bool = False
    recent_state_hashes: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    recent_action_keys: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    controlled_object_id: str = ""
    controlled_object_type_key: str = ""
    tentative_controlled_object_id: str = ""
    controlled_actor_confidence: float = 0.0
    actor_votes: dict[str, int] = field(default_factory=dict)
    walkable_color_votes: dict[int, int] = field(default_factory=dict)
    counter_cost_samples: deque[int] = field(default_factory=lambda: deque(maxlen=20))
    counter_refill_transitions: list[dict[str, Any]] = field(default_factory=list)
    last_failure: str = ""
    goal_instantiation: LevelGoalInstantiation | None = None
    action6_memory: Action6Memory = field(default_factory=Action6Memory)
    reject_reasons: Counter[str] = field(default_factory=Counter)
    last_selected_proposal: ActionProposal | None = None
    current_vlm_mode: str = ""
    loop_blocked_action_keys: set[str] = field(default_factory=set)
    recovery_mode: str = ""
    recovery_reason: str = ""
    last_recovery_source: str = ""
    last_recovery_reasoning: dict[str, Any] = field(default_factory=dict)
    recovery_until_step: int = 0
    terminal_reset_streak: int = 0
    no_safe_recovery_streak: int = 0
    fallback_recent_keys: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    blocked_state_action_pairs: set[tuple[str, str]] = field(default_factory=set)
    level_theories: list[LevelTheory] = field(default_factory=list)
    active_theory_id: str = ""
    initial_scene_ref: SceneSnapshot | None = None
    initial_rgb: Image.Image | None = None
    initial_annotated_rgb: Image.Image | None = None


@dataclass
class AgentRuntimeMemory:
    game: GameMemory = field(default_factory=GameMemory)
    level: LevelMemory = field(default_factory=LevelMemory)
    event_log: list[EventRecord] = field(default_factory=list)
    next_event_id: int = 1
    evicted_event_id_through: int = 0


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0] + 1) * max(0, box[3] - box[1] + 1)


def _bbox_intersection(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> int:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    return max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)


def _bbox_gap(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> tuple[int, int]:
    dx = max(0, max(left[0], right[0]) - min(left[2], right[2]) - 1)
    dy = max(0, max(left[1], right[1]) - min(left[3], right[3]) - 1)
    return dx, dy


def _bbox_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _hex_color(color: int) -> str:
    return "0123456789ABCDEF"[int(color) % 16]


def _pattern_rows(pattern: str) -> list[str]:
    return [row for row in pattern.split("/") if row]


def _compress_binary_rows(rows: Sequence[str]) -> tuple[str, ...]:
    if not rows:
        return ()
    normalized = [tuple(ch != "." for ch in row) for row in rows]
    dedup_rows: list[tuple[bool, ...]] = []
    for row in normalized:
        if not dedup_rows or row != dedup_rows[-1]:
            dedup_rows.append(row)
    if not dedup_rows:
        return ()
    columns = list(zip(*dedup_rows))
    dedup_cols: list[tuple[bool, ...]] = []
    for col in columns:
        if not dedup_cols or col != dedup_cols[-1]:
            dedup_cols.append(col)
    if not dedup_cols:
        return ()
    matrix = list(zip(*dedup_cols))
    return tuple("".join("#" if value else "." for value in row) for row in matrix)


def _rotate_binary(rows: tuple[str, ...]) -> tuple[str, ...]:
    if not rows:
        return ()
    width = len(rows[0])
    return tuple(
        "".join(rows[len(rows) - 1 - y][x] for y in range(len(rows)))
        for x in range(width)
    )


class Observer:
    """Extracts stable, object-level facts from settled 64x64 frames.

    V1.5 keeps three views separate:
    raw pixels, a navigation state with volatile counters masked, and compact
    multi-colour objects. This prevents a moving two-colour player or a framed
    target+glyph from being split into unrelated memory entries.
    """

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
        # Colours that repeatedly form large map regions are structural.  Persisting
        # this set prevents a moving object from cutting a floor component into a
        # small fragment that would otherwise be misclassified as a new object.
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
            raise ObservationError(f"grid must be 2-D, got shape {arr.shape}")
        height, width = int(arr.shape[0]), int(arr.shape[1])
        if height < 1 or width < 1 or height > 64 or width > 64:
            raise ObservationError(f"grid shape must be within 64x64, got {arr.shape}")
        rows: list[tuple[int, ...]] = []
        for y in range(height):
            row: list[int] = []
            for x in range(width):
                value = int(arr[y, x])
                if not 0 <= value <= 15:
                    raise ObservationError("grid cell values must be in 0..15")
                row.append(value)
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
                raise ObservationError("frame sequence is empty")
            grid = self._last_grid
        else:
            grid = self.normalize_grid_value(raw_grid)
            self._last_grid = grid
        return self.analyze_grid(grid, render_grid(grid, self.image_size))

    def analyze_grid(
        self,
        grid: tuple[tuple[int, ...], ...],
        rgb: Image.Image | None = None,
    ) -> SceneSnapshot:
        height = len(grid)
        width = len(grid[0])
        counts = Counter(value for row in grid for value in row)
        max_count = max(counts.values())
        background = min(color for color, count in counts.items() if count == max_count)

        raw_components = self._components(grid, background)
        self._update_hud_model(grid, raw_components, background)
        volatile_cells = self._volatile_cells(width, height)

        world_rows = [list(row) for row in grid]
        for x, y in volatile_cells:
            if 0 <= x < width and 0 <= y < height:
                world_rows[y][x] = background
        world_grid = tuple(tuple(row) for row in world_rows)
        world_components = self._components(world_grid, background)

        # Infer map/background colours from large regions, then persist that
        # inference for the rest of the level.  A moving sprite can split one
        # structural region into several smaller components; classifying each
        # frame independently turns those fragments into spurious objects.
        for comp in world_components:
            box_area = _bbox_area(comp.bbox)
            in_hud = self._hud_panel_bbox is not None and (
                _bbox_intersection(comp.bbox, self._hud_panel_bbox) >= max(1, comp.area // 2)
            )
            map_large = (
                comp.area >= max(64, int(width * height * 0.035))
                or box_area >= int(width * height * 0.18)
                or (comp.touches_border and comp.area >= max(32, int(width * height * 0.02)))
            )
            if map_large and not in_hud and self._component_frame_score(comp) < 0.62:
                self._structural_colors.add(comp.color)

        structural_indices: set[int] = set()
        structural_colors: Counter[int] = Counter()
        for index, comp in enumerate(world_components):
            box_area = _bbox_area(comp.bbox)
            in_hud = self._hud_panel_bbox is not None and (
                _bbox_intersection(comp.bbox, self._hud_panel_bbox) >= max(1, comp.area // 2)
            )
            map_large = (
                comp.area >= max(64, int(width * height * 0.035))
                or box_area >= int(width * height * 0.18)
                or (comp.touches_border and comp.area >= max(32, int(width * height * 0.02)))
            )
            comp_width = comp.bbox[2] - comp.bbox[0] + 1
            comp_height = comp.bbox[3] - comp.bbox[1] + 1
            comp_fill = comp.area / max(1, comp_width * comp_height)
            compact_dense_frame = (
                self._component_frame_score(comp) >= 0.62
                and max(comp_width, comp_height) <= 16
                and comp_fill >= 0.55
            )
            inherited_structural = (
                comp.color in self._structural_colors
                and comp.area >= 5
                and not compact_dense_frame
            )
            if in_hud or map_large or inherited_structural:
                structural_indices.add(index)
                structural_colors[comp.color] += comp.area

        objects = self._build_composite_objects(
            world_grid,
            world_components,
            structural_indices,
            background,
        )
        objects = self._assign_track_ids(objects)

        counter_value = self._measure_counter(grid)
        if counter_value is not None:
            self._counter_capacity = max(self._counter_capacity or 0, counter_value)
        counter_capacity = self._counter_capacity
        counter_ratio = None
        if counter_value is not None and counter_capacity:
            counter_ratio = max(0.0, min(1.0, counter_value / counter_capacity))
        life_count = self._measure_lives(grid)
        relations = self._template_relations(objects)

        scene = SceneSnapshot(
            grid=grid,
            world_grid=world_grid,
            width=width,
            height=height,
            state_hash=_stable_grid_hash(world_grid),
            full_state_hash=_stable_grid_hash(grid),
            background_candidate=background,
            structural_colors=tuple(color for color, _ in structural_colors.most_common()),
            components=world_components,
            objects=objects,
            volatile_cells=frozenset(volatile_cells),
            hud_panel_bbox=self._hud_panel_bbox,
            counter_bbox=self._counter_bbox,
            counter_value=counter_value,
            counter_capacity=counter_capacity,
            counter_ratio=counter_ratio,
            life_count=life_count,
            template_relations=relations,
            summary="",
            rgb=rgb or render_grid(grid, self.image_size),
        )
        scene.summary = self._scene_summary(scene)
        scene.annotated_rgb = self._annotated_image(scene)
        self._previous_objects = objects
        return scene

    def _components(
        self,
        grid: tuple[tuple[int, ...], ...],
        background: int,
    ) -> tuple[ComponentObservation, ...]:
        height = len(grid)
        width = len(grid[0])
        visited: set[tuple[int, int]] = set()
        components: list[ComponentObservation] = []
        for y in range(height):
            for x in range(width):
                if (x, y) in visited or grid[y][x] == background:
                    continue
                color = grid[y][x]
                queue: deque[tuple[int, int]] = deque([(x, y)])
                visited.add((x, y))
                cells: list[tuple[int, int]] = []
                while queue:
                    cx, cy = queue.popleft()
                    cells.append((cx, cy))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + dx, cy + dy
                        if (
                            0 <= nx < width
                            and 0 <= ny < height
                            and (nx, ny) not in visited
                            and grid[ny][nx] == color
                        ):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                xs = [cell[0] for cell in cells]
                ys = [cell[1] for cell in cells]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                signature = tuple(sorted((cx - x0, cy - y0) for cx, cy in cells))
                components.append(
                    ComponentObservation(
                        color=color,
                        area=len(cells),
                        bbox=(x0, y0, x1, y1),
                        centroid=(sum(xs) / len(cells), sum(ys) / len(cells)),
                        touches_border=x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1,
                        shape_signature=signature,
                        cells=tuple(sorted(cells, key=lambda point: (point[1], point[0]))),
                    )
                )
        components.sort(key=lambda comp: (-comp.area, comp.color, comp.bbox[1], comp.bbox[0]))
        return tuple(components)

    def _update_hud_model(
        self,
        grid: tuple[tuple[int, ...], ...],
        components: Sequence[ComponentObservation],
        background: int,
    ) -> None:
        height = len(grid)
        width = len(grid[0])
        candidates: list[tuple[float, ComponentObservation]] = []
        for comp in components:
            x0, y0, x1, y1 = comp.bbox
            w, h = x1 - x0 + 1, y1 - y0 + 1
            if (
                y1 >= height - 1
                and y0 >= int(height * 0.82)
                and w >= max(12, int(width * 0.45))
                and h <= max(8, height // 8)
                and w >= 4 * max(1, h)
            ):
                candidates.append((w * 4 + comp.area - h, comp))
        if candidates:
            panel = max(candidates, key=lambda item: item[0])[1]
            self._hud_panel_bbox = panel.bbox
            if self._counter_fill_color is None:
                self._initialize_counter_model(grid, panel, background)
            if not self._life_slots:
                self._initialize_life_slots(components, panel)

    def _initialize_counter_model(
        self,
        grid: tuple[tuple[int, ...], ...],
        panel: ComponentObservation,
        background: int,
    ) -> None:
        x0, y0, x1, y1 = panel.bbox
        inner_x0, inner_x1 = min(x1, x0 + 1), max(x0, x1 - 1)
        inner_y0, inner_y1 = min(y1, y0 + 1), max(y0, y1 - 1)
        run_groups: dict[tuple[int, int, int], list[int]] = {}
        for y in range(inner_y0, inner_y1 + 1):
            row = grid[y]
            start = inner_x0
            while start <= inner_x1:
                color = row[start]
                end = start
                while end + 1 <= inner_x1 and row[end + 1] == color:
                    end += 1
                length = end - start + 1
                if color not in {panel.color, background} and length >= max(4, (x1 - x0 + 1) // 5):
                    run_groups.setdefault((color, start, end), []).append(y)
                start = end + 1
        if not run_groups:
            # Some counters use the playfield colour as their empty/fill colour.
            for y in range(inner_y0, inner_y1 + 1):
                row = grid[y]
                start = inner_x0
                while start <= inner_x1:
                    color = row[start]
                    end = start
                    while end + 1 <= inner_x1 and row[end + 1] == color:
                        end += 1
                    length = end - start + 1
                    if color != panel.color and length >= max(4, (x1 - x0 + 1) // 5):
                        run_groups.setdefault((color, start, end), []).append(y)
                    start = end + 1
        if not run_groups:
            return
        (fill_color, run_x0, run_x1), rows = max(
            run_groups.items(),
            key=lambda item: ((item[0][2] - item[0][1] + 1) * len(item[1]), len(item[1])),
        )
        self._counter_fill_color = fill_color
        self._counter_bbox = (run_x0, min(rows), run_x1, max(rows))
        self._counter_capacity = run_x1 - run_x0 + 1

    def _initialize_life_slots(
        self,
        components: Sequence[ComponentObservation],
        panel: ComponentObservation,
    ) -> None:
        groups: dict[tuple[int, tuple[tuple[int, int], ...]], list[ComponentObservation]] = {}
        for comp in components:
            if comp.color in {panel.color, self._counter_fill_color}:
                continue
            if _bbox_intersection(comp.bbox, panel.bbox) < comp.area:
                continue
            w = comp.bbox[2] - comp.bbox[0] + 1
            h = comp.bbox[3] - comp.bbox[1] + 1
            if 1 <= w <= 5 and 1 <= h <= 5 and 1 <= comp.area <= 20:
                groups.setdefault((comp.color, comp.shape_signature), []).append(comp)
        repeated = [items for items in groups.values() if len(items) >= 2]
        if not repeated:
            return
        slots = max(repeated, key=lambda items: (len(items), sum(comp.area for comp in items)))
        slots.sort(key=lambda comp: (comp.bbox[0], comp.bbox[1]))
        self._life_slots = [(comp.bbox, comp.color) for comp in slots[:8]]

    def _measure_counter(self, grid: tuple[tuple[int, ...], ...]) -> int | None:
        if self._counter_bbox is None or self._counter_fill_color is None:
            return None
        x0, y0, x1, y1 = self._counter_bbox
        best = 0
        for y in range(max(0, y0), min(len(grid) - 1, y1) + 1):
            longest = 0
            current = 0
            for x in range(max(0, x0), min(len(grid[0]) - 1, x1) + 1):
                if grid[y][x] == self._counter_fill_color:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
            best = max(best, longest)
        return best

    def _measure_lives(self, grid: tuple[tuple[int, ...], ...]) -> int | None:
        if not self._life_slots:
            return None
        count = 0
        for (x0, y0, x1, y1), color in self._life_slots:
            present = any(
                grid[y][x] == color
                for y in range(max(0, y0), min(len(grid) - 1, y1) + 1)
                for x in range(max(0, x0), min(len(grid[0]) - 1, x1) + 1)
            )
            count += int(present)
        return count

    def _volatile_cells(self, width: int, height: int) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        if self._counter_bbox is not None:
            x0, y0, x1, y1 = self._counter_bbox
            for y in range(max(0, y0), min(height - 1, y1) + 1):
                for x in range(max(0, x0), min(width - 1, x1) + 1):
                    cells.add((x, y))
        for (x0, y0, x1, y1), _ in self._life_slots:
            for y in range(max(0, y0), min(height - 1, y1) + 1):
                for x in range(max(0, x0), min(width - 1, x1) + 1):
                    cells.add((x, y))
        return cells

    @staticmethod
    def _component_frame_score(comp: ComponentObservation) -> float:
        x0, y0, x1, y1 = comp.bbox
        width = x1 - x0 + 1
        height = y1 - y0 + 1
        if width < 5 or height < 5:
            return 0.0
        # Semantic frames in these tasks are compact panels/slots.  Large or
        # highly elongated map regions can also trace much of their bounding
        # perimeter and must not be mistaken for a frame.
        if max(width, height) > 16 or max(width, height) / max(1, min(width, height)) > 1.8:
            return 0.0
        perimeter = {
            (x, y)
            for y in range(y0, y1 + 1)
            for x in range(x0, x1 + 1)
            if x in {x0, x1} or y in {y0, y1}
        }
        if len(perimeter) < 12:
            return 0.0
        return len(perimeter.intersection(comp.cells)) / len(perimeter)

    def _build_composite_objects(
        self,
        grid: tuple[tuple[int, ...], ...],
        components: Sequence[ComponentObservation],
        structural_indices: set[int],
        background: int,
    ) -> tuple[ObjectObservation, ...]:
        """Build semantically useful multi-colour objects without chain-merging.

        The first V1 releases used a proximity connected-component graph.  That
        graph was transitive: when the controlled sprite approached a framed
        target, the target, corridor fragment, and player became one giant
        "object".  Here framed regions are assembled first by containment, then
        only compact non-frame components are joined.  This preserves a target
        frame and its inner glyph while keeping an adjacent player separate.
        """
        del background  # Kept in the signature for observer API stability.
        candidate_indices = [
            index for index in range(len(components)) if index not in structural_indices
        ]
        unassigned: set[int] = set(candidate_indices)
        groups: list[list[int]] = []

        # Frames are strong compositional anchors.  Attach only components that
        # are spatially contained by the frame; never attach merely adjacent
        # components such as the player standing next to the target.
        frame_indices = sorted(
            (
                index
                for index in candidate_indices
                if self._component_frame_score(components[index]) >= 0.62
            ),
            key=lambda index: (-components[index].area, components[index].bbox),
        )
        for frame_index in frame_indices:
            if frame_index not in unassigned:
                continue
            frame = components[frame_index]
            group = [frame_index]
            for other_index in sorted(unassigned):
                if other_index == frame_index:
                    continue
                other = components[other_index]
                contained = _bbox_contains(frame.bbox, other.bbox)
                mostly_inside = _bbox_intersection(frame.bbox, other.bbox) >= max(
                    1, int(other.area * 0.8)
                )
                if contained and mostly_inside:
                    group.append(other_index)
            for index in group:
                unassigned.discard(index)
            groups.append(group)

        # Join the colour pieces of compact sprites.  Bound the *whole* group
        # rather than each pair so a chain of nearby objects cannot grow into a
        # large accidental composite.
        while unassigned:
            root = max(
                unassigned,
                key=lambda index: (components[index].area, -components[index].bbox[1]),
            )
            group = [root]
            unassigned.remove(root)
            group_box = components[root].bbox
            changed = True
            while changed:
                changed = False
                for other_index in sorted(
                    unassigned,
                    key=lambda index: (-components[index].area, components[index].bbox),
                ):
                    other = components[other_index]
                    gap_x, gap_y = _bbox_gap(group_box, other.bbox)
                    union = (
                        min(group_box[0], other.bbox[0]),
                        min(group_box[1], other.bbox[1]),
                        max(group_box[2], other.bbox[2]),
                        max(group_box[3], other.bbox[3]),
                    )
                    union_width = union[2] - union[0] + 1
                    union_height = union[3] - union[1] + 1
                    # Components must touch/overlap in both axes.  A one-cell
                    # spatial gap is not enough evidence that two sprites are
                    # one object.
                    compact_join = (
                        gap_x == 0
                        and gap_y == 0
                        and union_width <= 7
                        and union_height <= 7
                        and _bbox_area(union) <= 49
                    )
                    if not compact_join:
                        continue
                    group.append(other_index)
                    unassigned.remove(other_index)
                    group_box = union
                    changed = True
                    break
            groups.append(group)

        untracked: list[ObjectObservation] = []
        for group in groups:
            group_components = [components[index] for index in group]
            xs = [x for comp in group_components for x, _ in comp.cells]
            ys = [y for comp in group_components for _, y in comp.cells]
            if not xs or not ys:
                continue
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            coloured_cells = tuple(
                sorted(
                    (
                        (x, y, comp.color)
                        for comp in group_components
                        for x, y in comp.cells
                    ),
                    key=lambda item: (item[1], item[0], item[2]),
                )
            )
            color_counts = Counter(color for _, _, color in coloured_cells)
            normalized = tuple((x - x0, y - y0, color) for x, y, color in coloured_cells)
            binary = tuple(sorted({(x - x0, y - y0) for x, y, _ in coloured_cells}))
            signature_hasher = hashlib.blake2b(digest_size=12)
            signature_hasher.update(repr(normalized).encode("utf-8"))
            signature = signature_hasher.hexdigest()
            pattern = self._object_pattern(coloured_cells, (x0, y0, x1, y1))
            frame_color = self._frame_color(group_components, (x0, y0, x1, y1))
            inner_pattern = self._inner_pattern(
                coloured_cells, (x0, y0, x1, y1), frame_color
            )
            shape_label = self._shape_label(
                binary,
                (x0, y0, x1, y1),
                len(color_counts),
                frame_color,
            )
            near_edge = (
                x0 <= 2
                or y0 <= 2
                or x1 >= len(grid[0]) - 3
                or y1 >= len(grid) - 3
            )
            salience = (
                len(color_counts) * 3.0
                + len(group_components) * 1.5
                + min(25.0, len(coloured_cells) / 4.0)
                + (4.0 if frame_color is not None else 0.0)
                - (3.0 if near_edge else 0.0)
            )
            untracked.append(
                ObjectObservation(
                    track_id="",
                    bbox=(x0, y0, x1, y1),
                    centroid=(sum(xs) / len(xs), sum(ys) / len(ys)),
                    area=len(coloured_cells),
                    colors=tuple(sorted(color_counts)),
                    color_areas=tuple(sorted(color_counts.items())),
                    component_count=len(group_components),
                    cells=coloured_cells,
                    intrinsic_signature=signature,
                    binary_signature=binary,
                    shape_label=shape_label,
                    pattern=pattern,
                    inner_pattern=inner_pattern,
                    frame_color=frame_color,
                    near_edge=near_edge,
                    salience=salience,
                )
            )
        untracked.sort(key=lambda obj: (-obj.salience, obj.bbox[1], obj.bbox[0]))
        return tuple(untracked)

    @staticmethod
    def _object_pattern(
        cells: Sequence[tuple[int, int, int]],
        bbox: tuple[int, int, int, int],
    ) -> str:
        x0, y0, x1, y1 = bbox
        if x1 - x0 + 1 > 14 or y1 - y0 + 1 > 14:
            return f"<{x1-x0+1}x{y1-y0+1} pattern omitted>"
        mapping = {(x, y): color for x, y, color in cells}
        return "/".join(
            "".join(_hex_color(mapping[(x, y)]) if (x, y) in mapping else "." for x in range(x0, x1 + 1))
            for y in range(y0, y1 + 1)
        )

    @staticmethod
    def _frame_color(
        components: Sequence[ComponentObservation],
        bbox: tuple[int, int, int, int],
    ) -> int | None:
        x0, y0, x1, y1 = bbox
        perimeter = {
            (x, y)
            for y in range(y0, y1 + 1)
            for x in range(x0, x1 + 1)
            if x in {x0, x1} or y in {y0, y1}
        }
        if len(perimeter) < 12:
            return None
        for comp in sorted(components, key=lambda item: item.area, reverse=True):
            coverage = len(perimeter.intersection(comp.cells)) / len(perimeter)
            if coverage >= 0.62:
                return comp.color
        return None

    @staticmethod
    def _inner_pattern(
        cells: Sequence[tuple[int, int, int]],
        bbox: tuple[int, int, int, int],
        frame_color: int | None,
    ) -> str:
        if frame_color is None:
            return ""
        inner = [(x, y, color) for x, y, color in cells if color != frame_color]
        if not inner:
            return ""
        xs = [x for x, _, _ in inner]
        ys = [y for _, y, _ in inner]
        inner_bbox = (min(xs), min(ys), max(xs), max(ys))
        return Observer._object_pattern(inner, inner_bbox)

    @staticmethod
    def _shape_label(
        binary: Sequence[tuple[int, int]],
        bbox: tuple[int, int, int, int],
        color_count: int,
        frame_color: int | None,
    ) -> str:
        if frame_color is not None:
            return "frame_with_inner_pattern"
        width = bbox[2] - bbox[0] + 1
        height = bbox[3] - bbox[1] + 1
        points = set(binary)
        if width <= 7 and height <= 7 and len(points) >= 3:
            degrees = []
            for x, y in points:
                degree = sum((x + dx, y + dy) in points for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                degrees.append(degree)
            if max(degrees, default=0) >= 3:
                return "cross_or_branch_like"
        fill = len(points) / max(1, width * height)
        if color_count >= 2 and fill >= 0.75:
            return "compact_multicolor_block"
        if color_count >= 2:
            return "compact_multicolor_glyph"
        return "single_color_glyph"

    def _assign_track_ids(
        self,
        objects: Sequence[ObjectObservation],
    ) -> tuple[ObjectObservation, ...]:
        previous = list(self._previous_objects)
        available_previous = set(range(len(previous)))
        assignments: dict[int, str] = {}

        exact_pairs: list[tuple[float, int, int]] = []
        for new_index, current in enumerate(objects):
            for old_index, old in enumerate(previous):
                if current.intrinsic_signature == old.intrinsic_signature:
                    distance = abs(current.centroid[0] - old.centroid[0]) + abs(current.centroid[1] - old.centroid[1])
                    exact_pairs.append((distance, new_index, old_index))
        for _distance, new_index, old_index in sorted(exact_pairs):
            if new_index in assignments or old_index not in available_previous:
                continue
            assignments[new_index] = previous[old_index].track_id
            available_previous.remove(old_index)

        overlap_pairs: list[tuple[float, int, int]] = []
        for new_index, current in enumerate(objects):
            if new_index in assignments:
                continue
            for old_index in available_previous:
                old = previous[old_index]
                intersection = _bbox_intersection(current.bbox, old.bbox)
                union = _bbox_area(current.bbox) + _bbox_area(old.bbox) - intersection
                iou = intersection / max(1, union)
                color_overlap = len(set(current.colors).intersection(old.colors)) / max(1, len(set(current.colors).union(old.colors)))
                area_ratio = min(current.area, old.area) / max(1, max(current.area, old.area))
                score = 2.5 * iou + color_overlap + area_ratio
                if score >= 1.35:
                    overlap_pairs.append((-score, new_index, old_index))
        for _negative_score, new_index, old_index in sorted(overlap_pairs):
            if new_index in assignments or old_index not in available_previous:
                continue
            assignments[new_index] = previous[old_index].track_id
            available_previous.remove(old_index)

        tracked: list[ObjectObservation] = []
        for index, obj in enumerate(objects):
            track_id = assignments.get(index)
            if track_id is None:
                track_id = f"O{self._next_track_id}"
                self._next_track_id += 1
            tracked.append(
                ObjectObservation(
                    track_id=track_id,
                    bbox=obj.bbox,
                    centroid=obj.centroid,
                    area=obj.area,
                    colors=obj.colors,
                    color_areas=obj.color_areas,
                    component_count=obj.component_count,
                    cells=obj.cells,
                    intrinsic_signature=obj.intrinsic_signature,
                    binary_signature=obj.binary_signature,
                    shape_label=obj.shape_label,
                    pattern=obj.pattern,
                    inner_pattern=obj.inner_pattern,
                    frame_color=obj.frame_color,
                    near_edge=obj.near_edge,
                    salience=obj.salience,
                )
            )
        tracked.sort(key=lambda obj: (int(obj.track_id[1:]) if obj.track_id[1:].isdigit() else 9999))
        return tuple(tracked)

    @staticmethod
    def _template_relations(
        objects: Sequence[ObjectObservation],
    ) -> tuple[dict[str, Any], ...]:
        framed = [obj for obj in objects if obj.frame_color is not None and obj.inner_pattern]
        relations: list[dict[str, Any]] = []
        for left_index, left in enumerate(framed):
            left_rows = _compress_binary_rows(_pattern_rows(left.inner_pattern))
            if not left_rows:
                continue
            for right in framed[left_index + 1 :]:
                right_rows = _compress_binary_rows(_pattern_rows(right.inner_pattern))
                if not right_rows:
                    continue
                rotated = left_rows
                matching_rotation: int | None = None
                for turns in range(4):
                    if rotated == right_rows:
                        matching_rotation = turns
                        break
                    rotated = _rotate_binary(rotated)
                if matching_rotation is None:
                    continue
                left_inner_colors = sorted({ch for row in _pattern_rows(left.inner_pattern) for ch in row if ch != "."})
                right_inner_colors = sorted({ch for row in _pattern_rows(right.inner_pattern) for ch in row if ch != "."})
                same_colors = left_inner_colors == right_inner_colors
                relations.append(
                    {
                        "left": left.track_id,
                        "right": right.track_id,
                        "same_shape_under_rotation": True,
                        "quarter_turns_left_to_right": matching_rotation,
                        "same_inner_colors": same_colors,
                        "exact_inner_match": matching_rotation == 0 and same_colors,
                        "edge_vs_world": left.near_edge != right.near_edge,
                    }
                )
        return tuple(relations)

    def compare(
        self,
        before: SceneSnapshot,
        after: SceneSnapshot,
    ) -> TransitionReport:
        changed: list[tuple[int, int]] = []
        world_changed: list[tuple[int, int]] = []
        volatile_union = set(before.volatile_cells) | set(after.volatile_cells)
        for y in range(max(before.height, after.height)):
            for x in range(max(before.width, after.width)):
                old = before.grid[y][x] if y < before.height and x < before.width else None
                new = after.grid[y][x] if y < after.height and x < after.width else None
                if old != new:
                    changed.append((x, y))
                    if (x, y) not in volatile_union:
                        world_changed.append((x, y))
        changed_bbox = None
        if changed:
            xs = [x for x, _ in changed]
            ys = [y for _, y in changed]
            changed_bbox = (min(xs), min(ys), max(xs), max(ys))

        before_by_id = {obj.track_id: obj for obj in before.objects}
        after_by_id = {obj.track_id: obj for obj in after.objects}
        shared_ids = sorted(set(before_by_id).intersection(after_by_id))
        moved: list[dict[str, Any]] = []
        transformed: list[dict[str, Any]] = []
        for track_id in shared_ids:
            old = before_by_id[track_id]
            new = after_by_id[track_id]
            dx = int(round(new.centroid[0] - old.centroid[0]))
            dy = int(round(new.centroid[1] - old.centroid[1]))
            if dx or dy:
                moved.append(
                    {
                        "object_id": track_id,
                        "from_bbox": old.bbox,
                        "to_bbox": new.bbox,
                        "dx": dx,
                        "dy": dy,
                        "shape": new.shape_label,
                    }
                )
            if old.intrinsic_signature != new.intrinsic_signature:
                transformed.append(
                    {
                        "object_id": track_id,
                        "before_type": old.shape_label,
                        "after_type": new.shape_label,
                        "before_pattern": old.pattern,
                        "after_pattern": new.pattern,
                    }
                )
        appeared = sorted(set(after_by_id) - set(before_by_id))
        disappeared = sorted(set(before_by_id) - set(after_by_id))

        counter_delta = None
        if before.counter_value is not None and after.counter_value is not None:
            counter_delta = after.counter_value - before.counter_value
        life_delta = None
        if before.life_count is not None and after.life_count is not None:
            life_delta = after.life_count - before.life_count

        candidate_moves = [
            item
            for item in moved
            if not after_by_id[item["object_id"]].near_edge
            and after_by_id[item["object_id"]].shape_label != "frame_with_inner_pattern"
            and after_by_id[item["object_id"]].area <= 160
        ]
        controlled_candidate = None
        if candidate_moves:
            candidate_moves.sort(
                key=lambda item: (
                    -abs(int(item["dx"])) - abs(int(item["dy"])),
                    after_by_id[item["object_id"]].area,
                )
            )
            controlled_candidate = candidate_moves[0]["object_id"]

        only_candidate_moved = bool(controlled_candidate) and all(
            item["object_id"] == controlled_candidate or after_by_id[item["object_id"]].near_edge
            for item in moved
        )
        is_simple_translation = (
            bool(controlled_candidate)
            and only_candidate_moved
            and not transformed
            and not appeared
            and not disappeared
        )
        world_noop = before.world_grid == after.world_grid
        full_noop = before.grid == after.grid
        effective_noop = world_noop and (counter_delta is None or counter_delta <= 0) and life_delta in (None, 0)

        counter_reset = (
            counter_delta is not None
            and before.counter_capacity is not None
            and counter_delta >= max(2, int(before.counter_capacity * 0.5))
        )
        retry_detected = bool(counter_reset and life_delta is not None and life_delta < 0)
        interaction_event = bool(
            retry_detected
            or (counter_delta is not None and counter_delta > 1)
            or life_delta not in (None, 0)
            or transformed
            or appeared
            or disappeared
        ) and not is_simple_translation

        parts = [
            f"changed_cells={len(changed)}",
            f"world_changed_cells={len(world_changed)}",
            f"effective_noop={str(effective_noop).lower()}",
            f"simple_translation={str(is_simple_translation).lower()}",
        ]
        if controlled_candidate:
            parts.append(f"controlled_candidate={controlled_candidate}")
        if moved:
            parts.append(
                "moved="
                + ",".join(
                    f"{item['object_id']}({item['dx']},{item['dy']})" for item in moved[:6]
                )
            )
        if transformed:
            parts.append("transformed=" + ",".join(item["object_id"] for item in transformed[:6]))
        if appeared:
            parts.append("appeared=" + ",".join(appeared[:6]))
        if disappeared:
            parts.append("disappeared=" + ",".join(disappeared[:6]))
        if counter_delta is not None:
            parts.append(f"counter_delta={counter_delta}")
        if after.counter_value is not None:
            parts.append(f"counter={after.counter_value}/{after.counter_capacity}")
        if life_delta is not None:
            parts.append(f"life_delta={life_delta}")
        if retry_detected:
            parts.append("retry_detected=true")

        return TransitionReport(
            changed_cell_count=len(changed),
            world_changed_cell_count=len(world_changed),
            hud_changed_cell_count=len(changed) - len(world_changed),
            changed_bbox=changed_bbox,
            world_noop=world_noop,
            full_visual_noop=full_noop,
            effective_noop=effective_noop,
            counter_delta=counter_delta,
            counter_ratio_after=after.counter_ratio,
            life_delta=life_delta,
            retry_detected=retry_detected,
            moved_objects=moved,
            transformed_objects=transformed,
            appeared_object_ids=appeared,
            disappeared_object_ids=disappeared,
            controlled_candidate_id=controlled_candidate,
            is_simple_translation=is_simple_translation,
            interaction_event=interaction_event,
            summary="; ".join(parts),
        )

    def _scene_summary(self, scene: SceneSnapshot) -> str:
        lines = [
            f"grid={scene.width}x{scene.height}",
            f"navigation_state_hash={scene.state_hash[:12]}",
            f"background_candidate={scene.background_candidate}",
            f"structural_colors={list(scene.structural_colors[:4])}",
        ]
        if scene.counter_value is not None:
            lines.append(
                f"step_counter_like={scene.counter_value}/{scene.counter_capacity} ratio={scene.counter_ratio}"
            )
        if scene.life_count is not None:
            lines.append(f"life_like_slots={scene.life_count}")
        for obj in sorted(scene.objects, key=lambda item: -item.salience)[:16]:
            lines.append(
                f"{obj.track_id}: type={obj.shape_label} bbox={obj.bbox} colors={dict(obj.color_areas)} "
                f"area={obj.area} near_edge={obj.near_edge} pattern={obj.pattern}"
                + (f" inner={obj.inner_pattern}" if obj.inner_pattern else "")
            )
        if scene.template_relations:
            lines.append(f"framed_template_relations={list(scene.template_relations)}")
        return "\n".join(lines)

    def _annotated_image(self, scene: SceneSnapshot) -> Image.Image:
        image = (scene.rgb or render_grid(scene.grid, self.image_size)).copy()
        draw = ImageDraw.Draw(image)
        sx = image.width / scene.width
        sy = image.height / scene.height
        for obj in scene.objects:
            x0, y0, x1, y1 = obj.bbox
            box = (
                int(x0 * sx),
                int(y0 * sy),
                int((x1 + 1) * sx - 1),
                int((y1 + 1) * sy - 1),
            )
            draw.rectangle(box, outline=(255, 255, 255), width=max(1, int(min(sx, sy) // 3)))
            label = f"{obj.track_id}:{obj.shape_label[:8]}"
            tx, ty = box[0] + 1, max(0, box[1] - 12)
            text_box = draw.textbbox((tx, ty), label)
            draw.rectangle(text_box, fill=(0, 0, 0))
            draw.text((tx, ty), label, fill=(255, 255, 255))
        if scene.counter_bbox is not None:
            x0, y0, x1, y1 = scene.counter_bbox
            draw.rectangle(
                (
                    int(x0 * sx),
                    int(y0 * sy),
                    int((x1 + 1) * sx - 1),
                    int((y1 + 1) * sy - 1),
                ),
                outline=(255, 255, 255),
                width=1,
            )
        return image

@dataclass
class VLMRequest:
    text_prompt: str
    current_rgb: Image.Image
    previous_rgb: Image.Image | None = None
    analysis_rgb: Image.Image | None = None
    max_new_tokens: int = 1100
class VLMBackend(Protocol):
    @property
    def available(self) -> bool:
        ...

    def decide(self, request: VLMRequest) -> Any:
        ...


class DecisionLogger:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.path: Path | None = None
        if config.log_dir:
            try:
                root = Path(config.log_dir)
                root.mkdir(parents=True, exist_ok=True)
                self.path = root / f"agent_v1_7_{int(time.time())}_{os.getpid()}.jsonl"
            except Exception:
                self.path = None

    def log_event(self, event: str, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        safe = {"event": event, "time": round(time.time(), 3), **_json_safe(payload)}
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(safe, ensure_ascii=True, default=str) + "\n")
        except Exception:
            pass

    def log_exception(self, exc: BaseException) -> None:
        self.log_event(
            "exception",
            {"type": type(exc).__name__, "message": str(exc)[:500], "trace": traceback.format_exc()[-2000:]},
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, deque)):
        return [_json_safe(v) for v in list(value)]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if dataclass_is_instance(value):
        return {
            str(name): _json_safe(getattr(value, name))
            for name in getattr(value, "__dataclass_fields__", {})
        }
    return value


def dataclass_is_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


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
        model_path = self.config.model_path
        if not Path(model_path).exists():
            self.disable_for_episode(f"model path missing: {model_path}")
            return False
        with _MODEL_CACHE_LOCK:
            if model_path in _MODEL_CACHE:
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
                if not torch.cuda.is_available() and not _bool_env_any(
                    ("ARC_V15_ALLOW_CPU_VLM", "ARC_V13_ALLOW_CPU_VLM", "ARC_V12_ALLOW_CPU_VLM", "ARC_V3_ALLOW_CPU_VLM", "ARC_V2_ALLOW_CPU_VLM", "ARC_V1_ALLOW_CPU_VLM"),
                    False,
                ):
                    self.disable_for_episode("CUDA unavailable and CPU VLM disabled")
                    return False
                processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
                kwargs = dict(local_files_only=True, device_map="auto", low_cpu_mem_usage=True)
                kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
                try:
                    model = ModelClass.from_pretrained(model_path, **kwargs)
                except TypeError:
                    dtype = kwargs.pop("torch_dtype", None)
                    if dtype is not None:
                        kwargs["dtype"] = dtype
                    model = ModelClass.from_pretrained(model_path, **kwargs)
                model.eval()
                _MODEL_CACHE[model_path] = (processor, model)
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
            messages = [
                {"role": "system", "content": [{"type": "text", "text": V1_SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ]
            kwargs = dict(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            try:
                inputs = processor.apply_chat_template(**kwargs, enable_thinking=False)
            except TypeError:
                inputs = processor.apply_chat_template(**kwargs)
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)
            elif isinstance(inputs, dict):
                inputs = {
                    key: value.to(model.device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
            input_len = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            return processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        except Exception as exc:
            self._handle_error(f"generate error: {exc}")
            return None


def _pil_image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _vlm_backend_mode(config: AgentConfig) -> str:
    return (config.vlm_backend or "local").strip().lower()


def vlm_uses_remote_api(config: AgentConfig | None = None) -> bool:
    backend = _vlm_backend_mode(config or AgentConfig.from_env())
    return backend in {"api", "openai", "remote", "http"}


def _extract_openai_message_text(message: Any) -> str | None:
    content = getattr(message, "content", None)
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    if content is not None and str(content).strip():
        return str(content)
    dump = message.model_dump() if hasattr(message, "model_dump") else {}
    for key in ("reasoning_content", "reasoning", "refusal"):
        value = dump.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


class OpenAICompatibleBackend:
    def __init__(self, config: AgentConfig, logger: DecisionLogger):
        self.config = config
        self.logger = logger
        self._client: Any | None = None
        self._errors = 0
        missing: list[str] = []
        if not config.vlm_api_base_url:
            missing.append("ARC_V15_VLM_API_BASE_URL")
        if not config.vlm_api_key:
            missing.append("ARC_V15_VLM_API_KEY/INF_API_KEY")
        if not config.vlm_api_model:
            missing.append("ARC_V15_VLM_API_MODEL")
        self._available = bool(config.enable_vlm and not missing)
        if config.enable_vlm and missing:
            self.disable_for_episode(f"missing remote VLM config: {', '.join(missing)}")

    @property
    def available(self) -> bool:
        return self._available

    def disable_for_episode(self, reason: str) -> None:
        self._available = False
        self.logger.log_event("vlm_disabled", {"reason": reason})

    def _handle_error(self, reason: str) -> None:
        self._errors += 1
        self.logger.log_event("vlm_error", {"reason": reason[:500], "count": self._errors})
        if self._errors >= 2:
            self.disable_for_episode(reason)

    def _client_instance(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config.vlm_api_key,
                base_url=self.config.vlm_api_base_url,
                timeout=self.config.vlm_api_timeout_s,
            )
        return self._client

    def decide(self, request: VLMRequest) -> str | None:
        if not self._available:
            return None
        content: list[dict[str, Any]] = []
        if request.previous_rgb is not None:
            content.append({"type": "image_url", "image_url": {"url": _pil_image_to_data_url(request.previous_rgb)}})
        content.append({"type": "image_url", "image_url": {"url": _pil_image_to_data_url(request.current_rgb)}})
        if request.analysis_rgb is not None:
            content.append({"type": "image_url", "image_url": {"url": _pil_image_to_data_url(request.analysis_rgb)}})
        content.append({"type": "text", "text": request.text_prompt})
        try:
            response = self._client_instance().chat.completions.create(
                model=self.config.vlm_api_model,
                messages=[
                    {"role": "system", "content": V1_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=request.max_new_tokens,
                temperature=0,
                timeout=self.config.vlm_api_timeout_s,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = _extract_openai_message_text(response.choices[0].message)
            if text is None or not text.strip():
                self._handle_error("empty model output")
                return None
            return text
        except Exception as exc:
            self._handle_error(f"remote generate error: {exc}")
            return None


def make_vlm_backend(config: AgentConfig, logger: DecisionLogger) -> VLMBackend:
    if vlm_uses_remote_api(config):
        return OpenAICompatibleBackend(config, logger)
    return Qwen35Backend(config, logger)





V1_SYSTEM_PROMPT = """You control an abstract ARC-AGI-3 grid game.
Use the raw image, the annotated image, and the exact object-pattern table together.
Annotated labels O1, O2, ... identify multi-colour composite objects; never split one labelled object back into unrelated colour components.
Facts from the observer are authoritative. Your interpretations are hypotheses until transitions verify them.
Do not infer cultural meaning from colours or icons.
Only propose actions listed in legal_actions_now, except NAVIGATE as an internal semantic waypoint.
ACTION1..ACTION4 have weak up/down/left/right priors. ACTION5 is contextual.
ACTION6 is coordinate-based and requires x,y. Prefer intent_proposals using click_candidate/click_object; the controller will compile candidate IDs or object IDs to coordinates. If ACTION6 is absent, clicking is impossible.
ACTION7 is Undo. Never propose ACTION7 for ordinary exploration, solving, fallback, or route replay. The controller alone may use ACTION7 for rollback after a controlled experiment.
Never propose RESET except when the state is terminal or the controller explicitly asks for reset recovery.
Do not output CLICK, MOVE, GO, PATHFIND, or free-form action names. Use intent_proposals, ACTION1..ACTION6, or NAVIGATE with target_object_id.
A pure translation of the controlled object is normal progress and does not invalidate a plan.
Compare edge/status framed patterns with world framed patterns. A mismatch can imply that an intermediate compact object must transform state before the frame can be completed.
Do not repeat a target or route listed under failed_targets unless new evidence changes the hypothesis.
Return one JSON object only. No markdown, no prose outside JSON."""


@dataclass
class VLMResult:
    observations: list[str] = field(default_factory=list)
    object_updates: list[dict[str, Any]] = field(default_factory=list)
    action_interpretation: dict[str, Any] = field(default_factory=dict)
    goal_update: dict[str, Any] = field(default_factory=dict)
    mechanic_updates: list[dict[str, Any]] = field(default_factory=list)
    counter_update: dict[str, Any] = field(default_factory=dict)
    rejected_hypotheses: list[str] = field(default_factory=list)
    ready_to_solve: bool = False
    plan_goal: str = ""
    target_object_id: str = ""
    game_summary: str = ""
    level_summary: str = ""
    mode: str = ""
    grounded_observations: list[str] = field(default_factory=list)
    hypothesis_updates: list[dict[str, Any]] = field(default_factory=list)
    goal_schema_patch: dict[str, Any] = field(default_factory=dict)
    level_instantiation_patch: dict[str, Any] = field(default_factory=dict)
    level_theory_patch: dict[str, Any] = field(default_factory=dict)
    intent_proposals: list[dict[str, Any]] = field(default_factory=list)
    failure_constraints: list[dict[str, Any]] = field(default_factory=list)
    proposed_memory_patch: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)


def _short_string(value: Any, limit: int = 300) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _has_explicit_value(value: Any) -> bool:
    return value not in (None, "", "None", "none", "null", "NULL")


def _extract_action6_xy_from_text(text: str) -> tuple[int, int] | None:
    upper = str(text).upper()
    patterns = [
        r"\bACTION6\s*[:(\[]\s*(-?\d+)\s*[,， ]\s*(-?\d+)\s*[)\]]?",
        r"\bACTION6\b.*?\bX\s*[:=]\s*(-?\d+).*?\bY\s*[:=]\s*(-?\d+)",
        r"\bACTION6\b.*?\(\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _extract_target_object_id_from_text(text: str) -> str:
    match = re.search(r"\bO\d+\b", str(text).upper())
    return match.group(0) if match else ""


def _balanced_objects(text: str) -> list[str]:
    objects: list[str] = []
    starts: list[int] = []
    in_string = False
    quote = ""
    escape = False
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                objects.append(text[start : index + 1])
                start = -1
    return objects


def _repair_truncated_json(text: str) -> str | None:
    """Best-effort recovery of a JSON object that was cut off mid-generation.

    VLM outputs frequently hit the token cap before closing the JSON, leaving an
    unbalanced object that neither `_balanced_objects` nor `json.loads` can read.
    We close the dangling string/brackets and drop a trailing incomplete pair so
    the already-completed fields (observations, object_updates, an early plan...)
    can still be salvaged into a structured result instead of being discarded.
    """
    start = text.find("{")
    if start < 0:
        return None
    fragment = text[start:]
    in_string = False
    escape = False
    quote = ""
    stack: list[str] = []
    for char in fragment:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in {"}", "]"} and stack:
            stack.pop()
    if not stack and not in_string:
        return None
    repaired = fragment
    if in_string:
        repaired += quote
    repaired = repaired.rstrip()

    def close(candidate: str) -> str:
        candidate = re.sub(r",\s*$", "", candidate.rstrip())
        return candidate + "".join(reversed(stack))

    closed = close(repaired)
    try:
        json.loads(closed)
        return closed
    except Exception:
        pass

    # Drop a dangling "key": or "key" with no value (a key emitted right before
    # the cutoff). We try this only after simple closing fails so truncated
    # string values such as {"observations":["foo are preserved.
    repaired = re.sub(r'(?:,\s*)?"(?:[^"\\]|\\.)*"\s*:\s*$', "", repaired).rstrip()
    repaired = re.sub(r'(?:,\s*)?"(?:[^"\\]|\\.)*"\s*$', "", repaired).rstrip()
    return close(repaired)


def _parse_payload(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    fenced = re.findall(r"```(?:json|python)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced + _balanced_objects(cleaned)
    if not candidates:
        candidates = [cleaned]
    candidates.sort(key=len, reverse=True)
    for candidate in candidates:
        candidate = candidate.strip()
        repairs = [
            candidate,
            re.sub(r",\s*([}\]])", r"\1", candidate),
        ]
        for repaired in repairs:
            try:
                value = json.loads(repaired)
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
            try:
                value = ast.literal_eval(repaired)
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
    # Last resort: the largest candidate is likely a truncated object; try to
    # close it so completed fields are recoverable instead of dropping everything.
    closed = _repair_truncated_json(candidates[0])
    if closed:
        for repaired in (closed, re.sub(r",\s*([}\]])", r"\1", closed)):
            try:
                value = json.loads(repaired)
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
    return None


def _safe_action_dict(value: Any) -> dict[str, Any] | None:
    original_value = value
    if isinstance(value, str):
        value = {"name": _short_string(value, 180)}
    if not isinstance(value, dict):
        return None
    raw_name = _short_string(value.get("name") or value.get("action"), 180)
    name = _short_string(raw_name, 40).upper()
    if not name:
        return None
    expected_predicates = value.get("expected_predicates")
    if not isinstance(expected_predicates, list):
        expected_predicates = []
    target_id = _short_string(
        value.get("target_object_id") or value.get("target") or value.get("object_id"),
        24,
    ).upper()
    if not target_id:
        target_id = _extract_target_object_id_from_text(raw_name)
    result = {
        "name": name,
        "target_object_id": target_id,
        "purpose": _short_string(value.get("purpose") or value.get("why"), 240),
        "expected_change": _short_string(
            value.get("expected_change") or value.get("checkpoint"), 240
        ),
        "expected_predicates": expected_predicates[:8],
        "risk": _short_string(value.get("risk"), 20).lower() or "low",
        "reversible": bool(value.get("reversible", True)),
        "information_gain": value.get("information_gain", value.get("expected_information_gain", 0.5)),
        "goal_progress": value.get("goal_progress", 0.45),
    }
    for coord_name in ("x", "y"):
        if _has_explicit_value(value.get(coord_name)):
            result[coord_name] = value.get(coord_name)
    xy = _extract_action6_xy_from_text(raw_name if isinstance(original_value, str) else str(raw_name))
    if xy is not None:
        result["name"] = "ACTION6"
        result["x"], result["y"] = xy
    if _has_explicit_value(value.get("action6_candidate_id") or value.get("candidate_id")):
        result["action6_candidate_id"] = _short_string(value.get("action6_candidate_id") or value.get("candidate_id"), 24).upper()
    if _has_explicit_value(value.get("intent")):
        result["intent"] = _short_string(value.get("intent"), 60).lower()
    if _has_explicit_value(value.get("intent_type")):
        result["intent_type"] = _short_string(value.get("intent_type"), 60).lower()
    return result


def _regex_action_plan(raw: str) -> list[dict[str, Any]]:
    text = raw.upper()
    result: list[dict[str, Any]] = []
    for name in re.findall(r"\b(ACTION[1-5])\b", text):
        result.append({"name": name, "target_object_id": "", "purpose": "regex recovered primitive action", "expected_change": ""})
        if len(result) >= 8:
            break
    for match in re.finditer(r"ACTION6[^\n;.]*", text):
        xy = _extract_action6_xy_from_text(match.group(0))
        if xy is not None:
            result.append({
                "name": "ACTION6",
                "x": xy[0],
                "y": xy[1],
                "target_object_id": "",
                "purpose": "regex recovered coordinate click",
                "expected_change": "",
            })
            break
    return result[:8]


def parse_vlm_result(raw: Any) -> VLMResult | None:
    payload = _parse_payload(raw)
    if payload is None:
        if isinstance(raw, str) and _short_string(raw, 1):
            excerpt = _short_string(raw, 1500)
            return VLMResult(
                observations=[_short_string(raw, 700)],
                proposed_memory_patch={"raw_invalid_vlm_excerpt": excerpt},
                mode="INVALID_PARTIAL",
            )
        return None

    _normalize_legacy_vlm_payload(payload)

    def string_list(key: str, limit: int) -> list[str]:
        value = payload.get(key)
        if not isinstance(value, list):
            return []
        return [_short_string(item, 260) for item in value[:limit] if _short_string(item, 260)]

    def dict_list(key: str, limit: int) -> list[dict[str, Any]]:
        value = payload.get(key)
        if not isinstance(value, list):
            return []
        result: list[dict[str, Any]] = []
        for item in value[:limit]:
            if not isinstance(item, dict):
                continue
            safe: dict[str, Any] = {}
            for field_name, field_value in item.items():
                key_name = str(field_name)[:50]
                if field_name == "confidence":
                    safe[key_name] = _clamp01(field_value)
                elif isinstance(field_value, (dict, list, int, float, bool)) or field_value is None:
                    safe[key_name] = field_value
                else:
                    safe[key_name] = _short_string(field_value, 280)
            result.append(safe)
        return result

    def safe_dict(key: str) -> dict[str, Any]:
        value = payload.get(key)
        return dict(value) if isinstance(value, dict) else {}

    goal = payload.get("goal_update") if isinstance(payload.get("goal_update"), dict) else {}
    safe_goal = {
        "claim": _short_string(goal.get("claim"), 320),
        "scope": _short_string(goal.get("scope"), 20).lower() or "level",
        "confidence": _clamp01(goal.get("confidence")),
        "evidence": _short_string(goal.get("evidence"), 320),
        "status": _short_string(goal.get("status"), 24).lower() or "candidate",
    }
    interpretation = (
        payload.get("action_interpretation")
        if isinstance(payload.get("action_interpretation"), dict)
        else {}
    )
    safe_interpretation = {
        "action": _short_string(interpretation.get("action"), 40).upper(),
        "effect": _short_string(interpretation.get("effect"), 280),
        "status": _short_string(interpretation.get("status"), 24).lower() or "unknown",
        "evidence": _short_string(interpretation.get("evidence"), 280),
    }
    counter = payload.get("counter_update") if isinstance(payload.get("counter_update"), dict) else {}
    safe_counter = {
        "claim": _short_string(counter.get("claim"), 240),
        "confidence": _clamp01(counter.get("confidence")),
        "strategy": _short_string(counter.get("strategy"), 260),
    }
    target = _short_string(
        payload.get("target_object_id")
        or payload.get("plan_target_id")
        or "",
        24,
    ).upper()

    # A semantically useful analysis without a plan is still valid. The
    # deterministic controller can navigate and explore from the object model.
    return VLMResult(
        observations=string_list("observations", 10),
        object_updates=dict_list("object_updates", 12),
        action_interpretation=safe_interpretation,
        goal_update=safe_goal,
        mechanic_updates=dict_list("mechanic_updates", 10),
        counter_update=safe_counter,
        rejected_hypotheses=string_list("rejected_hypotheses", 8),
        ready_to_solve=bool(payload.get("ready_to_solve", False)),
        plan_goal=_short_string(payload.get("plan_goal"), 320),
        target_object_id=target,
        game_summary=_short_string(payload.get("game_summary"), 700),
        level_summary=_short_string(payload.get("level_summary"), 700),
        mode=_short_string(payload.get("mode"), 60),
        grounded_observations=string_list("grounded_observations", 12),
        hypothesis_updates=dict_list("hypothesis_updates", 12),
        goal_schema_patch=safe_dict("goal_schema_patch"),
        level_instantiation_patch=safe_dict("level_instantiation_patch"),
        level_theory_patch=safe_dict("level_theory") or safe_dict("level_theory_patch"),
        intent_proposals=dict_list("intent_proposals", 12) or dict_list("intents", 12),
        failure_constraints=dict_list("failure_constraints", 12),
        proposed_memory_patch=safe_dict("proposed_memory_patch"),
        decision=safe_dict("decision"),
    )


_DECISION_SINGLE_KINDS = {"single_experiment", "single_step", "experiment", "action", "next_action"}
_DECISION_PLAN_KINDS = {"plan_candidate", "candidate_plan", "plan"}


def _normalize_legacy_vlm_payload(payload: dict[str, Any]) -> None:
    """Fold deprecated top-level VLM fields into the V1.7 decision contract once."""
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        decision = {}
        payload["decision"] = decision
    kind = _short_string(decision.get("kind"), 80).lower()
    protected_kinds = _DECISION_PLAN_KINDS | {"repair_decision", "memory_update_only"}

    next_action = payload.get("next_action")
    if isinstance(next_action, dict) and next_action and kind not in _DECISION_SINGLE_KINDS | protected_kinds:
        decision.setdefault("kind", "single_experiment")
        decision.setdefault("intents", [next_action])

    experiments = payload.get("recommended_experiments")
    if isinstance(experiments, list) and experiments:
        payload.setdefault("intent_proposals", experiments)
        kind = _short_string(decision.get("kind"), 80).lower()
        if kind not in _DECISION_SINGLE_KINDS | protected_kinds:
            decision["kind"] = "single_experiment"
            decision.setdefault("intents", experiments[:4])

    kind = _short_string(decision.get("kind"), 80).lower()
    if kind not in protected_kinds:
        for key in ("plan", "plan_proposals"):
            steps = payload.get(key)
            if isinstance(steps, list) and steps:
                decision["kind"] = "plan_candidate"
                decision.setdefault("steps", steps)
                decision.setdefault(
                    "goal",
                    _short_string(payload.get("plan_goal"), 320) or key,
                )
                break


def _decision_kind(decision: dict[str, Any]) -> str:
    if not isinstance(decision, dict):
        return ""
    return _short_string(decision.get("kind"), 80).lower()


def _coerce_decision_item(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and _short_string(value, 180):
        return {"action": _short_string(value, 180)}
    return None


def _decision_items_from_keys(
    decision: dict[str, Any],
    keys: Sequence[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not isinstance(decision, dict):
        return []
    result: list[dict[str, Any]] = []
    for key in keys:
        value = decision.get(key)
        if isinstance(value, list):
            for item in value:
                coerced = _coerce_decision_item(item)
                if coerced is not None:
                    result.append(coerced)
                if len(result) >= limit:
                    return result
        else:
            coerced = _coerce_decision_item(value)
            if coerced is not None:
                result.append(coerced)
                if len(result) >= limit:
                    return result
    return result


def _decision_action_items(decision: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    items = _decision_items_from_keys(
        decision,
        ("intents", "intent_proposals", "recommended_experiments", "intent", "next_action", "action"),
        limit=limit,
    )
    if not items and any(
        _has_explicit_value(decision.get(key))
        for key in ("name", "action", "intent", "intent_type", "target_object_id", "action6_candidate_id", "x", "y")
    ):
        item = _coerce_decision_item(decision)
        if item is not None:
            items.append(item)
    return items[:limit]


def _decision_plan_items(decision: dict[str, Any], limit: int = 48) -> list[dict[str, Any]]:
    return _decision_items_from_keys(decision, ("steps", "plan", "actions", "intents"), limit=limit)


def _vlm_contract_counts(result: VLMResult) -> dict[str, Any]:
    decision = result.decision if isinstance(result.decision, dict) else {}
    kind = _decision_kind(decision)
    decision_actions = _decision_action_items(decision)
    decision_plan_steps = _decision_plan_items(decision)
    experiments = len(decision_actions) if kind in _DECISION_SINGLE_KINDS else 0
    plan_proposals = len(decision_plan_steps) if kind in _DECISION_PLAN_KINDS else 0
    return {
        "experiments": experiments,
        "plan_proposals": plan_proposals,
        "intent_proposals": len(result.intent_proposals),
        "decision_kind": kind,
        "decision_intents": len(decision_actions),
        "decision_plan_steps": len(decision_plan_steps),
    }


def _normalize_claim(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", text.lower()))


def _claim_similarity(left: str, right: str) -> float:
    left_tokens = set(_normalize_claim(left).split())
    right_tokens = set(_normalize_claim(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / len(left_tokens.union(right_tokens))

class MyAgent(_BaseAgent):
    MAX_ACTIONS = _int_env_any(
        ("ARC_V15_MAX_ACTIONS", "ARC_V13_MAX_ACTIONS", "ARC_V12_MAX_ACTIONS", "ARC_V3_MAX_ACTIONS"),
        240,
        1,
        10_000,
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        backend = kwargs.pop("backend", None)
        config = kwargs.pop("config", None)
        super().__init__(*args, **kwargs)
        self.config = config or AgentConfig.from_env()
        self.MAX_ACTIONS = self.config.max_actions
        self.observer = Observer(self.config.image_size)
        self.logger = DecisionLogger(self.config)
        self.backend = backend if backend is not None else make_vlm_backend(self.config, self.logger)
        self.memory = AgentRuntimeMemory()
        self.undo_manager = UndoManager()
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
        return state_name(getattr(latest_frame, "state", None)) == "WIN" or self._done

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        self._call_index += 1
        state = state_name(getattr(latest_frame, "state", None))
        levels_completed = int(getattr(latest_frame, "levels_completed", 0) or 0)
        legal = normalize_legal_actions(
            getattr(latest_frame, "available_actions", None),
            self._env_action_space(),
            allow_env_fallback=(state == "NOT_PLAYED"),
        )
        try:
            if state == "NOT_PLAYED":
                return self._reset_action()

            current_scene = self.observer.scene_from_frame(latest_frame)
            level = self.memory.level
            transition: TransitionReport | None = None
            just_started = not bool(level.initial_scene_summary)
            advanced = bool(level.initial_scene_summary) and levels_completed > level.levels_completed_at_start

            if just_started:
                self.observer.reset_level()
                current_scene = self.observer.analyze_grid(current_scene.grid, current_scene.rgb)
                self._start_new_level(levels_completed, current_scene)
            elif advanced:
                transition = self._process_pending_transition(current_scene, latest_frame, cross_level=True)
                self._record_level_success(current_scene, levels_completed)
                if state == "WIN":
                    self._done = True
                    return self._reset_action()
                self.observer.reset_level()
                current_scene = self.observer.analyze_grid(current_scene.grid, current_scene.rgb)
                self._start_new_level(levels_completed, current_scene)
            else:
                transition = self._process_pending_transition(current_scene, latest_frame)

            if state == "WIN":
                if self.memory.level.initial_scene_summary:
                    if transition is None:
                        transition = self._process_pending_transition(current_scene, latest_frame)
                    self._record_level_success(
                        current_scene,
                        max(levels_completed, self.memory.level.level_index + 1),
                    )
                self._done = True
                return self._reset_action()

            if state == "GAME_OVER":
                if not self.memory.level.awaiting_reset:
                    self._record_game_over(current_scene, transition)
                return self._reset_action()

            if self.memory.level.awaiting_reset:
                self._on_reset_observed(current_scene)

            self._remember_current_state(current_scene)
            if not self.memory.level.active_plan:
                self._resume_near_success_route(current_scene, legal)
            if not self.memory.level.active_plan:
                self._seed_plan_from_strategy_rules(current_scene, legal)

            result: VLMResult | None = None
            decision = self._begin_decision_frame(current_scene, legal, state, transition)
            selected = self._try_active_plan_action(current_scene, legal, state, decision)
            if selected is None:
                result = self._maybe_request_repair_vlm_update(current_scene, transition, legal, decision, state=state)
                selected = self._try_repair_action(current_scene, legal, state, decision)

            if selected is None:
                mode = self._choose_vlm_mode(current_scene, transition, legal)
                if mode is not None:
                    decision["vlm_mode"] = mode.value if isinstance(mode, VLMMode) else str(mode)
                    result = self._request_vlm_once(current_scene, transition, legal, mode)
                    if result is not None:
                        self._apply_vlm_update(result, transition, current_scene, legal)
                        self._accept_vlm_candidate_plans(result, current_scene, legal, decision)
                        selected = self._try_vlm_single_action(result, current_scene, legal, state, decision)

            if selected is None:
                selected = self._try_exploration_action(current_scene, legal, state, decision)

            if selected is None:
                selected = self._last_safe_recovery_action(current_scene, legal, state, decision)

            if selected is None:
                self.memory.level.awaiting_reset = True
                selected = (
                    self._reset_action(),
                    {"name": "RESET", "owner": "terminal_reset", "reason": "no_safe_non_reset_action"},
                    "reset",
                )

            self._finalize_decision_frame(decision, selected)
            action, proposal, source = selected
            self._record_returned_action(action, proposal, current_scene, latest_frame, source=source)
            self._advance_stage(result, current_scene)
            return action
        except Exception as exc:
            self.logger.log_exception(exc)
            return self._emergency_action(latest_frame, legal)

    def _env_action_space(self) -> Iterable[Any] | None:
        return getattr(getattr(self, "arc_env", None), "action_space", None)

    def _game_id(self) -> str:
        return str(getattr(self, "game_id", "unknown"))

    def _append_event(self, event: EventRecord) -> None:
        self.memory.event_log.append(event)
        self.memory.next_event_id = max(self.memory.next_event_id, event.event_id + 1)
        max_len = self.config.max_event_log
        if len(self.memory.event_log) > max_len:
            evicted = self.memory.event_log[:-max_len]
            del self.memory.event_log[:-max_len]
            if evicted:
                evicted_through = max(e.event_id for e in evicted)
                self.memory.evicted_event_id_through = max(
                    self.memory.evicted_event_id_through, evicted_through
                )
                self._prune_dangling_event_refs(evicted_through)

    def _prune_dangling_event_refs(self, evicted_through: int) -> None:
        """Event_log is bounded, so ids referenced by active level theories can be
        evicted and dangle. Drop those references from the live theories so prompts
        and evidence lists only cite events that still exist. Cross-level schema /
        outcome evidence ids are left as historical labels (never re-queried)."""
        for theory in self.memory.level.level_theories:
            theory.supporting_event_ids = [
                eid for eid in theory.supporting_event_ids if eid > evicted_through
            ]
            theory.contradicting_event_ids = [
                eid for eid in theory.contradicting_event_ids if eid > evicted_through
            ]

    def _recent_event_prompt(self, limit: int | None = None) -> list[dict[str, Any]]:
        events = self.memory.event_log[-(limit or self.config.max_prompt_events) :]
        return [
            {
                "event_id": e.event_id,
                "level": e.level_index,
                "action": e.action_key,
                "source": e.source,
                "outcome": e.outcome,
                "summary": e.transition_summary[:300],
                "before": e.before_state_hash[:12],
                "after": e.after_state_hash[:12],
            }
            for e in events
        ]

    def _transition_delta_dict(self, report: TransitionReport) -> dict[str, Any]:
        return {
            "changed_cell_count": report.changed_cell_count,
            "world_changed_cell_count": report.world_changed_cell_count,
            "hud_changed_cell_count": report.hud_changed_cell_count,
            "world_noop": report.world_noop,
            "full_visual_noop": report.full_visual_noop,
            "effective_noop": report.effective_noop,
            "counter_delta": report.counter_delta,
            "life_delta": report.life_delta,
            "retry_detected": report.retry_detected,
            "moved_objects": report.moved_objects[:8],
            "transformed_objects": report.transformed_objects[:8],
            "appeared_object_ids": report.appeared_object_ids[:8],
            "disappeared_object_ids": report.disappeared_object_ids[:8],
            "controlled_candidate_id": report.controlled_candidate_id,
            "interaction_event": report.interaction_event,
        }

    def _classify_outcome(self, report: TransitionReport, current_scene: SceneSnapshot) -> str:
        if report.retry_detected:
            return "internal_retry"
        if report.effective_noop:
            return "noop"
        if report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids:
            return "transform"
        if report.is_simple_translation:
            return "movement"
        if report.counter_delta is not None or report.life_delta is not None:
            return "resource_delta"
        if report.interaction_event:
            return "interaction"
        return "state_change"

    def _transition_effect_flags(
        self,
        before_state: str,
        after_state: str,
        report: TransitionReport,
    ) -> dict[str, Any]:
        recent_states = list(self.memory.level.recent_state_hashes)
        state_changed = after_state != before_state
        state_novel = bool(state_changed and after_state not in recent_states[-8:])
        movement = bool(report.moved_objects or report.is_simple_translation or report.controlled_candidate_id)
        structural = bool(report.transformed_objects or report.appeared_object_ids or report.disappeared_object_ids)
        resource_delta = bool(report.counter_delta is not None or report.life_delta is not None)
        new_object_frontier = bool(report.appeared_object_ids or report.disappeared_object_ids)
        effectful = bool(
            not report.effective_noop
            and (
                state_changed
                or movement
                or structural
                or report.interaction_event
                or resource_delta
                or report.world_changed_cell_count > 0
            )
        )
        frontier = bool(
            state_novel
            or new_object_frontier
            or (effectful and (movement or structural or report.interaction_event or resource_delta))
        )
        return {
            "state_changed": state_changed,
            "state_novel": state_novel,
            "new_object_frontier": new_object_frontier,
            "effectful": effectful,
            "frontier": frontier,
            "movement": movement,
            "structural": structural,
            "resource_delta": resource_delta,
        }

    def _check_expected_predicates(
        self, predicates: list[dict[str, Any]], report: TransitionReport
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for pred in predicates[:8]:
            ptype = str(pred.get("type", ""))
            ok: bool | None
            if ptype == "not_noop":
                ok = not report.effective_noop
            elif ptype == "movement":
                ok = bool(report.moved_objects)
            elif ptype == "transform":
                ok = bool(
                    report.transformed_objects
                    or report.appeared_object_ids
                    or report.disappeared_object_ids
                )
            elif ptype == "no_retry":
                ok = not report.retry_detected
            elif ptype == "counter_delta_nonpositive":
                ok = report.counter_delta is None or report.counter_delta <= 0
            elif ptype in {"summary", "candidate_kind"}:
                ok = None
            else:
                ok = None
            results.append({"predicate": pred, "ok": ok})
        failed = [item for item in results if item.get("ok") is False]
        return {"results": results, "failed_count": len(failed), "checked_count": len(results)}

    def _action6_key_from_xy(self, x: int, y: int) -> str:
        return f"ACTION6:{int(x)},{int(y)}"

    def _action6_signature(
        self,
        scene: SceneSnapshot,
        x: int,
        y: int,
        target_object_id: str = "",
        candidate_kind: str = "",
    ) -> Action6Signature:
        obj = scene.object_by_id(target_object_id) if target_object_id else self._object_at_point(scene, x, y)
        object_id = obj.track_id if obj is not None else _short_string(target_object_id, 24).upper()
        return Action6Signature(
            state_hash=scene.state_hash,
            x=int(x),
            y=int(y),
            object_id=object_id,
            object_type_key=obj.type_key if obj is not None else "",
            candidate_kind=_short_string(candidate_kind or "unknown", 60),
            local_patch_signature=self._local_patch_signature(scene, x, y),
        )

    def _action6_avoid_keys(self, candidate: Action6Candidate, scene: SceneSnapshot) -> set[str]:
        sig = self._action6_signature(scene, candidate.x, candidate.y, candidate.target_object_id, candidate.candidate_kind)
        return {sig.xy_key, sig.exact_key, sig.local_key, candidate.key(), candidate.candidate_id}

    def _local_patch_signature(self, scene: SceneSnapshot, x: int, y: int, radius: int = 2) -> str:
        vals: list[int] = []
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                if 0 <= xx < scene.width and 0 <= yy < scene.height:
                    vals.append(int(scene.grid[yy][xx]))
                else:
                    vals.append(-1)
        h = hashlib.blake2b(digest_size=8)
        h.update(bytes((value + 1) % 256 for value in vals))
        return h.hexdigest()

    def _object_at_point(self, scene: SceneSnapshot, x: int, y: int) -> ObjectObservation | None:
        candidates = [
            obj
            for obj in scene.objects
            if obj.bbox[0] <= x <= obj.bbox[2] and obj.bbox[1] <= y <= obj.bbox[3]
        ]
        return max(candidates, key=lambda obj: obj.salience, default=None)

    def _object_click_point(self, scene: SceneSnapshot, obj: ObjectObservation) -> tuple[int, int]:
        cx = float(obj.centroid[0])
        cy = float(obj.centroid[1])
        cells = list(getattr(obj, "cells", ()) or ())
        if cells:
            x, y, *_ = min(
                cells,
                key=lambda cell: (
                    (float(cell[0]) - cx) ** 2 + (float(cell[1]) - cy) ** 2,
                    int(cell[1]),
                    int(cell[0]),
                ),
            )
        else:
            x = int(round(cx))
            y = int(round(cy))
        return max(0, min(scene.width - 1, int(x))), max(0, min(scene.height - 1, int(y)))

    def _component_click_point(self, scene: SceneSnapshot, comp: ComponentObservation) -> tuple[int, int]:
        cx = float(comp.centroid[0])
        cy = float(comp.centroid[1])
        cells = list(getattr(comp, "cells", ()) or ())
        if cells:
            x, y = min(
                cells,
                key=lambda cell: (
                    (float(cell[0]) - cx) ** 2 + (float(cell[1]) - cy) ** 2,
                    int(cell[1]),
                    int(cell[0]),
                ),
            )
        else:
            x = int(round(cx))
            y = int(round(cy))
        return max(0, min(scene.width - 1, int(x))), max(0, min(scene.height - 1, int(y)))

    def _action6_candidate_objects(self, scene: SceneSnapshot) -> list[Action6Candidate]:
        candidates: list[Action6Candidate] = []
        seen: set[tuple[int, int, str]] = set()

        def add(obj: ObjectObservation | None, x: int, y: int, kind: str, score: float) -> None:
            x = max(0, min(scene.width - 1, int(x)))
            y = max(0, min(scene.height - 1, int(y)))
            key = (x, y, kind)
            if key in seen:
                return
            seen.add(key)
            object_id = obj.track_id if obj else ""
            type_key = obj.type_key if obj else ""
            candidates.append(
                Action6Candidate(
                    x=x,
                    y=y,
                    target_object_id=object_id,
                    target_type_key=type_key,
                    candidate_kind=kind,
                    local_patch_signature=self._local_patch_signature(scene, x, y),
                    salience=float(obj.salience if obj else 0.0),
                    expected_role=str(self.memory.level.object_beliefs.get(object_id, {}).get("role", "")) if object_id else "",
                    prior_score=score,
                )
            )

        for obj in sorted(scene.objects, key=lambda item: -item.salience):
            click_x, click_y = self._object_click_point(scene, obj)
            add(obj, click_x, click_y, "centroid", obj.salience + 2.0)
            add(obj, (obj.bbox[0] + obj.bbox[2]) // 2, (obj.bbox[1] + obj.bbox[3]) // 2, "bbox_center", obj.salience + 1.0)
            x0, y0, x1, y1 = obj.bbox
            if obj.width >= 3 and obj.height >= 3:
                add(obj, x0, y0, "corner", obj.salience - 1.0)
                add(obj, x1, y0, "corner", obj.salience - 1.0)
                add(obj, x0, y1, "corner", obj.salience - 1.0)
                add(obj, x1, y1, "corner", obj.salience - 1.0)
                add(obj, (x0 + x1) // 2, y0, "edge_midpoint", obj.salience - 0.5)
                add(obj, (x0 + x1) // 2, y1, "edge_midpoint", obj.salience - 0.5)
                add(obj, x0, (y0 + y1) // 2, "edge_midpoint", obj.salience - 0.5)
                add(obj, x1, (y0 + y1) // 2, "edge_midpoint", obj.salience - 0.5)
            if len(candidates) >= self.config.max_action6_candidates * 2:
                break
        if not any(candidate.target_object_id for candidate in candidates):
            max_comp_area = max(16, int(scene.width * scene.height * 0.20))
            for comp in sorted(scene.components, key=lambda item: (-item.area, item.color, item.bbox[1], item.bbox[0])):
                if comp.area <= 0 or comp.area > max_comp_area:
                    continue
                if comp.touches_border and comp.area > 2:
                    continue
                x, y = self._component_click_point(scene, comp)
                add(None, x, y, "component_cell", min(4.0, comp.area / 2.0))
                if len(candidates) >= self.config.max_action6_candidates * 2:
                    break
        add(None, scene.width // 2, scene.height // 2, "blank_control", -5.0)
        candidates.sort(key=lambda item: (item.prior_score, item.salience), reverse=True)
        final = candidates[: self.config.max_action6_candidates]
        for candidate in final:
            sig = self._action6_signature(scene, candidate.x, candidate.y, candidate.target_object_id, candidate.candidate_kind)
            candidate.candidate_id = sig.exact_key.replace("A6E:", "A6_")
        return final

    def _action6_candidate_by_id(self, scene: SceneSnapshot, candidate_id: str) -> Action6Candidate | None:
        wanted = _short_string(candidate_id, 20).upper()
        if not wanted:
            return None
        for candidate in self._action6_candidate_objects(scene):
            if candidate.candidate_id.upper() == wanted:
                return candidate
        return None

    def _action6_candidate_blocked(
        self,
        candidate: Action6Candidate,
        scene: SceneSnapshot,
        *,
        ignore_family_disabled: bool = False,
    ) -> bool:
        level = self.memory.level
        if self._action_family_disabled("ACTION6") and not ignore_family_disabled:
            return True
        radius = self.config.action6_noop_radius
        sig = self._action6_signature(scene, candidate.x, candidate.y, candidate.target_object_id, candidate.candidate_kind)
        for rec in level.action6_memory.records:
            same_local = (
                rec.local_patch_signature == sig.local_patch_signature
                and bool(rec.target_type_key)
                and rec.target_type_key == sig.object_type_key
                and rec.candidate_kind == sig.candidate_kind
            )
            same_exact = rec.state_hash == sig.state_hash and rec.x == sig.x and rec.y == sig.y
            if rec.cooldown_until_step > self._controller_step() and (same_local or same_exact):
                return True
            if rec.state_hash == sig.state_hash and rec.outcome in {"noop", "nonprogress"}:
                if rec.x == sig.x and rec.y == sig.y:
                    return True
                if same_local:
                    return True
                if max(abs(rec.x - sig.x), abs(rec.y - sig.y)) <= radius:
                    if rec.local_patch_signature == sig.local_patch_signature or (
                        rec.target_type_key and rec.target_type_key == sig.object_type_key
                    ):
                        return True
        return False

    def _proposal_to_action6_candidate(
        self, proposal: ActionProposal, scene: SceneSnapshot
    ) -> Action6Candidate | None:
        if proposal.action_name != "ACTION6" or proposal.x is None or proposal.y is None:
            return None
        obj = scene.object_by_id(proposal.target_object_id) or self._object_at_point(scene, proposal.x, proposal.y)
        return Action6Candidate(
            x=proposal.x,
            y=proposal.y,
            target_object_id=obj.track_id if obj else proposal.target_object_id,
            target_type_key=obj.type_key if obj else "",
            candidate_kind=str(proposal.raw.get("candidate_kind") or "proposal"),
            local_patch_signature=self._local_patch_signature(scene, proposal.x, proposal.y),
            salience=float(obj.salience if obj else 0.0),
            expected_role=str(self.memory.level.object_beliefs.get(obj.track_id, {}).get("role", "")) if obj else "",
            prior_score=proposal.priority,
        )

    def _action6_should_probe(self, scene: SceneSnapshot) -> bool:
        level = self.memory.level
        mode = level.mechanism.mode
        if mode in {"click_selection", "global_transform", "hybrid"}:
            return True
        if not level.controlled_object_id and level.total_action_count >= 2:
            return True
        if scene.template_relations and level.total_action_count >= 3:
            return True
        return False

    def _loop_guard_decision(self, action_key: str, scene: SceneSnapshot) -> tuple[bool, float]:
        level = self.memory.level
        if action_key in level.loop_blocked_action_keys:
            return True, 1.0
        if action_key in level.noop_actions_by_state.get(scene.state_hash, set()):
            return True, 1.0
        if (scene.state_hash, action_key) in level.blocked_state_action_pairs:
            return True, 1.0

        recent_actions = list(level.recent_action_keys)
        recent_states = list(level.recent_state_hashes)
        if not recent_actions:
            return False, 0.0
        base = action_key.split(":", 1)[0]

        if base == "ACTION6":
            exact_seq = recent_actions[-3:] + [action_key]
            if len(exact_seq) >= 4 and all(item == action_key for item in exact_seq[-4:]):
                return True, 1.0
            return False, 0.2 if action_key in recent_actions[-6:] else 0.0

        if len(recent_states) >= 6 and recent_states[-1] == recent_states[-3] == recent_states[-5]:
            if recent_actions and recent_actions[-1].split(":", 1)[0] == base:
                return True, 1.0

        seq = [item.split(":", 1)[0] for item in recent_actions[-5:]] + [base]
        if len(seq) >= 5 and all(item == base for item in seq[-5:]):
            if len(set(recent_states[-5:])) >= 4:
                return False, 0.35
            return True, 1.0

        oscillations = [
            ["ACTION1", "ACTION2"] * 3,
            ["ACTION2", "ACTION1"] * 3,
            ["ACTION3", "ACTION4"] * 3,
            ["ACTION4", "ACTION3"] * 3,
        ]
        if len(seq) >= 6 and seq[-6:] in oscillations:
            if len(set(recent_states[-6:])) <= 3:
                return True, 1.0
            return False, 0.45
        return False, 0.0

    def _raw_loop_reason(self, names: Sequence[str]) -> str:
        seq = [_short_string(name, 40).upper() for name in names if _short_string(name, 40)]
        if len(seq) < 4:
            return ""
        oscillations = [
            ["ACTION1", "ACTION2"] * 3,
            ["ACTION2", "ACTION1"] * 3,
            ["ACTION3", "ACTION4"] * 3,
            ["ACTION4", "ACTION3"] * 3,
        ]
        if len(seq) >= 6 and seq[-6:] in oscillations:
            return "raw_direction_oscillation"
        if len(seq) >= 5 and len(set(seq[-5:])) == 1:
            return "raw_repeated_single_action"
        for period in (2, 3):
            width = period * 2
            if len(seq) >= width and seq[-width:-period] == seq[-period:]:
                return "raw_periodic_loop"
        return ""

    def _repeated_single_raw_without_grounding(self, names: Sequence[str]) -> bool:
        seq = [_short_string(name, 40).upper() for name in names if _short_string(name, 40)]
        if not seq or len(set(seq)) != 1:
            return False
        name = seq[0]
        if name not in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}:
            return False
        level = self.memory.level
        if level.mechanism.mode != "global_transform":
            return False
        recent = [item.split(":", 1)[0] for item in list(level.recent_action_keys)[-4:]]
        if len(recent) < 3 or not all(item == name for item in recent[-3:]):
            return False
        knowledge = self.memory.game.action_knowledge.get(name)
        return knowledge is not None and knowledge.attempts >= 3

    def _reject_plan_quality_gate(self, reason: str, names: Sequence[str]) -> None:
        level = self.memory.level
        level.plan_quality_rejections += 1
        level.reject_reasons[RejectReason.LOOP_RISK.value] += 1
        self.memory.game.controller_stats.reject_counts[RejectReason.LOOP_RISK.value] += 1
        level.force_vlm_reason = "plan_rejected_quality_gate"
        self.logger.log_event(
            "plan_quality_rejected",
            {"reason": reason, "names": list(names)[:12], "level": level.level_index},
        )

    def _vlm_plan_steps_pass_quality(
        self,
        steps: Sequence[dict[str, Any]],
        scene: SceneSnapshot,
        target_id: str = "",
    ) -> bool:
        proposed = [
            self._normalize_vlm_action_item(dict(raw))
            for raw in steps
            if isinstance(raw, dict)
        ]
        if target_id:
            for item in proposed:
                if not item.get("target_object_id"):
                    item["target_object_id"] = target_id
        names = [
            _short_string(item.get("name") or item.get("action"), 40).upper()
            for item in proposed
        ]
        loop_reason = self._raw_loop_reason(names)
        if loop_reason:
            self._reject_plan_quality_gate(loop_reason, names)
            return False
        if self._repeated_single_raw_without_grounding(names):
            self._reject_plan_quality_gate("repeated_single_global_raw_action", names)
            return False
        if len(names) > 6 and "NAVIGATE" not in names:
            has_grounding = bool(target_id) or any(
                item.get("target_object_id") or item.get("purpose") or item.get("expected_change")
                for item in proposed
            )
            if not has_grounding:
                self._reject_plan_quality_gate("long_ungrounded_raw_plan", names)
                return False
        if proposed and self._premature_frame_target(proposed[0], scene):
            rejected_target = _short_string(proposed[0].get("target_object_id"), 24).upper()
            self._record_target_failure(
                rejected_target,
                "framed target pattern mismatches status/template while an untested compact object remains",
            )
            self.memory.level.recent_events.append(f"rejected_premature_frame_target={rejected_target}")
            return False
        return True

    def _start_deterministic_active_plan(
        self,
        source: str,
        goal: str,
        target_id: str,
        steps: list[dict[str, Any]],
        scene: SceneSnapshot,
        legal: Sequence[Any],
        *,
        log_event: str,
        log_payload: dict[str, Any] | None = None,
    ) -> bool:
        level = self.memory.level
        if level.active_plan is not None:
            return False
        raw = {
            "source": source,
            "goal": goal,
            "target_object_id": target_id,
            "steps": steps[: self.config.plan_horizon],
            "confidence": 0.75,
            "evidence_level": 1,
        }
        plan = self._build_candidate_plan(raw, None, scene, legal)
        if plan is None or not self._preflight_candidate_plan(plan, scene, legal):
            return False
        level.plan_id += 1
        plan.status = "active"
        level.active_plan = plan
        level.plan_goal = goal
        level.plan_target_id = target_id
        level.stage = V1Phase.EXECUTE_PLAN
        payload = {**plan.as_prompt(), **(log_payload or {})}
        payload.setdefault("plan_id", level.plan_id)
        self.logger.log_event(log_event, payload)
        return True

    def _select_action6_candidate(
        self,
        scene: SceneSnapshot,
        avoid_keys: set[str] | None = None,
        *,
        allow_blocked: bool = False,
    ) -> Action6Candidate | None:
        level = self.memory.level
        avoid = avoid_keys or set()
        noops = level.noop_actions_by_state.get(scene.state_hash, set())
        attempts = level.action_attempt_counts_by_state.get(scene.state_hash, {})
        if self._action_family_disabled("ACTION6"):
            return None
        candidates = self._action6_candidate_objects(scene)
        if not candidates:
            return None

        def rank(candidate: Action6Candidate) -> tuple[int, int, int, int, int, float, float]:
            key = self._action6_key_from_xy(candidate.x, candidate.y)
            avoid_hit = bool(self._action6_avoid_keys(candidate, scene) & avoid)
            return (
                1 if self._action6_candidate_blocked(candidate, scene) else 0,
                1 if key in noops else 0,
                1 if avoid_hit else 0,
                1 if key in level.recent_action_keys else 0,
                int(attempts.get(key, 0)),
                -float(candidate.prior_score),
                -float(candidate.salience),
            )

        for candidate in sorted(candidates, key=rank):
            key = self._action6_key_from_xy(candidate.x, candidate.y)
            if not allow_blocked and (key in noops or self._action6_candidate_blocked(candidate, scene)):
                continue
            if not allow_blocked and self._action6_avoid_keys(candidate, scene) & avoid:
                continue
            return candidate
        return None

    def _select_action6_candidate_allowing_state_noops(
        self,
        scene: SceneSnapshot,
        avoid_keys: set[str] | None = None,
        *,
        allow_blocked: bool = False,
    ) -> Action6Candidate | None:
        if self._action_family_disabled("ACTION6"):
            return None
        avoid = avoid_keys or set()
        noops = self.memory.level.noop_actions_by_state.get(scene.state_hash, set())
        attempts = self.memory.level.action_attempt_counts_by_state.get(scene.state_hash, {})
        candidates = self._action6_candidate_objects(scene)

        def rank(candidate: Action6Candidate) -> tuple[int, int, int, int, int, float, float]:
            key = self._action6_key_from_xy(candidate.x, candidate.y)
            avoid_hit = bool(self._action6_avoid_keys(candidate, scene) & avoid)
            blocked = self._action6_candidate_blocked(candidate, scene)
            return (
                1 if blocked else 0,
                1 if key in noops else 0,
                1 if avoid_hit else 0,
                1 if key in self.memory.level.recent_action_keys else 0,
                int(attempts.get(key, 0)),
                -float(candidate.prior_score),
                -float(candidate.salience),
            )

        for candidate in sorted(candidates, key=rank):
            if not allow_blocked and self._action6_candidate_blocked(candidate, scene):
                continue
            return candidate
        return None

    def _complete_action6_proposal(
        self,
        proposal: dict[str, Any],
        result: VLMResult | None,
        scene: SceneSnapshot,
        avoid_keys: set[str] | None = None,
    ) -> dict[str, Any] | None:
        item = dict(proposal)
        for text_key in ("action", "name", "purpose"):
            xy = _extract_action6_xy_from_text(_short_string(item.get(text_key), 260))
            if xy is not None:
                item["x"], item["y"] = xy
                self.memory.game.controller_stats.contract_repairs["action6_coord_parsed_from_text"] += 1
                self.logger.log_event("action6_coord_parsed_from_text", {"field": text_key, "x": xy[0], "y": xy[1]})
                break

        explicit_x = _has_explicit_value(item.get("x"))
        explicit_y = _has_explicit_value(item.get("y"))
        has_any_explicit_coord = explicit_x or explicit_y
        x = self._coerce_int(item.get("x"))
        y = self._coerce_int(item.get("y"))
        target = _short_string(
            item.get("target_object_id") or (result.target_object_id if result is not None else ""),
            24,
        ).upper()
        if target:
            item["target_object_id"] = target

        if x is not None and y is not None and 0 <= x < scene.width and 0 <= y < scene.height:
            item["x"] = x
            item["y"] = y
            return item

        obj = scene.object_by_id(target) if target else None
        if obj is not None and not self._target_blocked(target, scene):
            # Target-object grounding is allowed to repair missing, partial, or out-of-range click coordinates.
            repaired_x, repaired_y = self._object_click_point(scene, obj)
            item["x"] = repaired_x
            item["y"] = repaired_y
            self.memory.game.controller_stats.contract_repairs["action6_coord_inferred_from_target"] += 1
            self.logger.log_event(
                "action6_coord_inferred_from_target",
                {"target_object_id": target, "x": repaired_x, "y": repaired_y, "had_explicit_coord": has_any_explicit_coord},
            )
            return item

        candidate_id = _short_string(item.get("action6_candidate_id") or item.get("candidate_id"), 24).upper()
        if candidate_id:
            candidate = self._action6_candidate_by_id(scene, candidate_id)
            if candidate is not None and not self._action6_candidate_blocked(candidate, scene):
                item["x"] = candidate.x
                item["y"] = candidate.y
                item["target_object_id"] = target or candidate.target_object_id
                item["candidate_kind"] = candidate.candidate_kind
                self.memory.game.controller_stats.contract_repairs["action6_coord_inferred_from_candidate_id"] += 1
                self.logger.log_event(
                    "action6_coord_inferred_from_candidate_id",
                    {"candidate_id": candidate_id, "x": candidate.x, "y": candidate.y, "target_object_id": item.get("target_object_id", "")},
                )
                return item
            self.logger.log_event(
                "vlm_contract_repair_failed",
                {"reason": "action6_candidate_id_not_found_or_blocked", "candidate_id": candidate_id},
            )
            if not target:
                candidate_id = ""

        if has_any_explicit_coord:
            return None
        if not target and not candidate_id:
            self.memory.game.controller_stats.contract_repairs["bare_action6_no_target_candidate_coord"] += 1
            self.logger.log_event(
                "vlm_contract_repair_failed",
                {"reason": "bare_action6_no_target_candidate_coord", "proposal": _json_safe(item)},
            )
            return None

        candidate = self._select_action6_candidate(scene, avoid_keys=avoid_keys)
        if candidate is None:
            return None
        item["x"] = candidate.x
        item["y"] = candidate.y
        if not target and candidate.target_object_id:
            item["target_object_id"] = candidate.target_object_id
        item["candidate_kind"] = item.get("candidate_kind") or candidate.candidate_kind
        self.memory.game.controller_stats.contract_repairs["action6_coord_inferred_from_candidate_beam"] += 1
        self.logger.log_event(
            "action6_coord_inferred_from_candidate_beam",
            {
                "target_object_id": item.get("target_object_id", ""),
                "candidate_kind": candidate.candidate_kind,
                "candidate_id": candidate.candidate_id,
                "x": candidate.x,
                "y": candidate.y,
            },
        )
        return item

    def _controller_step(self) -> int:
        level = self.memory.level
        return max(level.total_action_count, level.controller_tick_count)

    def _action_family_disabled(self, family: str) -> bool:
        family = _short_string(family, 40).upper()
        return self.memory.level.disabled_action_families_until.get(family, 0) > self._controller_step()

    def _disable_action_family(self, family: str, steps: int, reason: str) -> None:
        family = _short_string(family, 40).upper()
        until = self._controller_step() + max(1, int(steps))
        previous = self.memory.level.disabled_action_families_until.get(family, 0)
        self.memory.level.disabled_action_families_until[family] = max(previous, until)
        self.logger.log_event(
            "action_family_disabled",
            {"family": family, "until": self.memory.level.disabled_action_families_until[family], "reason": reason[:240]},
        )

    def _primitive_grounding_complete(self, name: str) -> bool:
        name = _short_string(name, 40).upper()
        if name not in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}:
            return False
        knowledge = self.memory.game.action_knowledge.get(name)
        if knowledge is None or knowledge.attempts <= 0:
            return False
        if knowledge.attempts >= 2:
            return True
        return bool(
            knowledge.best_vector() is not None
            or knowledge.blocked > 0
            or knowledge.structural_successes > 0
            or knowledge.global_changes > 0
            or knowledge.retry_failures > 0
        )

    def _primitive_transition_features(
        self,
        pending: PendingAction,
        report: TransitionReport,
    ) -> dict[str, Any]:
        movement = bool(report.moved_objects or report.is_simple_translation or report.controlled_candidate_id)
        interaction = bool(report.interaction_event)
        structural_count = (
            len(report.transformed_objects)
            + len(report.appeared_object_ids)
            + len(report.disappeared_object_ids)
        )
        resource_cost = bool(
            (report.counter_delta is not None and report.counter_delta < 0)
            or (report.life_delta is not None and report.life_delta < 0)
        )
        pure_structural = structural_count > 0 and not movement and not interaction
        low_visual_churn = (
            report.world_changed_cell_count <= 4
            and report.changed_cell_count <= 8
            and not movement
            and not interaction
        )
        global_churn = report.world_changed_cell_count >= 20 and not movement and not interaction
        suspicious = bool(
            pending.name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}
            and not report.effective_noop
            and not report.retry_detected
            and (pure_structural or global_churn or low_visual_churn or (resource_cost and not movement and not interaction))
        )
        return {
            "action": pending.name,
            "source": pending.source,
            "step": self.memory.level.total_action_count,
            "suspicious": suspicious,
            "movement": movement,
            "interaction": interaction,
            "resource_cost": resource_cost,
            "structural_count": structural_count,
            "world_changed_cells": report.world_changed_cell_count,
            "changed_cells": report.changed_cell_count,
            "counter_delta": report.counter_delta,
            "life_delta": report.life_delta,
            "summary": report.summary[:320],
        }

    def _detect_nonprogress_primitive_loop(
        self,
        pending: PendingAction,
        report: TransitionReport,
        current_scene: SceneSnapshot,
        event: EventRecord,
    ) -> None:
        if pending.name not in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}:
            return
        if report.retry_detected or report.effective_noop:
            return
        level = self.memory.level
        record = self._primitive_transition_features(pending, report)
        level.nonprogress_action_records.append(record)
        if not record.get("suspicious"):
            return

        action = pending.name
        recent = list(level.nonprogress_action_records)[-10:]
        same = [item for item in recent if item.get("action") == action]
        suspicious_same = [item for item in same if item.get("suspicious")]
        resource_same = [item for item in suspicious_same if item.get("resource_cost")]
        dominant_same = len(same) >= 4 and len(same) >= max(4, len(recent) // 2)
        should_repair = (
            len(resource_same) >= 2
            or len(suspicious_same) >= 3
            or (dominant_same and len(suspicious_same) >= 2)
        )
        if not should_repair:
            level.suspicious_action_counts[action] += 1
            return
        if self._action_family_disabled(action) and level.recovery_mode == "repair_nonprogress_loop":
            return

        level.suspicious_action_counts[action] += 1
        level.last_nonprogress_repair_step = self._controller_step()
        reason = f"nonprogress_primitive_loop:{action}"
        disable_steps = 10 if resource_same else 8
        self._disable_action_family(action, disable_steps, reason)
        level.blocked_state_action_pairs.add((pending.scene_before.state_hash, action))
        level.blocked_state_action_pairs.add((current_scene.state_hash, action))
        level.recent_events.append(f"{reason}; recent={[item.get('summary', '')[:80] for item in suspicious_same[-3:]]}")
        self._enter_repair_mode(reason)
        self.logger.log_event(
            "nonprogress_loop_detected",
            {
                "action": action,
                "event_id": event.event_id,
                "recent_same": len(same),
                "suspicious_same": len(suspicious_same),
                "resource_same": len(resource_same),
                "disabled_until": level.disabled_action_families_until.get(action, 0),
                "records": suspicious_same[-4:],
            },
        )

    def _action6_transition_features(
        self,
        pending: PendingAction,
        report: TransitionReport,
        current_scene: SceneSnapshot,
    ) -> dict[str, Any]:
        before = pending.scene_before
        level = self.memory.level
        obj = self._object_at_point(before, int(pending.x or 0), int(pending.y or 0)) if pending.x is not None and pending.y is not None else None
        candidate_kind = "unknown"
        for pred in pending.expected_predicates:
            if isinstance(pred, dict) and pred.get("candidate_kind"):
                candidate_kind = _short_string(pred.get("candidate_kind"), 60)
                break
        movement = bool(report.moved_objects or report.is_simple_translation or report.controlled_candidate_id)
        interaction = bool(report.interaction_event)
        structural_count = (
            len(report.transformed_objects)
            + len(report.appeared_object_ids)
            + len(report.disappeared_object_ids)
        )
        resource_cost = bool(
            (report.counter_delta is not None and report.counter_delta < 0)
            or (report.life_delta is not None and report.life_delta < 0)
        )
        state_changed = current_scene.state_hash != before.state_hash
        state_novel = bool(state_changed and current_scene.state_hash not in list(level.recent_state_hashes)[-8:])
        new_object_frontier = bool(report.appeared_object_ids or report.disappeared_object_ids)
        low_visual_churn = (
            report.world_changed_cell_count <= 4
            and report.changed_cell_count <= 8
            and not movement
            and not interaction
        )
        pure_structural = structural_count > 0 and not movement and not interaction
        frontier = bool(state_novel or new_object_frontier)
        suspicious = bool(
            pending.name == "ACTION6"
            and not report.retry_detected
            and (
                report.effective_noop
                or low_visual_churn
                or (resource_cost and not movement and not interaction)
                or (pure_structural and not frontier)
            )
        )
        local_patch = ""
        if pending.x is not None and pending.y is not None:
            local_patch = self._local_patch_signature(before, pending.x, pending.y)
        return {
            "action": "ACTION6",
            "source": pending.source,
            "step": self.memory.level.total_action_count,
            "state_hash": before.state_hash,
            "x": pending.x,
            "y": pending.y,
            "target_object_id": obj.track_id if obj else pending.target_object_id,
            "target_type_key": obj.type_key if obj else "",
            "candidate_kind": candidate_kind,
            "local_patch_signature": local_patch,
            "suspicious": suspicious,
            "state_changed": state_changed,
            "state_frontier": state_novel,
            "new_object_frontier": new_object_frontier,
            "pure_structural": pure_structural,
            "movement": movement,
            "interaction": interaction,
            "resource_cost": resource_cost,
            "structural_count": structural_count,
            "world_changed_cells": report.world_changed_cell_count,
            "changed_cells": report.changed_cell_count,
            "counter_delta": report.counter_delta,
            "life_delta": report.life_delta,
            "summary": report.summary[:320],
        }

    def _detect_action6_nonprogress_loop(
        self,
        pending: PendingAction,
        report: TransitionReport,
        current_scene: SceneSnapshot,
        event: EventRecord,
    ) -> None:
        if pending.name != "ACTION6" or pending.x is None or pending.y is None:
            return
        level = self.memory.level
        record = self._action6_transition_features(pending, report, current_scene)
        level.nonprogress_action_records.append(record)
        if not record.get("suspicious"):
            return

        recent = [
            item for item in list(level.nonprogress_action_records)[-14:]
            if item.get("action") == "ACTION6"
        ]
        suspicious = [
            item for item in recent
            if item.get("suspicious")
            and not item.get("state_frontier")
            and not item.get("new_object_frontier")
        ]
        same_local = [
            item for item in suspicious
            if item.get("local_patch_signature") == record.get("local_patch_signature")
            and item.get("target_type_key") == record.get("target_type_key")
            and item.get("candidate_kind") == record.get("candidate_kind")
        ]
        same_exact = [
            item for item in suspicious
            if item.get("state_hash") == record.get("state_hash")
            and item.get("x") == record.get("x")
            and item.get("y") == record.get("y")
        ]
        local_families = {
            (
                item.get("local_patch_signature"),
                item.get("target_type_key"),
                item.get("candidate_kind"),
            )
            for item in suspicious
        }
        should_repair = len(same_exact) >= 2 or len(same_local) >= 3 or (len(suspicious) >= 4 and len(local_families) <= 2)
        if not should_repair:
            return
        if self._controller_step() - level.last_nonprogress_repair_step < 3 and level.recovery_mode:
            return

        cooldown_until = self._controller_step() + 10
        for rec in level.action6_memory.records[-30:]:
            same_rec_local = (
                rec.local_patch_signature == record.get("local_patch_signature")
                and rec.target_type_key == record.get("target_type_key")
                and rec.candidate_kind == record.get("candidate_kind")
            )
            same_rec_exact = (
                rec.state_hash == record.get("state_hash")
                and rec.x == record.get("x")
                and rec.y == record.get("y")
            )
            if same_rec_local or same_rec_exact:
                rec.cooldown_until_step = max(rec.cooldown_until_step, cooldown_until)
                if rec.outcome != "noop":
                    rec.outcome = "nonprogress"

        key = self._action6_key_from_xy(pending.x, pending.y)
        level.blocked_state_action_pairs.add((pending.scene_before.state_hash, key))
        level.blocked_state_action_pairs.add((current_scene.state_hash, key))
        level.last_nonprogress_repair_step = self._controller_step()
        only_action6 = self._action6_is_only_forward_name_set(pending.available_actions_before)
        reason = "click_candidate_nonprogress_loop" if only_action6 else "action6_nonprogress_loop"
        level.recent_events.append(
            f"{reason}; recent={[item.get('summary', '')[:80] for item in suspicious[-3:]]}"
        )
        self._enter_repair_mode(reason)
        self.logger.log_event(
            "action6_nonprogress_loop_detected",
            {
                "event_id": event.event_id,
                "only_action6_forward": only_action6,
                "recent_action6": len(recent),
                "suspicious": len(suspicious),
                "same_local": len(same_local),
                "same_exact": len(same_exact),
                "blocked_key": key,
                "cooldown_until": cooldown_until,
                "records": suspicious[-4:],
            },
        )

    def _action6_is_only_forward_name_set(self, names: Sequence[str]) -> bool:
        normalized = {_short_string(name, 40).upper() for name in names}
        if "ACTION6" not in normalized:
            return False
        return not any(
            name in normalized for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")
        )

    def _action6_is_only_forward_action(self, legal: Sequence[Any]) -> bool:
        names = [action_name(action) for action in legal]
        return self._action6_is_only_forward_name_set(names)

    def _action6_budget_available(self, legal: Sequence[Any] | None = None) -> bool:
        # When ACTION6 is the only forward action, throttling it would deadlock
        # the controller into an endless RESET storm (no other legal action can
        # ever be proposed), so the budget must not apply in that case.
        if legal is not None and self._action6_is_only_forward_action(legal):
            return True
        if self._action_family_disabled("ACTION6"):
            return False
        recent = list(self.memory.level.recent_action_keys)[-8:]
        return sum(1 for item in recent if item.startswith("ACTION6")) < 2

    def _classify_recovery_mode(self, reason: str) -> str:
        text = str(reason).lower()
        if "action6" in text:
            return "escape_from_action6_loop"
        if "nonprogress" in text or "resource" in text or "undo" in text or "fallback" in text:
            return "repair_nonprogress_loop"
        if "nav" in text or "compile" in text:
            return "repair_compile_failure"
        if "retry" in text or "game_over" in text:
            return "recover_after_retry"
        return "proposal_recovery"

    def _enter_repair_mode(self, reason: str) -> None:
        level = self.memory.level
        mode = self._classify_recovery_mode(reason)
        normalized_reason = _short_string(reason, 240)
        current_step = self._controller_step()
        if (
            level.recovery_mode == mode
            and level.recovery_reason == normalized_reason
            and current_step <= level.recovery_until_step
        ):
            return
        level.recovery_mode = mode
        level.recovery_reason = normalized_reason
        level.recovery_until_step = max(level.recovery_until_step, current_step + 8)
        level.force_vlm_reason = "repair_decision"
        if level.recovery_mode == "escape_from_action6_loop":
            level.anti_loop_escape_failures = 0
            self._disable_action_family("ACTION6", 8, reason)
        else:
            level.anti_loop_escape_failures = 0
        self.memory.game.controller_stats.recovery_counts[level.recovery_mode] += 1
        self.logger.log_event(
            "repair_mode_entered",
            {"mode": level.recovery_mode, "reason": level.recovery_reason, "until": level.recovery_until_step},
        )

    def _proposal_summary(self, proposal: ActionProposal | None) -> dict[str, Any]:
        if proposal is None:
            return {}
        return {
            "source": proposal.source,
            "action": proposal.action_key(),
            "target_object_id": proposal.target_object_id,
            "purpose": proposal.purpose[:240],
            "priority": round(float(proposal.priority), 4),
            "information_gain": round(float(proposal.information_gain), 4),
            "goal_progress": round(float(proposal.goal_progress), 4),
            "raw": _json_safe(proposal.raw),
        }

    def _active_plan_prompt(self) -> dict[str, Any]:
        plan = self.memory.level.active_plan
        return plan.as_prompt() if plan is not None else {}

    def _begin_decision_frame(
        self,
        scene: SceneSnapshot,
        legal: Sequence[Any],
        state: str,
        transition: TransitionReport | None,
    ) -> dict[str, Any]:
        level = self.memory.level
        return {
            "decision_id": self._call_index,
            "level": level.level_index,
            "step": level.total_action_count,
            "state": scene.state_hash[:12],
            "phase": level.stage.value,
            "state_name": state,
            "legal": [action_name(action) for action in legal],
            "owner_attempts": [],
            "vlm_mode": "",
            "active_plan": self._active_plan_prompt(),
            "recovery": {
                "mode": level.recovery_mode,
                "reason": level.recovery_reason,
                "until": level.recovery_until_step,
            },
            "candidates": [],
            "compile_failures": [],
            "scored": [],
            "selected": {},
            "transition": transition.summary if transition is not None else None,
        }

    def _record_compile_result(
        self,
        decision: dict[str, Any] | None,
        compiled: CompileResult,
        *,
        owner: str,
    ) -> None:
        if compiled.ok:
            if decision is not None and compiled.proposal is not None:
                decision.setdefault("candidates", []).append(
                    {"owner": owner, "proposal": self._proposal_summary(compiled.proposal), "compile_status": compiled.status.value}
                )
            return
        entry = {
            "owner": owner,
            "status": compiled.status.value,
            "detail": compiled.detail[:300],
            "severity": compiled.severity,
            "intent": _json_safe(compiled.intent),
        }
        self.memory.level.last_compile_failures.append(entry)
        self._record_compile_target_failure(compiled)
        if (
            owner != "active_plan"
            and compiled.intent.intent_type == IntentType.NAVIGATE_TO_OBJECT.value
            and compiled.status
            in {
                CompileStatus.ACTOR_UNKNOWN,
                CompileStatus.ACTION_VECTORS_UNKNOWN,
                CompileStatus.WALKABLE_UNKNOWN,
                CompileStatus.NO_PATH_KNOWN,
            }
        ):
            self._enter_repair_mode(f"nav_compile_failed:{compiled.status.value}")
        if decision is not None:
            decision.setdefault("compile_failures", []).append(entry)

    def _record_compile_target_failure(self, compiled: CompileResult) -> None:
        if compiled.ok or compiled.status is None:
            return
        target_id = compiled.intent.target_object_id
        if not target_id or compiled.intent.intent_type != IntentType.NAVIGATE_TO_OBJECT.value:
            return
        if compiled.status in {
            CompileStatus.TARGET_BLOCKED,
            CompileStatus.TARGET_NOT_VISIBLE,
        }:
            self._record_target_failure(target_id, f"nav_compile:{compiled.status.value}")

    def _finalize_decision_frame(self, frame: dict[str, Any], selected: tuple[Any, dict[str, Any], str] | None) -> None:
        if selected is not None:
            action, proposal, source = selected
            frame["selected"] = {
                "owner": proposal.get("owner") or source,
                "source": source,
                "action": self._action_key_for_action(action),
                "proposal": _json_safe(proposal),
            }
        else:
            frame["selected"] = {"owner": "none"}
        self.memory.level.decision_frames.append(frame)
        self.logger.log_event("decision_frame", frame)

    def _risk_value(self, risk: str) -> float:
        return {"low": 0.0, "medium": 0.35, "high": 1.0}.get(str(risk).lower(), 0.3)

    def _compile_reject_reason(self, status: CompileStatus | None) -> RejectReason:
        return {
            CompileStatus.ACTOR_UNKNOWN: RejectReason.NAV_ACTOR_UNKNOWN,
            CompileStatus.ACTION_VECTORS_UNKNOWN: RejectReason.NAV_ACTION_VECTORS_UNKNOWN,
            CompileStatus.WALKABLE_UNKNOWN: RejectReason.NAV_WALKABLE_UNKNOWN,
            CompileStatus.NO_PATH_KNOWN: RejectReason.NAV_NO_PATH_KNOWN,
            CompileStatus.TARGET_MISSING: RejectReason.BAD_NAVIGATE_TARGET,
            CompileStatus.TARGET_NOT_VISIBLE: RejectReason.BAD_NAVIGATE_TARGET,
            CompileStatus.TARGET_BLOCKED: RejectReason.TARGET_BLOCKED,
            CompileStatus.ILLEGAL_ACTION: RejectReason.ILLEGAL_ACTION,
            CompileStatus.ACTION6_BAD_COORD: RejectReason.ACTION6_BAD_COORD,
            CompileStatus.ACTION6_NO_CANDIDATE: RejectReason.ACTION6_BAD_COORD,
        }.get(status, RejectReason.CONTRACT_REPAIR_FAILED)

    def _validate_proposal(
        self, proposal: ActionProposal, scene: SceneSnapshot, legal: Sequence[Any], state: str = "NOT_FINISHED"
    ) -> ValidationResult:
        level = self.memory.level
        legal_by_name = {action_name(action): action for action in legal}
        name = _short_string(proposal.action_name, 40).upper()
        proposal.action_name = name

        def reject(reason: RejectReason, detail: str = "") -> ValidationResult:
            level.reject_reasons[reason.value] += 1
            self.memory.game.controller_stats.reject_counts[reason.value] += 1
            if reason == RejectReason.ACTION7_UNDO_ONLY:
                self.memory.game.controller_stats.action7_forbidden_count += 1
            self.logger.log_event(
                "proposal_rejected",
                {
                    "reason": reason.value,
                    "detail": detail[:300],
                    "proposal": _json_safe(proposal),
                    "level": level.level_index,
                    "step": level.total_action_count,
                },
            )
            return ValidationResult(False, proposal, reason, detail)

        if name in {"CLICK", "MOVE", "GO", "PATHFIND"}:
            return reject(RejectReason.UNSUPPORTED_ACTION_TOKEN, name)
        if name == "NAVIGATE":
            intent = ActionIntent(
                source=proposal.source,
                intent_type=IntentType.NAVIGATE_TO_OBJECT.value,
                target_object_id=proposal.target_object_id,
                purpose=proposal.purpose,
                expected_predicates=proposal.expected_predicates,
                risk=proposal.risk,
                reversible=proposal.reversible,
                information_gain=proposal.information_gain,
                goal_progress=proposal.goal_progress,
                novelty=proposal.novelty,
                priority=proposal.priority,
                raw=dict(proposal.raw),
            )
            compiled = self._compile_intent(intent, scene, legal)
            if compiled.ok and compiled.proposal is not None:
                proposal.action_name = compiled.proposal.action_name
                proposal.raw.update(compiled.proposal.raw)
                proposal.target_object_id = compiled.proposal.target_object_id
                name = proposal.action_name
            else:
                self._record_compile_result(None, compiled, owner=proposal.source or "proposal")
                return reject(self._compile_reject_reason(compiled.status), compiled.detail)
        if name == "RESET":
            if state not in {"NOT_PLAYED", "GAME_OVER", "WIN"}:
                return reject(RejectReason.RESET_NOT_ALLOWED)
            return ValidationResult(True, proposal)
        if name == "ACTION7":
            if proposal.source != "undo_rollback" or not self.undo_manager.can_emit(level.total_action_count):
                return reject(RejectReason.ACTION7_UNDO_ONLY)
        if name not in legal_by_name:
            return reject(RejectReason.ILLEGAL_ACTION, f"{name} not in {sorted(legal_by_name)}")
        allow_disabled_action6_frontier = bool(
            name == "ACTION6"
            and proposal.source == "action6_frontier"
            and proposal.raw.get("family_disabled_bypass")
        )
        if self._action_family_disabled(name) and not allow_disabled_action6_frontier:
            return reject(RejectReason.LOOP_RISK, f"{name} family disabled until {self.memory.level.disabled_action_families_until.get(name, 0)}")
        if name == "ACTION6":
            if proposal.x is None or proposal.y is None or not (0 <= proposal.x < scene.width and 0 <= proposal.y < scene.height):
                return reject(RejectReason.ACTION6_BAD_COORD)
            candidate = self._proposal_to_action6_candidate(proposal, scene)
            ignore_candidate_block = bool(allow_disabled_action6_frontier)
            if candidate is not None and not ignore_candidate_block and self._action6_candidate_blocked(
                candidate,
                scene,
                ignore_family_disabled=allow_disabled_action6_frontier,
            ):
                level.action6_memory.duplicate_suppressed += 1
                self.memory.game.controller_stats.action6_duplicate_suppressed += 1
                self.logger.log_event("action6_candidate_suppressed", {"candidate": _json_safe(candidate), "state": scene.state_hash[:12]})
                return reject(RejectReason.ACTION6_DUPLICATE_OR_COOLDOWN)
        key = proposal.action_key()
        if key in level.noop_actions_by_state.get(scene.state_hash, set()):
            return reject(RejectReason.KNOWN_NOOP)
        hard_loop, soft_penalty = self._loop_guard_decision(key, scene)
        if hard_loop:
            self.logger.log_event("loop_hard_reject", {"action": key, "level": level.level_index, "step": level.total_action_count})
            return reject(RejectReason.LOOP_RISK)
        proposal.raw["loop_soft_penalty"] = soft_penalty
        if soft_penalty > 0:
            self.memory.game.controller_stats.loop_soft_penalties += 1
            self.logger.log_event("loop_soft_penalty", {"action": key, "penalty": soft_penalty, "level": level.level_index, "step": level.total_action_count})
        if self.memory.game.failure_model.forbids(list(level.recent_action_keys), key):
            return reject(RejectReason.FAILURE_SUFFIX)
        if name == "ACTION6" and proposal.target_object_id and self._target_blocked(proposal.target_object_id, scene):
            return reject(RejectReason.TARGET_BLOCKED)
        return ValidationResult(True, proposal)

    def _history_penalty(self, p: ActionProposal, scene: SceneSnapshot) -> float:
        level = self.memory.level
        key = p.action_key()
        if key in level.noop_actions_by_state.get(scene.state_hash, set()):
            return 999.0
        penalty = 1.2 * level.action_attempt_counts_by_state.get(scene.state_hash, {}).get(key, 0)
        if p.target_object_id:
            penalty += 1.5 * level.target_failure_counts.get(p.target_object_id, 0)
        allow_disabled_action6_frontier = bool(
            p.action_name == "ACTION6"
            and p.source == "action6_frontier"
            and p.raw.get("family_disabled_bypass")
        )
        if self._action_family_disabled(p.action_name) and not allow_disabled_action6_frontier:
            return 999.0
        if p.action_name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}:
            recent_suspicious = [
                item for item in list(level.nonprogress_action_records)[-12:]
                if item.get("action") == p.action_name and item.get("suspicious")
            ]
            penalty += 1.25 * len(recent_suspicious)
            penalty += 1.5 * min(3, level.suspicious_action_counts.get(p.action_name, 0))
        if p.action_name == "ACTION6" and p.x is not None and p.y is not None:
            sig = self._action6_signature(scene, p.x, p.y, p.target_object_id, p.raw.get("candidate_kind", ""))
            for rec in level.action6_memory.records[-80:]:
                if rec.outcome not in {"noop", "nonprogress"}:
                    continue
                if rec.x == sig.x and rec.y == sig.y and rec.state_hash == sig.state_hash:
                    penalty += 4.0
                elif rec.local_patch_signature == sig.local_patch_signature and rec.target_type_key == sig.object_type_key:
                    penalty += 2.0
            for item in list(level.nonprogress_action_records)[-80:]:
                if item.get("action") != "ACTION6" or not item.get("suspicious"):
                    continue
                if item.get("x") == sig.x and item.get("y") == sig.y and item.get("state_hash") == sig.state_hash:
                    penalty += 3.0
                elif (
                    item.get("local_patch_signature") == sig.local_patch_signature
                    and item.get("target_type_key") == sig.object_type_key
                    and item.get("candidate_kind") == sig.candidate_kind
                ):
                    penalty += 1.5
            recent_family = sum(1 for action in list(level.recent_action_keys)[-8:] if action.startswith("ACTION6"))
            penalty += 0.6 * recent_family
            if self._action_family_disabled("ACTION6") and not allow_disabled_action6_frontier:
                penalty += 999.0
        return penalty

    def _score_proposal(self, result: ValidationResult, scene: SceneSnapshot) -> ValidationResult:
        p = result.proposal
        schema_support = 1.0 if p.source in {"schema_transfer", "vlm_transfer", "active_plan"} else 0.0
        try:
            loop_risk = float(p.raw.get("loop_soft_penalty", 0.0) or 0.0)
        except Exception:
            loop_risk = 0.0
        terminal_risk = self._risk_value(p.risk)
        irreversibility = 0.0 if p.reversible else (0.2 if p.risk == "low" else 0.5)
        history_penalty = self._history_penalty(p, scene)
        plan_bonus = 0.6 if p.raw.get("_from_active_plan") else 0.0
        score = (
            2.5 * p.information_gain
            + 2.0 * p.goal_progress
            + 1.2 * schema_support
            + 0.8 * p.novelty
            + p.priority
            + plan_bonus
            - 1.0 * p.cost
            - 3.0 * terminal_risk
            - 2.0 * irreversibility
            - 4.0 * loop_risk
            - history_penalty
        )
        result.score = round(score, 4)
        return result

    def _choose_best_proposal(
        self,
        proposals: list[ActionProposal],
        scene: SceneSnapshot,
        legal: Sequence[Any],
        state: str,
        decision: dict[str, Any] | None = None,
    ) -> ActionProposal | None:
        stats = self.memory.game.controller_stats
        valid: list[ValidationResult] = []
        for proposal in proposals:
            stats.proposal_counts[proposal.source] += 1
            result = self._validate_proposal(proposal, scene, legal, state)
            if result.ok:
                scored = self._score_proposal(result, scene)
                valid.append(scored)
                if decision is not None:
                    decision.setdefault("scored", []).append({**self._proposal_summary(scored.proposal), "score": scored.score})
            elif decision is not None:
                decision.setdefault("candidates", []).append(
                    {
                        "proposal": self._proposal_summary(result.proposal),
                        "reject": result.reason.value if result.reason is not None else "unknown",
                        "detail": result.detail[:240],
                    }
                )
        if not valid:
            return None
        valid.sort(key=lambda item: (item.score, item.proposal.priority, item.proposal.information_gain), reverse=True)
        chosen = valid[0].proposal
        stats.selected_counts[chosen.source] += 1
        self.memory.level.last_selected_proposal = chosen
        self.logger.log_event(
            "proposal_selected",
            {
                "source": chosen.source,
                "action": chosen.action_key(),
                "score": valid[0].score,
                "purpose": chosen.purpose,
                "phase": chosen.phase.value if isinstance(chosen.phase, V1Phase) else str(chosen.phase),
            },
        )
        return chosen

    def _instantiate_proposal(
        self, proposal: ActionProposal | None, scene: SceneSnapshot, legal: Sequence[Any]
    ) -> tuple[Any, dict[str, Any], str] | None:
        if proposal is None:
            return None
        step = proposal.to_plan_step()
        action = self._make_valid_action(step, scene, legal, allow_reset=False)
        if action is None:
            return None
        return action, step, proposal.source

    def _nav_reject_reason(
        self,
        scene: SceneSnapshot,
        target_id: str,
        legal: Sequence[Any],
    ) -> str | None:
        level = self.memory.level
        target = scene.object_by_id(target_id)
        if target is None:
            return "bad_navigation_target"
        if level.mechanism.mode == "click_selection":
            click_score = float(level.mechanism.scores.get("click_selection", 0.0))
            nav_score = float(level.mechanism.scores.get("direct_navigation", 0.0))
            if level.controlled_actor_confidence < 0.85 and click_score >= nav_score:
                return "click_selection_not_direct_navigation"
        if not level.controlled_object_id:
            return "missing_controlled_actor"
        if self._plan_path_to_object(scene, target_id, legal) is None:
            return "no_path_to_target"
        return None

    def _semantic_nav_proposal(self, scene: SceneSnapshot, legal: Sequence[Any]) -> ActionProposal | None:
        target_id = self._choose_semantic_target(scene)
        if not target_id:
            return None
        target = scene.object_by_id(target_id)
        return ActionProposal(
            source="frontier_nav" if target is None else "object_probe",
            action_name="NAVIGATE",
            target_object_id=target_id,
            purpose=f"approach semantic target {target_id}",
            phase=V1Phase.MECHANIC_EXPLORATION,
            expected_predicates=[{"type": "movement"}, {"type": "no_retry"}],
            information_gain=0.35,
            goal_progress=0.35,
            novelty=0.25,
            priority=0.12,
        )

    def _level_theory_proposals(self, scene: SceneSnapshot, legal: Sequence[Any]) -> list[ActionProposal]:
        level = self.memory.level
        if not level.level_theories:
            return []
        theory = level.level_theories[0]
        if theory.status == "rejected" or theory.confidence < 0.25 or theory.evidence_level <= 0:
            return []
        raw_intents: list[dict[str, Any]] = [
            item for item in theory.discriminating_tests[:2] if isinstance(item, dict)
        ]
        if theory.evidence_level >= 2 and theory.confidence >= 0.55 and theory.solve_sketch:
            plan = self._build_candidate_plan(
                {
                    "source": "level_theory",
                    "goal": theory.win_condition_hypothesis or theory.mechanism_hypothesis,
                    "steps": theory.solve_sketch[: self.config.plan_horizon],
                    "confidence": min(0.75, theory.confidence),
                    "evidence_level": theory.evidence_level,
                    "theory_id": theory.theory_id,
                },
                None,
                scene,
                legal,
            )
            if plan is not None and self._preflight_candidate_plan(plan, scene, legal):
                if not any(existing.plan_id == plan.plan_id for existing in level.candidate_plans):
                    level.candidate_plans.append(plan)
                    self.logger.log_event("candidate_plan_created", plan.as_prompt())
        intents: list[ActionIntent] = []
        for raw in raw_intents:
            intent = self._intent_from_action_item(raw, source="level_theory")
            if intent is not None:
                intent.priority += 0.12
                intent.information_gain = min(intent.information_gain, 0.55)
                intent.goal_progress = min(intent.goal_progress, 0.2 if theory.evidence_level == 1 else intent.goal_progress)
                intents.append(intent)
        return self._compile_intents_to_proposals(intents, scene, legal)

    def _experiment_scheduler_proposals(self, scene: SceneSnapshot, legal: Sequence[Any]) -> list[ActionProposal]:
        level = self.memory.level
        legal_names = {action_name(action) for action in legal}
        proposals: list[ActionProposal] = []
        state = scene.state_hash
        tried = level.tried_actions_by_state.setdefault(state, set())
        noops = level.noop_actions_by_state.get(state, set())
        simple = [
            name for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")
            if name in legal_names and not self._action_family_disabled(name)
        ]
        if level.mechanism.mode == "click_selection":
            simple_probes = []
        elif level.mechanism.mode == "global_transform":
            simple_probes = [name for name in ("ACTION3", "ACTION4", "ACTION5") if name in simple] or simple
        else:
            simple_probes = simple
        if not level.controlled_object_id or level.stage in {V1Phase.INIT, V1Phase.ACTION_GROUNDING}:
            for idx, name in enumerate(simple_probes):
                if name not in tried and name not in noops and not self._primitive_grounding_complete(name):
                    contextual = name == "ACTION5"
                    proposals.append(
                        ActionProposal(
                            source="action_grounding_probe",
                            action_name=name,
                            purpose=f"ground {name}: identify controlled object/effect/resource cost",
                            phase=V1Phase.ACTION_GROUNDING,
                            expected_predicates=[{"type": "no_retry"}],
                            information_gain=0.45 if contextual else 0.55,
                            novelty=0.55 if contextual else 0.7,
                            cost=1.25 if contextual else 1.0,
                            priority=0.08 if contextual else 0.2 - 0.02 * idx,
                        )
                    )
        nav = None if level.mechanism.mode == "click_selection" else self._frontier_navigation_action(state, simple)
        if nav is not None:
            proposals.append(
                ActionProposal(
                    source="frontier_nav",
                    action_name=nav,
                    purpose="move toward an untested state-action frontier",
                    phase=V1Phase.MECHANIC_EXPLORATION,
                    expected_predicates=[{"type": "movement"}, {"type": "no_retry"}],
                    information_gain=0.35,
                    novelty=0.4,
                    priority=0.05,
                )
            )
        semantic = self._semantic_nav_proposal(scene, legal)
        if semantic is not None:
            proposals.append(semantic)
        if "ACTION6" in legal_names and self._action6_should_probe(scene) and self._action6_budget_available(legal):
            action6_count = 0
            for cand in self._action6_candidate_objects(scene):
                cand_key = self._action6_key_from_xy(cand.x, cand.y)
                if cand_key in noops or self._action6_candidate_blocked(cand, scene):
                    continue
                proposals.append(
                    ActionProposal(
                        source="action6_probe",
                        action_name="ACTION6",
                        x=cand.x,
                        y=cand.y,
                        target_object_id=cand.target_object_id,
                        purpose=f"object-centered ACTION6 probe {cand.candidate_kind} on {cand.target_object_id or 'blank'}",
                        phase=V1Phase.MECHANIC_EXPLORATION,
                        expected_predicates=[
                            {"type": "no_retry"},
                            {"type": "summary", "summary": "observe clicked object response"},
                            {"type": "candidate_kind", "candidate_kind": cand.candidate_kind},
                        ],
                        information_gain=0.55 if cand.target_object_id else 0.2,
                        novelty=0.5,
                        priority=cand.prior_score / 100.0,
                        raw={"candidate_kind": cand.candidate_kind, "local_patch_signature": cand.local_patch_signature},
                    )
                )
                action6_count += 1
                if action6_count >= 6:
                    break
        return proposals

    def _choose_schema_target(self, scene: SceneSnapshot, schema: GameGoalSchema) -> str | None:
        text = f"{schema.name} {schema.statement}".lower()
        candidates = [obj for obj in scene.objects if not obj.near_edge]
        if any(token in text for token in ("frame", "pattern", "match", "symbol")):
            framed = [obj for obj in candidates if obj.frame_color is not None or obj.inner_pattern]
            if framed:
                return max(framed, key=lambda obj: obj.salience).track_id
        if any(token in text for token in ("height", "volume", "adjust")) and candidates:
            return max(candidates, key=lambda obj: obj.salience).track_id
        return None

    def _transfer_bootstrap_proposals(self, scene: SceneSnapshot, legal: Sequence[Any]) -> list[ActionProposal]:
        if not self.config.enable_transfer_bootstrap or not self.memory.game.goal_schemas:
            return []
        level = self.memory.level
        if level.total_action_count > 8:
            return []
        best_schema = self.memory.game.goal_schemas[0]
        proposals: list[ActionProposal] = []
        target_id = self._choose_schema_target(scene, best_schema) or self._choose_semantic_target(scene)
        if target_id:
            proposals.append(
                ActionProposal(
                    source="schema_transfer",
                    action_name="NAVIGATE",
                    target_object_id=target_id,
                    purpose=f"instantiate prior success schema {best_schema.name} on current level",
                    phase=V1Phase.TRANSFER_BOOTSTRAP,
                    expected_predicates=[{"type": "movement"}, {"type": "no_retry"}],
                    information_gain=0.45,
                    goal_progress=0.55,
                    novelty=0.2,
                    priority=0.35,
                    raw={"schema_id": best_schema.schema_id},
                )
            )
        legal_names = {action_name(action) for action in legal}
        if "ACTION6" in legal_names and best_schema.confidence >= 0.5 and self._action6_budget_available(legal):
            for cand in self._action6_candidate_objects(scene)[:6]:
                if self._action6_candidate_blocked(cand, scene):
                    continue
                proposals.append(
                    ActionProposal(
                        source="schema_transfer",
                        action_name="ACTION6",
                        x=cand.x,
                        y=cand.y,
                        target_object_id=cand.target_object_id,
                        purpose=f"test schema {best_schema.name} via object-centered click candidate",
                        phase=V1Phase.TRANSFER_BOOTSTRAP,
                        expected_predicates=[{"type": "no_retry"}],
                        information_gain=0.45,
                        goal_progress=0.4,
                        novelty=0.25,
                        priority=0.2,
                        raw={"schema_id": best_schema.schema_id, "candidate_kind": cand.candidate_kind},
                    )
                )
                break
        if proposals:
            self.logger.log_event("schema_transfer_proposed", {"count": len(proposals), "schema_id": best_schema.schema_id})
        return proposals

    def _normalize_vlm_action_item(self, item: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(item)
        raw_action = _short_string(normalized.get("action") or normalized.get("name"), 180)
        text = re.sub(r"\s+", " ", raw_action).strip()
        upper = text.upper()
        target_id = _short_string(
            normalized.get("target_object_id") or normalized.get("target") or normalized.get("object_id"),
            24,
        ).upper()
        if not target_id:
            target_id = _extract_target_object_id_from_text(upper)

        def commit(name: str) -> dict[str, Any]:
            normalized["name"] = name
            normalized["action"] = name
            if target_id:
                normalized["target_object_id"] = target_id
            if text and text != name and not normalized.get("purpose"):
                normalized["purpose"] = text[:260]
            return normalized

        if re.search(r"\bACTION6\b", upper):
            xy = _extract_action6_xy_from_text(upper)
            if xy is not None:
                normalized["x"], normalized["y"] = xy
            return commit("ACTION6")

        action_match = re.search(r"\bACTION[1-5]\b", upper)
        if action_match:
            return commit(action_match.group(0))
        if re.search(r"\bACTION7\b", upper):
            return commit("ACTION7")
        if upper == "RESET" or upper.startswith("RESET "):
            return commit("RESET")
        if target_id and re.search(r"\b(NAVIGATE|APPROACH|REACH|GO TO|MOVE TO)\b", upper):
            return commit("NAVIGATE")
        if re.search(r"\bMOVE\b", upper):
            direction_to_action = {
                "UP": "ACTION1",
                "DOWN": "ACTION2",
                "LEFT": "ACTION3",
                "RIGHT": "ACTION4",
            }
            for direction, action in direction_to_action.items():
                if re.search(rf"\b{direction}\b", upper):
                    return commit(action)
        if target_id and re.search(r"\b(CLICK|TAP|SELECT|PRESS|TEST)\b", upper):
            return commit("ACTION6")
        return normalized

    def _intent_from_action_item(self, item: dict[str, Any], source: str) -> ActionIntent | None:
        if not isinstance(item, dict):
            return None
        normalized = self._normalize_vlm_action_item(dict(item))
        name = _short_string(normalized.get("action") or normalized.get("name"), 40).upper()
        raw_intent = _short_string(normalized.get("intent") or normalized.get("intent_type"), 80).lower()
        target_id = _short_string(
            normalized.get("target_object_id") or normalized.get("target") or normalized.get("object_id"),
            24,
        ).upper()
        candidate_id = _short_string(
            normalized.get("action6_candidate_id") or normalized.get("candidate_id"),
            24,
        ).upper()
        expected = normalized.get("expected_predicates") if isinstance(normalized.get("expected_predicates"), list) else []
        if not expected and normalized.get("expected_change"):
            expected = [{"type": "summary", "summary": _short_string(normalized.get("expected_change"), 260)}]
        if not expected and normalized.get("predictions_by_hypothesis"):
            expected = [{"type": "summary", "summary": _short_string(normalized.get("predictions_by_hypothesis"), 260)}]

        intent_type = ""
        if raw_intent in {item.value for item in IntentType}:
            intent_type = raw_intent
        elif name == "NAVIGATE":
            intent_type = IntentType.NAVIGATE_TO_OBJECT.value
        elif name == "ACTION6":
            x = self._coerce_int(normalized.get("x"))
            y = self._coerce_int(normalized.get("y"))
            if candidate_id:
                intent_type = IntentType.CLICK_CANDIDATE.value
            elif target_id:
                intent_type = IntentType.CLICK_OBJECT.value
            elif x is not None and y is not None:
                intent_type = IntentType.CLICK_CANDIDATE.value
            else:
                self.memory.game.controller_stats.contract_repairs["bare_action6_no_target_candidate_coord"] += 1
                self.logger.log_event(
                    "vlm_contract_repair_failed",
                    {"reason": "bare_action6_no_target_candidate_coord", "item": _json_safe(normalized)},
                )
                return None
        elif name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7", "RESET"}:
            intent_type = IntentType.PRIMITIVE_ACTION.value
        elif target_id and raw_intent in {"click", "click_object", "test", "test_object"}:
            intent_type = IntentType.TEST_OBJECT.value if "test" in raw_intent else IntentType.CLICK_OBJECT.value
        elif not name and raw_intent in {IntentType.CLICK_OBJECT.value, IntentType.TEST_OBJECT.value} and target_id:
            name = "ACTION6"
            intent_type = raw_intent
        if not intent_type:
            if name:
                intent_type = CompileStatus.UNSUPPORTED_INTENT.value
            else:
                return None

        return ActionIntent(
            source=source,
            intent_type=intent_type,
            action_name=name,
            target_object_id=target_id,
            action6_candidate_id=candidate_id,
            x=self._coerce_int(normalized.get("x")),
            y=self._coerce_int(normalized.get("y")),
            purpose=_short_string(normalized.get("purpose") or normalized.get("why"), 260),
            expected_predicates=expected[:8],
            risk=_short_string(normalized.get("risk"), 20).lower() or "low",
            reversible=bool(normalized.get("reversible", True)),
            information_gain=_clamp01(normalized.get("expected_information_gain", normalized.get("information_gain", 0.5))) or 0.5,
            goal_progress=_clamp01(normalized.get("goal_progress", 0.45)) or 0.45,
            novelty=_clamp01(normalized.get("novelty", 0.45)) or 0.45,
            priority=_clamp01(normalized.get("priority", 0.0)),
            raw=dict(normalized),
        )

    def _vlm_result_to_intents(self, result: VLMResult, scene: SceneSnapshot) -> list[ActionIntent]:
        intents: list[ActionIntent] = []
        if result.mode == VLMMode.REPAIR_DECISION.value:
            self.logger.log_event("vlm_repair_free_actions_ignored", {"reason": "repair_decision_candidate_choice_only"})
            return intents

        def add(raw: dict[str, Any], source: str) -> None:
            intent = self._intent_from_action_item(raw, source)
            if intent is not None:
                intents.append(intent)

        decision = result.decision if isinstance(result.decision, dict) else {}
        decision_kind = _decision_kind(decision)
        if decision_kind in _DECISION_SINGLE_KINDS:
            for item in _decision_action_items(decision, limit=4):
                add(item, "vlm_decision")
        for item in result.intent_proposals:
            add(item, "vlm_intent")
        return intents

    def _navigation_preflight(
        self,
        scene: SceneSnapshot,
        legal: Sequence[Any],
        target_id: str,
    ) -> tuple[bool, CompileStatus, str]:
        if not target_id:
            return False, CompileStatus.TARGET_MISSING, "missing target"
        target = scene.object_by_id(target_id)
        if target is None:
            return False, CompileStatus.TARGET_NOT_VISIBLE, target_id
        if self._target_blocked(target_id, scene):
            return False, CompileStatus.TARGET_BLOCKED, target_id
        actor = scene.object_by_id(self.memory.level.controlled_object_id)
        if actor is None:
            return False, CompileStatus.ACTOR_UNKNOWN, "controlled actor unknown"
        vectors = self._grounded_action_vectors(legal)
        if not vectors:
            return False, CompileStatus.ACTION_VECTORS_UNKNOWN, "grounded vectors unknown"
        floor_colors = self._walkable_colors(scene, actor)
        if not floor_colors:
            return False, CompileStatus.WALKABLE_UNKNOWN, "walkable colors unknown"
        path = self._plan_path_to_object(scene, target_id, legal, use_weak_vectors=False)
        if not path:
            return False, CompileStatus.NO_PATH_KNOWN, "no grounded path"
        return True, CompileStatus.OK, ""

    def _grounding_proposals_for_nav_failure(
        self,
        status: CompileStatus,
        scene: SceneSnapshot,
        legal: Sequence[Any],
    ) -> list[ActionProposal]:
        if status not in {
            CompileStatus.ACTOR_UNKNOWN,
            CompileStatus.ACTION_VECTORS_UNKNOWN,
            CompileStatus.WALKABLE_UNKNOWN,
            CompileStatus.NO_PATH_KNOWN,
        }:
            return []
        legal_names = {action_name(action) for action in legal}
        tried = self.memory.level.tried_actions_by_state.setdefault(scene.state_hash, set())
        noops = self.memory.level.noop_actions_by_state.get(scene.state_hash, set())
        proposals: list[ActionProposal] = []
        for idx, name in enumerate(("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")):
            if name not in legal_names or name in tried or name in noops:
                continue
            proposals.append(
                ActionProposal(
                    source="nav_grounding_repair",
                    action_name=name,
                    purpose=f"ground controls before NAVIGATE after {status.value}",
                    phase=V1Phase.ACTION_GROUNDING,
                    expected_predicates=[{"type": "no_retry"}],
                    information_gain=0.6,
                    goal_progress=0.05,
                    novelty=0.65,
                    priority=0.35 - 0.02 * idx,
                    raw={"nav_compile_status": status.value},
                )
            )
        return proposals

    def _compile_intent(self, intent: ActionIntent, scene: SceneSnapshot, legal: Sequence[Any]) -> CompileResult:
        legal_names = {action_name(action) for action in legal}

        def legal_allows(name: str) -> bool:
            return not legal_names or name in legal_names

        def fail(status: CompileStatus, detail: str = "", severity: str = "hard") -> CompileResult:
            self.memory.game.controller_stats.compile_failures[status.value] += 1
            event = {
                CompileStatus.ACTOR_UNKNOWN: "nav_compile_failed_actor_unknown",
                CompileStatus.ACTION_VECTORS_UNKNOWN: "nav_compile_failed_vectors_unknown",
                CompileStatus.WALKABLE_UNKNOWN: "nav_compile_failed_walkable_unknown",
                CompileStatus.NO_PATH_KNOWN: "nav_compile_failed_no_path_known",
            }.get(status, "vlm_contract_repair_failed")
            self.logger.log_event(event, {"status": status.value, "detail": detail[:300], "intent": _json_safe(intent)})
            return CompileResult(False, intent, status=status, detail=detail, severity=severity)

        def make(action_name_: str, *, x: int | None = None, y: int | None = None, target: str = "", raw: dict[str, Any] | None = None, repaired: bool = False) -> CompileResult:
            proposal = ActionProposal(
                source=intent.source,
                action_name=action_name_,
                x=x,
                y=y,
                target_object_id=target or intent.target_object_id,
                purpose=intent.purpose,
                phase=self.memory.level.stage,
                expected_predicates=intent.expected_predicates[:8],
                risk=intent.risk,
                reversible=intent.reversible,
                information_gain=intent.information_gain,
                goal_progress=intent.goal_progress,
                novelty=intent.novelty,
                priority=(0.8 if intent.source.startswith("vlm") else 0.0) + intent.priority,
                raw={**intent.raw, **(raw or {})},
            )
            status = CompileStatus.REPAIRED if repaired else CompileStatus.OK
            if repaired:
                self.memory.game.controller_stats.contract_repairs["compiled_repaired"] += 1
                self.logger.log_event("vlm_contract_repaired", {"intent": _json_safe(intent), "proposal": _json_safe(proposal)})
            return CompileResult(True, intent, proposal=proposal, status=status, severity="repaired" if repaired else "ok")

        kind = str(intent.intent_type)
        name = _short_string(intent.action_name, 40).upper()
        if kind == IntentType.PRIMITIVE_ACTION.value:
            if name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7", "RESET"}:
                if not legal_allows(name):
                    return fail(CompileStatus.ILLEGAL_ACTION, f"{name} not legal")
                return make(name)
            return fail(CompileStatus.ILLEGAL_ACTION if name else CompileStatus.UNSUPPORTED_INTENT, name)

        if kind in {IntentType.CLICK_CANDIDATE.value, IntentType.CLICK_OBJECT.value, IntentType.TEST_OBJECT.value}:
            if not legal_allows("ACTION6"):
                return fail(CompileStatus.ILLEGAL_ACTION, "ACTION6 not legal")
            item = {**intent.raw, "name": "ACTION6", "target_object_id": intent.target_object_id}
            if intent.action6_candidate_id:
                item["action6_candidate_id"] = intent.action6_candidate_id
            if intent.x is not None:
                item["x"] = intent.x
            if intent.y is not None:
                item["y"] = intent.y
            if kind in {IntentType.CLICK_OBJECT.value, IntentType.TEST_OBJECT.value} and not intent.target_object_id:
                return fail(CompileStatus.TARGET_MISSING, "click/test object missing target")
            if intent.target_object_id and self._target_blocked(intent.target_object_id, scene):
                return fail(CompileStatus.TARGET_BLOCKED, intent.target_object_id, severity="soft")
            completed = self._complete_action6_proposal(item, None, scene)
            if completed is None:
                return fail(CompileStatus.ACTION6_BAD_COORD if (intent.x is not None or intent.y is not None) else CompileStatus.ACTION6_NO_CANDIDATE, str(intent.raw))
            x = self._coerce_int(completed.get("x"))
            y = self._coerce_int(completed.get("y"))
            if x is None or y is None or not (0 <= x < scene.width and 0 <= y < scene.height):
                return fail(CompileStatus.ACTION6_BAD_COORD, str(completed))
            target = _short_string(completed.get("target_object_id"), 24).upper()
            return make("ACTION6", x=x, y=y, target=target, raw=dict(completed), repaired=(intent.x != x or intent.y != y))

        if kind == IntentType.NAVIGATE_TO_OBJECT.value:
            target_id = intent.target_object_id
            ok, status, detail = self._navigation_preflight(scene, legal, target_id)
            if not ok:
                severity = "need_grounding" if status in {
                    CompileStatus.ACTOR_UNKNOWN,
                    CompileStatus.ACTION_VECTORS_UNKNOWN,
                    CompileStatus.WALKABLE_UNKNOWN,
                    CompileStatus.NO_PATH_KNOWN,
                } else "soft"
                return fail(status, detail, severity=severity)
            path = self._plan_path_to_object(scene, target_id, legal, use_weak_vectors=False)
            if not path:
                return fail(CompileStatus.NO_PATH_KNOWN, "no grounded path", severity="soft")
            first = path[0]
            if not legal_allows(first):
                return fail(CompileStatus.ILLEGAL_ACTION, f"compiled {first} not legal")
            return make(first, target=target_id, raw={"expanded_from": "NAVIGATE", "nav_path_len": len(path)}, repaired=True)

        return fail(CompileStatus.UNSUPPORTED_INTENT, kind)

    def _compile_intents_to_proposals(
        self,
        intents: list[ActionIntent],
        scene: SceneSnapshot,
        legal: Sequence[Any],
        decision: dict[str, Any] | None = None,
        owner: str = "intent_compiler",
    ) -> list[ActionProposal]:
        proposals: list[ActionProposal] = []
        for intent in intents:
            compiled = self._compile_intent(intent, scene, legal)
            if decision is not None or not compiled.ok:
                self._record_compile_result(decision, compiled, owner=owner or intent.source)
            if compiled.ok and compiled.proposal is not None:
                proposals.append(compiled.proposal)
        return proposals

    def _vlm_result_to_proposals(self, result: VLMResult, scene: SceneSnapshot, legal: Sequence[Any] = ()) -> list[ActionProposal]:
        return self._compile_intents_to_proposals(self._vlm_result_to_intents(result, scene), scene, legal)

    def _try_vlm_single_action(
        self,
        result: VLMResult,
        scene: SceneSnapshot,
        legal: Sequence[Any],
        state: str,
        decision: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any], str] | None:
        if result.mode in {VLMMode.REPAIR_DECISION.value, "INVALID_PARTIAL"}:
            return None
        decision = result.decision if isinstance(result.decision, dict) else {}
        if _decision_kind(decision) not in _DECISION_SINGLE_KINDS and not result.intent_proposals:
            return None
        intents = self._vlm_result_to_intents(result, scene)
        if not intents:
            return None
        if decision is not None:
            decision.setdefault("owner_attempts", []).append({"owner": "vlm_single", "intents": len(intents)})
        proposals = self._compile_intents_to_proposals(intents, scene, legal, decision=decision, owner="vlm")
        chosen = self._choose_best_proposal(proposals, scene, legal, state, decision=decision)
        selected = self._instantiate_proposal(chosen, scene, legal) if chosen else None
        if selected is None:
            return None
        action, proposal, source = selected
        return action, {**proposal, "owner": "vlm"}, source

    def _build_candidate_plan(
        self,
        raw_plan: dict[str, Any],
        result: VLMResult | None,
        scene: SceneSnapshot,
        legal: Sequence[Any],
    ) -> ActivePlan | None:
        steps_raw = raw_plan.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            return None
        source = _short_string(raw_plan.get("source") or "candidate_plan", 80)
        goal = _short_string(raw_plan.get("goal") or (result.plan_goal if result is not None else "") or source, 400)
        default_target = _short_string(raw_plan.get("target_object_id") or (result.target_object_id if result is not None else ""), 24).upper()
        steps: list[ActivePlanStep] = []
        for idx, item in enumerate(steps_raw[: self.config.plan_horizon]):
            if not isinstance(item, dict):
                continue
            raw_item = dict(item)
            if default_target and not raw_item.get("target_object_id"):
                raw_item["target_object_id"] = default_target
            intent = self._intent_from_action_item(raw_item, source=source)
            if intent is None:
                self.logger.log_event("candidate_plan_step_rejected", {"source": source, "step": _json_safe(raw_item), "reason": "unsupported_intent"})
                continue
            step_seed = json.dumps({"source": source, "idx": idx, "intent": _json_safe(intent)}, sort_keys=True, default=str)
            steps.append(
                ActivePlanStep(
                    step_id="APS:" + hashlib.sha1(step_seed.encode()).hexdigest()[:10],
                    intent=intent,
                    expected_predicates=list(intent.expected_predicates),
                    preconditions=raw_item.get("preconditions") if isinstance(raw_item.get("preconditions"), list) else [],
                    max_attempts=max(1, int(raw_item.get("max_attempts", 1) or 1)) if not isinstance(raw_item.get("max_attempts"), bool) else 1,
                )
            )
        if not steps:
            return None
        plan_seed = json.dumps(
            {"source": source, "goal": goal, "steps": [_json_safe(step.intent) for step in steps]},
            sort_keys=True,
            default=str,
        )
        return ActivePlan(
            plan_id="AP:" + hashlib.sha1(plan_seed.encode()).hexdigest()[:12],
            source=source,
            goal=goal,
            theory_id=_short_string(raw_plan.get("theory_id"), 80),
            steps=steps,
            confidence=_clamp01(raw_plan.get("confidence", 0.4)),
            evidence_level=int(raw_plan.get("evidence_level", 0) or 0),
        )

    def _preflight_candidate_plan(self, plan: ActivePlan, scene: SceneSnapshot, legal: Sequence[Any]) -> bool:
        step = plan.current_step()
        if step is None:
            return False
        compiled = self._compile_intent(step.intent, scene, legal)
        if not compiled.ok:
            self.logger.log_event(
                "candidate_plan_preflight_failed",
                {"plan_id": plan.plan_id, "status": compiled.status.value, "detail": compiled.detail[:240]},
            )
            return False
        if compiled.proposal is not None and self._action_family_disabled(compiled.proposal.action_name):
            self.logger.log_event(
                "candidate_plan_preflight_failed",
                {"plan_id": plan.plan_id, "status": "ACTION_FAMILY_DISABLED", "detail": compiled.proposal.action_name},
            )
            return False
        return True

    def _accept_vlm_candidate_plans(
        self,
        result: VLMResult,
        scene: SceneSnapshot,
        legal: Sequence[Any],
        decision: dict[str, Any] | None = None,
    ) -> None:
        decision_payload = result.decision if isinstance(result.decision, dict) else {}
        if _decision_kind(decision_payload) not in _DECISION_PLAN_KINDS:
            return
        decision_steps = _decision_plan_items(decision_payload)
        if not decision_steps:
            return
        if not self._vlm_plan_steps_pass_quality(decision_steps, scene, result.target_object_id):
            return
        raw = {
            "source": "vlm_decision_plan",
            "goal": _short_string(decision_payload.get("goal") or result.plan_goal or "vlm_decision_plan", 320),
            "target_object_id": result.target_object_id,
            "steps": decision_steps,
            "confidence": _clamp01(decision_payload.get("confidence", 0.45)),
            "evidence_level": int(decision_payload.get("evidence_level", 1) or 1),
        }
        plan = self._build_candidate_plan(raw, result, scene, legal)
        if plan is None:
            return
        if not self._preflight_candidate_plan(plan, scene, legal):
            if decision is not None:
                decision.setdefault("candidates", []).append({"owner": "vlm", "candidate_plan_rejected": plan.as_prompt()})
            return
        if any(existing.plan_id == plan.plan_id for existing in self.memory.level.candidate_plans):
            return
        self.memory.level.candidate_plans.append(plan)
        if decision is not None:
            decision.setdefault("candidates", []).append({"owner": "vlm", "candidate_plan": plan.as_prompt()})
        self.logger.log_event("candidate_plan_created", plan.as_prompt())

    def _maybe_activate_candidate_plan(self, scene: SceneSnapshot, legal: Sequence[Any], decision: dict[str, Any] | None = None) -> bool:
        level = self.memory.level
        if level.active_plan is not None:
            return False
        candidates = sorted(
            list(level.candidate_plans),
            key=lambda p: (p.evidence_level, p.confidence, -p.failures),
            reverse=True,
        )
        for plan in candidates:
            if plan.status in {"failed", "completed"}:
                continue
            if not self._preflight_candidate_plan(plan, scene, legal):
                continue
            plan.status = "active"
            level.active_plan = plan
            if decision is not None:
                decision.setdefault("owner_attempts", []).append({"owner": "active_plan", "activated": plan.plan_id})
            self.logger.log_event("active_plan_started", plan.as_prompt())
            return True
        return False

    def _fail_active_plan(self, reason: str) -> None:
        level = self.memory.level
        plan = level.active_plan
        if plan is not None:
            plan.status = "failed"
            plan.failures += 1
            self.logger.log_event("active_plan_failed", {**plan.as_prompt(), "reason": reason[:240]})
        level.active_plan = None
        self._enter_repair_mode(reason)

    def _try_active_plan_action(
        self,
        scene: SceneSnapshot,
        legal: Sequence[Any],
        state: str,
        decision: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any], str] | None:
        plan = self.memory.level.active_plan
        if plan is None or plan.status not in {"active", "candidate"}:
            return None
        plan.status = "active"
        step = plan.current_step()
        if step is None:
            plan.status = "completed"
            self.memory.level.active_plan = None
            self.logger.log_event("active_plan_completed", plan.as_prompt())
            return None
        if decision is not None:
            decision.setdefault("owner_attempts", []).append({"owner": "active_plan", "plan_id": plan.plan_id, "step_id": step.step_id})
        compiled = self._compile_intent(step.intent, scene, legal)
        self._record_compile_result(decision, compiled, owner="active_plan")
        if not compiled.ok or compiled.proposal is None:
            step.attempts += 1
            step.last_failure = compiled.status.value
            step.status = "failed" if step.attempts >= step.max_attempts else step.status
            self._fail_active_plan(f"active_plan_compile_failed:{compiled.status.value}")
            return None
        if self._action_family_disabled(compiled.proposal.action_name):
            step.attempts += 1
            step.last_failure = f"action_family_disabled:{compiled.proposal.action_name}"
            step.status = "failed"
            self._fail_active_plan(step.last_failure)
            self._enter_repair_mode(f"active_plan_{step.last_failure}")
            return None
        compiled.proposal.raw["_from_active_plan"] = True
        compiled.proposal.raw["active_plan_id"] = plan.plan_id
        compiled.proposal.raw["active_plan_step_id"] = step.step_id
        selected = self._instantiate_proposal(compiled.proposal, scene, legal)
        if selected is None:
            step.attempts += 1
            step.last_failure = "instantiate_failed"
            self._fail_active_plan("active_plan_instantiate_failed")
            return None
        step.status = "running"
        action, proposal, _source = selected
        proposal = {
            **proposal,
            "owner": "active_plan",
            "active_plan_id": plan.plan_id,
            "active_plan_step_id": step.step_id,
        }
        return action, proposal, "active_plan"

    def _update_active_plan_after_transition(self, pending: PendingAction, report: TransitionReport, event: EventRecord) -> None:
        plan = self.memory.level.active_plan
        if plan is None or plan.status != "active":
            return
        step = plan.current_step()
        if step is None:
            return
        predicate_check = self._check_expected_predicates(step.expected_predicates or pending.expected_predicates, report)
        if predicate_check.get("failed_count", 0) > 0 or report.retry_detected:
            step.status = "failed"
            step.last_failure = "checkpoint_failed_or_retry"
            plan.failures += 1
            self._invalidate_or_downgrade_active_theory("active_plan_checkpoint_failed", event.event_id)
            self._fail_active_plan("checkpoint_failed_or_retry")
            return
        if report.effective_noop and pending.name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"}:
            step.status = "failed"
            step.last_failure = "noop"
            plan.failures += 1
            self._invalidate_or_downgrade_active_theory("active_plan_noop", event.event_id)
            self._fail_active_plan("noop")
            return
        step.status = "done"
        plan.cursor += 1
        self.logger.log_event("active_plan_step_done", {"plan_id": plan.plan_id, "step_id": step.step_id, "cursor": plan.cursor})
        if plan.current_step() is None:
            plan.status = "completed"
            self.memory.level.active_plan = None
            self.logger.log_event("active_plan_completed", plan.as_prompt())

    def _repair_candidate_id(self, proposal: ActionProposal) -> str:
        seed = json.dumps(
            {
                "source": proposal.source,
                "action": proposal.action_key(),
                "target": proposal.target_object_id,
                "purpose": proposal.purpose[:120],
                "candidate_kind": proposal.raw.get("candidate_kind", ""),
            },
            sort_keys=True,
            default=str,
        )
        return "RP_" + hashlib.sha1(seed.encode()).hexdigest()[:10].upper()

    def _vlm_repair_target_hints(self) -> set[str]:
        choice = self.memory.level.vlm_repair_choice if isinstance(self.memory.level.vlm_repair_choice, dict) else {}
        hints: set[str] = set()
        target = _short_string(choice.get("target_object_id"), 24).upper()
        if target:
            hints.add(target)
        raw_hints = choice.get("target_hints")
        if isinstance(raw_hints, list):
            for item in raw_hints:
                text = _short_string(item, 24).upper()
                if text:
                    hints.add(text)
        return hints

    def _action6_frontier_proposals(
        self,
        scene: SceneSnapshot,
        legal: Sequence[Any],
        reason: str,
    ) -> list[ActionProposal]:
        legal_names = {action_name(action) for action in legal}
        if "ACTION6" not in legal_names:
            return []
        level = self.memory.level
        only_action6 = self._action6_is_only_forward_action(legal)
        allow_family_disabled = False
        if self._action_family_disabled("ACTION6"):
            return []
        target_hints = self._vlm_repair_target_hints()
        attempts = level.action_attempt_counts_by_state.get(scene.state_hash, {})
        noops = level.noop_actions_by_state.get(scene.state_hash, set())
        candidates = self._action6_candidate_objects(scene)

        def history(candidate: Action6Candidate) -> dict[str, Any]:
            sig = self._action6_signature(scene, candidate.x, candidate.y, candidate.target_object_id, candidate.candidate_kind)
            seen_exact = False
            seen_local = False
            effects = 0
            failures = 0
            for rec in level.action6_memory.records[-160:]:
                exact = rec.state_hash == sig.state_hash and rec.x == sig.x and rec.y == sig.y
                local = (
                    rec.local_patch_signature == sig.local_patch_signature
                    and rec.target_type_key == sig.object_type_key
                    and rec.candidate_kind == sig.candidate_kind
                )
                if not (exact or local):
                    continue
                seen_exact = seen_exact or exact
                seen_local = seen_local or local
                if rec.outcome in {"noop", "nonprogress"}:
                    failures += 1
                else:
                    effects += 1
            return {"seen_exact": seen_exact, "seen_local": seen_local, "effects": effects, "failures": failures}

        ranked: list[tuple[tuple[int, int, int, int, int, float, float], Action6Candidate, dict[str, Any]]] = []
        suppressed = 0
        for candidate in candidates:
            key = self._action6_key_from_xy(candidate.x, candidate.y)
            blocked = self._action6_candidate_blocked(candidate, scene)
            hist = history(candidate)
            hinted = bool(candidate.target_object_id and candidate.target_object_id in target_hints)
            seen = bool(hist["seen_exact"] or hist["seen_local"] or key in noops)
            rank = (
                1 if seen else 0,
                -int(hist["effects"]),
                0 if hinted else 1,
                int(hist["failures"]),
                int(attempts.get(key, 0)),
                -float(candidate.prior_score),
                -float(candidate.salience),
            )
            if blocked or key in noops:
                suppressed += 1
                continue
            ranked.append((rank, candidate, hist))
        if not ranked and suppressed:
            self.logger.log_event(
                "action6_frontier_exhausted",
                {"reason": reason, "suppressed": suppressed, "only_action6": only_action6, "state": scene.state_hash[:12]},
            )
        ranked.sort(key=lambda item: item[0])

        proposals: list[ActionProposal] = []
        for idx, (_rank, candidate, hist) in enumerate(ranked[:8]):
            hinted = bool(candidate.target_object_id and candidate.target_object_id in target_hints)
            proposals.append(
                ActionProposal(
                    source="action6_frontier",
                    action_name="ACTION6",
                    x=candidate.x,
                    y=candidate.y,
                    target_object_id=candidate.target_object_id,
                    purpose=f"ACTION6 candidate frontier search after {reason}",
                    phase=V1Phase.RECOVER if level.recovery_mode else V1Phase.MECHANIC_EXPLORATION,
                    expected_predicates=[
                        {"type": "no_retry"},
                        {"type": "summary", "summary": "test untried or historically effective ACTION6 signature"},
                        {"type": "candidate_kind", "candidate_kind": candidate.candidate_kind},
                    ],
                    information_gain=0.62 if not hist["seen_exact"] and not hist["seen_local"] else 0.36,
                    goal_progress=0.18 + min(0.24, 0.08 * int(hist["effects"])),
                    novelty=0.75 if not hist["seen_exact"] and not hist["seen_local"] else 0.3,
                    priority=0.35 + (0.18 if hinted else 0.0) + min(0.2, 0.07 * int(hist["effects"])) - 0.08 * idx,
                    raw={
                        "candidate_id": candidate.candidate_id,
                        "candidate_kind": candidate.candidate_kind,
                        "local_patch_signature": candidate.local_patch_signature,
                        "frontier_history": hist,
                        "vlm_target_hint": hinted,
                        "only_action6_frontier": only_action6,
                        "family_disabled_bypass": allow_family_disabled,
                        "relaxed_frontier": False,
                    },
                )
            )
        return proposals

    def _repair_policy_proposals(self, scene: SceneSnapshot, legal: Sequence[Any]) -> list[ActionProposal]:
        level = self.memory.level
        reason = level.recovery_reason or level.recovery_mode or "repair"
        if self._action6_is_only_forward_action(legal):
            return self._action6_frontier_proposals(scene, legal, reason)
        proposals: list[ActionProposal] = []
        if level.recovery_mode == "escape_from_action6_loop":
            self._disable_action_family("ACTION6", 1, "active action6 recovery")
            proposals.extend(self._anti_loop_escape_proposals(scene, legal, reason))
        elif level.recovery_mode == "repair_nonprogress_loop":
            proposals.extend(self._anti_loop_escape_proposals(scene, legal, reason))
            proposals.extend(self._level_theory_proposals(scene, legal))
            proposals.extend(self._experiment_scheduler_proposals(scene, legal))
        elif level.recovery_mode == "repair_compile_failure":
            last = level.last_compile_failures[-1] if level.last_compile_failures else {}
            status = CompileStatus(str(last.get("status", CompileStatus.UNSUPPORTED_INTENT.value))) if last.get("status") in {item.value for item in CompileStatus} else CompileStatus.UNSUPPORTED_INTENT
            proposals.extend(self._grounding_proposals_for_nav_failure(status, scene, legal))
            proposals.extend(self._anti_loop_escape_proposals(scene, legal, reason or status.value))
        elif level.recovery_mode == "recover_after_retry":
            proposals.extend(self._anti_loop_escape_proposals(scene, legal, reason or "retry"))
            proposals.extend(self._experiment_scheduler_proposals(scene, legal))
        elif level.recovery_mode:
            proposals.extend(self._anti_loop_escape_proposals(scene, legal, reason or "proposal_recovery"))
            proposals.extend(self._experiment_scheduler_proposals(scene, legal))
        return proposals

    def _cache_repair_action_candidates(
        self,
        proposals: list[ActionProposal],
        scene: SceneSnapshot,
        legal: Sequence[Any],
        decision: dict[str, Any] | None = None,
    ) -> None:
        payloads: list[dict[str, Any]] = []
        for proposal in proposals[:12]:
            repair_id = self._repair_candidate_id(proposal)
            proposal.raw["repair_candidate_id"] = repair_id
            payload = {
                "repair_candidate_id": repair_id,
                "source": proposal.source,
                "action": proposal.action_key(),
                "target_object_id": proposal.target_object_id,
                "purpose": proposal.purpose[:240],
                "expected_predicates": proposal.expected_predicates[:6],
                "priority": round(float(proposal.priority), 4),
                "information_gain": round(float(proposal.information_gain), 4),
                "goal_progress": round(float(proposal.goal_progress), 4),
                "raw": _json_safe(proposal.raw),
            }
            payloads.append(payload)
        self.memory.level.repair_action_candidates = payloads
        if decision is not None and payloads:
            decision.setdefault("candidates", []).append({"owner": "repair_policy", "repair_candidates": payloads[:8]})

    def _apply_vlm_repair_choice(self, result: VLMResult) -> None:
        decision = result.decision if isinstance(result.decision, dict) else {}
        if _short_string(decision.get("kind"), 80) != "repair_decision":
            return
        level = self.memory.level
        selected_id = _short_string(
            decision.get("selected_repair_candidate_id")
            or decision.get("repair_candidate_id")
            or decision.get("candidate_id"),
            40,
        ).upper()
        target_hints: list[str] = []
        for raw in decision.get("intents", []) if isinstance(decision.get("intents"), list) else []:
            if not isinstance(raw, dict):
                continue
            target = _short_string(raw.get("target_object_id") or raw.get("target") or raw.get("object_id"), 24).upper()
            if target and target not in target_hints:
                target_hints.append(target)
        explicit_target = _short_string(decision.get("target_object_id"), 24).upper()
        if explicit_target and explicit_target not in target_hints:
            target_hints.append(explicit_target)
        known = {
            _short_string(item.get("repair_candidate_id"), 40).upper(): item
            for item in level.repair_action_candidates
            if isinstance(item, dict)
        }
        choice: dict[str, Any] = {
            "step": self._controller_step(),
            "reason": level.recovery_reason,
            "target_hints": target_hints,
            "rationale": _short_string(decision.get("rationale") or decision.get("goal"), 240),
        }
        if selected_id and selected_id in known:
            choice.update(known[selected_id])
            choice["repair_candidate_id"] = selected_id
            level.vlm_repair_choice = choice
            self.logger.log_event("vlm_repair_candidate_selected", {"repair_candidate_id": selected_id, "target_hints": target_hints})
        else:
            if target_hints:
                level.vlm_repair_choice = choice
            if selected_id:
                self.logger.log_event("vlm_repair_candidate_ignored", {"repair_candidate_id": selected_id, "known": sorted(known)[:12]})
        if decision.get("intents"):
            self.logger.log_event("vlm_repair_free_actions_ignored", {"target_hints": target_hints, "reason": "candidate_choice_only"})

    def _bias_repair_proposals_from_vlm_choice(self, proposals: list[ActionProposal]) -> None:
        choice = self.memory.level.vlm_repair_choice if isinstance(self.memory.level.vlm_repair_choice, dict) else {}
        selected_id = _short_string(choice.get("repair_candidate_id"), 40).upper()
        if not selected_id:
            return
        for proposal in proposals:
            if _short_string(proposal.raw.get("repair_candidate_id"), 40).upper() == selected_id:
                proposal.priority += 0.9
                proposal.raw["vlm_repair_selected"] = True
                break

    def _maybe_request_repair_vlm_update(
        self,
        scene: SceneSnapshot,
        transition: TransitionReport | None,
        legal: Sequence[Any],
        decision: dict[str, Any] | None = None,
        *,
        state: str = "NOT_FINISHED",
    ) -> VLMResult | None:
        level = self.memory.level
        if not level.recovery_mode:
            return None
        if level.recovery_until_step and self._controller_step() > level.recovery_until_step:
            return None
        if (
            level.last_repair_vlm_reason == level.recovery_reason
            and self._controller_step() - level.last_repair_vlm_step < 6
        ):
            return None
        proposals = self._repair_policy_proposals(scene, legal)
        self._cache_repair_action_candidates(proposals, scene, legal, decision)
        mode = self._choose_vlm_mode(scene, transition, legal)
        if mode != VLMMode.REPAIR_DECISION:
            return None
        level.last_repair_vlm_reason = level.recovery_reason
        level.last_repair_vlm_step = self._controller_step()
        if decision is not None:
            decision["vlm_mode"] = mode.value
            decision.setdefault("owner_attempts", []).append(
                {"owner": "vlm_repair_analysis", "mode": mode.value, "reason": level.recovery_reason}
            )
        result = self._request_vlm_once(scene, transition, legal, mode)
        if result is not None:
            self._apply_vlm_update(result, transition, scene, legal)
            self._apply_vlm_repair_choice(result)
        return result

    def _try_repair_action(
        self,
        scene: SceneSnapshot,
        legal: Sequence[Any],
        state: str,
        decision: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any], str] | None:
        level = self.memory.level
        if not level.recovery_mode:
            return None
        if level.recovery_until_step and self._controller_step() > level.recovery_until_step:
            self.logger.log_event(
                "repair_mode_expired",
                {"mode": level.recovery_mode, "reason": level.recovery_reason, "until": level.recovery_until_step},
            )
            level.recovery_mode = ""
            level.recovery_reason = ""
            level.recovery_until_step = 0
            level.repair_action_candidates = []
            if level.force_vlm_reason == "repair_decision":
                level.force_vlm_reason = ""
            return None
        if decision is not None:
            decision.setdefault("owner_attempts", []).append({"owner": "repair", "mode": level.recovery_mode, "reason": level.recovery_reason})
        proposals = self._repair_policy_proposals(scene, legal)
        self._cache_repair_action_candidates(proposals, scene, legal, decision)
        self._bias_repair_proposals_from_vlm_choice(proposals)
        chosen = self._choose_best_proposal(proposals, scene, legal, state, decision=decision)
        selected = self._instantiate_proposal(chosen, scene, legal) if chosen else None
        if selected is None:
            return None
        action, proposal, source = selected
        return action, {**proposal, "owner": "repair", "repair_mode": level.recovery_mode}, source

    def _try_exploration_action(
        self,
        scene: SceneSnapshot,
        legal: Sequence[Any],
        state: str,
        decision: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any], str] | None:
        if self._maybe_activate_candidate_plan(scene, legal, decision):
            selected = self._try_active_plan_action(scene, legal, state, decision)
            if selected is not None:
                return selected
        if decision is not None:
            decision.setdefault("owner_attempts", []).append({"owner": "exploration"})
        proposals: list[ActionProposal] = []
        proposals.extend(self._transfer_bootstrap_proposals(scene, legal))
        need_grounding = not self.memory.level.controlled_object_id or not self._grounded_action_vectors(legal)
        scheduled = self._experiment_scheduler_proposals(scene, legal)
        if need_grounding:
            grounding = [proposal for proposal in scheduled if proposal.source == "action_grounding_probe"]
            chosen = self._choose_best_proposal(grounding, scene, legal, state, decision=decision)
            selected = self._instantiate_proposal(chosen, scene, legal) if chosen else None
            if selected is not None:
                action, proposal, source = selected
                return action, {**proposal, "owner": "exploration"}, source
        proposals.extend(self._level_theory_proposals(scene, legal))
        proposals.extend(scheduled)
        chosen = self._choose_best_proposal(proposals, scene, legal, state, decision=decision)
        selected = self._instantiate_proposal(chosen, scene, legal) if chosen else None
        if selected is None:
            return None
        action, proposal, source = selected
        return action, {**proposal, "owner": "exploration"}, source

    def _note_fallback_selection(self, source: str, key: str, scene: SceneSnapshot) -> None:
        fallback_sources = {"least_repeated_safe", "last_resort_non_reset"}
        self.memory.level.fallback_action_records.append(
            {
                "step": self._controller_step(),
                "source": _short_string(source, 80),
                "action": _short_string(key, 80),
                "state": scene.state_hash[:12],
                "before": scene.state_hash[:12],
                "after": "",
                "pending": True,
                "is_fallback": source in fallback_sources,
                "frontier": False,
                "effectful": False,
                "state_novel": False,
            }
        )

    def _update_fallback_transition_record(
        self,
        pending: PendingAction,
        report: TransitionReport,
        current_scene: SceneSnapshot,
        effect_flags: dict[str, Any],
    ) -> None:
        key = pending.action_key()
        level = self.memory.level
        matched: dict[str, Any] | None = None
        for item in reversed(level.fallback_action_records):
            if not item.get("pending"):
                continue
            if item.get("action") != key:
                continue
            if item.get("before") != pending.scene_before.state_hash[:12]:
                continue
            matched = item
            break
        if matched is None:
            return
        matched.update(
            {
                "after": current_scene.state_hash[:12],
                "pending": False,
                "effective_noop": bool(report.effective_noop),
                "changed_cells": report.changed_cell_count,
                "world_changed_cells": report.world_changed_cell_count,
                "state_changed": bool(effect_flags.get("state_changed")),
                "state_novel": bool(effect_flags.get("state_novel")),
                "new_object_frontier": bool(effect_flags.get("new_object_frontier")),
                "effectful": bool(effect_flags.get("effectful")),
                "frontier": bool(effect_flags.get("frontier")),
                "summary": report.summary[:240],
            }
        )
        if matched.get("is_fallback"):
            self.logger.log_event(
                "fallback_effect_recorded",
                {
                    "source": matched.get("source"),
                    "action": key,
                    "before": matched.get("before"),
                    "after": matched.get("after"),
                    "frontier": matched.get("frontier"),
                    "effectful": matched.get("effectful"),
                    "state_novel": matched.get("state_novel"),
                },
            )

    def _fallback_record_has_frontier(self, item: dict[str, Any]) -> bool:
        if item.get("pending"):
            return False
        return bool(item.get("state_novel") or item.get("new_object_frontier"))

    def _fallback_overuse_detected(self) -> bool:
        level = self.memory.level
        if self._controller_step() < 8:
            return False
        if self._controller_step() - level.fallback_overuse_repair_step < 5:
            return False
        recent = list(level.fallback_action_records)[-10:]
        if len(recent) < 6:
            return False
        fallback_recent = [item for item in recent if item.get("is_fallback")]
        fallback_count = len(fallback_recent)
        consecutive = 0
        for item in reversed(recent):
            if not item.get("is_fallback"):
                break
            consecutive += 1
        repeated_sources = {item.get("source") for item in recent[-6:] if item.get("is_fallback")}
        source_pressure = consecutive >= 4 or fallback_count >= 7 or (fallback_count >= 6 and len(repeated_sources) <= 2)
        if not source_pressure:
            return False

        frontier_count = sum(1 for item in fallback_recent if self._fallback_record_has_frontier(item))
        stale_count = fallback_count - frontier_count
        stale_consecutive = 0
        for item in reversed(recent):
            if not item.get("is_fallback"):
                break
            if self._fallback_record_has_frontier(item):
                break
            stale_consecutive += 1
        if frontier_count >= 2 and frontier_count >= max(2, fallback_count // 3):
            self.logger.log_event(
                "fallback_overuse_guard_skipped",
                {
                    "reason": "fallback_has_meaningful_frontier",
                    "fallback_count": fallback_count,
                    "frontier_count": frontier_count,
                    "stale_count": stale_count,
                    "consecutive": consecutive,
                },
            )
            return False
        return stale_consecutive >= 4 or stale_count >= max(5, fallback_count - 1)

    def _record_anti_loop_escape_outcome(
        self,
        pending: PendingAction,
        report: TransitionReport,
        current_scene: SceneSnapshot,
        effect_flags: dict[str, Any],
        event: EventRecord,
    ) -> None:
        level = self.memory.level
        if level.recovery_mode != "escape_from_action6_loop":
            return
        if pending.source != "anti_loop_escape" or pending.name not in {"ACTION1", "ACTION2"}:
            return
        progress = bool(
            effect_flags.get("frontier")
            or (
                effect_flags.get("effectful")
                and (
                    effect_flags.get("movement")
                    or effect_flags.get("resource_delta")
                    or report.world_changed_cell_count > 12
                )
            )
        )
        if progress:
            level.anti_loop_escape_failures = 0
            return
        level.anti_loop_escape_failures += 1
        level.last_anti_loop_escape_failure_step = self._controller_step()
        self._disable_action_family(pending.name, 3, "anti_loop_escape_no_progress")
        self.logger.log_event(
            "anti_loop_escape_no_progress",
            {
                "action": pending.name,
                "event_id": event.event_id,
                "failures": level.anti_loop_escape_failures,
                "summary": report.summary[:240],
            },
        )
        if level.anti_loop_escape_failures >= 2:
            level.recovery_until_step = min(level.recovery_until_step or self._controller_step(), self._controller_step())
            level.recovery_mode = ""
            level.recovery_reason = ""
            if level.force_vlm_reason == "repair_decision":
                level.force_vlm_reason = ""
            self.logger.log_event(
                "anti_loop_escape_budget_exhausted",
                {"event_id": event.event_id, "failures": level.anti_loop_escape_failures},
            )

    def _last_safe_recovery_action(
        self,
        scene: SceneSnapshot,
        legal: Sequence[Any],
        state: str,
        decision: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any], str] | None:
        if self._action6_is_only_forward_action(legal):
            proposals = self._action6_frontier_proposals(scene, legal, "only_action6_fallback")
            self._cache_repair_action_candidates(proposals, scene, legal, decision)
            self._bias_repair_proposals_from_vlm_choice(proposals)
            chosen = self._choose_best_proposal(proposals, scene, legal, state, decision=decision)
            selected = self._instantiate_proposal(chosen, scene, legal) if chosen else None
            if selected is not None:
                action, proposal, source = selected
                return action, {**proposal, "owner": "action6_frontier"}, source
            action = self._last_resort_non_reset_legal_action(scene, legal)
            if action is not None:
                proposal = getattr(action, "reasoning", {})
                return action, {**(proposal if isinstance(proposal, dict) else {}), "owner": "fallback"}, "last_resort_non_reset"
            return None
        if self.memory.level.recovery_mode:
            selected = self._try_repair_action(scene, legal, state, decision)
            if selected is not None:
                return selected
        if self._fallback_overuse_detected():
            self.memory.level.fallback_overuse_repair_step = self._controller_step()
            self._enter_repair_mode("fallback_overuse_loop")
            if decision is not None:
                decision.setdefault("owner_attempts", []).append(
                    {"owner": "fallback_guard", "reason": "fallback_overuse_loop"}
                )
            selected = self._try_repair_action(scene, legal, state, decision)
            if selected is not None:
                return selected
            return None
        action = self._least_repeated_safe_legal_action(scene, legal)
        if action is not None:
            proposal = getattr(action, "reasoning", {})
            return action, {**(proposal if isinstance(proposal, dict) else {}), "owner": "fallback"}, "least_repeated_safe"
        action = self._last_resort_non_reset_legal_action(scene, legal)
        if action is not None:
            proposal = getattr(action, "reasoning", {})
            return action, {**(proposal if isinstance(proposal, dict) else {}), "owner": "fallback"}, "last_resort_non_reset"
        if state in {"NOT_PLAYED", "GAME_OVER", "WIN"}:
            return self._reset_action(), {"name": "RESET", "owner": "terminal_reset"}, "reset"
        return None

    def _anti_loop_escape_proposals(self, scene: SceneSnapshot, legal: Sequence[Any], reason: str) -> list[ActionProposal]:
        level = self.memory.level
        legal_names = {action_name(action) for action in legal}
        state = scene.state_hash
        noops = level.noop_actions_by_state.get(state, set())
        attempts = level.action_attempt_counts_by_state.get(state, {})
        proposals: list[ActionProposal] = []
        for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"):
            if level.recovery_mode == "escape_from_action6_loop" and level.anti_loop_escape_failures >= 2 and name in {"ACTION1", "ACTION2"}:
                continue
            if self._action_family_disabled(name):
                continue
            if name in legal_names and name not in noops and attempts.get(name, 0) == 0:
                proposals.append(
                    ActionProposal(
                        source="anti_loop_escape",
                        action_name=name,
                        purpose=f"escape loop after {reason} by trying untested primitive",
                        phase=V1Phase.RECOVER,
                        expected_predicates=[{"type": "not_noop"}, {"type": "no_retry"}],
                        information_gain=0.45,
                        novelty=0.8,
                        priority=0.45,
                    )
                )
        if "ACTION6" in legal_names and level.recovery_mode != "escape_from_action6_loop" and self._action6_budget_available(legal):
            cand = self._select_action6_candidate(scene, avoid_keys=set(level.recent_action_keys))
            if cand is not None:
                proposals.append(
                    ActionProposal(
                        source="anti_loop_escape",
                        action_name="ACTION6",
                        x=cand.x,
                        y=cand.y,
                        target_object_id=cand.target_object_id,
                        purpose=f"escape loop after {reason} by testing nonduplicate click candidate",
                        phase=V1Phase.RECOVER,
                        expected_predicates=[{"type": "no_retry"}, {"type": "summary", "summary": "observe non-loop click response"}],
                        information_gain=0.55,
                        novelty=0.75,
                        priority=0.5,
                        raw={"candidate_kind": cand.candidate_kind, "candidate_id": cand.candidate_id},
                    )
                )
        return proposals

    def _least_repeated_safe_legal_action(self, scene: SceneSnapshot, legal: Sequence[Any]) -> Any | None:
        legal_by_name = {action_name(action): action for action in legal}
        state = scene.state_hash
        attempts = self.memory.level.action_attempt_counts_by_state.get(state, {})
        noops = self.memory.level.noop_actions_by_state.get(state, set())
        recent = list(self.memory.level.recent_action_keys)
        candidates: list[tuple[tuple[int, int, int], str, Any]] = []
        for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"):
            if name not in legal_by_name or self._action_family_disabled(name):
                continue
            key = name
            if key in noops or (state, key) in self.memory.level.blocked_state_action_pairs:
                continue
            hard_loop, _soft = self._loop_guard_decision(key, scene)
            if hard_loop:
                continue
            score = (attempts.get(key, 0), recent[-8:].count(key), recent.count(key))
            candidates.append((score, key, legal_by_name[name]))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            _score, key, action = candidates[0]
            return self._attach_reasoning(
                action,
                {"name": key, "purpose": "least repeated recovery action", "expected_change": "escape rejected proposal state"},
            )
        if "ACTION6" in legal_by_name and self._action6_budget_available(legal):
            cand = self._select_action6_candidate(scene)
            if cand is not None:
                return self._make_action6(
                    legal_by_name["ACTION6"],
                    cand.x,
                    cand.y,
                    {
                        "name": "ACTION6",
                        "x": cand.x,
                        "y": cand.y,
                        "target_object_id": cand.target_object_id,
                        "purpose": "least repeated recovery click",
                        "expected_change": "escape rejected proposal state",
                        "expected_predicates": [{"type": "not_noop"}],
                        "relaxed_recovery": False,
                    },
                )
        return None

    def _last_resort_non_reset_legal_action(self, scene: SceneSnapshot, legal: Sequence[Any]) -> Any | None:
        legal_by_name = {action_name(action): action for action in legal}
        state = scene.state_hash
        attempts = self.memory.level.action_attempt_counts_by_state.get(state, {})
        noops = self.memory.level.noop_actions_by_state.get(state, set())
        recent = list(self.memory.level.recent_action_keys)
        primitive_choices: list[tuple[tuple[int, int, int, int], str, Any]] = []
        for index, name in enumerate(("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")):
            if name not in legal_by_name or self._action_family_disabled(name):
                continue
            if name in noops or (state, name) in self.memory.level.blocked_state_action_pairs:
                continue
            hard_loop, _soft = self._loop_guard_decision(name, scene)
            if hard_loop or self.memory.game.failure_model.forbids(list(recent), name):
                continue
            score = (recent[-6:].count(name), attempts.get(name, 0), recent.count(name), index)
            primitive_choices.append((score, name, legal_by_name[name]))
        if primitive_choices:
            primitive_choices.sort(key=lambda item: item[0])
            _score, key, action = primitive_choices[0]
            return self._attach_reasoning(
                action,
                {
                    "name": key,
                    "purpose": "last-resort non-reset recovery action",
                    "expected_change": "avoid active-level reset storm after proposal exhaustion",
                    "expected_predicates": [{"type": "no_retry"}],
                },
            )
        only_action6 = self._action6_is_only_forward_action(legal)
        if "ACTION6" in legal_by_name and self._action6_budget_available(legal):
            avoid = set(recent[-8:]) | set(self.memory.level.fallback_recent_keys)
            cand = self._select_action6_candidate(scene, avoid_keys=avoid)
            relaxed_recovery = False
            if cand is None and only_action6:
                cand = self._select_action6_candidate_allowing_state_noops(scene, avoid_keys=avoid)
                relaxed_recovery = cand is not None
            if cand is not None:
                x, y = cand.x, cand.y
                key = self._action6_key_from_xy(x, y)
                self.logger.log_event(
                    "recovery_last_resort_action6_selected",
                    {
                        "action": key,
                        "state": scene.state_hash[:12],
                        "candidate_id": cand.candidate_id,
                        "only_action6": only_action6,
                        "relaxed_recovery": relaxed_recovery,
                    },
                )
                return self._make_action6(
                    legal_by_name["ACTION6"],
                    x,
                    y,
                    {
                        "name": "ACTION6",
                        "x": x,
                        "y": y,
                        "target_object_id": cand.target_object_id,
                        "purpose": "last-resort non-reset recovery click",
                        "expected_change": "avoid active-level reset storm after proposal exhaustion",
                        "expected_predicates": [{"type": "no_retry"}],
                        "relaxed_recovery": relaxed_recovery,
                    },
                )
        return None

    def _recovery_action(self, scene: SceneSnapshot, legal: Sequence[Any], state: str, reason: str) -> Any:
        level = self.memory.level
        level.last_recovery_source = "recovery"
        level.last_recovery_reasoning = {"reason": reason, "source": "recovery"}
        if state in {"NOT_PLAYED", "GAME_OVER", "WIN"}:
            level.last_recovery_source = "reset"
            level.last_recovery_reasoning = {"reason": reason, "source": "reset"}
            return self._reset_action()
        level.recovery_mode = "proposal_recovery"
        level.recovery_reason = reason
        level.recovery_until_step = max(level.recovery_until_step, self._controller_step() + 6)
        self.memory.game.controller_stats.recovery_counts[reason] += 1

        proposals: list[ActionProposal] = []
        proposals.extend(self._anti_loop_escape_proposals(scene, legal, reason))
        proposals.extend(self._experiment_scheduler_proposals(scene, legal))
        chosen = self._choose_best_proposal(proposals, scene, legal, state)
        selected = self._instantiate_proposal(chosen, scene, legal) if chosen else None
        if selected is not None:
            action, proposal, source = selected
            key = self._action_key_for_action(action)
            level.fallback_recent_keys.append(key)
            level.last_recovery_source = source
            level.last_recovery_reasoning = {**proposal, "reason": reason, "source": source, "action": key}
            self.logger.log_event("recovery_selected", {"reason": reason, "source": source, "action": key})
            if source == "anti_loop_escape":
                self.logger.log_event("anti_loop_escape_selected", {"reason": reason, "action": key})
            return action

        action = self._least_repeated_safe_legal_action(scene, legal)
        if action is not None:
            key = self._action_key_for_action(action)
            level.fallback_recent_keys.append(key)
            level.last_recovery_source = "least_repeated_safe"
            level.last_recovery_reasoning = {"reason": reason, "source": "least_repeated_safe", "action": key}
            self.logger.log_event("recovery_selected", {"reason": reason, "source": "least_repeated_safe", "action": key})
            return action

        action = self._last_resort_non_reset_legal_action(scene, legal)
        if action is not None:
            key = self._action_key_for_action(action)
            level.fallback_recent_keys.append(key)
            level.last_recovery_source = "last_resort_non_reset"
            level.last_recovery_reasoning = {"reason": reason, "source": "last_resort_non_reset", "action": key}
            self.logger.log_event("recovery_selected", {"reason": reason, "source": "last_resort_non_reset", "action": key})
            return action

        level.awaiting_reset = True
        level.force_vlm_reason = "recovery_no_safe_action"
        level.last_recovery_source = "reset"
        level.last_recovery_reasoning = {"reason": reason, "source": "reset", "detail": "recovery_no_safe_action"}
        self.logger.log_event("recovery_no_safe_action", {"state": state, "legal": [action_name(a) for a in legal], "reason": reason})
        return self._reset_action()

    def _emergency_safe_action(self, scene: SceneSnapshot, legal: Sequence[Any], state: str) -> Any:
        return self._recovery_action(scene, legal, state, reason="proposal_exhausted")

    def _cap_vlm_evidence_level(
        self, requested: Any, transition: TransitionReport | None, event_ids: list[int] | None = None
    ) -> int:
        try:
            requested_int = int(requested or 0)
        except (TypeError, ValueError):
            requested_int = 0
        if transition is None:
            cap = 1 if event_ids else 0
        elif transition.interaction_event or transition.moved_objects or transition.transformed_objects or transition.retry_detected:
            cap = 2
        else:
            cap = 1
        return max(0, min(requested_int, cap))

    def _upsert_hypothesis(
        self,
        kind: str,
        statement: str,
        confidence: float,
        scope: str,
        event_ids: list[int] | None = None,
        evidence_level: int = 0,
    ) -> str:
        statement = _short_string(statement, 500)
        if not statement:
            return ""
        hid = f"H:{kind}:{hashlib.sha1(_normalize_claim(statement).encode()).hexdigest()[:10]}"
        hyp = self.memory.game.hypotheses.get(hid)
        if hyp is None:
            hyp = Hypothesis(hid, scope, kind, statement, confidence=round(_clamp01(confidence), 3), evidence_level=evidence_level)
            self.memory.game.hypotheses[hid] = hyp
        else:
            hyp.confidence = round(max(hyp.confidence, _clamp01(confidence)), 3)
            hyp.evidence_level = max(hyp.evidence_level, evidence_level)
        if event_ids:
            hyp.supporting_event_ids = sorted(set(hyp.supporting_event_ids + event_ids))[-20:]
        return hid

    def _apply_level_instantiation_patch(self, patch: dict[str, Any]) -> None:
        if not isinstance(patch, dict):
            return
        schema_id = _short_string(patch.get("schema_id"), 80)
        if not schema_id:
            return
        inst = self.memory.level.goal_instantiation or LevelGoalInstantiation(self.memory.level.level_index, schema_id=schema_id)
        if isinstance(patch.get("role_bindings"), dict):
            inst.role_bindings.update({str(k): _short_string(v, 24).upper() for k, v in patch["role_bindings"].items()})
        if isinstance(patch.get("concrete_values"), dict):
            inst.concrete_values.update(patch["concrete_values"])
        if isinstance(patch.get("unknown_slots"), list):
            inst.unknown_slots = [_short_string(v, 80) for v in patch["unknown_slots"]]
        inst.confidence = max(inst.confidence, _clamp01(patch.get("confidence", 0.0)))
        self.memory.level.goal_instantiation = inst

    def _seed_initial_level_theory(self, scene: SceneSnapshot) -> None:
        if self.memory.level.level_theories:
            return
        critical = [
            {
                "object_id": obj.track_id,
                "role": "salient_candidate",
                "confidence": 0.25,
                "evidence": f"visible {obj.shape_label} area={obj.area}",
            }
            for obj in sorted(scene.objects, key=lambda item: -item.salience)[:6]
        ]
        has_frame = bool(scene.template_relations)
        win = "Find the object interaction or navigation target that advances the level."
        mech = "Initial observer prior: test salient non-edge objects and framed/pattern relations before committing."
        if has_frame:
            win = "Satisfy or match the framed/status pattern relation, then interact with the resulting goal."
            mech = "Framed template relations suggest a transformer or gate may need verification."
        solve_sketch = []
        if critical:
            solve_sketch.append({"intent": "test_object", "target_object_id": critical[0]["object_id"], "purpose": "test first salient object for mechanism evidence"})
        seed = json.dumps({"level": self.memory.level.level_index, "state": scene.state_hash, "win": win, "mech": mech}, sort_keys=True)
        theory = LevelTheory(
            theory_id="LT:" + hashlib.sha1(seed.encode()).hexdigest()[:10],
            level_index=self.memory.level.level_index,
            created_step=self.memory.level.total_action_count,
            source="observer_init",
            confidence=0.28,
            evidence_level=0,
            win_condition_hypothesis=win,
            mechanism_hypothesis=mech,
            critical_objects=critical,
            expected_progress_signals=[{"type": "movement"}, {"type": "transform"}, {"type": "counter_delta_nonpositive"}],
            solve_sketch=solve_sketch,
            discriminating_tests=solve_sketch[:2],
            invalidating_evidence=["Repeated no-op or retry on the same object/action in the same state."],
        )
        self.memory.level.level_theories.append(theory)
        self.memory.level.active_theory_id = theory.theory_id
        self.memory.game.controller_stats.level_theory_updates["created"] += 1
        self.logger.log_event("level_theory_created", theory.as_prompt())

    def _apply_level_theory_patch(
        self, patch: dict[str, Any], source: str, transition: TransitionReport | None = None
    ) -> None:
        if not isinstance(patch, dict):
            return
        win = _short_string(patch.get("win_condition_hypothesis") or patch.get("win_condition") or patch.get("goal"), 500)
        mech = _short_string(patch.get("mechanism_hypothesis") or patch.get("mechanism"), 500)
        if not win and not mech:
            return
        confidence = _clamp01(patch.get("confidence", 0.45))
        if transition is None:
            confidence = min(confidence, 0.62)
            evidence_level = 0
        else:
            evidence_level = self._cap_vlm_evidence_level(patch.get("evidence_level", 1), transition)
        seed = json.dumps({"win": win, "mech": mech, "level": self.memory.level.level_index}, sort_keys=True)
        theory_id = "LT:" + hashlib.sha1(seed.encode()).hexdigest()[:10]
        existing = next((t for t in self.memory.level.level_theories if t.theory_id == theory_id), None)
        if existing is None:
            theory = LevelTheory(
                theory_id=theory_id,
                level_index=self.memory.level.level_index,
                created_step=self.memory.level.total_action_count,
                source=source,
                confidence=confidence,
                evidence_level=evidence_level,
                status="supported" if evidence_level >= 1 else "candidate",
                win_condition_hypothesis=win,
                mechanism_hypothesis=mech,
                critical_objects=patch.get("critical_objects") if isinstance(patch.get("critical_objects"), list) else [],
                expected_progress_signals=patch.get("expected_progress_signals") if isinstance(patch.get("expected_progress_signals"), list) else [],
                solve_sketch=patch.get("solve_sketch") if isinstance(patch.get("solve_sketch"), list) else [],
                discriminating_tests=patch.get("discriminating_tests") if isinstance(patch.get("discriminating_tests"), list) else [],
                invalidating_evidence=[_short_string(x, 200) for x in patch.get("invalidating_evidence", [])] if isinstance(patch.get("invalidating_evidence"), list) else [],
            )
            self.memory.level.level_theories.append(theory)
            self.memory.game.controller_stats.level_theory_updates["created"] += 1
            self.logger.log_event("level_theory_created", theory.as_prompt())
        else:
            old_confidence = existing.confidence
            existing.confidence = max(existing.confidence, confidence)
            existing.evidence_level = max(existing.evidence_level, evidence_level)
            if evidence_level >= 1:
                existing.status = "supported"
            if confidence >= old_confidence:
                existing.win_condition_hypothesis = win or existing.win_condition_hypothesis
                existing.mechanism_hypothesis = mech or existing.mechanism_hypothesis
                if isinstance(patch.get("critical_objects"), list):
                    existing.critical_objects = patch["critical_objects"][:8]
                if isinstance(patch.get("expected_progress_signals"), list):
                    existing.expected_progress_signals = patch["expected_progress_signals"][:8]
                if isinstance(patch.get("solve_sketch"), list):
                    existing.solve_sketch = patch["solve_sketch"][:10]
                if isinstance(patch.get("discriminating_tests"), list):
                    existing.discriminating_tests = patch["discriminating_tests"][:8]
            self.memory.game.controller_stats.level_theory_updates["supported" if evidence_level >= 1 else "updated"] += 1
            self.logger.log_event("level_theory_supported" if evidence_level >= 1 else "level_theory_created", existing.as_prompt())
        self.memory.level.level_theories.sort(key=lambda t: (t.status == "verified", t.evidence_level, t.confidence), reverse=True)
        self.memory.level.level_theories = self.memory.level.level_theories[:6]
        self.memory.level.active_theory_id = self.memory.level.level_theories[0].theory_id if self.memory.level.level_theories else ""

    def _invalidate_or_downgrade_active_theory(self, reason: str, event_id: int) -> None:
        tid = self.memory.level.active_theory_id
        if not tid:
            return
        for theory in self.memory.level.level_theories:
            if theory.theory_id != tid:
                continue
            if event_id not in theory.contradicting_event_ids:
                theory.contradicting_event_ids.append(event_id)
            theory.confidence = max(0.0, theory.confidence - 0.18)
            if len(theory.contradicting_event_ids) >= 2 or theory.confidence < 0.2:
                theory.status = "rejected"
                self.memory.game.controller_stats.level_theory_updates["rejected"] += 1
                self.logger.log_event("level_theory_invalidated", {**theory.as_prompt(), "reason": reason[:240]})
            else:
                self.memory.game.controller_stats.level_theory_updates["downgraded"] += 1
                self.logger.log_event("level_theory_downgraded", {**theory.as_prompt(), "reason": reason[:240]})
            break

    def _start_new_level(self, levels_completed: int, scene: SceneSnapshot) -> None:
        self.memory.game.levels_seen += 1
        start_stage = (
            V1Phase.TRANSFER_BOOTSTRAP
            if self.config.enable_transfer_bootstrap and (self.memory.game.goal_schemas or self.memory.game.level_outcomes)
            else V1Phase.INIT
        )
        self.memory.level = LevelMemory(
            level_index=levels_completed,
            levels_completed_at_start=levels_completed,
            stage=start_stage,
            initial_scene_summary=scene.summary[:2200],
            initial_state_hash=scene.state_hash,
            recent_events=deque(maxlen=self.config.max_recent_events),
            force_vlm_reason="initial_scene",
            actions_since_vlm=999,
        )
        self.memory.level.initial_scene_ref = scene
        self.memory.level.initial_rgb = scene.rgb
        self.memory.level.initial_annotated_rgb = scene.annotated_rgb
        self._seed_initial_level_theory(scene)
        self.memory.level.recent_state_hashes.append(scene.state_hash)
        self.memory.level.recent_events.append(
            f"new_level={levels_completed} state={scene.state_hash[:12]} "
            f"counter={scene.counter_value}/{scene.counter_capacity} lives={scene.life_count}"
        )
        self._update_mechanism_from_scene(scene)
        self.logger.log_event(
            "new_level",
            {
                "level": levels_completed,
                "state": scene.state_hash[:12],
                "objects": [obj.track_id for obj in scene.objects],
                "counter": scene.counter_value,
                "capacity": scene.counter_capacity,
                "life_count": scene.life_count,
            },
        )

    def _process_pending_transition(
        self,
        current_scene: SceneSnapshot,
        latest_frame: Any,
        *,
        cross_level: bool = False,
    ) -> TransitionReport | None:
        pending = self.memory.level.pending_action
        if pending is None or id(latest_frame) == pending.source_frame_id:
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

    def _record_transition(
        self,
        pending: PendingAction,
        report: TransitionReport,
        current_scene: SceneSnapshot,
        *,
        cross_level: bool = False,
    ) -> None:
        level = self.memory.level
        before_state = pending.scene_before.state_hash
        after_state = current_scene.state_hash
        action_key = pending.action_key()

        # A cross-level transition compares the previous level's final scene with
        # the next level's initial scene; that diff is a whole-screen artifact and
        # must not pollute the persistent game-level action_knowledge (attempts,
        # successes, movement vectors, effect scores) nor the soon-discarded
        # level-state maps. We still record the event below for success causality.
        if not cross_level:
            attempts = level.action_attempt_counts_by_state.setdefault(before_state, {})
            attempts[action_key] = attempts.get(action_key, 0) + 1

            knowledge = self.memory.game.action_knowledge.setdefault(
                pending.name,
                ActionKnowledge(),
            )
            knowledge.attempts += 1
            if report.effective_noop:
                knowledge.blocked += 1
                level.noop_actions_by_state.setdefault(before_state, set()).add(action_key)
            else:
                knowledge.successes += 1
                level.transition_graph.setdefault(before_state, {})[action_key] = after_state
            self._record_action_effect(knowledge, report)

            if report.controlled_candidate_id:
                self._update_actor_identity(pending, report, current_scene, knowledge)
            elif knowledge.attempts:
                knowledge.status = "tentative"
            knowledge.last_evidence = report.summary[:320]

        if report.counter_delta is not None:
            if report.counter_delta < 0:
                level.counter_cost_samples.append(report.counter_delta)
            elif report.counter_delta > 1 and not report.retry_detected:
                transition = {
                    "state": before_state,
                    "action": action_key,
                    "gain": report.counter_delta,
                    "after": after_state,
                }
                if not any(
                    item["state"] == before_state and item["action"] == action_key
                    for item in level.counter_refill_transitions
                ):
                    level.counter_refill_transitions.append(transition)
                    level.counter_refill_transitions = level.counter_refill_transitions[-12:]

        self._update_object_interactions(pending, report, current_scene)

        level.recent_events.append(
            f"{action_key}[{pending.source}] -> {report.summary}"
        )
        level.recent_action_keys.append(action_key)
        level.recent_state_hashes.append(after_state)

        self._update_mechanism_state(pending, report, current_scene)

        predicate_check = self._check_expected_predicates(pending.expected_predicates, report)
        event = EventRecord(
            event_id=self.memory.next_event_id,
            level_index=level.level_index,
            attempt_index=level.attempt_index,
            life_index=level.life_index,
            step_index=level.total_action_count,
            action_name=pending.name,
            action_key=action_key,
            action_params={"x": pending.x, "y": pending.y} if pending.name == "ACTION6" else {},
            source=pending.source,
            before_state_hash=pending.scene_before.state_hash,
            after_state_hash=current_scene.state_hash,
            before_full_hash=pending.scene_before.full_state_hash,
            after_full_hash=current_scene.full_state_hash,
            before_summary=pending.scene_before.summary[:1200],
            after_summary=current_scene.summary[:1200],
            transition_summary=report.summary[:1200],
            transition_delta=self._transition_delta_dict(report),
            available_actions_before=pending.available_actions_before,
            available_actions_after=[],
            expected_predicates=pending.expected_predicates,
            predicate_check=predicate_check,
            outcome=self._classify_outcome(report, current_scene),
            vlm_mode=pending.vlm_mode,
        )
        self._append_event(event)
        effect_flags = self._transition_effect_flags(before_state, after_state, report)
        self._update_fallback_transition_record(pending, report, current_scene, effect_flags)
        self._record_anti_loop_escape_outcome(pending, report, current_scene, effect_flags, event)
        self._update_action6_memory_from_event(pending, report, current_scene, event)
        self._detect_action6_nonprogress_loop(pending, report, current_scene, event)
        self._update_failure_model_from_event(pending, report, current_scene, event)
        self._update_active_plan_after_transition(pending, report, event)
        self._detect_nonprogress_primitive_loop(pending, report, current_scene, event)
        if pending.source == "level_theory" and (report.effective_noop or report.retry_detected or predicate_check.get("failed_count", 0) > 0):
            self._invalidate_or_downgrade_active_theory("level_theory_action_failed", event.event_id)
        self._detect_action6_drift_loop()
        if predicate_check.get("failed_count", 0) > 0 and pending.source in {"vlm_plan", "symbolic_nav", "active_plan"}:
            self._abort_plan("checkpoint_mismatch")

        if report.retry_detected:
            self._record_route_outcome(pending, report, current_scene)
            self._record_internal_retry(current_scene, report)
            return

        if pending.source in {"vlm_plan", "symbolic_nav"} and report.effective_noop:
            self._record_target_failure(
                pending.target_object_id,
                "planned movement was blocked or had no physical effect",
            )
            self._abort_plan("planned_action_was_noop")
        else:
            self._finish_waypoint_if_reached(
                pending.target_object_id,
                current_scene,
            )
            if report.interaction_event:
                level.force_vlm_reason = "verified_object_or_state_transformation"
                level.stage = V1Phase.OBJECT_EXPLORATION
            if pending.source in {"vlm_plan", "symbolic_nav", "active_plan"} and level.active_plan is None:
                level.active_strategy_rule_id = ""

        self._detect_oscillation()

    def _update_action6_memory_from_event(
        self,
        pending: PendingAction,
        report: TransitionReport,
        scene: SceneSnapshot,
        event: EventRecord,
    ) -> None:
        if pending.name != "ACTION6" or pending.x is None or pending.y is None:
            return
        before = pending.scene_before
        obj = self._object_at_point(before, pending.x, pending.y)
        outcome = "noop" if report.effective_noop else event.outcome
        candidate_kind = "unknown"
        for pred in pending.expected_predicates:
            if isinstance(pred, dict) and pred.get("candidate_kind"):
                candidate_kind = _short_string(pred.get("candidate_kind"), 60)
                break
        rec = Action6ProbeRecord(
            state_hash=before.state_hash,
            abstract_state_hash=before.state_hash,
            x=pending.x,
            y=pending.y,
            target_object_id=obj.track_id if obj else pending.target_object_id,
            target_type_key=obj.type_key if obj else "",
            candidate_kind=candidate_kind,
            local_patch_signature=self._local_patch_signature(before, pending.x, pending.y),
            outcome=outcome,
            event_id=event.event_id,
            cooldown_until_step=self._controller_step() + (self.config.action6_cooldown_steps if outcome == "noop" else 3),
        )
        self.memory.level.action6_memory.remember(rec)

    def _update_failure_model_from_event(
        self,
        pending: PendingAction,
        report: TransitionReport,
        scene: SceneSnapshot,
        event: EventRecord,
    ) -> None:
        if not (report.retry_detected or event.outcome in {"internal_retry", "game_over"}):
            return
        recent = list(self.memory.level.action_trace)[-8:]
        self.memory.game.failure_model.add_suffix(
            recent,
            report.summary[:300],
            self.memory.level.total_action_count,
            self.memory.level.level_index,
            self.memory.level.attempt_index,
        )
        self.logger.log_event(
            "failure_model_updated",
            {"reason": report.summary[:300], "suffix": recent, "event_id": event.event_id},
        )

    def _detect_action6_drift_loop(self) -> None:
        records = [rec for rec in self.memory.level.action6_memory.records[-8:] if rec.outcome == "noop"]
        if len(records) < 5:
            return
        same_type = len({rec.target_type_key for rec in records}) <= 2
        same_patch = len({rec.local_patch_signature for rec in records}) <= 2
        xs = [rec.x for rec in records]
        ys = [rec.y for rec in records]
        monotone_x = xs == sorted(xs) or xs == sorted(xs, reverse=True)
        monotone_y = ys == sorted(ys) or ys == sorted(ys, reverse=True)
        if same_type and same_patch and (monotone_x or monotone_y):
            level = self.memory.level
            self._abort_plan("action6_coordinate_drift_loop")
            level.recovery_mode = "escape_from_action6_loop"
            level.recovery_reason = "action6_drift"
            level.recovery_until_step = self._controller_step() + 8
            self._disable_action_family("ACTION6", 8, "action6_drift_loop")
            if records:
                last_key = self._action6_key_from_xy(records[-1].x, records[-1].y)
                level.blocked_state_action_pairs.add((records[-1].state_hash, last_key))
            self.memory.level.recent_events.append("action6_coordinate_drift_loop_detected")
            self.logger.log_event("loop_detected", {"kind": "action6_drift", "points": [(rec.x, rec.y) for rec in records]})

    def _record_action_effect(self, knowledge: ActionKnowledge, report: TransitionReport) -> None:
        structural = (
            len(report.transformed_objects)
            + len(report.appeared_object_ids)
            + len(report.disappeared_object_ids)
        )
        if structural or report.interaction_event:
            knowledge.structural_successes += 1
        if report.world_changed_cell_count >= 20 and not report.is_simple_translation:
            knowledge.global_changes += 1
        if report.retry_detected:
            knowledge.retry_failures += 1
        score = 0.0
        if report.effective_noop:
            score -= 2.0
        else:
            score += min(2.5, report.world_changed_cell_count / 35.0)
        if report.is_simple_translation:
            score += 0.35
        if report.interaction_event:
            score += 2.0
        if structural:
            score += min(2.5, 0.8 * structural)
        if report.retry_detected:
            score -= 4.0
        knowledge.recent_effect_scores.append(round(score, 3))

    def _learn_walkable_colour(
        self,
        before: SceneSnapshot,
        after: SceneSnapshot,
        object_id: str,
    ) -> None:
        old_actor = before.object_by_id(object_id)
        new_actor = after.object_by_id(object_id)
        if old_actor is None or new_actor is None or old_actor.bbox == new_actor.bbox:
            return
        counts: Counter[int] = Counter()
        nx0, ny0, nx1, ny1 = new_actor.bbox
        for y in range(old_actor.bbox[1], old_actor.bbox[3] + 1):
            for x in range(old_actor.bbox[0], old_actor.bbox[2] + 1):
                if nx0 <= x <= nx1 and ny0 <= y <= ny1:
                    continue
                if (x, y) in after.volatile_cells:
                    continue
                color = after.grid[y][x]
                if color != after.background_candidate:
                    counts[color] += 1
        if counts:
            color, count = counts.most_common(1)[0]
            self.memory.level.walkable_color_votes[color] = (
                self.memory.level.walkable_color_votes.get(color, 0) + count
            )

    def _update_object_interactions(
        self,
        pending: PendingAction,
        report: TransitionReport,
        scene: SceneSnapshot,
    ) -> None:
        level = self.memory.level
        actor = scene.object_by_id(level.controlled_object_id)
        if actor is None:
            return
        overlapped: list[str] = []
        for obj in scene.objects:
            if obj.track_id == actor.track_id or obj.near_edge:
                continue
            if self._object_reached(actor, obj):
                overlapped.append(obj.track_id)
                level.object_visit_counts[obj.track_id] = (
                    level.object_visit_counts.get(obj.track_id, 0) + 1
                )
        if report.interaction_event:
            likely_sources = overlapped or (
                [pending.target_object_id] if pending.target_object_id else []
            )
            for object_id in likely_sources:
                if object_id:
                    level.object_effect_counts[object_id] = (
                        level.object_effect_counts.get(object_id, 0) + 1
                    )
                    belief = level.object_beliefs.setdefault(object_id, {})
                    belief["observed_effect"] = report.summary[:300]
                    belief["status"] = "supported"
                    belief["confidence"] = max(
                        0.72,
                        float(belief.get("confidence", 0.0)),
                    )

    def _record_internal_retry(
        self,
        scene: SceneSnapshot,
        report: TransitionReport,
    ) -> None:
        level = self.memory.level
        failure = {
            "attempt": level.attempt_index,
            "life": level.life_index,
            "steps": level.life_action_count,
            "last_actions": list(level.action_trace)[-28:],
            "counter_after": scene.counter_value,
            "lives_after": scene.life_count,
            "transition": report.summary,
            "reason": "counter reset with a lost life; automatic retry detected",
        }
        level.failed_lives.append(failure)
        level.life_index += 1
        level.life_action_count = 0
        level.active_plan = None
        level.candidate_plans.clear()
        level.plan_goal = ""
        level.plan_target_id = ""
        level.active_strategy_rule_id = ""
        level.active_near_success_route_id = ""
        level.recent_state_hashes.clear()
        level.recent_state_hashes.append(scene.state_hash)
        level.recent_action_keys.clear()
        level.force_vlm_reason = "internal_retry_counter_exhausted"
        level.stage = V1Phase.FAILURE_RECOVERY
        level.actions_since_vlm = max(
            level.actions_since_vlm,
            self.config.vlm_min_action_gap,
        )
        self.logger.log_event("internal_retry", failure)

    def _finish_waypoint_if_reached(
        self,
        target_object_id: str,
        scene: SceneSnapshot,
    ) -> None:
        if not target_object_id:
            return
        actor = scene.object_by_id(self.memory.level.controlled_object_id)
        target = scene.object_by_id(target_object_id)
        if actor is None or target is None or not self._object_reached(actor, target):
            return
        level = self.memory.level
        plan = level.active_plan
        if plan is None or plan.status != "active":
            return
        step = plan.current_step()
        if step is None:
            return
        intent = step.intent
        nav_target = intent.target_object_id
        if intent.intent_type != IntentType.NAVIGATE_TO_OBJECT.value and intent.action_name != "NAVIGATE":
            return
        if nav_target and nav_target == target_object_id:
            step.status = "done"
            plan.cursor += 1
        level.recent_events.append(f"waypoint_reached={target_object_id}")

    def _record_target_failure(self, target_object_id: str, reason: str) -> None:
        if not target_object_id:
            return
        level = self.memory.level
        reason = _short_string(reason, 240)
        level.target_failure_counts[target_object_id] = (
            level.target_failure_counts.get(target_object_id, 0) + 1
        )
        signature = level.recent_state_hashes[-1][:12] if level.recent_state_hashes else ""
        for record in level.target_failure_records:
            if (
                record.target_id == target_object_id
                and record.reason == reason
                and record.life == level.life_index
                and record.attempt == level.attempt_index
                and record.state_signature == signature
            ):
                record.count += 1
                break
        else:
            level.target_failure_records.append(
                TargetFailureRecord(
                    target_id=target_object_id,
                    reason=reason,
                    life=level.life_index,
                    attempt=level.attempt_index,
                    state_signature=signature,
                )
            )
            level.target_failure_records = level.target_failure_records[-24:]
        belief = level.object_beliefs.setdefault(target_object_id, {})
        belief["contradictions"] = int(belief.get("contradictions", 0)) + 1
        belief["confidence"] = max(0.0, float(belief.get("confidence", 0.5)) - 0.18)
        belief["last_contradiction"] = reason

    def _detect_oscillation(self) -> None:
        level = self.memory.level
        states = list(level.recent_state_hashes)
        actions = list(level.recent_action_keys)
        if len(states) >= 6 and states[-1] == states[-3] == states[-5]:
            if actions:
                level.blocked_state_action_pairs.add((states[-1], actions[-1]))
            self._abort_plan("repeated_two_state_cycle")
            level.force_vlm_reason = "repeated_two_state_cycle"
            level.recovery_mode = "escape_from_primitive_loop"
            level.recovery_reason = "two_state_cycle"
            level.recovery_until_step = self._controller_step() + 6
            level.recent_events.append("two_state_cycle_detected")
            self.logger.log_event("loop_detected", {"kind": "two_state", "actions": actions[-8:], "states": [state[:12] for state in states[-8:]]})
            return
        if len(actions) >= 6:
            bases = [action.split(":", 1)[0] for action in actions[-6:]]
            oscillations = [
                ["ACTION1", "ACTION2"] * 3,
                ["ACTION2", "ACTION1"] * 3,
                ["ACTION3", "ACTION4"] * 3,
                ["ACTION4", "ACTION3"] * 3,
            ]
            if bases in oscillations:
                if actions and states:
                    level.blocked_state_action_pairs.add((states[-1], actions[-1]))
                self._abort_plan("opposite_action_oscillation")
                level.recovery_mode = "escape_from_primitive_loop"
                level.recovery_reason = "opposite_action_oscillation"
                level.recovery_until_step = self._controller_step() + 6
                level.recent_events.append("opposite_action_oscillation_detected")
                self.logger.log_event("loop_detected", {"kind": "opposite_actions", "actions": actions[-8:]})

    def _abort_plan(self, reason: str) -> None:
        level = self.memory.level
        abort_target = level.plan_target_id
        if not abort_target and level.active_plan is not None:
            step = level.active_plan.current_step()
            if step is not None:
                abort_target = _short_string(step.intent.target_object_id, 24).upper()
        if abort_target and reason in {
            "repeated_two_state_cycle",
            "opposite_action_oscillation",
            "action6_coordinate_drift_loop",
            "planned_action_was_noop",
        }:
            self._record_target_failure(abort_target, reason)
        self._record_strategy_rule_failure(reason)
        if level.plan_goal == "resume_near_success_route":
            self._record_near_success_route_failure(reason)
        if level.active_plan is not None:
            level.active_plan.status = "failed"
            level.active_plan.failures += 1
            self.logger.log_event("active_plan_failed", {**level.active_plan.as_prompt(), "reason": reason[:240]})
            self.logger.log_event(
                "plan_aborted",
                {
                    "reason": reason,
                    "level": level.level_index,
                    "attempt": level.attempt_index,
                    "life": level.life_index,
                },
            )
            level.active_plan = None
        level.plan_goal = ""
        level.plan_target_id = ""
        level.active_strategy_rule_id = ""
        level.active_near_success_route_id = ""
        level.force_vlm_reason = reason
        level.stage = V1Phase.FAILURE_RECOVERY

    def _success_causal_slice(self, max_events: int = 30) -> list[EventRecord]:
        events = [event for event in self.memory.event_log if event.level_index == self.memory.level.level_index]
        if not events:
            return []
        interesting: list[EventRecord] = []
        last_action = self.memory.level.action_trace[-1] if self.memory.level.action_trace else ""
        for event in events[-max_events:]:
            delta = event.transition_delta
            if (
                event.outcome in {"transform", "interaction", "resource_delta", "movement"}
                or delta.get("transformed_objects")
                or delta.get("appeared_object_ids")
                or delta.get("disappeared_object_ids")
                or delta.get("counter_delta") not in (None, 0)
                or event.action_key == last_action
            ):
                interesting.append(event)
        return interesting[-max_events:]

    def _summarize_scene_diff(self, before: SceneSnapshot | None, after: SceneSnapshot | None) -> dict[str, Any]:
        if before is None or after is None:
            return {}
        report = self.observer.compare(before, after)
        delta = self._transition_delta_dict(report)
        delta["summary"] = report.summary[:700]
        return delta

    def _record_level_outcome_memory(
        self, scene: SceneSnapshot, outcome: dict[str, Any], new_levels_completed: int
    ) -> LevelOutcomeMemory | None:
        level = self.memory.level
        initial = level.initial_scene_ref
        pending = level.pending_action or level.last_resolved_pending_action
        pre_success = pending.scene_before if pending is not None else None
        if initial is None:
            return None
        success_action = pending.action_key() if pending else (level.action_trace[-1] if level.action_trace else "unknown")
        causal = self._success_causal_slice()
        mem = LevelOutcomeMemory(
            level_index=level.level_index,
            initial_state_hash=initial.state_hash,
            pre_success_state_hash=pre_success.state_hash if pre_success else "",
            post_success_state_hash=scene.state_hash,
            post_success_is_next_level_start=new_levels_completed > level.levels_completed_at_start,
            success_action_key=success_action,
            action_trace=list(level.action_trace)[-100:],
            causal_event_ids=[event.event_id for event in causal],
            initial_summary=initial.summary,
            pre_success_summary=pre_success.summary if pre_success else "",
            post_success_summary=scene.summary,
            start_to_pre_success_diff=self._summarize_scene_diff(initial, pre_success),
            success_transition_summary=causal[-1].transition_summary if causal else "",
            confidence=0.5,
        )
        self.memory.game.level_outcomes.append(mem)
        self.memory.game.level_outcomes = self.memory.game.level_outcomes[-8:]
        self.logger.log_event("level_outcome_recorded", mem.as_prompt())
        return mem

    def _make_success_comparison_image(
        self, initial: SceneSnapshot | None, pre_success: SceneSnapshot | None, post_success: SceneSnapshot | None
    ) -> Image.Image | None:
        scenes = [initial, pre_success, post_success]
        if not any(scene is not None for scene in scenes):
            return None
        images: list[Image.Image] = []
        for scene in scenes:
            if scene is None:
                images.append(Image.new("RGB", (self.config.image_size, self.config.image_size), (0, 0, 0)))
                continue
            image = scene.rgb or render_grid(scene.grid, self.config.image_size)
            images.append(image.convert("RGB").resize((self.config.image_size, self.config.image_size), Image.Resampling.NEAREST))
        combined = Image.new("RGB", (self.config.image_size * 3, self.config.image_size), (0, 0, 0))
        for idx, image in enumerate(images):
            combined.paste(image, (idx * self.config.image_size, 0))
        draw = ImageDraw.Draw(combined)
        for idx, label in enumerate(["INITIAL_RAW", "PRE_SUCCESS_RAW", "POST_ACTION_RAW"]):
            draw.rectangle((idx * self.config.image_size, 0, idx * self.config.image_size + 170, 22), fill=(0, 0, 0))
            draw.text((idx * self.config.image_size + 4, 4), label, fill=(255, 255, 255))
        self.logger.log_event("success_comparison_prompt_built", {"level": self.memory.level.level_index})
        return combined

    def _request_vlm_success_consolidation(
        self,
        scene: SceneSnapshot,
        outcome: dict[str, Any],
        causal: list[EventRecord],
        pre_success_scene: SceneSnapshot | None = None,
    ) -> VLMResult | None:
        if not (self.config.enable_vlm and self.config.enable_success_consolidation_vlm and getattr(self.backend, "available", False)):
            return None
        level = self.memory.level
        if level.vlm_calls_this_level >= self.config.max_vlm_calls_per_level:
            return None
        prompt_scene = pre_success_scene or scene
        prompt = self._build_vlm_prompt(prompt_scene, None, (), VLMMode.SUCCESS_CONSOLIDATION.value)
        try:
            payload = json.loads(prompt)
        except Exception:
            payload = {"base_prompt": prompt}
        pending = level.pending_action or level.last_resolved_pending_action
        initial = level.initial_scene_ref
        pre_success = pre_success_scene or (pending.scene_before if pending is not None else None)
        comparison = self._make_success_comparison_image(initial, pre_success, scene)
        outcome_mem = self.memory.game.level_outcomes[-1].as_prompt() if self.memory.game.level_outcomes else {}
        payload["success_outcome"] = outcome
        payload["level_outcome_memory"] = outcome_mem
        payload["causal_slice"] = [_json_safe(event) for event in causal]
        payload["mode_instructions"] = [
            "The level was completed. Infer the reusable success condition.",
            "Image order for success consolidation: 1 INITIAL_RAW first frame, 2 PRE_SUCCESS_RAW frame immediately before the completing action, 3 COMPARISON_OR_POST may include post-action/next-level first frame.",
            "Do not infer the previous level's final board solely from the post-action frame if levels_completed already advanced.",
            "Separate game_goal_schema from level_goal_instantiation.",
            "Do not return an action unless needed as next-level hint.",
            "Every claim must cite event_id values from causal_slice when possible.",
        ]
        request = VLMRequest(
            text_prompt=json.dumps(payload, ensure_ascii=True, default=str),
            current_rgb=(pre_success.rgb if pre_success and pre_success.rgb else scene.rgb or render_grid(scene.grid, self.config.image_size)),
            previous_rgb=(initial.rgb if initial and initial.rgb else None),
            analysis_rgb=comparison or scene.annotated_rgb,
            max_new_tokens=self.config.vlm_max_new_tokens,
        )
        level.vlm_calls_this_level += 1
        level.last_vlm_action_count = level.total_action_count
        level.current_vlm_mode = VLMMode.SUCCESS_CONSOLIDATION.value
        self.memory.game.vlm_calls_total += 1
        self.memory.game.controller_stats.vlm_mode_counts[VLMMode.SUCCESS_CONSOLIDATION.value] += 1
        self.logger.log_event("vlm_call", {"mode": VLMMode.SUCCESS_CONSOLIDATION.value, "reason": "level_success", "level": level.level_index})
        try:
            raw = self.backend.decide(request)
        except Exception as exc:
            self.memory.game.total_vlm_errors += 1
            self.logger.log_event("vlm_exception", {"message": str(exc)[:300], "mode": VLMMode.SUCCESS_CONSOLIDATION.value})
            return None
        result = parse_vlm_result(raw)
        if result is not None:
            result.mode = result.mode or VLMMode.SUCCESS_CONSOLIDATION.value
        return result

    def _deterministic_success_schema_patch(
        self, scene: SceneSnapshot, outcome: dict[str, Any], causal: list[EventRecord]
    ) -> dict[str, Any]:
        last_action = outcome.get("win_action") or (self.memory.level.action_trace[-1] if self.memory.level.action_trace else "")
        has_frames = bool(scene.template_relations)
        has_transform = any(event.outcome == "transform" for event in causal)
        has_resource = any(event.outcome == "resource_delta" for event in causal)
        name = "generic_reach_or_interact_success"
        statement = "Reach or interact with the success target after satisfying local preconditions."
        role_slots = {"actor": {"must_be_controllable": True}, "target": {"visual_target": True}}
        predicates = [{"type": "terminal_level_advanced", "levels_completed": outcome.get("level", 0) + 1}]
        if has_frames:
            name = "frame_or_pattern_gate_success"
            statement = "Satisfy a framed or pattern relation, then reach or interact with the target."
            role_slots["pattern_frame"] = {"shape_label": "frame_with_inner_pattern"}
            predicates.append({"type": "pattern_or_frame_relation_satisfied"})
        if has_transform:
            role_slots["transformer"] = {"causes": "object_or_status_pattern_change"}
            predicates.append({"type": "required_transform_before_terminal"})
        if has_resource:
            role_slots["resource"] = {"affects": "counter_or_life"}
        required = list(dict.fromkeys(event.outcome for event in causal if event.outcome != "noop"))
        return {
            "name": name,
            "statement": statement,
            "role_slots": role_slots,
            "success_predicates": predicates,
            "trigger_action_patterns": [str(last_action).split(":", 1)[0]],
            "required_mechanics": required,
            "known_variations": [],
            "evidence_event_ids": [event.event_id for event in causal],
            "confidence": 0.55,
            "evidence_level": 2,
        }

    def _accept_goal_schema_patch(
        self, patch: dict[str, Any], causal: list[EventRecord], level_index: int
    ) -> GameGoalSchema | None:
        if not isinstance(patch, dict):
            return None
        name = _short_string(patch.get("name") or "unnamed_success_schema", 80)
        statement = _short_string(patch.get("statement") or patch.get("claim"), 500)
        if not statement:
            return None
        evidence_ids = [int(value) for value in patch.get("evidence_event_ids", []) if isinstance(value, int)]
        if not evidence_ids:
            evidence_ids = [event.event_id for event in causal]
        confidence = _clamp01(patch.get("confidence", 0.5))
        evidence_level = int(patch.get("evidence_level", 2 if evidence_ids else 1) or 1)
        schema_id = "schema:" + hashlib.sha1(_normalize_claim(statement).encode()).hexdigest()[:12]
        for schema in self.memory.game.goal_schemas:
            if schema.schema_id == schema_id or _claim_similarity(schema.statement, statement) >= 0.72:
                schema.source_levels = sorted(set(schema.source_levels + [level_index]))
                schema.evidence_event_ids = sorted(set(schema.evidence_event_ids + evidence_ids))[-40:]
                schema.confidence = round(max(schema.confidence, confidence), 3)
                schema.evidence_level = max(schema.evidence_level, evidence_level, 4 if len(set(schema.source_levels)) >= 2 else schema.evidence_level)
                if confidence >= schema.confidence:
                    schema.statement = statement
                if isinstance(patch.get("known_variations"), list):
                    schema.known_variations = list(dict.fromkeys(schema.known_variations + [_short_string(v, 160) for v in patch["known_variations"]]))[-12:]
                return schema
        schema = GameGoalSchema(
            schema_id=schema_id,
            name=name,
            statement=statement,
            role_slots=patch.get("role_slots") if isinstance(patch.get("role_slots"), dict) else {},
            success_predicates=patch.get("success_predicates") if isinstance(patch.get("success_predicates"), list) else [],
            trigger_action_patterns=patch.get("trigger_action_patterns") if isinstance(patch.get("trigger_action_patterns"), list) else [],
            required_mechanics=patch.get("required_mechanics") if isinstance(patch.get("required_mechanics"), list) else [],
            known_variations=patch.get("known_variations") if isinstance(patch.get("known_variations"), list) else [],
            source_levels=[level_index],
            evidence_event_ids=evidence_ids[-40:],
            confidence=round(confidence, 3),
            evidence_level=evidence_level,
        )
        self.memory.game.goal_schemas.append(schema)
        self.memory.game.goal_schemas = sorted(self.memory.game.goal_schemas, key=lambda item: (item.evidence_level, item.confidence), reverse=True)[:12]
        return schema

    def _consolidate_success(
        self,
        scene: SceneSnapshot,
        outcome: dict[str, Any],
        new_levels_completed: int,
        pre_success_scene: SceneSnapshot | None = None,
    ) -> None:
        level = self.memory.level
        causal = self._success_causal_slice()
        analysis_scene = pre_success_scene or scene
        patch: dict[str, Any] | None = None
        result = self._request_vlm_success_consolidation(scene, outcome, causal, pre_success_scene=pre_success_scene)
        if result is not None:
            patch = result.goal_schema_patch
            if not patch and isinstance(result.proposed_memory_patch, dict):
                maybe_patch = result.proposed_memory_patch.get("goal_schema")
                patch = maybe_patch if isinstance(maybe_patch, dict) else None
            self._apply_vlm_update(result, None, analysis_scene, ())
        if not patch:
            patch = self._deterministic_success_schema_patch(analysis_scene, outcome, causal)
        schema = self._accept_goal_schema_patch(patch, causal, level.level_index)
        if schema is not None:
            self.logger.log_event(
                "success_consolidated",
                {
                    "schema_id": schema.schema_id,
                    "name": schema.name,
                    "confidence": schema.confidence,
                    "evidence_level": schema.evidence_level,
                    "source_level": level.level_index,
                    "causal_event_ids": [event.event_id for event in causal],
                },
            )

    def _record_level_success(
        self,
        scene: SceneSnapshot,
        new_levels_completed: int,
    ) -> None:
        level = self.memory.level
        if not level.initial_scene_summary:
            return
        pending = level.pending_action or level.last_resolved_pending_action
        outcome = {
            "level": level.level_index,
            "attempt": level.attempt_index,
            "life": level.life_index,
            "steps": level.total_action_count,
            "goal": level.local_goal,
            "win_action": pending.action_key() if pending else (
                level.action_trace[-1] if level.action_trace else "unknown"
            ),
            "action_trace": list(level.action_trace)[-100:],
            "evidence": f"levels_completed advanced to {new_levels_completed}",
        }
        pre_success_scene = pending.scene_before if pending is not None else None
        analysis_scene = pre_success_scene or scene
        self.memory.game.successful_levels.append(outcome)
        self.memory.game.successful_levels = self.memory.game.successful_levels[-8:]
        self._record_level_outcome_memory(scene, outcome, new_levels_completed)
        self._consolidate_success(scene, outcome, new_levels_completed, pre_success_scene=pre_success_scene)
        self._extract_success_strategy(level, analysis_scene, outcome)
        if level.local_goal:
            self._merge_belief(
                self.memory.game.goal_hypotheses,
                claim=level.local_goal,
                confidence=0.96,
                evidence=outcome["evidence"],
                scope="game",
                verified=True,
            )
        level.pending_action = None
        level.last_resolved_pending_action = None
        self.logger.log_event("level_success", outcome)

    def _record_game_over(
        self,
        scene: SceneSnapshot,
        transition: TransitionReport | None,
    ) -> None:
        level = self.memory.level
        failure = {
            "attempt": level.attempt_index,
            "life": level.life_index,
            "total_steps": level.total_action_count,
            "counter": scene.counter_value,
            "capacity": scene.counter_capacity,
            "life_count": scene.life_count,
            "last_actions": list(level.action_trace)[-40:],
            "last_transition": transition.summary if transition else None,
            "reason": "GAME_OVER after all visible/hidden retries or another terminal failure",
        }
        level.failed_lives.append(failure)
        level.last_failure = _short_string(failure, 700)
        level.attempt_index += 1
        level.life_index = 0
        level.life_action_count = 0
        level.active_plan = None
        level.candidate_plans.clear()
        level.plan_goal = ""
        level.plan_target_id = ""
        level.active_strategy_rule_id = ""
        level.pending_action = None
        level.awaiting_reset = True
        level.recent_state_hashes.clear()
        level.recent_action_keys.clear()
        level.force_vlm_reason = "retry_after_game_over"
        level.stage = V1Phase.FAILURE_RECOVERY
        recent = list(level.action_trace)[-8:]
        self.memory.game.failure_model.add_suffix(
            recent,
            "GAME_OVER after retries or terminal failure",
            level.total_action_count,
            level.level_index,
            level.attempt_index,
        )
        self.logger.log_event("failure_model_updated", {"reason": "GAME_OVER", "suffix": recent})
        self.logger.log_event("attempt_failed", failure)

    def _on_reset_observed(self, scene: SceneSnapshot) -> None:
        level = self.memory.level
        level.awaiting_reset = False
        level.recent_state_hashes.clear()
        level.recent_state_hashes.append(scene.state_hash)
        # The pre-reset action history is stale after a restart; clearing it also
        # lets the ACTION6 throttle window decay so a stuck window can never
        # freeze the controller into a repeated terminal-reset loop.
        level.recent_action_keys.clear()
        level.force_vlm_reason = "retry_after_game_over"
        level.recent_events.append(
            f"reset_observed attempt={level.attempt_index} state={scene.state_hash[:12]}"
        )

    def _remember_current_state(self, scene: SceneSnapshot) -> None:
        level = self.memory.level
        if not level.recent_state_hashes or level.recent_state_hashes[-1] != scene.state_hash:
            level.recent_state_hashes.append(scene.state_hash)

    def _object_descriptor(self, obj: ObjectObservation | None, role_hint: str = "") -> dict[str, Any]:
        if obj is None:
            return {}
        return {
            "type_key": obj.type_key,
            "shape_label": obj.shape_label,
            "frame_color": obj.frame_color,
            "inner_pattern": obj.inner_pattern[:160],
            "near_edge": obj.near_edge,
            "role_hint": role_hint,
        }

    def _descriptor_matches(self, descriptor: dict[str, Any], obj: ObjectObservation) -> bool:
        if not descriptor:
            return False
        score = 0
        if descriptor.get("type_key") and descriptor.get("type_key") == obj.type_key:
            score += 3
        if descriptor.get("shape_label") and descriptor.get("shape_label") == obj.shape_label:
            score += 2
        if descriptor.get("frame_color") is not None and descriptor.get("frame_color") == obj.frame_color:
            score += 1
        if descriptor.get("near_edge") == obj.near_edge:
            score += 1
        expected_inner = _short_string(descriptor.get("inner_pattern"), 160)
        if expected_inner and obj.inner_pattern.startswith(expected_inner[:48]):
            score += 2
        return score >= 4

    def _update_mechanism_from_scene(self, scene: SceneSnapshot) -> None:
        mechanism = self.memory.level.mechanism
        if scene.counter_value is not None or scene.life_count is not None:
            mechanism.note("resource", 1.2, "visible counter/lives")
        if scene.template_relations:
            mechanism.note("frame_gate", 0.9, "framed template relations visible")
        old_mode = mechanism.mode
        mechanism.mode = mechanism.choose_mode()
        if mechanism.mode != old_mode:
            mechanism.last_switch_step = self.memory.level.total_action_count
            self.logger.log_event("strategy_switch", mechanism.summary())

    def _update_mechanism_state(
        self,
        pending: PendingAction,
        report: TransitionReport,
        scene: SceneSnapshot,
    ) -> None:
        level = self.memory.level
        mechanism = level.mechanism
        if pending.name == "ACTION6":
            if report.effective_noop:
                mechanism.note("click_selection", -0.15, "ACTION6 produced no visible effect")
            else:
                mechanism.note("click_selection", 1.6, "ACTION6 changed the scene")
        if report.controlled_candidate_id and report.is_simple_translation:
            mechanism.note("direct_navigation", 1.2, f"controlled candidate {report.controlled_candidate_id} translated")
        if max(level.actor_votes.values(), default=0) >= 2:
            mechanism.note("direct_navigation", 0.8, "stable actor votes")
        moved_count = len(report.moved_objects)
        structural_events = len(report.transformed_objects) + len(report.appeared_object_ids) + len(report.disappeared_object_ids)
        if (
            report.world_changed_cell_count >= 20
            and (moved_count >= 2 or structural_events >= 1)
            and not report.is_simple_translation
        ):
            mechanism.note("global_transform", 1.5, "large non-translation world change")
        if not level.controlled_object_id and moved_count >= 2:
            mechanism.note("global_transform", 0.8, "multiple objects moved while actor unknown")
        if scene.template_relations:
            mismatched = any(
                relation.get("edge_vs_world") and not relation.get("exact_inner_match")
                for relation in scene.template_relations
            )
            if mismatched:
                mechanism.note("frame_gate", 1.0, "frame inner pattern mismatch")
        if report.counter_delta is not None or report.life_delta is not None or report.retry_detected:
            mechanism.note("resource", 1.0 if not report.retry_detected else 2.0, "counter/life changed")
        old_mode = mechanism.mode
        mechanism.mode = mechanism.choose_mode()
        self.logger.log_event(
            "mechanism_update",
            {
                **mechanism.summary(),
                "action": pending.action_key(),
                "level": level.level_index,
                "attempt": level.attempt_index,
                "life": level.life_index,
            },
        )
        if mechanism.mode != old_mode:
            mechanism.last_switch_step = level.total_action_count
            self.logger.log_event(
                "strategy_switch",
                {
                    "from": old_mode,
                    "to": mechanism.mode,
                    "level": level.level_index,
                    "attempt": level.attempt_index,
                    "life": level.life_index,
                    "scores": dict(mechanism.scores),
                },
            )

    def _update_actor_identity(
        self,
        pending: PendingAction,
        report: TransitionReport,
        current_scene: SceneSnapshot,
        knowledge: ActionKnowledge,
    ) -> None:
        level = self.memory.level
        candidate = report.controlled_candidate_id or ""
        if not candidate:
            return
        level.actor_votes[candidate] = level.actor_votes.get(candidate, 0) + 1
        knowledge.controlled_object_votes[candidate] = (
            knowledge.controlled_object_votes.get(candidate, 0) + 1
        )
        move = next((item for item in report.moved_objects if item["object_id"] == candidate), None)
        if move is not None:
            vector_key = f"{int(move['dx'])},{int(move['dy'])}"
            knowledge.movement_vectors[vector_key] = knowledge.movement_vectors.get(vector_key, 0) + 1
            self._learn_walkable_colour(pending.scene_before, current_scene, candidate)
        best_actor, votes = max(level.actor_votes.items(), key=lambda item: item[1])
        unique_simple_translation = (
            report.is_simple_translation
            and len(report.moved_objects) == 1
            and not report.transformed_objects
            and not report.appeared_object_ids
            and not report.disappeared_object_ids
        )
        if votes >= 2 or unique_simple_translation:
            actor = current_scene.object_by_id(best_actor)
            level.controlled_object_id = best_actor
            level.controlled_actor_confidence = 1.0 if votes >= 2 else 0.72
            level.tentative_controlled_object_id = ""
            if actor is not None:
                level.controlled_object_type_key = actor.type_key
        else:
            level.tentative_controlled_object_id = best_actor
            level.controlled_actor_confidence = max(level.controlled_actor_confidence, 0.35)
        knowledge.status = "verified" if max(knowledge.movement_vectors.values(), default=0) >= 2 else "tentative"

    def _record_route_outcome(
        self,
        pending: PendingAction,
        report: TransitionReport,
        scene: SceneSnapshot,
    ) -> None:
        level = self.memory.level
        trace = list(level.action_trace)
        if len(trace) < 2:
            return
        target_id = pending.target_object_id or level.plan_target_id
        route = NearSuccessRoute(
            target_id=target_id,
            safe_prefix=trace[:-1][-28:],
            last_bad_action=trace[-1],
            counter_before=pending.scene_before.counter_value,
            counter_after=scene.counter_value,
            lives_after=scene.life_count,
            state_before_last=pending.scene_before.state_hash,
            life=level.life_index,
            attempt=level.attempt_index,
            progress=report.summary[:240],
        )
        route.route_id = route.stable_id()
        if level.plan_goal == "resume_near_success_route":
            level.failed_route_signatures.add(route.failure_signature())
            self._record_near_success_route_failure("retry_detected_after_resume")
            self.logger.log_event(
                "near_success_route_not_saved",
                {
                    **route.as_prompt(),
                    "level": level.level_index,
                    "reason": "resume_route_failed",
                },
            )
            return
        if level.active_strategy_rule_id:
            self._record_strategy_rule_failure("retry_detected_after_strategy_rule")
        level.near_success_routes.append(route)
        self.logger.log_event(
            "near_success_route_saved",
            {
                **route.as_prompt(),
                "level": level.level_index,
                "reason": "retry_detected",
            },
        )

    def _resume_near_success_route(self, scene: SceneSnapshot, legal: Sequence[Any]) -> bool:
        level = self.memory.level
        if not level.near_success_routes:
            return False
        legal_by_name = {action_name(action): action for action in legal}
        legal_names = set(legal_by_name)
        for route in reversed(level.near_success_routes):
            route.route_id = route.stable_id()
            if route.failures >= 2:
                level.recent_events.append(f"near_success_route_skipped_failed={route.route_id}")
                continue
            if route.cooldown_until > level.total_action_count:
                level.recent_events.append(f"near_success_route_skipped_cooldown={route.route_id}")
                continue
            if route.last_resumed_life == level.life_index and route.last_resumed_attempt == level.attempt_index:
                level.recent_events.append(f"near_success_route_skipped_same_life={route.route_id}")
                continue
            if route.state_before_last and route.state_before_last == scene.state_hash:
                continue
            signature = route.failure_signature()
            legacy_signature = f"{route.target_id}:{route.last_bad_action}:{route.state_before_last[:12]}"
            if signature in level.failed_route_signatures or legacy_signature in level.failed_route_signatures:
                level.recent_events.append(f"near_success_route_skipped_signature={signature[:40]}")
                continue
            if (
                scene.counter_value is not None
                and route.counter_before is not None
                and scene.counter_value < max(1, min(route.counter_before, len(route.safe_prefix)))
            ):
                level.recent_events.append("near_success_route_skipped_low_counter")
                continue
            bad_base = route.last_bad_action.split(":", 1)[0]
            safe: list[str] = []
            for raw_name in route.safe_prefix[-self.config.plan_horizon :]:
                name = _short_string(raw_name, 40).upper().split(":", 1)[0]
                if name == "RESET" or name == bad_base or name not in legal_names:
                    continue
                if self._is_known_noop_action(legal_by_name[name], scene):
                    continue
                safe.append(name)
            if not safe:
                route.failures += 1
                route.cooldown_until = level.total_action_count + 12
                level.recent_events.append(f"near_success_route_skipped_no_safe_steps={route.route_id}")
                continue
            route.uses += 1
            route.last_resumed_state = scene.state_hash
            route.last_resumed_life = level.life_index
            route.last_resumed_attempt = level.attempt_index
            route.cooldown_until = level.total_action_count + max(8, min(24, len(safe) + 4))
            steps = [
                {
                    "action": name,
                    "target_object_id": route.target_id,
                    "purpose": "resume near-success route without repeating the terminal action",
                    "expected_change": "make progress while avoiding the last retry-causing suffix",
                }
                for name in safe
            ]
            level.active_strategy_rule_id = ""
            level.active_near_success_route_id = route.route_id
            started = self._start_deterministic_active_plan(
                "near_success_route",
                "resume_near_success_route",
                route.target_id,
                steps,
                scene,
                legal,
                log_event="near_success_route_resumed",
                log_payload={
                    **route.as_prompt(),
                    "level": level.level_index,
                    "steps": safe,
                },
            )
            if not started:
                route.failures += 1
                route.cooldown_until = level.total_action_count + 12
                level.recent_events.append(f"near_success_route_skipped_preflight={route.route_id}")
                continue
            return True
        return False

    def _record_near_success_route_failure(self, reason: str) -> None:
        level = self.memory.level
        route_id = level.active_near_success_route_id
        if not route_id:
            return
        for route in level.near_success_routes:
            route.route_id = route.stable_id()
            if route.route_id != route_id:
                continue
            route.failures += 1
            route.cooldown_until = max(route.cooldown_until, level.total_action_count + 16)
            level.failed_route_signatures.add(route.failure_signature())
            self.logger.log_event(
                "near_success_route_failed",
                {
                    **route.as_prompt(),
                    "level": level.level_index,
                    "attempt": level.attempt_index,
                    "life": level.life_index,
                    "reason": reason,
                },
            )
            break

    def _extract_success_strategy(
        self,
        level: LevelMemory,
        scene: SceneSnapshot,
        outcome: dict[str, Any],
    ) -> None:
        trace = [name for name in outcome.get("action_trace", []) if isinstance(name, str) and name != "RESET"]
        if not trace:
            return
        frame_rule = bool(scene.template_relations) or "frame" in str(outcome.get("goal", "")).lower()
        kind = "frame_gate_interaction" if frame_rule else "success_macro"
        target = None
        if level.plan_target_id:
            target = scene.object_by_id(level.plan_target_id)
        if target is None:
            all_candidates = list(scene.objects)
            candidates = [obj for obj in all_candidates if not obj.near_edge] or all_candidates
            if frame_rule:
                framed = [obj for obj in candidates if obj.frame_color is not None]
                candidates = framed or candidates
            target = max(candidates, key=lambda obj: obj.salience, default=None)
        descriptor = self._object_descriptor(target, role_hint="success_target")
        steps = [
            {
                "name": name.split(":", 1)[0],
                "target_object_id": "",
                "purpose": "replay successful macro prefix",
                "expected_change": "follow a previously successful action pattern",
            }
            for name in trace[-self.config.plan_horizon :]
            if name.split(":", 1)[0].startswith("ACTION")
        ]
        if not steps:
            return
        rule_id = f"{kind}:{hashlib.sha1(json.dumps(descriptor, sort_keys=True, default=str).encode()).hexdigest()[:10]}"
        for rule in self.memory.game.strategy_rules:
            if rule.rule_id == rule_id:
                rule.support += 1
                rule.confidence = min(0.98, max(rule.confidence, 0.75) + 0.05)
                rule.steps = steps
                return
        self.memory.game.strategy_rules.append(
            ExecutableStrategyRule(
                rule_id=rule_id,
                kind=kind,
                trigger_features={
                    "mode": level.mechanism.mode,
                    "has_counter": scene.counter_value is not None,
                    "has_frames": bool(scene.template_relations),
                    "object_count": len(scene.objects),
                    "frame_relation_count": len(scene.template_relations),
                    "grid_size": [scene.width, scene.height],
                },
                preconditions={"legal_actions": sorted({step["name"] for step in steps})},
                policy="execute successful macro when scene descriptors match",
                steps=steps,
                target_descriptor=descriptor,
                actor_required=any(step["name"] == "NAVIGATE" for step in steps),
                confidence=0.82,
                source_level=level.level_index,
            )
        )
        self.memory.game.strategy_rules = self.memory.game.strategy_rules[-12:]

    def _seed_plan_from_strategy_rules(self, scene: SceneSnapshot, legal: Sequence[Any]) -> bool:
        level = self.memory.level
        legal_by_name = {action_name(action): action for action in legal}
        legal_names = set(legal_by_name)
        if level.total_action_count > 4:
            return False
        for rule in sorted(self.memory.game.strategy_rules, key=lambda item: (item.confidence, item.support), reverse=True):
            if rule.failures >= 2 or rule.confidence < 0.7:
                continue
            if rule.actor_required and not level.controlled_object_id and level.mechanism.mode not in {"click_selection", "global_transform"}:
                continue
            if not self._strategy_rule_scene_compatible(rule, scene):
                continue
            safe_steps = []
            for step in rule.steps:
                name = _short_string(step.get("name"), 40).upper()
                if name not in legal_names or name == "RESET":
                    continue
                if self._is_known_noop_action(legal_by_name[name], scene):
                    continue
                safe_steps.append(dict(step))
            if not safe_steps:
                rule.failures += 1
                rule.confidence = max(0.0, round(rule.confidence - 0.12, 3))
                self.logger.log_event(
                    "strategy_rule_skipped",
                    {
                        "rule_id": rule.rule_id,
                        "kind": rule.kind,
                        "reason": "no_safe_non_noop_steps",
                        "failures": rule.failures,
                        "confidence": rule.confidence,
                        "level": level.level_index,
                        "attempt": level.attempt_index,
                        "life": level.life_index,
                    },
                )
                continue
            level.active_strategy_rule_id = rule.rule_id
            return self._start_deterministic_active_plan(
                f"strategy_rule:{rule.kind}",
                f"strategy_rule:{rule.kind}:{rule.rule_id}",
                "",
                safe_steps[: self.config.plan_horizon],
                scene,
                legal,
                log_event="strategy_rule_applied",
                log_payload={
                    "rule_id": rule.rule_id,
                    "kind": rule.kind,
                    "level": level.level_index,
                    "attempt": level.attempt_index,
                    "life": level.life_index,
                    "steps": safe_steps[: self.config.plan_horizon],
                },
            )
        return False

    def _strategy_rule_scene_compatible(self, rule: ExecutableStrategyRule, scene: SceneSnapshot) -> bool:
        descriptor = rule.target_descriptor
        matches = [obj for obj in scene.objects if self._descriptor_matches(descriptor, obj)] if descriptor else []
        if descriptor and not matches:
            return False
        features = rule.trigger_features or {}
        if features.get("has_frames") and not scene.template_relations:
            return False
        if features.get("has_counter") and scene.counter_value is None:
            return False
        if rule.kind == "frame_gate_interaction":
            if not scene.template_relations:
                return False
            source_level = int(rule.source_level)
            if source_level != self.memory.level.level_index and rule.support < 2:
                object_count = int(features.get("object_count") or 0)
                frame_relations = int(features.get("frame_relation_count") or 0)
                if object_count and abs(len(scene.objects) - object_count) > 1:
                    return False
                if frame_relations and len(scene.template_relations) != frame_relations:
                    return False
            if descriptor and len(matches) > 3:
                return False
        return True

    def _record_strategy_rule_failure(self, reason: str) -> None:
        level = self.memory.level
        rule_id = level.active_strategy_rule_id
        if not rule_id:
            return
        for rule in self.memory.game.strategy_rules:
            if rule.rule_id != rule_id:
                continue
            rule.failures += 1
            rule.confidence = max(0.0, round(rule.confidence - 0.08, 3))
            self.logger.log_event(
                "strategy_rule_failed",
                {
                    "rule_id": rule_id,
                    "kind": rule.kind,
                    "reason": reason,
                    "failures": rule.failures,
                    "confidence": rule.confidence,
                    "level": level.level_index,
                    "attempt": level.attempt_index,
                    "life": level.life_index,
                },
            )
            break

    def _target_blocked(self, target_id: str, scene: SceneSnapshot | None = None) -> bool:
        if not target_id:
            return False
        level = self.memory.level
        current_life_failures = [
            item
            for item in level.target_failure_records
            if item.target_id == target_id
            and item.life == level.life_index
            and item.attempt == level.attempt_index
        ]
        if sum(item.count for item in current_life_failures) >= self.config.target_failure_limit:
            return True
        if scene is not None:
            signature = scene.state_hash[:12]
            repeated_here = [
                item
                for item in level.target_failure_records
                if item.target_id == target_id and item.state_signature == signature
            ]
            if sum(item.count for item in repeated_here) >= max(3, self.config.target_failure_limit + 1):
                return True
        old_count = level.target_failure_counts.get(target_id, 0)
        if old_count >= self.config.target_failure_limit:
            self.logger.log_event(
                "target_failure_softened",
                {
                    "target": target_id,
                    "count": old_count,
                    "life": level.life_index,
                    "attempt": level.attempt_index,
                    "level": level.level_index,
                },
            )
        return False

    def _repair_window_active(self) -> bool:
        level = self.memory.level
        return bool(
            level.recovery_mode
            and (not level.recovery_until_step or self._controller_step() <= level.recovery_until_step)
        )

    def _choose_vlm_mode(
        self,
        scene: SceneSnapshot,
        transition: TransitionReport | None,
        legal: Sequence[Any],
    ) -> VLMMode | None:
        level = self.memory.level
        if (
            not self.config.enable_vlm
            or not getattr(self.backend, "available", False)
            or level.vlm_calls_this_level >= self.config.max_vlm_calls_per_level
            or level.last_vlm_action_count == level.total_action_count
        ):
            return None
        reason = level.force_vlm_reason.strip()
        if (reason == "repair_decision" or level.recovery_mode) and self._repair_window_active():
            return VLMMode.REPAIR_DECISION
        if reason in {"retry_after_game_over", "internal_retry_counter_exhausted"}:
            return VLMMode.FAILURE_REPAIR
        if reason in {"planned_action_was_noop", "repeated_two_state_cycle", "checkpoint_mismatch", "action6_coordinate_drift_loop", "retry_invalid_vlm_output"}:
            return VLMMode.EXPERIMENT_DESIGN
        if reason == "initial_scene" or level.stage == V1Phase.INIT:
            return VLMMode.INIT_ANALYSIS
        if level.stage == V1Phase.TRANSFER_BOOTSTRAP and (self.memory.game.goal_schemas or self.memory.game.level_outcomes):
            return VLMMode.TRANSFER_INSTANTIATION
        if transition is not None and transition.interaction_event:
            return VLMMode.TRANSITION_EXPLANATION
        if level.stage in {V1Phase.PLAN_SYNTHESIS, V1Phase.GOAL_VALIDATION} and (
            level.local_goal or level.goal_instantiation
        ):
            return VLMMode.PLAN_SYNTHESIS
        if not level.active_plan and level.actions_since_vlm >= self.config.vlm_min_action_gap:
            if self._choose_semantic_target(scene) is None:
                remaining = self.config.max_vlm_calls_per_level - level.vlm_calls_this_level
                if level.life_index == 0 and remaining <= self.config.vlm_retry_reserve:
                    return None
                return VLMMode.EXPERIMENT_DESIGN
        return None

    def _request_vlm_once(
        self,
        scene: SceneSnapshot,
        transition: TransitionReport | None,
        legal: Sequence[Any],
        reason: str | VLMMode,
    ) -> VLMResult | None:
        level = self.memory.level
        mode_value = reason.value if isinstance(reason, VLMMode) else str(reason)
        prompt = self._build_vlm_prompt(scene, transition, legal, mode_value)
        request = VLMRequest(
            text_prompt=prompt,
            current_rgb=scene.rgb or render_grid(scene.grid, self.config.image_size),
            previous_rgb=None if transition is None else transition.previous_rgb,
            analysis_rgb=scene.annotated_rgb,
            max_new_tokens=self.config.vlm_max_new_tokens,
        )
        level.vlm_calls_this_level += 1
        level.last_vlm_action_count = level.total_action_count
        level.actions_since_vlm = 0
        level.force_vlm_reason = ""
        level.current_vlm_mode = mode_value
        self.memory.game.vlm_calls_total += 1
        self.memory.game.controller_stats.vlm_mode_counts[mode_value] += 1
        self.logger.log_event(
            "vlm_call",
            {
                "reason": mode_value,
                "mode": mode_value,
                "level": level.level_index,
                "attempt": level.attempt_index,
                "life": level.life_index,
                "call": level.vlm_calls_this_level,
            },
        )
        try:
            raw = self.backend.decide(request)
        except Exception as exc:
            self.memory.game.total_vlm_errors += 1
            level.vlm_invalid_streak += 1
            self.logger.log_event("vlm_exception", {"message": str(exc)[:300]})
            return None
        result = parse_vlm_result(raw)
        if result is None:
            self.memory.game.total_vlm_errors += 1
            level.vlm_invalid_streak += 1
            if level.vlm_invalid_streak <= 1:
                level.force_vlm_reason = "retry_invalid_vlm_output"
            self.logger.log_event(
                "vlm_invalid",
                {
                    "raw_type": type(raw).__name__,
                    "reason": mode_value,
                    "raw_excerpt": _short_string(raw, 500),
                },
            )
            return None
        if result.mode == "INVALID_PARTIAL":
            self.memory.game.total_vlm_errors += 1
            level.vlm_invalid_streak += 1
            if level.vlm_invalid_streak <= 1:
                level.force_vlm_reason = "retry_invalid_vlm_output"
            self.logger.log_event(
                "vlm_invalid_salvaged",
                {"reason": mode_value, "raw_excerpt": result.proposed_memory_patch.get("raw_invalid_vlm_excerpt", "")[:500]},
            )
            return result
        level.vlm_invalid_streak = 0
        result.mode = result.mode or mode_value
        contract_counts = _vlm_contract_counts(result)
        self.logger.log_event(
            "vlm_result_parsed",
            {
                "mode": result.mode,
                **contract_counts,
                "has_schema_patch": bool(result.goal_schema_patch),
            },
        )
        return result

    def _build_vlm_prompt(
        self,
        scene: SceneSnapshot,
        transition: TransitionReport | None,
        legal: Sequence[Any],
        reason: str,
    ) -> str:
        legal_names = [action_name(action) for action in legal]
        objects = [
            {
                "id": obj.track_id,
                "type": obj.shape_label,
                "bbox": obj.bbox,
                "colors": dict(obj.color_areas),
                "area": obj.area,
                "near_edge": obj.near_edge,
                "pattern": obj.pattern,
                "inner_pattern": obj.inner_pattern,
                "visit_count": self.memory.level.object_visit_counts.get(obj.track_id, 0),
                "effect_count": self.memory.level.object_effect_counts.get(obj.track_id, 0),
                "belief": self.memory.level.object_beliefs.get(obj.track_id, {}),
            }
            for obj in sorted(scene.objects, key=lambda item: -item.salience)[
                : self.config.max_prompt_objects
            ]
        ]
        transition_payload = None
        if transition is not None:
            transition_payload = {
                "action": transition.action_key,
                "source": transition.action_source,
                "summary": transition.summary,
                "moved_objects": transition.moved_objects[:8],
                "transformed_objects": transition.transformed_objects[:8],
                "appeared_object_ids": transition.appeared_object_ids,
                "disappeared_object_ids": transition.disappeared_object_ids,
                "retry_detected": transition.retry_detected,
            }
        controller_target = self._choose_semantic_target(scene)
        mode = str(reason)
        payload = {
            "task": "update evidence-backed memory and return structured experiments, proposals, or memory patches",
            "memory_readme": {
                "format": "All memory fields are JSON and object ids match the annotated image labels. Treat them as evidence summaries, not hidden state.",
                "trust_order": [
                    "observer and last_transition are authoritative",
                    "recent_event_log and action_knowledge are measured controller evidence",
                    "goal_schemas, hypotheses, object_beliefs, and advisory summaries are hypotheses until verified",
                ],
                "how_to_use": [
                    "Use level_memory.current_state.known_noops to avoid repeating known no-op actions in this exact state.",
                    "Use level_memory.target_failure_policy.hard_blocks_current_life for current-life hard target blocks; failed_targets counts are soft history after a new life.",
                    "Prefer decision.kind/intents over raw action fields. Use action6_candidates candidate_id via click_candidate when available; do not hand-copy coordinates unless needed.",
                    "Use active level_theories and update them from transition evidence.",
                    "Use game_memory.action_knowledge vectors/effect_score to infer action effects, but keep uncertainty when evidence is weak.",
                    "Use recent_event_log event_ids when writing hypothesis_updates or goal_schema_patch evidence.",
                ],
            },
            "mode": mode,
            "game_id": self._game_id(),
            "decision_reason": mode,
            "stage": self.memory.level.stage.value,
            "image_order": (
                ["CURRENT_RAW", "CURRENT_ANNOTATED"]
                if transition is None
                else ["BEFORE_RAW", "CURRENT_RAW", "CURRENT_ANNOTATED"]
            ),
            "legal_actions_now": legal_names,
            "action6_candidates": (
                [_json_safe(candidate) for candidate in self._action6_candidate_objects(scene)]
                if "ACTION6" in legal_names
                else []
            ),
            "observer": {
                "scene_summary": scene.summary[:5200],
                "objects": objects,
                "framed_template_relations": list(scene.template_relations),
                "controlled_object_id": self.memory.level.controlled_object_id or None,
                "walkable_color_votes": self.memory.level.walkable_color_votes,
                "step_counter_like": {
                    "value": scene.counter_value,
                    "capacity": scene.counter_capacity,
                    "ratio": scene.counter_ratio,
                    "lives": scene.life_count,
                },
            },
            "last_transition": transition_payload,
            "game_memory": self._game_memory_summary(),
            "level_memory": self._level_memory_summary(scene, legal),
            "goal_schemas": [_json_safe(schema) for schema in self.memory.game.goal_schemas[-6:]],
            "level_goal_instantiation": _json_safe(self.memory.level.goal_instantiation),
            "hypotheses": [_json_safe(hyp) for hyp in list(self.memory.game.hypotheses.values())[-12:]],
            "recent_event_log": self._recent_event_prompt(self.config.max_prompt_events),
            "reject_reasons": dict(self.memory.level.reject_reasons),
            "hard_action_rules": {
                "ACTION7": "undo-only; do not propose",
                "RESET": "terminal/recovery only; do not propose for exploration",
                "ACTION6": "compiler-owned coordinate action; prefer click_candidate/action6_candidate_id or click_object/target_object_id",
                "disabled_action_families": dict(self.memory.level.disabled_action_families_until),
                "nonprogress_policy": "When a primitive repeatedly causes structural/resource changes without movement, interaction, or goal progress, treat it as suspected undo/resource churn and avoid it unless new evidence requires it.",
            },
            "output_action_contract": {
                "valid_names": sorted(set(legal_names + ["NAVIGATE"])),
                "internal_navigation": "Use {intent: navigate_to_object, target_object_id: O3}; legacy {action: NAVIGATE, target_object_id: O3} is also accepted.",
                "simple_action_example": "Use {intent: primitive_action, action: ACTION1, target_object_id: O1, purpose: ...}; do not emit free text like MOVE O1 UP.",
                "action6_example": "Prefer {intent: click_candidate, action6_candidate_id: A6_xxxxxx}. For object clicks use {intent: click_object, target_object_id: O2}.",
            },
            "mechanism_classifier": self.memory.level.mechanism.summary(),
            "strategy_rules": [
                rule.as_prompt() for rule in self.memory.game.strategy_rules[-6:]
            ],
            "near_success_routes": [
                route.as_prompt() for route in list(self.memory.level.near_success_routes)[-4:]
            ],
            "target_failure_policy": {
                "hard_block_scope": "current life only unless the same scene signature repeats three times",
                "current_life_limit": self.config.target_failure_limit,
                "counter_exhaustion_is_not_target_failure": True,
            },
            "controller_evidence": {
                "suggested_visible_target": controller_target,
                "failed_targets": self.memory.level.target_failure_counts,
                "known_noops_here": sorted(
                    self.memory.level.noop_actions_by_state.get(scene.state_hash, set())
                ),
            },
            "repair_candidates": list(self.memory.level.repair_action_candidates)[:12],
            "requirements": [
                "Use object IDs from the annotated image and object table.",
                "Return experiments when the mode is EXPERIMENT_DESIGN; include predictions and expected_predicates.",
                "Do not call a framed world object directly reachable when its inner pattern mismatches an edge/status frame unless the mismatch is irrelevant by evidence.",
                "When a compact object is an untested possible transformer, prefer testing it before repeatedly colliding with a mismatched frame.",
                "Use NAVIGATE plus target_object_id for visible waypoints; do not invent low-level maze paths when the controller can pathfind.",
                "Treat failed_targets as soft penalties; only avoid targets listed as current hard blocks by target_failure_policy unless new evidence supports retry.",
                "Prefer decision.kind/intents over raw plan. NAVIGATE is only an intent and the controller will compile it to a primitive action.",
                "Do not hand-copy ACTION6 coordinates when an action6_candidate_id is available. Use click_candidate with action6_candidate_id.",
                "During INIT_ANALYSIS, produce level_theory before proposing actions. During later modes, update or invalidate level_theory based on transition evidence.",
                "Use discriminating_tests from active level_theory before falling back to random exploration.",
                "Only emit legal environment actions, except the internal plan token NAVIGATE.",
                "Never emit ACTION7, RESET, CLICK, MOVE, GO, PATHFIND, or free-form action names.",
            ],
            "output_schema": {
                "observations": ["specific object-level facts"],
                "object_updates": [
                    {
                        "object_id": "O3",
                        "role": "candidate transformer/goal/player/status",
                        "confidence": 0.0,
                        "status": "candidate|supported|verified|rejected",
                        "evidence": "specific image or transition",
                    }
                ],
                "action_interpretation": {
                    "action": "ACTION1",
                    "effect": "which labelled object moved and by what vector",
                    "status": "unknown|tentative|verified",
                    "evidence": "",
                },
                "goal_update": {
                    "claim": "",
                    "scope": "game|level",
                    "confidence": 0.0,
                    "status": "candidate|supported|verified|rejected",
                    "evidence": "",
                },
                "mechanic_updates": [
                    {
                        "claim": "",
                        "scope": "game|level",
                        "confidence": 0.0,
                        "evidence": "",
                    }
                ],
                "counter_update": {
                    "claim": "",
                    "confidence": 0.0,
                    "strategy": "",
                },
                "rejected_hypotheses": [],
                "ready_to_solve": False,
                "plan_goal": "",
                "target_object_id": "O3",
                "level_theory": {
                    "win_condition_hypothesis": "what likely completes this level",
                    "mechanism_hypothesis": "how objects/actions may cause progress",
                    "critical_objects": [
                        {"object_id": "O2", "role": "candidate transformer/goal/player/status", "confidence": 0.5, "evidence": "visible pattern or transition"}
                    ],
                    "solve_sketch": [
                        {"intent": "test_object", "target_object_id": "O2", "purpose": "..."}
                    ],
                    "discriminating_tests": [
                        {"intent": "click_candidate", "action6_candidate_id": "A6_xxxxxx", "expected_predicates": [{"type": "transform"}]}
                    ],
                    "expected_progress_signals": [{"type": "transform"}, {"type": "movement"}, {"type": "counter_delta_nonpositive"}],
                    "invalidating_evidence": ["what observation would disprove this theory"],
                    "confidence": 0.0,
                },
                "decision": {
                    "kind": "memory_update_only|single_experiment|plan_candidate|repair_decision",
                    "goal": "",
                    "confidence": 0.0,
                    "evidence_level": 0,
                    "selected_repair_candidate_id": "RP_XXXXXXXXXX",
                    "rationale": "why this controller-provided repair candidate is preferred",
                    "intents": [
                        {
                            "intent": "primitive_action|navigate_to_object|click_candidate|click_object|test_object",
                            "action": "ACTION1",
                            "target_object_id": "O2",
                            "action6_candidate_id": "A6_xxxxxx",
                            "purpose": "",
                            "expected_predicates": [{"type": "not_noop"}],
                            "risk": "low",
                            "information_gain": 0.0,
                            "goal_progress": 0.0,
                        }
                    ],
                },
                "intent_proposals": [],
                "goal_schema_patch": {},
                "level_instantiation_patch": {},
                "failure_constraints": [],
                "game_summary": "advisory hypotheses only",
                "level_summary": "advisory current-level hypotheses only",
            },
        }
        if mode == VLMMode.SUCCESS_CONSOLIDATION.value:
            payload["mode_instructions"] = [
                "Infer the reusable game-level success condition from the terminal event and causal slice.",
                "Separate game_goal_schema from level_goal_instantiation.",
                "Identify which actions were navigation and which changed necessary conditions.",
                "Every schema claim must cite event_ids when possible.",
            ]
        elif mode == VLMMode.EXPERIMENT_DESIGN.value:
            payload["mode_instructions"] = [
                "Return experiments, not a long plan.",
                "Each experiment must distinguish hypotheses and include predictions.",
                "Avoid rejected actions and known noops.",
            ]
        elif mode == VLMMode.TRANSFER_INSTANTIATION.value:
            payload["mode_instructions"] = [
                "Compare current initial frame with previous level_outcomes: initial_summary, pre_success_summary, and start_to_pre_success_diff.",
                "Infer which object roles transfer and which slots changed in this level.",
                "Bind prior goal_schema role_slots to current visible object IDs and concrete values.",
                "List unknown_slots that still need evidence before committing to a plan.",
                "Return level_theory and level_instantiation_patch before proposing solve intents.",
                "Prefer schema_transfer or vlm_transfer proposals only when role bindings are grounded in observer facts.",
                "Return intent_proposals and short validating experiments for uncertain slots.",
            ]
        elif mode == VLMMode.REPAIR_DECISION.value:
            payload["mode_instructions"] = [
                "Explain the controller failure using recent_event_log, recovery, compile_failures, nonprogress_action_records, and failure_model.",
                "Choose from repair_candidates only: return decision.kind=repair_decision with selected_repair_candidate_id and rationale, or memory_update_only if none are safe.",
                "Do not emit free-form repair intents/actions in REPAIR_DECISION; target_object_id hints may be included only to rank ACTION6 candidate frontier search.",
                "If recovery.reason mentions nonprogress/resource/undo, identify which primitive is likely churn or resource-consuming and state what evidence would justify trying it again.",
                "The deterministic RepairPolicy validates and emits the final action.",
            ]
        elif mode == VLMMode.FAILURE_REPAIR.value:
            payload["mode_instructions"] = [
                "Explain the likely failure cause using recent_event_log, retry/game-over evidence, and failure_model.",
                "Return failure_constraints that forbid unsafe suffixes or target/action combinations.",
                "Suggest safe retry experiments with low risk, reversibility, and explicit expected_predicates.",
                "Do not propose the same failed action suffix unless new evidence changes the constraint.",
            ]
        elif mode == VLMMode.TRANSITION_EXPLANATION.value:
            payload["mode_instructions"] = [
                "Explain the last transition in terms of object roles, action effects, resources, and hypotheses.",
                "Update hypotheses only to the evidence level supported by the observed transition.",
                "Recommend the next discriminating experiment if the mechanism remains ambiguous.",
            ]
        elif mode == VLMMode.PLAN_SYNTHESIS.value:
            payload["mode_instructions"] = [
                "Synthesize a short validated plan from current goal_instantiation and known mechanics.",
                "Return decision.kind=plan_candidate with intents; each intent should include target_object_id where relevant and expected_predicates checkpoints.",
                "Prefer NAVIGATE for visible waypoints and avoid long raw action loops.",
            ]
        else:
            payload["mode_instructions"] = [
                "Describe only observer-grounded facts and mark interpretations as hypotheses.",
                "During INIT_ANALYSIS, return level_theory before proposing actions.",
                "Identify plausible controlled object, interactive objects, goal cues, and first low-risk experiments.",
                "Return intent_proposals or recommended_experiments with predictions rather than a long plan.",
            ]
        return json.dumps(payload, ensure_ascii=True, default=str)

    def _game_memory_summary(self) -> dict[str, Any]:
        return {
            "advisory_summary": self.memory.game.advisory_summary[:700],
            "action_knowledge": {
                name: {
                    "status": item.status,
                    "attempts": item.attempts,
                    "successes": item.successes,
                    "blocked": item.blocked,
                    "structural_successes": item.structural_successes,
                    "global_changes": item.global_changes,
                    "retry_failures": item.retry_failures,
                    "effect_score": item.effect_score(),
                    "vectors": item.movement_vectors,
                    "last_evidence": item.last_evidence[:220],
                }
                for name, item in sorted(self.memory.game.action_knowledge.items())
            },
            "goal_hypotheses": self.memory.game.goal_hypotheses[:8],
            "mechanics": self.memory.game.mechanics[:10],
            "object_type_roles": list(self.memory.game.object_type_roles.values())[:10],
            "successful_levels": self.memory.game.successful_levels[-4:],
            "level_outcomes": [outcome.as_prompt() for outcome in self.memory.game.level_outcomes[-4:]],
            "mechanism_library": self.memory.game.mechanism_library[-8:],
            "strategy_rules": [
                rule.as_prompt() for rule in self.memory.game.strategy_rules[-6:]
            ],
            "goal_schemas": [_json_safe(schema) for schema in self.memory.game.goal_schemas[-6:]],
            "hypotheses": [_json_safe(hyp) for hyp in list(self.memory.game.hypotheses.values())[-12:]],
            "failure_model": _json_safe(self.memory.game.failure_model),
            "controller_stats": _json_safe(self.memory.game.controller_stats),
        }

    def _level_memory_summary(
        self,
        scene: SceneSnapshot,
        legal: Sequence[Any],
    ) -> dict[str, Any]:
        level = self.memory.level
        legal_names = [action_name(action) for action in legal]
        return {
            "level_index": level.level_index,
            "attempt": level.attempt_index,
            "life": level.life_index,
            "stage": level.stage.value,
            "advisory_summary": level.advisory_summary[:700],
            "local_goal": level.local_goal[:320],
            "controlled_object_id": level.controlled_object_id,
            "tentative_controlled_object_id": level.tentative_controlled_object_id,
            "controlled_actor_confidence": round(level.controlled_actor_confidence, 3),
            "mechanism_classifier": level.mechanism.summary(),
            "current_plan_goal": level.plan_goal,
            "active_plan": self._active_plan_prompt(),
            "candidate_plans": [plan.as_prompt() for plan in list(level.candidate_plans)[:6]],
            "disabled_action_families_until": dict(level.disabled_action_families_until),
            "nonprogress_action_records": list(level.nonprogress_action_records)[-12:],
            "suspicious_action_counts": dict(level.suspicious_action_counts),
            "fallback_action_records": list(level.fallback_action_records)[-12:],
            "repair_action_candidates": list(level.repair_action_candidates)[:12],
            "vlm_repair_choice": _json_safe(level.vlm_repair_choice),
            "anti_loop_escape_failures": level.anti_loop_escape_failures,
            "last_repair_vlm": {
                "reason": level.last_repair_vlm_reason,
                "step": level.last_repair_vlm_step,
            },
            "last_compile_failures": list(level.last_compile_failures)[-8:],
            "total_steps": level.total_action_count,
            "life_steps": level.life_action_count,
            "vlm_calls": level.vlm_calls_this_level,
            "vlm_remaining": self.config.max_vlm_calls_per_level - level.vlm_calls_this_level,
            "failed_lives": list(level.failed_lives)[-4:],
            "failed_targets": level.target_failure_counts,
            "target_failure_records": [
                item.as_prompt() for item in level.target_failure_records[-8:]
            ],
            "target_failure_policy": {
                "hard_blocks_current_life": [
                    target
                    for target in sorted(level.target_failure_counts)
                    if sum(
                        item.count
                        for item in level.target_failure_records
                        if item.target_id == target
                        and item.life == level.life_index
                        and item.attempt == level.attempt_index
                    ) >= self.config.target_failure_limit
                ],
                "legacy_counts_are_soft_after_new_life": True,
            },
            "near_success_routes": [
                route.as_prompt() for route in list(level.near_success_routes)[-4:]
            ],
            "plan_quality_rejections": level.plan_quality_rejections,
            "reject_reasons": dict(level.reject_reasons),
            "recovery": {
                "mode": level.recovery_mode,
                "reason": level.recovery_reason,
                "until_step": level.recovery_until_step,
                "fallback_recent_keys": list(level.fallback_recent_keys),
                "blocked_state_action_pairs": [(state[:12], action) for state, action in sorted(level.blocked_state_action_pairs)],
            },
            "active_theory_id": level.active_theory_id,
            "level_theories": [theory.as_prompt() for theory in level.level_theories[:6]],
            "goal_instantiation": _json_safe(level.goal_instantiation),
            "action6_duplicate_suppressed": level.action6_memory.duplicate_suppressed,
            "loop_blocked_action_keys": sorted(level.loop_blocked_action_keys),
            "counter_cost_samples": list(level.counter_cost_samples),
            "counter_refills": level.counter_refill_transitions[-5:],
            "current_state": {
                "state_hash": scene.state_hash[:12],
                "legal_actions": legal_names,
                "tried": sorted(level.tried_actions_by_state.get(scene.state_hash, set())),
                "known_noops": sorted(level.noop_actions_by_state.get(scene.state_hash, set())),
                "counter": scene.counter_value,
                "capacity": scene.counter_capacity,
                "lives": scene.life_count,
            },
            "recent_events": list(level.recent_events)[-self.config.max_prompt_events :],
        }

    def _merge_belief(
        self,
        target: list[dict[str, Any]],
        *,
        claim: str,
        confidence: float,
        evidence: str,
        scope: str,
        verified: bool = False,
    ) -> None:
        claim = _short_string(claim, 340)
        if not claim:
            return
        confidence = _clamp01(confidence)
        best = None
        best_similarity = 0.0
        for item in target:
            similarity = _claim_similarity(claim, str(item.get("claim", "")))
            if similarity > best_similarity:
                best_similarity = similarity
                best = item
        if best is None or best_similarity < 0.70:
            target.append(
                {
                    "claim": claim,
                    "confidence": round(confidence, 3),
                    "scope": scope,
                    "supports": 1,
                    "contradictions": 0,
                    "status": "verified" if verified else "candidate",
                    "evidence": [_short_string(evidence, 300)] if evidence else [],
                    "level": self.memory.level.level_index,
                }
            )
        else:
            supports = int(best.get("supports", 1)) + 1
            old_confidence = float(best.get("confidence", 0.0))
            best["supports"] = supports
            best["confidence"] = round(
                max(old_confidence, (old_confidence * (supports - 1) + confidence) / supports),
                3,
            )
            if confidence >= old_confidence:
                best["claim"] = claim
            if evidence:
                evidence_list = list(best.get("evidence", []))
                evidence_list.append(_short_string(evidence, 300))
                best["evidence"] = evidence_list[-5:]
            if verified:
                best["status"] = "verified"
                best["confidence"] = max(0.9, float(best["confidence"]))
        target.sort(
            key=lambda item: (
                item.get("status") == "verified",
                float(item.get("confidence", 0.0)),
                int(item.get("supports", 0)) - int(item.get("contradictions", 0)),
            ),
            reverse=True,
        )
        del target[18:]

    def _reject_belief(self, claim: str, reason: str) -> None:
        for collection in (self.memory.game.goal_hypotheses, self.memory.game.mechanics):
            for item in collection:
                if _claim_similarity(claim, str(item.get("claim", ""))) >= 0.65:
                    item["contradictions"] = int(item.get("contradictions", 0)) + 1
                    item["confidence"] = max(0.0, float(item.get("confidence", 0.5)) - 0.22)
                    item["status"] = "contradicted"
                    evidence = list(item.get("evidence", []))
                    evidence.append(f"CONTRADICTION: {_short_string(reason, 260)}")
                    item["evidence"] = evidence[-5:]

    def _apply_vlm_update(
        self,
        result: VLMResult,
        transition: TransitionReport | None,
        scene: SceneSnapshot,
        legal: Sequence[Any],
    ) -> None:
        level = self.memory.level
        if result.game_summary:
            self.memory.game.advisory_summary = result.game_summary[:700]
        if result.level_summary:
            level.advisory_summary = result.level_summary[:700]
        invalid_excerpt = ""
        if isinstance(result.proposed_memory_patch, dict):
            invalid_excerpt = _short_string(result.proposed_memory_patch.get("raw_invalid_vlm_excerpt"), 700)
        if invalid_excerpt:
            level.advisory_summary = _short_string(f"{level.advisory_summary} Invalid VLM advisory excerpt: {invalid_excerpt}", 900)

        for claim in result.rejected_hypotheses:
            self._reject_belief(claim, "VLM explicitly rejected it after new evidence")

        goal = result.goal_update
        goal_claim = _short_string(goal.get("claim"), 340)
        if goal_claim:
            status = _short_string(goal.get("status"), 24).lower()
            confidence = _clamp01(goal.get("confidence"))
            verified = status == "verified" and transition is not None and transition.interaction_event
            if transition is None and not verified:
                confidence = min(confidence, 0.62)
            self._merge_belief(
                self.memory.game.goal_hypotheses,
                claim=goal_claim,
                confidence=confidence,
                evidence=_short_string(goal.get("evidence"), 320),
                scope=_short_string(goal.get("scope"), 20).lower() or "level",
                verified=verified,
            )
            if confidence >= 0.55 and status != "rejected":
                level.local_goal = goal_claim

        for update in result.mechanic_updates:
            claim = _short_string(update.get("claim"), 340)
            if claim:
                confidence = _clamp01(update.get("confidence"))
                if transition is None:
                    confidence = min(confidence, 0.62)
                self._merge_belief(
                    self.memory.game.mechanics,
                    claim=claim,
                    confidence=confidence,
                    evidence=_short_string(update.get("evidence"), 320),
                    scope=_short_string(update.get("scope"), 20).lower() or "level",
                    verified=False,
                )
        if result.counter_update.get("claim"):
            self._merge_belief(
                self.memory.game.mechanics,
                claim=_short_string(result.counter_update.get("claim"), 320),
                confidence=min(0.8, _clamp01(result.counter_update.get("confidence"))),
                evidence=_short_string(result.counter_update.get("strategy"), 300),
                scope="game",
                verified=False,
            )

        valid_ids = {obj.track_id for obj in scene.objects}
        for update in result.object_updates:
            object_id = _short_string(
                update.get("object_id") or update.get("id"),
                24,
            ).upper()
            if object_id not in valid_ids:
                continue
            confidence = _clamp01(update.get("confidence"))
            status = _short_string(update.get("status"), 24).lower() or "candidate"
            evidence = _short_string(update.get("evidence"), 320)
            if transition is None and status != "verified":
                confidence = min(confidence, 0.62)
            belief = level.object_beliefs.setdefault(object_id, {})
            if status == "rejected":
                belief["contradictions"] = int(belief.get("contradictions", 0)) + 1
                belief["confidence"] = max(0.0, float(belief.get("confidence", 0.5)) - 0.2)
                belief["status"] = "rejected"
            else:
                old_conf = float(belief.get("confidence", 0.0))
                if confidence >= old_conf:
                    belief["role"] = _short_string(
                        update.get("role") or update.get("role_hypothesis"),
                        180,
                    )
                belief["confidence"] = max(old_conf, confidence)
                belief["status"] = status
                belief["supports"] = int(belief.get("supports", 0)) + 1
            if evidence:
                evidence_list = list(belief.get("evidence", []))
                evidence_list.append(evidence)
                belief["evidence"] = evidence_list[-5:]
            obj = scene.object_by_id(object_id)
            if obj is not None and belief.get("role"):
                type_entry = self.memory.game.object_type_roles.setdefault(
                    obj.type_key,
                    {
                        "type_key": obj.type_key,
                        "visual_type": obj.shape_label,
                        "role": belief["role"],
                        "confidence": belief.get("confidence", 0.0),
                        "supports": 0,
                    },
                )
                type_entry["supports"] = int(type_entry.get("supports", 0)) + 1
                if float(belief.get("confidence", 0.0)) >= float(type_entry.get("confidence", 0.0)):
                    type_entry["role"] = belief["role"]
                    type_entry["confidence"] = belief.get("confidence", 0.0)

        interp = result.action_interpretation
        action = _short_string(interp.get("action"), 40).upper()
        transition_action = transition.action_key.split(":", 1)[0] if transition else ""
        if action and (not transition_action or action == transition_action):
            knowledge = self.memory.game.action_knowledge.setdefault(action, ActionKnowledge())
            if interp.get("evidence"):
                knowledge.last_evidence = _short_string(interp.get("evidence"), 300)

        for update in result.hypothesis_updates:
            statement = _short_string(update.get("statement") or update.get("claim"), 500)
            if statement:
                event_ids = (
                    [eid for eid in update.get("event_ids", []) if isinstance(eid, int)]
                    if isinstance(update.get("event_ids"), list)
                    else None
                )
                self._upsert_hypothesis(
                    kind=_short_string(update.get("kind"), 40) or "vlm",
                    statement=statement,
                    confidence=_clamp01(update.get("confidence", 0.4)),
                    scope=_short_string(update.get("scope"), 40) or "level",
                    event_ids=event_ids,
                    evidence_level=self._cap_vlm_evidence_level(update.get("evidence_level", 0), transition, event_ids),
                )
        if goal_claim:
            self._upsert_hypothesis(
                kind="goal",
                statement=goal_claim,
                confidence=_clamp01(goal.get("confidence")),
                scope=_short_string(goal.get("scope"), 20).lower() or "level",
                event_ids=[self.memory.event_log[-1].event_id] if self.memory.event_log and transition is not None else None,
                evidence_level=2 if transition is not None else 0,
            )
        if result.level_instantiation_patch:
            self._apply_level_instantiation_patch(result.level_instantiation_patch)
        if result.level_theory_patch:
            self._apply_level_theory_patch(result.level_theory_patch, source=result.mode or "vlm", transition=transition)
        if result.goal_schema_patch:
            known_ids = {event.event_id for event in self.memory.event_log}
            evidence_ids = result.goal_schema_patch.get("evidence_event_ids", [])
            has_known = any(eid in known_ids for eid in evidence_ids if isinstance(eid, int))
            if has_known or result.mode == VLMMode.SUCCESS_CONSOLIDATION.value:
                self._accept_goal_schema_patch(result.goal_schema_patch, [], self.memory.level.level_index)
        for constraint in result.failure_constraints:
            suffix = constraint.get("suffix")
            if isinstance(suffix, list):
                self.memory.game.failure_model.add_suffix(
                    [_short_string(item, 80) for item in suffix],
                    _short_string(constraint.get("reason"), 240),
                    level.total_action_count,
                    level.level_index,
                    level.attempt_index,
                )

    def _premature_frame_target(
        self,
        step: dict[str, Any],
        scene: SceneSnapshot,
    ) -> bool:
        if _short_string(step.get("name"), 40).upper() != "NAVIGATE":
            return False
        target_id = _short_string(step.get("target_object_id"), 24).upper()
        target = scene.object_by_id(target_id)
        if target is None or target.frame_color is None or target.near_edge:
            return False
        mismatch = any(
            relation.get("edge_vs_world")
            and not relation.get("exact_inner_match")
            and target_id in {relation.get("left"), relation.get("right")}
            for relation in scene.template_relations
        )
        if not mismatch:
            return False
        return any(
            obj.track_id != self.memory.level.controlled_object_id
            and not obj.near_edge
            and obj.frame_color is None
            and obj.area <= 120
            and self.memory.level.object_effect_counts.get(obj.track_id, 0) == 0
            for obj in scene.objects
        )

    def _semantic_fallback_action(
        self,
        scene: SceneSnapshot,
        legal: Sequence[Any],
    ) -> tuple[Any, dict[str, Any], str] | None:
        target_id = self._choose_semantic_target(scene)
        if target_id is None:
            return None
        path = self._plan_path_to_object(scene, target_id, legal)
        if not path:
            return None
        action = get_action_by_name(path[0])
        legal_by_value = {action_value(item): item for item in legal}
        if action is None or action_value(action) not in legal_by_value:
            return None
        target = scene.object_by_id(target_id)
        proposal = {
            "name": path[0],
            "target_object_id": target_id,
            "purpose": f"symbolic navigation toward {target_id} ({target.shape_label if target else 'object'})",
            "expected_change": f"controlled object moves along a collision-free route toward {target_id}",
        }
        made = self._attach_reasoning(
            legal_by_value[action_value(action)],
            proposal,
        )
        return made, proposal, "symbolic_nav"

    def _choose_semantic_target(self, scene: SceneSnapshot) -> str | None:
        level = self.memory.level
        actor = scene.object_by_id(level.controlled_object_id)
        if actor is None:
            return None

        exact_frames: set[str] = set()
        mismatched_frames: set[str] = set()
        for relation in scene.template_relations:
            if not relation.get("edge_vs_world"):
                continue
            left = scene.object_by_id(str(relation.get("left")))
            right = scene.object_by_id(str(relation.get("right")))
            world = left if left is not None and not left.near_edge else right
            if world is None:
                continue
            if relation.get("exact_inner_match"):
                exact_frames.add(world.track_id)
            else:
                mismatched_frames.add(world.track_id)

        candidates: list[tuple[float, str]] = []
        for obj in scene.objects:
            if obj.track_id == actor.track_id or obj.near_edge:
                continue
            if self._target_blocked(obj.track_id, scene):
                continue
            belief = level.object_beliefs.get(obj.track_id, {})
            role = str(belief.get("role", "")).lower()
            confidence = float(belief.get("confidence", 0.0))
            distance = abs(obj.centroid[0] - actor.centroid[0]) + abs(obj.centroid[1] - actor.centroid[1])
            visits = level.object_visit_counts.get(obj.track_id, 0)
            effects = level.object_effect_counts.get(obj.track_id, 0)
            score = obj.salience - 0.12 * distance - 2.0 * visits

            if obj.track_id in exact_frames:
                score += 45.0
            if obj.track_id in mismatched_frames:
                score -= 35.0
            if any(token in role for token in ("goal", "target", "exit", "destination")):
                score += 24.0 * confidence
            if any(token in role for token in ("modifier", "transform", "switch", "tool", "pickup")):
                score += 28.0 * max(0.4, confidence)
            if obj.frame_color is None and obj.area <= 120 and visits == 0:
                score += 24.0
            if effects > 0 and mismatched_frames and obj.frame_color is None:
                score += 20.0
            candidates.append((score, obj.track_id))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        best_score, best_id = candidates[0]
        return best_id if best_score >= 4.0 else None

    @staticmethod
    def _object_reached(
        actor: ObjectObservation,
        target: ObjectObservation,
    ) -> bool:
        tx = int(round(target.centroid[0]))
        ty = int(round(target.centroid[1]))
        return (
            actor.bbox[0] <= tx <= actor.bbox[2]
            and actor.bbox[1] <= ty <= actor.bbox[3]
        )

    def _plan_path_to_object(
        self,
        scene: SceneSnapshot,
        target_id: str,
        legal: Sequence[Any],
        *,
        use_weak_vectors: bool = True,
    ) -> list[str] | None:
        level = self.memory.level
        actor = scene.object_by_id(level.controlled_object_id)
        target = scene.object_by_id(target_id)
        if actor is None or target is None:
            return None
        vectors = self._action_vectors(actor, legal) if use_weak_vectors else self._grounded_action_vectors(legal)
        if not vectors:
            return None
        floor_colors = self._walkable_colors(scene, actor)
        if not floor_colors:
            return None

        start = (actor.bbox[0], actor.bbox[1])
        width, height = actor.width, actor.height
        legal_names = {action_name(action) for action in legal}
        vectors = {
            name: vector
            for name, vector in vectors.items()
            if name in legal_names and vector != (0, 0)
        }
        if not vectors:
            return None

        queue: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])
        seen = {start}
        while queue:
            position, path = queue.popleft()
            test_actor = (
                position[0],
                position[1],
                position[0] + width - 1,
                position[1] + height - 1,
            )
            tx = int(round(target.centroid[0]))
            ty = int(round(target.centroid[1]))
            if test_actor[0] <= tx <= test_actor[2] and test_actor[1] <= ty <= test_actor[3]:
                return path
            if len(path) >= self.config.navigation_max_depth:
                continue
            for name in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"):
                if name not in vectors:
                    continue
                dx, dy = vectors[name]
                next_position = (position[0] + dx, position[1] + dy)
                if next_position in seen:
                    continue
                if not self._position_passable(
                    scene,
                    next_position,
                    width,
                    height,
                    floor_colors,
                    actor,
                    target,
                ):
                    continue
                seen.add(next_position)
                queue.append((next_position, [*path, name]))
        return None

    def _grounded_action_vectors(self, legal: Sequence[Any]) -> dict[str, tuple[int, int]]:
        legal_names = {action_name(action) for action in legal}
        vectors: dict[str, tuple[int, int]] = {}
        for name, knowledge in self.memory.game.action_knowledge.items():
            vector = knowledge.best_vector()
            if vector is not None and name in legal_names:
                vectors[name] = vector
        return vectors

    def _action_vectors(
        self,
        actor: ObjectObservation,
        legal: Sequence[Any],
    ) -> dict[str, tuple[int, int]]:
        vectors = self._grounded_action_vectors(legal)
        legal_names = {action_name(action) for action in legal}
        weak = {
            "ACTION1": (0, -actor.height),
            "ACTION2": (0, actor.height),
            "ACTION3": (-actor.width, 0),
            "ACTION4": (actor.width, 0),
        }
        for name, vector in weak.items():
            if name in legal_names and name not in vectors:
                vectors[name] = vector
        return vectors

    def _walkable_colors(
        self,
        scene: SceneSnapshot,
        actor: ObjectObservation,
    ) -> set[int]:
        votes = self.memory.level.walkable_color_votes
        if votes:
            best = max(votes.values())
            return {color for color, count in votes.items() if count >= max(1, best // 2)}
        counts: Counter[int] = Counter()
        x0, y0, x1, y1 = actor.bbox
        for y in range(max(0, y0 - 1), min(scene.height - 1, y1 + 1) + 1):
            for x in range(max(0, x0 - 1), min(scene.width - 1, x1 + 1) + 1):
                if x0 <= x <= x1 and y0 <= y <= y1:
                    continue
                color = scene.grid[y][x]
                if color != scene.background_candidate and color not in actor.colors:
                    counts[color] += 1
        if counts:
            color = counts.most_common(1)[0][0]
            return {color}
        return set()

    def _position_passable(
        self,
        scene: SceneSnapshot,
        position: tuple[int, int],
        width: int,
        height: int,
        floor_colors: set[int],
        actor: ObjectObservation,
        target: ObjectObservation,
    ) -> bool:
        x0, y0 = position
        x1, y1 = x0 + width - 1, y0 + height - 1
        if x0 < 0 or y0 < 0 or x1 >= scene.width or y1 >= scene.height:
            return False
        small_objects = [
            obj
            for obj in scene.objects
            if obj.track_id != actor.track_id
            and not obj.near_edge
            and obj.area <= 180
            and obj.frame_color is None
        ]
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if actor.bbox[0] <= x <= actor.bbox[2] and actor.bbox[1] <= y <= actor.bbox[3]:
                    continue
                if target.bbox[0] <= x <= target.bbox[2] and target.bbox[1] <= y <= target.bbox[3]:
                    continue
                if any(
                    obj.bbox[0] <= x <= obj.bbox[2]
                    and obj.bbox[1] <= y <= obj.bbox[3]
                    for obj in small_objects
                ):
                    continue
                if scene.grid[y][x] in floor_colors:
                    continue
                return False
        return True

    def _find_path_to_states(
        self,
        state_hash: str,
        target_states: set[str],
        allowed_actions: Sequence[str],
        max_depth: int = 32,
    ) -> list[str] | None:
        if state_hash in target_states:
            return []
        allowed = set(allowed_actions)
        queue: deque[tuple[str, list[str]]] = deque([(state_hash, [])])
        seen = {state_hash}
        while queue:
            current, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for name, next_state in self.memory.level.transition_graph.get(current, {}).items():
                if name not in allowed or next_state in seen:
                    continue
                next_path = [*path, name]
                if next_state in target_states:
                    return next_path
                seen.add(next_state)
                queue.append((next_state, next_path))
        return None

    def _frontier_navigation_action(
        self,
        state_hash: str,
        simple_legal: Sequence[str],
    ) -> str | None:
        targets: set[str] = set()
        known_states = set(self.memory.level.transition_graph) | set(
            self.memory.level.tried_actions_by_state
        )
        for known in known_states:
            tried = self.memory.level.tried_actions_by_state.get(known, set())
            noops = self.memory.level.noop_actions_by_state.get(known, set())
            if any(name not in tried and name not in noops for name in simple_legal):
                targets.add(known)
        path = self._find_path_to_states(state_hash, targets, simple_legal, max_depth=30)
        return path[0] if path else None

    def _graph_fallback_action(
        self,
        scene: SceneSnapshot,
        legal: Sequence[Any],
    ) -> Any:
        self.memory.game.controller_stats.unstructured_fallback_count += 1
        proposals = self._experiment_scheduler_proposals(scene, legal)
        chosen = self._choose_best_proposal(proposals, scene, legal, "NOT_FINISHED")
        selected = self._instantiate_proposal(chosen, scene, legal) if chosen else None
        if selected is not None:
            action, _proposal, _source = selected
            return action
        return self._emergency_safe_action(scene, legal, "NOT_FINISHED")

    def _make_valid_action(
        self,
        proposal: dict[str, Any] | None,
        scene: SceneSnapshot,
        legal: Sequence[Any],
        *,
        allow_reset: bool,
    ) -> Any | None:
        if not proposal:
            return None
        name = _short_string(proposal.get("name"), 40).upper()
        if name in {"CLICK", "MOVE", "GO", "PATHFIND", "NAVIGATE"}:
            return None
        if name == "ACTION7":
            source = _short_string(proposal.get("source"), 40)
            if source != "undo_rollback" or not self.undo_manager.can_emit(self.memory.level.total_action_count):
                return None
        if name == "RESET" and not allow_reset:
            return None
        action = get_action_by_name(name)
        if action is None:
            return None
        allow_disabled_action6_frontier = bool(
            name == "ACTION6"
            and _short_string(proposal.get("source"), 80) == "action6_frontier"
            and proposal.get("family_disabled_bypass")
        )
        if name != "RESET" and self._action_family_disabled(name) and not allow_disabled_action6_frontier:
            return None
        legal_by_value = {action_value(item): item for item in legal}
        if action_value(action) not in legal_by_value and name != "RESET":
            return None
        if name == "ACTION6":
            completed = self._complete_action6_proposal(dict(proposal), None, scene)
            if completed is None:
                return None
            proposal = completed
            x = self._coerce_int(proposal.get("x"))
            y = self._coerce_int(proposal.get("y"))
            if x is None or y is None or not (0 <= x < scene.width and 0 <= y < scene.height):
                return None
            return self._make_action6(legal_by_value.get(action_value(action), action), x, y, proposal)
        return self._attach_reasoning(legal_by_value.get(action_value(action), action), proposal)

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"[-+]?\d+", text):
                return int(text)
        return None

    def _make_action6(
        self,
        action: Any,
        x: int,
        y: int,
        proposal: dict[str, Any],
    ) -> Any:
        data = {"x": int(x), "y": int(y)}
        if hasattr(action, "validate_data"):
            try:
                valid = action.validate_data(data)
            except Exception:
                valid = True
            if valid is False:
                raise ObservationError(f"ACTION6 data rejected: {data}")
        made = None
        if hasattr(action, "set_data"):
            try:
                made = action.set_data(data)
            except Exception:
                made = None
        if made is None:
            made = action
        if action6_data(made) != data:
            for attr in ("action_data", "data"):
                try:
                    setattr(made, attr, dict(data))
                    if action6_data(made) == data:
                        break
                except Exception:
                    pass
        if action6_data(made) != data:
            base = get_action_by_name("ACTION6") or action
            made = ActionWithData(base, dict(data))
        proposal = {**proposal, "x": data["x"], "y": data["y"]}
        return self._attach_reasoning(made, proposal)

    def _attach_reasoning(
        self,
        action: Any,
        proposal: dict[str, Any],
    ) -> Any:
        reasoning = {
            "agent": self.config.agent_version,
            "purpose": _short_string(proposal.get("purpose"), 220),
            "expected_change": _short_string(proposal.get("expected_change"), 220),
            "target_object_id": _short_string(proposal.get("target_object_id"), 24).upper(),
        }
        if proposal.get("name") == "ACTION6" or proposal.get("x") is not None or proposal.get("y") is not None:
            reasoning["x"] = proposal.get("x")
            reasoning["y"] = proposal.get("y")
        for extra_key in ("name", "candidate_id", "candidate_kind", "relaxed_recovery"):
            if extra_key in proposal:
                reasoning[extra_key] = proposal.get(extra_key)
        try:
            setattr(action, "reasoning", reasoning)
            return action
        except Exception:
            data = action6_data(action) if action_name(action) == "ACTION6" else None
            if data is not None:
                base = getattr(action, "id", None) or get_action_by_name("ACTION6") or action
                return ActionWithData(base, data, reasoning)
        return action

    def _action_key_for_action(self, action: Any) -> str:
        name = action_name(action)
        if name != "ACTION6":
            return name
        data = action6_data(action)
        if data is None:
            reasoning = getattr(action, "reasoning", None)
            data = action6_data(ActionWithData(get_action_by_name("ACTION6") or action, reasoning)) if isinstance(reasoning, dict) else None
        if data is None:
            return "ACTION6:None,None"
        return f"ACTION6:{data['x']},{data['y']}"

    def _is_known_noop_action(
        self,
        action: Any,
        scene: SceneSnapshot,
    ) -> bool:
        return self._action_key_for_action(action) in self.memory.level.noop_actions_by_state.get(
            scene.state_hash,
            set(),
        )

    def _record_returned_action(
        self,
        action: Any,
        proposal: dict[str, Any],
        scene: SceneSnapshot,
        latest_frame: Any,
        *,
        source: str,
    ) -> None:
        name = action_name(action)
        if not proposal:
            reasoning = getattr(action, "reasoning", None)
            proposal = reasoning if isinstance(reasoning, dict) else {}
        key = self._action_key_for_action(action)
        level = self.memory.level
        if name == "RESET":
            level.controller_tick_count += 1
            level.terminal_reset_streak += 1
            if _short_string(proposal.get("reason"), 80) == "no_safe_non_reset_action":
                level.no_safe_recovery_streak += 1
                level.force_vlm_reason = "recovery_no_safe_action"
            level.pending_action = None
            self.logger.log_event(
                "terminal_reset_returned",
                {
                    "source": source,
                    "reason": _short_string(proposal.get("reason"), 160),
                    "streak": level.terminal_reset_streak,
                    "no_safe_streak": level.no_safe_recovery_streak,
                    "controller_step": self._controller_step(),
                    "recovery_mode": level.recovery_mode,
                    "recovery_until": level.recovery_until_step,
                },
            )
            return

        level.terminal_reset_streak = 0
        level.no_safe_recovery_streak = 0
        level.tried_actions_by_state.setdefault(scene.state_hash, set()).add(key)
        level.total_action_count += 1
        level.controller_tick_count = max(level.controller_tick_count + 1, level.total_action_count)
        level.life_action_count += 1
        level.actions_since_vlm += 1
        level.action_trace.append(key)
        self._note_fallback_selection(source, key, scene)

        x = y = None
        if name == "ACTION6" and ":" in key:
            try:
                x_text, y_text = key.split(":", 1)[1].split(",", 1)
                x, y = int(x_text), int(y_text)
            except Exception:
                x = y = None
        target = _short_string(
            proposal.get("target_object_id"),
            24,
        ).upper()
        expected_predicates = proposal.get("expected_predicates")
        if not isinstance(expected_predicates, list):
            expected_predicates = []
        available_before = [
            action_name(action)
            for action in normalize_legal_actions(
                getattr(latest_frame, "available_actions", None),
                self._env_action_space(),
                allow_env_fallback=False,
            )
        ]
        level.pending_action = PendingAction(
            name=name,
            x=x,
            y=y,
            purpose=_short_string(proposal.get("purpose"), 220),
            expected_change=_short_string(proposal.get("expected_change"), 220),
            target_object_id=target,
            scene_before=scene,
            source_frame_id=id(latest_frame),
            issued_call=self._call_index,
            source=source,
            plan_id=level.plan_id if source in {"vlm_plan", "symbolic_nav", "active_plan"} else None,
            source_frame_ref=latest_frame,
            expected_predicates=expected_predicates[:8],
            available_actions_before=available_before,
            vlm_mode=level.current_vlm_mode,
        )
        self.logger.log_event(
            "action",
            {
                "stage": level.stage.value,
                "state": scene.state_hash[:12],
                "action": key,
                "source": source,
                "target_object_id": target,
                "plan_id": level.plan_id if source in {"vlm_plan", "symbolic_nav", "active_plan"} else None,
                "plan_remaining": (
                    max(0, len(level.active_plan.steps) - level.active_plan.cursor)
                    if level.active_plan is not None
                    else 0
                ),
                "level": level.level_index,
                "attempt": level.attempt_index,
                "life": level.life_index,
                "life_step": level.life_action_count,
                "counter": scene.counter_value,
                "capacity": scene.counter_capacity,
                "lives": scene.life_count,
            },
        )

    def _advance_stage(
        self,
        result: VLMResult | None,
        scene: SceneSnapshot,
    ) -> None:
        level = self.memory.level
        if level.awaiting_reset:
            level.stage = V1Phase.FAILURE_RECOVERY
        elif level.active_plan is not None:
            level.stage = V1Phase.EXECUTE_PLAN
        elif level.total_action_count <= 2 and (self.memory.game.goal_schemas or self.memory.game.level_outcomes):
            level.stage = V1Phase.TRANSFER_BOOTSTRAP
        elif not level.controlled_object_id and level.mechanism.mode not in {"click_selection", "global_transform"}:
            level.stage = V1Phase.ACTION_GROUNDING
        elif level.local_goal or level.goal_instantiation:
            level.stage = V1Phase.PLAN_SYNTHESIS
        else:
            level.stage = V1Phase.MECHANIC_EXPLORATION

    def _action6_candidates(
        self,
        scene: SceneSnapshot,
    ) -> list[tuple[int, int]]:
        candidates = self._action6_candidate_objects(scene)
        return [(candidate.x, candidate.y) for candidate in candidates] or [(scene.width // 2, scene.height // 2)]

    def _reset_action(self) -> Any:
        return get_action_by_name("RESET") or getattr(GameAction, "RESET")

    def _emergency_action(
        self,
        latest_frame: Any,
        legal: Sequence[Any],
    ) -> Any:
        state = state_name(getattr(latest_frame, "state", None))
        if state in {"NOT_PLAYED", "GAME_OVER", "WIN"}:
            return self._reset_action()
        try:
            scene = self.observer.scene_from_frame(latest_frame)
            semantic = self._semantic_fallback_action(scene, legal)
            if semantic is not None:
                return semantic[0]
            return self._graph_fallback_action(scene, legal)
        except Exception:
            return self._reset_action()


__all__ = [
    "Action6Candidate",
    "ActionIntent",
    "CompileResult",
    "CompileStatus",
    "ActionWithData",
    "Action6Memory",
    "Action6ProbeRecord",
    "ActionKnowledge",
    "ActionProposal",
    "AgentConfig",
    "AgentRuntimeMemory",
    "ComponentObservation",
    "ControllerStats",
    "DecisionLogger",
    "EventRecord",
    "ExecutableStrategyRule",
    "FailureModel",
    "IntentType",
    "GameAction",
    "GameGoalSchema",
    "GameMemory",
    "GameState",
    "Hypothesis",
    "LevelGoalInstantiation",
    "LevelOutcomeMemory",
    "LevelTheory",
    "LevelMechanismState",
    "LevelMemory",
    "MyAgent",
    "NearSuccessRoute",
    "ObjectObservation",
    "ObservationError",
    "Observer",
    "OpenAICompatibleBackend",
    "PendingAction",
    "Qwen35Backend",
    "RejectReason",
    "SceneSnapshot",
    "TargetFailureRecord",
    "TransitionReport",
    "UndoContext",
    "UndoManager",
    "V1Phase",
    "ValidationResult",
    "VLMMode",
    "action6_data",
    "VLMRequest",
    "VLMResult",
    "action_name",
    "action_value",
    "get_action_by_id",
    "get_action_by_name",
    "make_vlm_backend",
    "normalize_legal_actions",
    "normalize_one_action",
    "parse_vlm_result",
    "render_grid",
    "state_name",
    "vlm_uses_remote_api",
]
