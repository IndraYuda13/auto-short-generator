"""Pipeline package for Auto Short Generator (Blueprint Bab 18, 21, 22)."""

from pipeline.state_machine import (
    PipelineStatus,
    HAPPY_STATES,
    TERMINAL_REJECT_STATES,
    TERMINAL_STATES,
    VALID_TRANSITIONS,
    InvalidStateTransitionError,
    GuardInvariantError,
    StateTransition,
    parse_state,
    can_transition,
    validate_transition,
    record_transition,
    transition,
    is_terminal,
    is_happy,
    is_reject,
    utc_now_iso,
)

__all__ = [
    "PipelineStatus",
    "HAPPY_STATES",
    "TERMINAL_REJECT_STATES",
    "TERMINAL_STATES",
    "VALID_TRANSITIONS",
    "InvalidStateTransitionError",
    "GuardInvariantError",
    "StateTransition",
    "parse_state",
    "can_transition",
    "validate_transition",
    "record_transition",
    "transition",
    "is_terminal",
    "is_happy",
    "is_reject",
    "utc_now_iso",
    "AutoClipperOrchestrator",
    "PipelineResult",
]


def __getattr__(name: str):
    if name in ("AutoClipperOrchestrator", "PipelineResult"):
        from pipeline.orchestrator import AutoClipperOrchestrator, PipelineResult
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
