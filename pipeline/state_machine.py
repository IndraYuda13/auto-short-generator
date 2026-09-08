"""Pipeline State Machine & Transition Validator (Blueprint Bab 18 & Bab 21).

Enforces strict lifecycle state transitions and guard invariants for the Auto Clipper pipeline:
1. 11 Happy States:
   - discovered
   - eligible
   - transcribed
   - candidates_found
   - candidate_selected
   - visual_verified
   - rendering
   - rendered
   - qc_passed
   - uploading
   - completed

2. 6 Terminal Reject States:
   - rejected_language
   - no_good_clip
   - rejected_visual
   - render_failed
   - qc_failed
   - upload_failed

Guard Invariants:
- Mandatory Guard: Cannot transition to 'uploading' unless current state is 'qc_passed'.
- Terminal Guard: Terminal states (rejects and completed) cannot transition to any other state.
- Pipeline Order Guard: States cannot skip mandatory pipeline verification stages.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Set, Union
from pydantic import BaseModel, Field


class PipelineStatus(str, Enum):
    """11 Happy states + 6 Terminal Reject states for the Auto Clipper pipeline."""

    # --- 11 Happy Path States ---
    DISCOVERED = "discovered"
    ELIGIBLE = "eligible"
    TRANSCRIBED = "transcribed"
    CANDIDATES_FOUND = "candidates_found"
    CANDIDATE_SELECTED = "candidate_selected"
    VISUAL_VERIFIED = "visual_verified"
    RENDERING = "rendering"
    RENDERED = "rendered"
    QC_PASSED = "qc_passed"
    UPLOADING = "uploading"
    COMPLETED = "completed"

    # --- 6 Terminal Reject States ---
    REJECTED_LANGUAGE = "rejected_language"
    NO_GOOD_CLIP = "no_good_clip"
    REJECTED_VISUAL = "rejected_visual"
    RENDER_FAILED = "render_failed"
    QC_FAILED = "qc_failed"
    UPLOAD_FAILED = "upload_failed"


# Categorization Sets
HAPPY_STATES: Set[PipelineStatus] = {
    PipelineStatus.DISCOVERED,
    PipelineStatus.ELIGIBLE,
    PipelineStatus.TRANSCRIBED,
    PipelineStatus.CANDIDATES_FOUND,
    PipelineStatus.CANDIDATE_SELECTED,
    PipelineStatus.VISUAL_VERIFIED,
    PipelineStatus.RENDERING,
    PipelineStatus.RENDERED,
    PipelineStatus.QC_PASSED,
    PipelineStatus.UPLOADING,
    PipelineStatus.COMPLETED,
}

TERMINAL_REJECT_STATES: Set[PipelineStatus] = {
    PipelineStatus.REJECTED_LANGUAGE,
    PipelineStatus.NO_GOOD_CLIP,
    PipelineStatus.REJECTED_VISUAL,
    PipelineStatus.RENDER_FAILED,
    PipelineStatus.QC_FAILED,
    PipelineStatus.UPLOAD_FAILED,
}

TERMINAL_STATES: Set[PipelineStatus] = TERMINAL_REJECT_STATES | {PipelineStatus.COMPLETED}


# Strict Transition Graph
VALID_TRANSITIONS: Dict[PipelineStatus, Set[PipelineStatus]] = {
    PipelineStatus.DISCOVERED: {
        PipelineStatus.ELIGIBLE,
        PipelineStatus.REJECTED_LANGUAGE,
    },
    PipelineStatus.ELIGIBLE: {
        PipelineStatus.TRANSCRIBED,
        PipelineStatus.REJECTED_LANGUAGE,
    },
    PipelineStatus.TRANSCRIBED: {
        PipelineStatus.CANDIDATES_FOUND,
        PipelineStatus.REJECTED_LANGUAGE,
        PipelineStatus.NO_GOOD_CLIP,
    },
    PipelineStatus.CANDIDATES_FOUND: {
        PipelineStatus.CANDIDATE_SELECTED,
        PipelineStatus.NO_GOOD_CLIP,
    },
    PipelineStatus.CANDIDATE_SELECTED: {
        PipelineStatus.VISUAL_VERIFIED,
        PipelineStatus.REJECTED_VISUAL,
        PipelineStatus.NO_GOOD_CLIP,
    },
    PipelineStatus.VISUAL_VERIFIED: {
        PipelineStatus.RENDERING,
        PipelineStatus.REJECTED_VISUAL,
    },
    PipelineStatus.RENDERING: {
        PipelineStatus.RENDERED,
        PipelineStatus.RENDER_FAILED,
    },
    PipelineStatus.RENDERED: {
        PipelineStatus.QC_PASSED,
        PipelineStatus.QC_FAILED,
    },
    # Guard invariant: 'uploading' ONLY accessible from 'qc_passed'
    PipelineStatus.QC_PASSED: {
        PipelineStatus.UPLOADING,
        PipelineStatus.COMPLETED,  # Allowed for dry-run or local-only processing
    },
    PipelineStatus.UPLOADING: {
        PipelineStatus.COMPLETED,
        PipelineStatus.UPLOAD_FAILED,
    },
    # Terminal states have no outbound transitions
    PipelineStatus.COMPLETED: set(),
    PipelineStatus.REJECTED_LANGUAGE: set(),
    PipelineStatus.NO_GOOD_CLIP: set(),
    PipelineStatus.REJECTED_VISUAL: set(),
    PipelineStatus.RENDER_FAILED: set(),
    PipelineStatus.QC_FAILED: set(),
    PipelineStatus.UPLOAD_FAILED: set(),
}


class InvalidStateTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""

    def __init__(self, current_state: PipelineStatus, target_state: PipelineStatus, reason: str):
        self.current_state = current_state
        self.target_state = target_state
        self.reason = reason
        super().__init__(
            f"Invalid state transition from '{current_state.value}' to '{target_state.value}': {reason}"
        )


class GuardInvariantError(InvalidStateTransitionError):
    """Raised when a specific pipeline safety invariant is violated."""
    pass


def parse_state(state: Union[str, PipelineStatus]) -> PipelineStatus:
    """Safely converts string or enum to PipelineStatus."""
    if isinstance(state, PipelineStatus):
        return state
    try:
        return PipelineStatus(str(state).lower())
    except ValueError:
        raise ValueError(f"Unknown pipeline state: '{state}'")


def can_transition(
    current_state: Union[str, PipelineStatus],
    target_state: Union[str, PipelineStatus],
) -> bool:
    """Returns True if the transition from current_state to target_state is permitted."""
    try:
        curr = parse_state(current_state)
        target = parse_state(target_state)
    except ValueError:
        return False

    # Guard Invariant: Target 'uploading' is STRICTLY forbidden unless current is 'qc_passed'
    if target == PipelineStatus.UPLOADING and curr != PipelineStatus.QC_PASSED:
        return False

    # Terminal state check
    if curr in TERMINAL_STATES:
        return False

    allowed_next = VALID_TRANSITIONS.get(curr, set())
    return target in allowed_next


def validate_transition(
    current_state: Union[str, PipelineStatus],
    target_state: Union[str, PipelineStatus],
) -> PipelineStatus:
    """Validates transition and returns the new PipelineStatus.

    Raises GuardInvariantError or InvalidStateTransitionError if the transition is disallowed.
    """
    curr = parse_state(current_state)
    target = parse_state(target_state)

    # Invariant Guard: Terminal state immutability
    if curr in TERMINAL_STATES:
        raise InvalidStateTransitionError(
            current_state=curr,
            target_state=target,
            reason=f"Current state '{curr.value}' is a terminal state and cannot transition further.",
        )

    # Invariant Guard: Transition to 'uploading' ONLY from 'qc_passed'
    if target == PipelineStatus.UPLOADING and curr != PipelineStatus.QC_PASSED:
        raise GuardInvariantError(
            current_state=curr,
            target_state=target,
            reason="Guard Invariant Violated: Cannot transition to 'uploading' unless current state is 'qc_passed'!",
        )

    allowed_next = VALID_TRANSITIONS.get(curr, set())
    if target not in allowed_next:
        raise InvalidStateTransitionError(
            current_state=curr,
            target_state=target,
            reason=f"Transition not allowed in state graph. Allowed next states: {[s.value for s in allowed_next]}",
        )

    return target


def is_terminal(state: Union[str, PipelineStatus]) -> bool:
    """Checks whether the state is a terminal state."""
    return parse_state(state) in TERMINAL_STATES


def is_happy(state: Union[str, PipelineStatus]) -> bool:
    """Checks whether the state belongs to the 11 happy path states."""
    return parse_state(state) in HAPPY_STATES


def is_reject(state: Union[str, PipelineStatus]) -> bool:
    """Checks whether the state is a terminal rejection state."""
    return parse_state(state) in TERMINAL_REJECT_STATES


def utc_now_iso() -> str:
    """Returns current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class StateTransition(BaseModel):
    """Represents a validated transition event in the pipeline lifecycle."""
    from_state: PipelineStatus
    to_state: PipelineStatus
    timestamp: str = Field(default_factory=utc_now_iso)
    is_terminal: bool = False
    reason: Optional[str] = None
    error_message: Optional[str] = None


def record_transition(
    current_state: Union[str, PipelineStatus],
    target_state: Union[str, PipelineStatus],
    reason: Optional[str] = None,
    error_message: Optional[str] = None,
) -> StateTransition:
    """Validates the state transition and records reason and timestamp for terminal states.

    Raises:
        GuardInvariantError: If attempting to transition to 'uploading' when current != 'qc_passed'.
        InvalidStateTransitionError: If attempting to transition out of a terminal state or illegal edge.

    When target_state is a terminal state (any rejection or completed),
    explicitly captures reason and UTC timestamp.
    """
    curr = parse_state(current_state)
    target = parse_state(target_state)

    validated_target = validate_transition(curr, target)
    now = utc_now_iso()
    term = is_terminal(validated_target)

    # Invariant: Terminal transitions must record reason & timestamp
    recorded_reason = reason
    if term and not recorded_reason:
        if validated_target == PipelineStatus.COMPLETED:
            recorded_reason = "Completed successfully."
        else:
            recorded_reason = error_message or f"Terminated with rejection state: '{validated_target.value}'"

    return StateTransition(
        from_state=curr,
        to_state=validated_target,
        timestamp=now,
        is_terminal=term,
        reason=recorded_reason,
        error_message=error_message,
    )


# Alias
transition = record_transition

