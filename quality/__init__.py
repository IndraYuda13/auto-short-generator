"""Three-Tier Quality Control Gate Package for Auto Short Generator (Blueprint Bab 16).

Modules:
- technical_qc (Bab 16.1): ffprobe spec verification & stream integrity
- visual_qc (Bab 16.2): OpenCV local frame sampling, blank/flash, face framing & subtitle safety
- perceptual_qc (Bab 16.3): Gemini Visual Director multimodal evaluation via 9router
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from pydantic import BaseModel, Field

from quality.technical_qc import (
    TechnicalQC,
    TechnicalQCResult,
    evaluate_technical_qc,
)
from quality.visual_qc import (
    VisualQC,
    VisualQCResult,
    evaluate_visual_qc,
)
from quality.perceptual_qc import (
    PerceptualQC,
    PerceptualQCResult,
    evaluate_perceptual_qc,
)


class ThreeTierQCReport(BaseModel):
    """Consolidated report across all three Quality Control tiers."""
    passed: bool = Field(..., description="Overall pass status (all 3 tiers must pass)")
    publishable: bool = Field(..., description="Approved for automatic platform publishing")
    technical: TechnicalQCResult = Field(..., description="Technical QC result (Tier 1)")
    visual: VisualQCResult = Field(..., description="Visual QC result (Tier 2)")
    perceptual: PerceptualQCResult = Field(..., description="Perceptual QC result (Tier 3)")
    errors: List[str] = Field(default_factory=list, description="Consolidated list of defects")


class ThreeTierQCGate:
    """Orchestrates the complete Three-Tier Quality Control Gate (Blueprint Bab 16)."""

    def __init__(
        self,
        technical_qc: Optional[TechnicalQC] = None,
        visual_qc: Optional[VisualQC] = None,
        perceptual_qc: Optional[PerceptualQC] = None,
    ):
        self.technical_qc = technical_qc or TechnicalQC()
        self.visual_qc = visual_qc or VisualQC()
        self.perceptual_qc = perceptual_qc or PerceptualQC()

    def evaluate(
        self,
        video_path: Union[str, Path],
        transcript_text: str = "",
        edit_plan: Optional[Any] = None,
        repair_handler: Optional[Callable[[List[str], Any], Tuple[bool, Optional[str], Optional[str]]]] = None,
    ) -> ThreeTierQCReport:
        """Executes Tier 1 (Technical), Tier 2 (Visual), and Tier 3 (Perceptual) QC in sequence."""
        # Tier 1: Technical QC
        tech_res = self.technical_qc.evaluate(video_path)

        # Tier 2: Visual QC
        sub_policy = getattr(edit_plan, "subtitle_policy", None) if edit_plan else None
        layout_mode = getattr(edit_plan, "layout", None) if edit_plan else None
        vis_res = self.visual_qc.evaluate(video_path, subtitle_policy=sub_policy, layout=layout_mode)

        # Tier 3: Perceptual QC
        perc_res = self.perceptual_qc.evaluate(
            video_path=video_path,
            transcript_text=transcript_text,
            edit_plan=edit_plan,
            technical_qc=tech_res,
            visual_qc=vis_res,
            repair_handler=repair_handler,
        )

        all_errors: List[str] = []
        all_errors.extend(tech_res.errors)
        all_errors.extend(vis_res.errors)
        all_errors.extend(perc_res.blocking_issues)

        passed = tech_res.passed and vis_res.passed and perc_res.passed
        publishable = passed and perc_res.publishable

        return ThreeTierQCReport(
            passed=passed,
            publishable=publishable,
            technical=tech_res,
            visual=vis_res,
            perceptual=perc_res,
            errors=all_errors,
        )


def evaluate_three_tier_qc(
    video_path: Union[str, Path],
    transcript_text: str = "",
    edit_plan: Optional[Any] = None,
) -> ThreeTierQCReport:
    """Convenience function to evaluate full three-tier QC pipeline."""
    gate = ThreeTierQCGate()
    return gate.evaluate(video_path, transcript_text=transcript_text, edit_plan=edit_plan)


__all__ = [
    "TechnicalQC",
    "TechnicalQCResult",
    "evaluate_technical_qc",
    "VisualQC",
    "VisualQCResult",
    "evaluate_visual_qc",
    "PerceptualQC",
    "PerceptualQCResult",
    "evaluate_perceptual_qc",
    "ThreeTierQCGate",
    "ThreeTierQCReport",
    "evaluate_three_tier_qc",
]
