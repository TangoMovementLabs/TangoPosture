"""Couple analysis and Auto dancer-mode detection package."""

from .classification import classify_opening_window, detect_post_window_scene_changes
from .csv_export import export_couple_csv
from .measurements import (
    compute_centred_regression_velocity,
    rebuild_couple_result_with_annotations,
    run_couple_analysis,
)
from .models import (
    AutoDecisionResult,
    CandidatePreview,
    CoupleAnalysisRequest,
    CoupleAnalysisResult,
    CoupleAnnotationItem,
    CoupleAnnotations,
    CoupleEvent,
    CoupleSummaryValue,
    DancerAnchor,
    DancerId,
    DancerMode,
    PairedFrameSample,
    PivotFootBehavior,
    PlannedTransitionMark,
    ResolvedDancerMode,
    SampleQualityStatus,
    TrackedDancerSample,
)
from .overlay import render_couple_frame_overlay, smooth_couple_frame_overlays
from .tracking import (
    CoupleTrackingOutput,
    canonical_xy,
    extract_single_target_observations,
    filter_and_deduplicate_poses,
    resolve_close_embrace_frame_poses,
    track_couple_frames,
)

__all__ = [
    "AutoDecisionResult",
    "CandidatePreview",
    "CoupleAnalysisRequest",
    "CoupleAnalysisResult",
    "CoupleAnnotationItem",
    "CoupleAnnotations",
    "CoupleEvent",
    "CoupleSummaryValue",
    "CoupleTrackingOutput",
    "DancerAnchor",
    "DancerId",
    "DancerMode",
    "PairedFrameSample",
    "PivotFootBehavior",
    "PlannedTransitionMark",
    "ResolvedDancerMode",
    "SampleQualityStatus",
    "TrackedDancerSample",
    "canonical_xy",
    "classify_opening_window",
    "compute_centred_regression_velocity",
    "detect_post_window_scene_changes",
    "export_couple_csv",
    "extract_single_target_observations",
    "filter_and_deduplicate_poses",
    "rebuild_couple_result_with_annotations",
    "render_couple_frame_overlay",
    "resolve_close_embrace_frame_poses",
    "run_couple_analysis",
    "smooth_couple_frame_overlays",
    "track_couple_frames",
]
