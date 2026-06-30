"""
Data models for the Automated Insurance Claim Reviewer.

Defines all input/output structures using Pydantic for validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────


class ClaimObject(str, Enum):
    CAR = "car"
    LAPTOP = "laptop"
    PACKAGE = "package"


class IssueType(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    BROKEN_PART = "broken_part"
    MISSING_PART = "missing_part"
    TORN_PACKAGING = "torn_packaging"
    CRUSHED_PACKAGING = "crushed_packaging"
    WATER_DAMAGE = "water_damage"
    STAIN = "stain"
    NONE = "none"
    UNKNOWN = "unknown"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class RiskFlag(str, Enum):
    BLURRY_IMAGE = "blurry_image"
    CROPPED_OR_OBSTRUCTED = "cropped_or_obstructed"
    LOW_LIGHT_OR_GLARE = "low_light_or_glare"
    WRONG_ANGLE = "wrong_angle"
    WRONG_OBJECT = "wrong_object"
    WRONG_OBJECT_PART = "wrong_object_part"
    DAMAGE_NOT_VISIBLE = "damage_not_visible"
    CLAIM_MISMATCH = "claim_mismatch"
    POSSIBLE_MANIPULATION = "possible_manipulation"
    NON_ORIGINAL_IMAGE = "non_original_image"
    TEXT_INSTRUCTION_PRESENT = "text_instruction_present"
    USER_HISTORY_RISK = "user_history_risk"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


# ── Car Parts ──────────────────────────────────────────────────────────────────


class CarPart(str, Enum):
    FRONT_BUMPER = "front_bumper"
    REAR_BUMPER = "rear_bumper"
    DOOR = "door"
    HOOD = "hood"
    WINDSHIELD = "windshield"
    SIDE_MIRROR = "side_mirror"
    HEADLIGHT = "headlight"
    TAILLIGHT = "taillight"
    FENDER = "fender"
    QUARTER_PANEL = "quarter_panel"
    BODY = "body"
    UNKNOWN = "unknown"


# ── Laptop Parts ───────────────────────────────────────────────────────────────


class LaptopPart(str, Enum):
    SCREEN = "screen"
    KEYBOARD = "keyboard"
    TRACKPAD = "trackpad"
    HINGE = "hinge"
    LID = "lid"
    CORNER = "corner"
    PORT = "port"
    BASE = "base"
    BODY = "body"
    UNKNOWN = "unknown"


# ── Package Parts ──────────────────────────────────────────────────────────────


class PackagePart(str, Enum):
    BOX = "box"
    PACKAGE_CORNER = "package_corner"
    PACKAGE_SIDE = "package_side"
    SEAL = "seal"
    LABEL = "label"
    CONTENTS = "contents"
    ITEM = "item"
    UNKNOWN = "unknown"


# ── Input Models ───────────────────────────────────────────────────────────────


class UserHistory(BaseModel):
    """Past claim history for the user."""

    past_claim_count: int = Field(
        default=0,
        description="Total number of past claims submitted by this user.",
    )
    accepted_count: int = Field(
        default=0,
        description="Number of previously accepted claims.",
    )
    rejected_count: int = Field(
        default=0,
        description="Number of previously rejected claims.",
    )
    manual_count: int = Field(
        default=0,
        description="Number of claims escalated to manual review.",
    )
    history_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Risk flags from history analysis, e.g. "
            "'user_history_risk', 'manual_review_required'."
        ),
    )
    history_summary: str = Field(
        default="",
        description="Free-text summary of the user's claim history.",
    )


class EvidenceRequirements(BaseModel):
    """Minimum visual evidence bar for this claim type."""

    min_resolution: str = Field(
        default="640x480",
        description="Minimum acceptable image resolution (WxH).",
    )
    required_visibility: str = Field(
        default="claimed_part_clearly_visible",
        description="What must be visible in the image.",
    )
    acceptable_formats: list[str] = Field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp"],
        description="Accepted image file formats.",
    )
    notes: str = Field(
        default="",
        description="Additional evidence requirement notes.",
    )


class ClaimInput(BaseModel):
    """Complete input payload for a single claim review."""

    claim_object: ClaimObject = Field(
        ...,
        description="The type of object being claimed: car, laptop, or package.",
    )
    user_claim: str = Field(
        ...,
        description=(
            "Chat transcript from which to extract the final specific claim."
        ),
    )
    image_paths: list[str] = Field(
        ...,
        min_length=1,
        description="One or more file paths to claim evidence images.",
    )
    user_history: UserHistory = Field(
        default_factory=UserHistory,
        description="User's past claim history context.",
    )
    evidence_requirements: EvidenceRequirements = Field(
        default_factory=EvidenceRequirements,
        description="Minimum evidence standard for this claim type.",
    )


# ── Output Model ───────────────────────────────────────────────────────────────


class ClaimReviewResult(BaseModel):
    """Structured verdict from the automated claim reviewer."""

    evidence_standard_met: str = Field(
        ...,
        description="'true' or 'false' — whether the image set meets the evidence bar.",
    )
    evidence_standard_met_reason: str = Field(
        ...,
        description="One sentence explaining why the evidence standard is or isn't met.",
    )
    risk_flags: str = Field(
        ...,
        description="Semicolon-separated risk flags, or 'none'.",
    )
    issue_type: str = Field(
        ...,
        description="Closest damage type: dent, scratch, crack, etc.",
    )
    object_part: str = Field(
        ...,
        description="The part of the object that is claimed damaged.",
    )
    claim_status: str = Field(
        ...,
        description="'supported', 'contradicted', or 'not_enough_information'.",
    )
    claim_status_justification: str = Field(
        ...,
        description="1-2 sentence justification grounded in specific images.",
    )
    supporting_image_ids: str = Field(
        ...,
        description="Semicolon-separated image IDs that support the decision, or 'none'.",
    )
    valid_image: str = Field(
        ...,
        description="'true' or 'false' — whether the image set is usable for review.",
    )
    severity: str = Field(
        ...,
        description="none | low | medium | high | unknown.",
    )


class BatchReviewResult(BaseModel):
    """Wrapper for batch of claim review results."""

    results: list[ClaimReviewResult] = Field(
        ...,
        description="The review results for the claims in the batch, in the exact same order as requested.",
    )

