from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from visualact_r.data.task_policy import normalized_center, point_in_polygon
from visualact_r.schemas import Action, AgentStep, AgentTrace, CropRecord, TaskRecord

VALID_TOOLS = {"INSPECT_REGION", "FINALIZE", "ROUTE_REVIEW"}
VALID_ACTIONS = {"AUTO_CLEAR", "REVIEW", "ESCALATE"}


def _strict_json_object(text: str) -> dict[str, Any]:
    """Accept a JSON object or one complete ```json fenced object, and reject all extra prose."""
    candidate = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        candidate = fenced.group(1)

    def _reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant is forbidden: {value}")

    decoder = json.JSONDecoder(parse_constant=_reject_constant)
    payload, position = decoder.raw_decode(candidate)
    if candidate[position:].strip():
        raise ValueError("Typed output contains text outside the JSON object")
    if not isinstance(payload, dict):
        raise ValueError("Typed output must be a JSON object")
    return payload


def parse_typed_output(text: str) -> dict[str, Any]:
    """Validate a narrow fail-closed JSON contract for local VLM actions.

    The prompt prohibits extra keys, so accepting an underspecified or over-specified
    payload would silently weaken the serving boundary. Missing mandatory audit fields
    become an explicit REVIEW through the Qwen adapter's fail-closed path.
    """
    payload = _strict_json_object(text)
    tool = payload.get("tool")
    if tool not in VALID_TOOLS:
        raise ValueError(f"Unknown tool: {tool}")
    allowed_keys = {
        "INSPECT_REGION": {"tool", "region_id", "confidence", "reason_code"},
        "FINALIZE": {"tool", "decision", "confidence", "evidence_ids", "reason_code"},
        "ROUTE_REVIEW": {"tool", "confidence", "evidence_ids", "reason_code"},
    }[tool]
    extras = sorted(set(payload) - allowed_keys)
    if extras:
        raise ValueError(f"Typed output contains unsupported keys: {extras}")
    if tool == "FINALIZE":
        required = {"decision", "confidence", "evidence_ids", "reason_code"}
        if not required.issubset(payload):
            raise ValueError(
                f"FINALIZE is missing required keys: {sorted(required - set(payload))}"
            )
        if payload.get("decision") not in VALID_ACTIONS:
            raise ValueError("FINALIZE requires a valid decision")
    elif tool == "ROUTE_REVIEW":
        required = {"confidence", "reason_code"}
        if not required.issubset(payload):
            raise ValueError(
                f"ROUTE_REVIEW is missing required keys: {sorted(required - set(payload))}"
            )
    else:
        required = {"region_id", "confidence", "reason_code"}
        if not required.issubset(payload):
            raise ValueError(
                f"INSPECT_REGION is missing required keys: {sorted(required - set(payload))}"
            )
        if not isinstance(payload.get("region_id"), str) or not payload["region_id"]:
            raise ValueError("INSPECT_REGION requires a non-empty region_id")
    if "confidence" in payload:
        try:
            confidence = float(payload["confidence"])
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be numeric") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        payload["confidence"] = confidence
    if "evidence_ids" in payload:
        evidence_ids = payload["evidence_ids"]
        if not isinstance(evidence_ids, list) or not all(
            isinstance(x, str) and x for x in evidence_ids
        ):
            raise ValueError("evidence_ids must be a list of non-empty strings")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence_ids must not contain duplicates")
    if not isinstance(payload.get("reason_code"), str) or not payload["reason_code"].strip():
        raise ValueError("reason_code must be a non-empty string")
    return payload


@dataclass
class MockVisualAgent:
    """Deterministic fixture-only agent exercising the typed-tool contract.

    It reads detector/CEF crop provenance only. It cannot read TaskRecord GT labels or supporting IDs.
    """

    candidate_id: str

    def rollout(
        self, record: TaskRecord, crops: list[CropRecord], config: dict[str, Any]
    ) -> AgentTrace:
        start = time.perf_counter()
        steps: list[AgentStep] = []
        primary_crops = []
        for crop in crops:
            x, y = normalized_center(crop.box_xyxy, record.width, record.height)
            if point_in_polygon(x, y, config["task_policy"]["primary_zone"]):
                primary_crops.append(crop)
        if crops:
            chosen = primary_crops[0] if primary_crops else crops[0]
            steps.append(
                AgentStep(
                    step=1,
                    typed_action="INSPECT_REGION",
                    region_id=chosen.crop_id,
                    decision=None,
                    confidence=min(0.95, 0.50 + 0.5 * chosen.coverage_proxy),
                    schema_valid=True,
                    reason_code="candidate_evidence_selected",
                    latency_ms=0.0,
                )
            )
        max_area = max(
            (x.box_xyxy.area / (record.width * record.height) for x in primary_crops), default=0.0
        )
        max_score = max((x.coverage_proxy for x in crops), default=0.0)
        if (
            primary_crops
            and max_area >= config["task_policy"]["primary_large_area_fraction"] * 0.55
        ):
            decision: Action = "ESCALATE"
            reason = "large_primary_evidence"
        elif primary_crops or max_score >= 0.55:
            decision = "REVIEW"
            reason = "ambiguous_or_contextual_evidence"
        else:
            decision = "AUTO_CLEAR"
            reason = "no_high_value_evidence"
        confidence = float(min(0.98, max(0.45, 0.55 + 0.35 * max_score)))
        steps.append(
            AgentStep(
                step=len(steps) + 1,
                typed_action="FINALIZE",
                region_id=None,
                decision=decision,
                confidence=confidence,
                schema_valid=True,
                reason_code=reason,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        )
        return AgentTrace(
            trace_id=f"{record.image_id}:{self.candidate_id}",
            image_id=record.image_id,
            task_id=record.task_id,
            candidate_id=self.candidate_id,
            visible_crop_ids=[x.crop_id for x in crops],
            steps=steps,
            final_decision=decision,
            final_confidence=confidence,
            schema_valid=True,
            latency_ms=(time.perf_counter() - start) * 1000,
            final_evidence_ids=[steps[0].region_id] if steps and steps[0].region_id else [],
        )
