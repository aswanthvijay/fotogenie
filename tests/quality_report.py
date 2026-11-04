"""Real offline quality report.

This replaces the old hardcoded 150-point checker. Only checks that can be
measured on a reference/output pair are reported; nothing is marked PASS
without a computed value. Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import config
from backend.analyzer import analyze_reference
from backend.face_guard import face_metrics, quality_ok, verify


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure identity, skin, texture and framing.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--full-body", action="store_true", help="Require a full-body framing check.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    ref_bytes = args.reference.read_bytes()
    gen_bytes = args.generated.read_bytes()
    profile = analyze_reference(ref_bytes)
    metrics = verify(profile.face_crop_png, gen_bytes, profile.skin_lab)
    gen_metrics = face_metrics(gen_bytes)

    checks = []

    def add(name, passed, measured, threshold):
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "measured": measured,
                "threshold": threshold,
            }
        )

    add("face_detected", metrics["face_detected"], metrics["face_detected"], "True")
    add("face_count", metrics["face_count"] == 1, metrics["face_count"], "== 1")
    add("detection_score", metrics["det_score"] >= 0.6, round(metrics["det_score"], 3), ">= 0.6")
    add(
        "identity_similarity",
        metrics["identity_similarity"] >= config.RESCUE_MIN_SIM,
        metrics["identity_similarity"],
        f">= {config.RESCUE_MIN_SIM}",
    )
    ref_texture = metrics.get("reference_texture") or profile.face_texture
    add(
        "face_texture",
        config.MIN_RESCUE_TEXTURE <= metrics["texture"] <= config.MAX_FACE_TEXTURE,
        round(metrics["texture"], 1),
        f"{config.MIN_RESCUE_TEXTURE:.0f}-{config.MAX_FACE_TEXTURE:.0f}",
    )
    add(
        "face_texture_target",
        metrics["texture"] >= 100.0,
        round(metrics["texture"], 1),
        ">= 100 (photographic detail)",
    )
    add(
        "rescue_quality_gate",
        quality_ok(metrics, ref_texture, min_sim=config.RESCUE_MIN_SIM),
        quality_ok(metrics, ref_texture, min_sim=config.RESCUE_MIN_SIM),
        "identity + texture floor + skin chroma",
    )
    # 3D delta-E is reported for context, but scene lighting legitimately changes L*.
    # The actionable, lighting-independent gate is the a*/b* chroma difference.
    if metrics["skin_delta_chroma"] is not None:
        add(
            "skin_delta_chroma",
            metrics["skin_delta_chroma"] <= config.MAX_SKIN_DELTA_CHROMA,
            metrics["skin_delta_chroma"],
            f"<= {config.MAX_SKIN_DELTA_CHROMA}",
        )
    if args.full_body and gen_metrics["bbox"]:
        x1, y1, x2, y2 = gen_metrics["bbox"]
        ratio = (y2 - y1) / float(max(1, gen_metrics["height"]))
        add("full_body_framing", ratio <= 0.35, round(ratio, 3), "<= 0.35")

    relative_min = max(config.MIN_RESCUE_TEXTURE, ref_texture * 0.75) if ref_texture else None
    report = {
        "reference": str(args.reference),
        "generated": str(args.generated),
        "image_size": [gen_metrics["width"], gen_metrics["height"]],
        "checks": checks,
        "passed": sum(1 for c in checks if c["passed"]),
        "total": len(checks),
        "notes": {
            "reference_texture": round(float(ref_texture or 0), 1),
            "generated_texture": round(float(metrics["texture"]), 1),
            "relative_texture_target": round(relative_min, 1) if relative_min else None,
            "skin_delta_e": metrics.get("skin_delta_e"),
            "skin_delta_chroma": metrics.get("skin_delta_chroma"),
        },
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
