"""Two-pass auto-gated Motion Viterbi reranking.

This is an add-only experimental script. It keeps all reference flights in the
candidate set, but automatically downweights reference dataset/segment choices
that look temporally unstable in the first-pass path.

It is designed for GNSS-denied-style use: the gate is based on visual/temporal
path consistency, not ground-truth error. If the query manifest contains ground
truth, metrics are printed for evaluation only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

EARTH_R = 6_378_137.0


def parse_manifest_arg(value: str) -> tuple[Path, str]:
    if "=" not in value:
        path = Path(value)
        return path, path.stem
    dataset_id, path_text = value.split("=", 1)
    return Path(path_text), dataset_id


def load_manifest(path: Path, dataset_id: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    for row in rows:
        row["dataset_id"] = dataset_id
    return rows


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return default
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def local_xy_from_latlon(latitude: float, longitude: float, origin_latitude: float, origin_longitude: float) -> tuple[float, float]:
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    lat0 = math.radians(origin_latitude)
    lon0 = math.radians(origin_longitude)
    return ((lon - lon0) * math.cos(lat0) * EARTH_R, (lat - lat0) * EARTH_R)


def ground_xy(row: dict[str, Any], origin: tuple[float, float]) -> tuple[float, float]:
    return local_xy_from_latlon(float(row["ground_latitude"]), float(row["ground_longitude"]), origin[0], origin[1])


def distance_ref_m(a: dict[str, Any], b: dict[str, Any], origin: tuple[float, float]) -> float:
    ax, ay = ground_xy(a, origin)
    bx, by = ground_xy(b, origin)
    return math.hypot(bx - ax, by - ay)


def has_truth(row: dict[str, Any]) -> bool:
    return safe_float(row.get("ground_latitude"), None) is not None and safe_float(row.get("ground_longitude"), None) is not None


def position_error_m(query: dict[str, Any], reference: dict[str, Any], origin: tuple[float, float]) -> float | None:
    if not has_truth(query):
        return None
    qx, qy = ground_xy(query, origin)
    rx, ry = ground_xy(reference, origin)
    return math.hypot(qx - rx, qy - ry)


def ref_segment_id(row: dict[str, Any], segment_frame_span: int) -> int:
    try:
        return int(float(row.get("frame_count", 0))) // max(1, int(segment_frame_span))
    except Exception:
        return 0


def ref_key(row: dict[str, Any], segment_frame_span: int) -> str:
    return f"{row.get('dataset_id', '')}:{ref_segment_id(row, segment_frame_span)}"


def unary_cost(candidate: dict[str, Any], dino_weight: float, inlier_weight: float, ratio_weight: float) -> float:
    return (
        -dino_weight * float(candidate["dino_similarity"])
        -inlier_weight * math.log1p(float(candidate["lg_inlier_count"]))
        -ratio_weight * float(candidate["lg_inlier_ratio"])
    )


def step_cost(
    previous_reference: dict[str, Any],
    current_reference: dict[str, Any],
    origin: tuple[float, float],
    max_step_m: float,
    transition_weight: float,
) -> float:
    distance = distance_ref_m(previous_reference, current_reference, origin)
    excess = max(0.0, distance - max_step_m)
    return transition_weight * (excess / max(max_step_m, 1e-6)) ** 2


def acceleration_cost(
    older_reference: dict[str, Any],
    previous_reference: dict[str, Any],
    current_reference: dict[str, Any],
    origin: tuple[float, float],
    acceleration_scale_m: float,
    acceleration_weight: float,
) -> float:
    ox, oy = ground_xy(older_reference, origin)
    px, py = ground_xy(previous_reference, origin)
    cx, cy = ground_xy(current_reference, origin)
    acceleration = math.hypot((cx - px) - (px - ox), (cy - py) - (py - oy))
    return acceleration_weight * (acceleration / max(acceleration_scale_m, 1e-6)) ** 2


def direction_change_cost(
    older_reference: dict[str, Any],
    previous_reference: dict[str, Any],
    current_reference: dict[str, Any],
    origin: tuple[float, float],
    direction_scale_degrees: float,
    direction_weight: float,
    min_direction_step_m: float,
) -> float:
    if direction_weight <= 0.0:
        return 0.0
    ox, oy = ground_xy(older_reference, origin)
    px, py = ground_xy(previous_reference, origin)
    cx, cy = ground_xy(current_reference, origin)
    v1 = (px - ox, py - oy)
    v2 = (cx - px, cy - py)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < min_direction_step_m or n2 < min_direction_step_m:
        return 0.0
    cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cosine = min(1.0, max(-1.0, cosine))
    angle_degrees = math.degrees(math.acos(cosine))
    return direction_weight * (angle_degrees / max(direction_scale_degrees, 1e-6)) ** 2


def candidate_extra_penalty(candidate: dict[str, Any], reference_rows: list[dict[str, Any]], gate: dict[str, Any] | None, args: argparse.Namespace) -> float:
    if not gate:
        return 0.0
    ref = reference_rows[int(candidate["reference_index"])]
    key = ref_key(ref, args.segment_frame_span)
    penalty = 0.0
    if key in gate.get("suspicious_keys", set()):
        penalty += args.autogate_suspicious_segment_penalty
    dataset = str(ref.get("dataset_id", ""))
    if dataset in gate.get("suspicious_datasets", set()):
        penalty += args.autogate_suspicious_dataset_penalty
    return penalty


def switch_penalty(prev_ref: dict[str, Any], cur_ref: dict[str, Any], gate: dict[str, Any] | None, args: argparse.Namespace) -> float:
    if not gate:
        return 0.0
    cost = 0.0
    if prev_ref.get("dataset_id") != cur_ref.get("dataset_id"):
        cost += args.autogate_dataset_switch_penalty
    if ref_segment_id(prev_ref, args.segment_frame_span) != ref_segment_id(cur_ref, args.segment_frame_span):
        cost += args.autogate_segment_switch_penalty
    return cost


def first_order_viterbi(candidates: list[list[dict[str, Any]]], reference_rows: list[dict[str, Any]], origin: tuple[float, float], args: argparse.Namespace, gate: dict[str, Any] | None) -> list[int]:
    costs: list[list[float]] = []
    parents: list[list[int]] = []
    for query_idx, query_candidates in enumerate(candidates):
        current_costs = []
        current_parents = []
        for cand_idx, candidate in enumerate(query_candidates):
            base = unary_cost(candidate, args.dino_weight, args.inlier_weight, args.ratio_weight)
            base += candidate_extra_penalty(candidate, reference_rows, gate, args)
            if query_idx == 0:
                current_costs.append(base)
                current_parents.append(-1)
                continue
            cur_ref = reference_rows[int(candidate["reference_index"])]
            best_cost = math.inf
            best_parent = -1
            for prev_idx, previous in enumerate(candidates[query_idx - 1]):
                prev_ref = reference_rows[int(previous["reference_index"])]
                cost = costs[query_idx - 1][prev_idx] + base + step_cost(prev_ref, cur_ref, origin, args.max_step_m, args.transition_weight) + switch_penalty(prev_ref, cur_ref, gate, args)
                if cost < best_cost:
                    best_cost = cost
                    best_parent = prev_idx
            current_costs.append(best_cost)
            current_parents.append(best_parent)
        costs.append(current_costs)
        parents.append(current_parents)
    last_idx = int(np.argmin(np.array(costs[-1])))
    selected: list[int] = []
    for query_idx in range(len(candidates) - 1, -1, -1):
        selected.append(last_idx)
        last_idx = parents[query_idx][last_idx]
    selected.reverse()
    return selected


def second_order_viterbi(candidates: list[list[dict[str, Any]]], reference_rows: list[dict[str, Any]], origin: tuple[float, float], args: argparse.Namespace, gate: dict[str, Any] | None = None) -> list[int]:
    if len(candidates) < 3:
        return first_order_viterbi(candidates, reference_rows, origin, args, gate)

    pair_costs: dict[tuple[int, int], float] = {}
    pair_parents: list[dict[tuple[int, int], int]] = [{} for _ in candidates]

    for i, cand0 in enumerate(candidates[0]):
        ref0 = reference_rows[int(cand0["reference_index"])]
        cost0 = unary_cost(cand0, args.dino_weight, args.inlier_weight, args.ratio_weight) + candidate_extra_penalty(cand0, reference_rows, gate, args)
        for j, cand1 in enumerate(candidates[1]):
            ref1 = reference_rows[int(cand1["reference_index"])]
            cost1 = unary_cost(cand1, args.dino_weight, args.inlier_weight, args.ratio_weight) + candidate_extra_penalty(cand1, reference_rows, gate, args)
            pair_costs[(i, j)] = cost0 + cost1 + step_cost(ref0, ref1, origin, args.max_step_m, args.transition_weight) + switch_penalty(ref0, ref1, gate, args)

    for query_idx in range(2, len(candidates)):
        next_pair_costs: dict[tuple[int, int], float] = {}
        for (older_idx, previous_idx), prev_cost in pair_costs.items():
            older_ref = reference_rows[int(candidates[query_idx - 2][older_idx]["reference_index"])]
            previous_ref = reference_rows[int(candidates[query_idx - 1][previous_idx]["reference_index"])]
            for current_idx, current_candidate in enumerate(candidates[query_idx]):
                current_ref = reference_rows[int(current_candidate["reference_index"])]
                cost = (
                    prev_cost
                    + unary_cost(current_candidate, args.dino_weight, args.inlier_weight, args.ratio_weight)
                    + candidate_extra_penalty(current_candidate, reference_rows, gate, args)
                    + step_cost(previous_ref, current_ref, origin, args.max_step_m, args.transition_weight)
                    + switch_penalty(previous_ref, current_ref, gate, args)
                    + acceleration_cost(older_ref, previous_ref, current_ref, origin, args.acceleration_scale_m, args.acceleration_weight)
                    + direction_change_cost(older_ref, previous_ref, current_ref, origin, args.direction_scale_degrees, args.direction_weight, args.min_direction_step_m)
                )
                pair = (previous_idx, current_idx)
                if cost < next_pair_costs.get(pair, math.inf):
                    next_pair_costs[pair] = cost
                    pair_parents[query_idx][pair] = older_idx
        pair_costs = next_pair_costs

    last_pair = min(pair_costs, key=pair_costs.get)
    selected = [0 for _ in candidates]
    selected[-2], selected[-1] = last_pair
    for query_idx in range(len(candidates) - 1, 1, -1):
        selected[query_idx - 2] = pair_parents[query_idx][(selected[query_idx - 1], selected[query_idx])]
    return selected


def selected_refs(candidates: list[list[dict[str, Any]]], selected_indices: list[int], reference_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [reference_rows[int(cands[idx]["reference_index"])] for cands, idx in zip(candidates, selected_indices)]


def analyze_first_pass(selected: list[dict[str, Any]], origin: tuple[float, float], args: argparse.Namespace) -> dict[str, Any]:
    suspicious_keys: set[str] = set()
    suspicious_datasets: set[str] = set()
    reasons: dict[str, list[str]] = {}

    steps = [distance_ref_m(a, b, origin) for a, b in zip(selected, selected[1:])]
    for i, step in enumerate(steps, start=1):
        if step > args.autogate_jump_m:
            for j in (i - 1, i):
                key = ref_key(selected[j], args.segment_frame_span)
                suspicious_keys.add(key)
                reasons.setdefault(key, []).append(f"jump_{step:.1f}m_at_{i}")

    # Runs by dataset. Very short isolated dataset runs are suspicious because a
    # real flight path should not flicker into another reference for one or two frames.
    runs: list[tuple[int, int, str]] = []
    start = 0
    while start < len(selected):
        ds = str(selected[start].get("dataset_id", ""))
        end = start + 1
        while end < len(selected) and str(selected[end].get("dataset_id", "")) == ds:
            end += 1
        runs.append((start, end, ds))
        start = end

    for r_idx, (lo, hi, ds) in enumerate(runs):
        length = hi - lo
        prev_ds = runs[r_idx - 1][2] if r_idx > 0 else None
        next_ds = runs[r_idx + 1][2] if r_idx + 1 < len(runs) else None
        isolated = prev_ds is not None and next_ds is not None and prev_ds != ds and next_ds != ds
        near_jump = False
        if lo > 0 and steps[lo - 1] > args.autogate_near_jump_m:
            near_jump = True
        if hi < len(selected) and steps[hi - 1] > args.autogate_near_jump_m:
            near_jump = True
        if length <= args.autogate_isolated_run_frames and (isolated or near_jump):
            for j in range(lo, hi):
                key = ref_key(selected[j], args.segment_frame_span)
                suspicious_keys.add(key)
                reasons.setdefault(key, []).append(f"short_run_len_{length}_dataset_{ds}")

    # Dataset-level suspicion is softer and only used when one dataset appears as many unstable short runs.
    dataset_bad_counts: dict[str, int] = {}
    for key in suspicious_keys:
        ds = key.split(":", 1)[0]
        dataset_bad_counts[ds] = dataset_bad_counts.get(ds, 0) + 1
    for ds, count in dataset_bad_counts.items():
        if count >= args.autogate_dataset_bad_key_count:
            suspicious_datasets.add(ds)

    return {
        "suspicious_keys": suspicious_keys,
        "suspicious_datasets": suspicious_datasets,
        "reasons": reasons,
        "first_pass_step_mean_m": float(np.mean(steps)) if steps else 0.0,
        "first_pass_step_median_m": float(np.median(steps)) if steps else 0.0,
        "first_pass_step_max_m": float(max(steps)) if steps else 0.0,
        "first_pass_steps_gt_jump": int(sum(s > args.autogate_jump_m for s in steps)),
        "first_pass_dataset_switches": int(sum(a.get("dataset_id") != b.get("dataset_id") for a, b in zip(selected, selected[1:]))),
        "first_pass_segment_switches": int(sum(ref_segment_id(a, args.segment_frame_span) != ref_segment_id(b, args.segment_frame_span) for a, b in zip(selected, selected[1:]))),
    }


def summarize(results: list[dict[str, Any]], has_ground_truth: bool, prefix: str = "motion_viterbi") -> dict[str, Any]:
    out: dict[str, Any] = {"queries": len(results), "has_ground_truth": bool(has_ground_truth)}
    if not has_ground_truth:
        return out
    dino_errors = np.array([float(row["dino_position_error_m"]) for row in results])
    v_errors = np.array([float(row[f"{prefix}_position_error_m"]) for row in results])
    out.update({
        "dino_mean_error_m": float(dino_errors.mean()) if len(dino_errors) else 0.0,
        "dino_median_error_m": float(np.median(dino_errors)) if len(dino_errors) else 0.0,
        "dino_p90_error_m": float(np.percentile(dino_errors, 90)) if len(dino_errors) else 0.0,
        f"{prefix}_mean_error_m": float(v_errors.mean()) if len(v_errors) else 0.0,
        f"{prefix}_median_error_m": float(np.median(v_errors)) if len(v_errors) else 0.0,
        f"{prefix}_p90_error_m": float(np.percentile(v_errors, 90)) if len(v_errors) else 0.0,
        f"{prefix}_max_error_m": float(v_errors.max()) if len(v_errors) else 0.0,
        "improved_queries": int((v_errors < dino_errors).sum()),
        "worsened_queries": int((v_errors > dino_errors).sum()),
        "unchanged_queries": int((v_errors == dino_errors).sum()),
    })
    return out


def build_results(
    candidates: list[list[dict[str, Any]]],
    selected_indices: list[int],
    query_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    origin: tuple[float, float],
    gate: dict[str, Any] | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_index, (query, selected_idx) in enumerate(zip(query_rows, selected_indices)):
        dino_ref = reference_rows[int(candidates[query_index][0]["reference_index"])]
        cand = candidates[query_index][selected_idx]
        sel_ref = reference_rows[int(cand["reference_index"])]
        dino_err = position_error_m(query, dino_ref, origin)
        v_err = position_error_m(query, sel_ref, origin)
        key = ref_key(sel_ref, args.segment_frame_span)
        rows.append({
            "query_dataset": query["dataset_id"],
            "query_frame_count": int(query["frame_count"]),
            "query_frame_path": query["frame_path"],
            "dino_reference_dataset": dino_ref["dataset_id"],
            "dino_reference_frame_count": int(dino_ref["frame_count"]),
            "dino_position_error_m": "" if dino_err is None else dino_err,
            "motion_viterbi_reference_dataset": sel_ref["dataset_id"],
            "motion_viterbi_reference_frame_count": int(sel_ref["frame_count"]),
            "motion_viterbi_reference_frame_path": sel_ref["frame_path"],
            "motion_viterbi_rank": int(cand["rank"]),
            "motion_viterbi_dino_similarity": float(cand["dino_similarity"]),
            "lg_match_count": int(cand["lg_match_count"]),
            "lg_inlier_count": int(cand["lg_inlier_count"]),
            "lg_inlier_ratio": float(cand["lg_inlier_ratio"]),
            "motion_viterbi_position_error_m": "" if v_err is None else v_err,
            "autogate_reference_key": key,
            "autogate_penalized": "1" if gate and key in gate.get("suspicious_keys", set()) else "0",
        })
    return rows


def write_results(results: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in results:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", action="append", required=True)
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--candidates-cache", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--dino-weight", type=float, default=4.0)
    parser.add_argument("--inlier-weight", type=float, default=1.0)
    parser.add_argument("--ratio-weight", type=float, default=1.0)
    parser.add_argument("--max-step-m", type=float, default=20.0)
    parser.add_argument("--transition-weight", type=float, default=12.0)
    parser.add_argument("--acceleration-scale-m", type=float, default=20.0)
    parser.add_argument("--acceleration-weight", type=float, default=0.5)
    parser.add_argument("--direction-scale-degrees", type=float, default=45.0)
    parser.add_argument("--direction-weight", type=float, default=2.0)
    parser.add_argument("--min-direction-step-m", type=float, default=3.0)
    parser.add_argument("--candidate-limit", type=int, default=25)
    parser.add_argument("--segment-frame-span", type=int, default=3000)
    parser.add_argument("--autogate-jump-m", type=float, default=100.0)
    parser.add_argument("--autogate-near-jump-m", type=float, default=50.0)
    parser.add_argument("--autogate-isolated-run-frames", type=int, default=4)
    parser.add_argument("--autogate-dataset-bad-key-count", type=int, default=4)
    parser.add_argument("--autogate-suspicious-segment-penalty", type=float, default=3.0)
    parser.add_argument("--autogate-suspicious-dataset-penalty", type=float, default=0.0)
    parser.add_argument("--autogate-dataset-switch-penalty", type=float, default=2.0)
    parser.add_argument("--autogate-segment-switch-penalty", type=float, default=0.5)
    args = parser.parse_args()

    reference_rows: list[dict[str, Any]] = []
    for value in args.reference_manifest:
        path, dataset_id = parse_manifest_arg(value)
        reference_rows.extend(load_manifest(path, dataset_id))

    query_path, query_dataset_id = parse_manifest_arg(args.query_manifest)
    query_rows = load_manifest(query_path, query_dataset_id)

    with args.candidates_cache.open("rb") as f:
        candidates = pickle.load(f)
    if args.candidate_limit > 0:
        candidates = [qc[: args.candidate_limit] for qc in candidates]
    if any(len(qc) == 0 for qc in candidates):
        raise RuntimeError("Candidate cache contains empty query candidate lists")

    origin = (float(reference_rows[0]["ground_latitude"]), float(reference_rows[0]["ground_longitude"]))
    has_gt = bool(query_rows and all(has_truth(row) for row in query_rows[: min(10, len(query_rows))]))

    first_indices = second_order_viterbi(candidates, reference_rows, origin, args, gate=None)
    first_refs = selected_refs(candidates, first_indices, reference_rows)
    gate = analyze_first_pass(first_refs, origin, args)

    second_indices = second_order_viterbi(candidates, reference_rows, origin, args, gate=gate)
    results = build_results(candidates, second_indices, query_rows, reference_rows, origin, gate, args)
    write_results(results, args.output_csv)

    summary = summarize(results, has_gt)
    summary.update({
        "two_pass_autogate": True,
        "candidate_limit": args.candidate_limit,
        "max_step_m": args.max_step_m,
        "transition_weight": args.transition_weight,
        "acceleration_weight": args.acceleration_weight,
        "direction_weight": args.direction_weight,
        "autogate_jump_m": args.autogate_jump_m,
        "autogate_near_jump_m": args.autogate_near_jump_m,
        "autogate_isolated_run_frames": args.autogate_isolated_run_frames,
        "autogate_suspicious_segment_penalty": args.autogate_suspicious_segment_penalty,
        "autogate_suspicious_dataset_penalty": args.autogate_suspicious_dataset_penalty,
        "autogate_dataset_switch_penalty": args.autogate_dataset_switch_penalty,
        "autogate_segment_switch_penalty": args.autogate_segment_switch_penalty,
        "suspicious_keys": sorted(gate["suspicious_keys"]),
        "suspicious_datasets": sorted(gate["suspicious_datasets"]),
        "suspicious_reasons": gate["reasons"],
        "first_pass_step_mean_m": gate["first_pass_step_mean_m"],
        "first_pass_step_median_m": gate["first_pass_step_median_m"],
        "first_pass_step_max_m": gate["first_pass_step_max_m"],
        "first_pass_steps_gt_jump": gate["first_pass_steps_gt_jump"],
        "first_pass_dataset_switches": gate["first_pass_dataset_switches"],
        "first_pass_segment_switches": gate["first_pass_segment_switches"],
    })

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.gate_json:
        args.gate_json.parent.mkdir(parents=True, exist_ok=True)
        gate_out = dict(gate)
        gate_out["suspicious_keys"] = sorted(gate["suspicious_keys"])
        gate_out["suspicious_datasets"] = sorted(gate["suspicious_datasets"])
        args.gate_json.write_text(json.dumps(gate_out, indent=2), encoding="utf-8")

    for key, value in summary.items():
        if key in {"suspicious_reasons"}:
            continue
        print(f"{key}: {value}")
    print(f"wrote: {args.output_csv}")
    print(f"wrote: {args.summary_json}")
    if args.gate_json:
        print(f"wrote: {args.gate_json}")


if __name__ == "__main__":
    main()
