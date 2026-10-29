"""Pre-registered, repeated-seed paired experiments for prompt/model comparisons.

The primary metric and scoring rules are read from a versioned JSON spec before
inference. Metrics are transparent deterministic response functions, not semantic
quality or safety judges. Results retain per-case/per-seed paired observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPEC_VERSION = "experiment-scientist.spec.v1"
SUPPORTED_METRICS = {"word_count", "character_count", "contains", "regex_match", "exact_match"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_spec(path: Path) -> tuple[dict[str, Any], str]:
    """Load and validate a pre-registration spec, returning its exact file hash."""
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read experiment spec {path}: {exc}") from exc
    if not isinstance(spec, dict) or spec.get("schema_version") != SPEC_VERSION:
        raise ValueError(f"schema_version must be {SPEC_VERSION!r}")
    for key in ("title", "hypothesis", "primary_metric", "metrics", "cases"):
        if key not in spec:
            raise ValueError(f"Missing required spec field: {key}")
    if not isinstance(spec["title"], str) or not spec["title"].strip():
        raise ValueError("title must be a non-empty string")
    if not isinstance(spec["hypothesis"], str) or not spec["hypothesis"].strip():
        raise ValueError("hypothesis must be a non-empty string")
    metrics = spec["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("metrics must be a non-empty list")
    metric_ids: set[str] = set()
    for metric in metrics:
        if not isinstance(metric, dict):
            raise ValueError("each metric must be an object")
        metric_id, kind = metric.get("id"), metric.get("kind")
        if not isinstance(metric_id, str) or not metric_id.strip() or metric_id in metric_ids:
            raise ValueError("metric IDs must be unique non-empty strings")
        metric_ids.add(metric_id)
        if kind not in SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric kind for {metric_id}: {kind!r}")
        if metric.get("direction", "higher") not in {"higher", "lower", "neutral"}:
            raise ValueError(f"Invalid direction for metric {metric_id}")
        if kind in {"contains", "exact_match"} and not isinstance(metric.get("value"), str):
            raise ValueError(f"Metric {metric_id} requires a string 'value'")
        if kind == "regex_match":
            pattern = metric.get("pattern")
            if not isinstance(pattern, str):
                raise ValueError(f"Metric {metric_id} requires a string 'pattern'")
            import re
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid regex for metric {metric_id}: {exc}") from exc
    if spec["primary_metric"] not in metric_ids:
        raise ValueError("primary_metric must reference a declared metric ID")
    cases = spec["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids:
            raise ValueError("case IDs must be unique non-empty strings")
        case_ids.add(case_id)
        for arm in ("baseline", "treatment"):
            if not isinstance(case.get(arm), str) or not case[arm].strip():
                raise ValueError(f"case {case_id} requires non-empty {arm} prompt")
    return spec, sha256_file(path)


def score_response(response: str, metric: dict[str, Any]) -> float:
    """Score a response with a declared, deterministic metric."""
    kind = metric["kind"]
    if kind == "word_count":
        return float(len(response.split()))
    if kind == "character_count":
        return float(len(response))
    if kind == "contains":
        return float(metric["value"].casefold() in response.casefold())
    if kind == "exact_match":
        return float(response.strip() == metric["value"])
    if kind == "regex_match":
        import re
        return float(re.search(metric["pattern"], response, flags=re.MULTILINE) is not None)
    raise ValueError(f"Unsupported metric kind: {kind}")


def bootstrap_mean_ci(values: list[float], *, iterations: int = 4000, seed: int = 17,
                      confidence: float = 0.95) -> dict[str, Any]:
    """Percentile bootstrap over paired deltas; exploratory, not a substitute for design."""
    if not values:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    if iterations < 100:
        raise ValueError("iterations must be >= 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "lower": None, "upper": None,
                "confidence": confidence, "method": "not estimable for n < 2"}
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(iterations))
    lo = max(0, min(len(means) - 1, math.floor(((1 - confidence) / 2) * iterations)))
    hi = max(0, min(len(means) - 1, math.ceil((1 - (1 - confidence) / 2) * iterations) - 1))
    return {"n": len(values), "mean": mean, "lower": means[lo], "upper": means[hi],
            "confidence": confidence, "iterations": iterations,
            "method": "percentile bootstrap over paired deltas; exploratory"}


def summarize_pairs(values: list[float], *, seed: int) -> dict[str, Any]:
    ci = bootstrap_mean_ci(values, seed=seed)
    sd = statistics.stdev(values) if len(values) > 1 else None
    effect = (statistics.fmean(values) / sd) if sd and sd > 0 else None
    return {**ci, "sd_paired_delta": sd, "cohens_dz": effect,
            "interpretation": "Positive delta favors treatment only when the metric direction is higher; rule-based metric, not semantic quality."}


def run_matrix(spec_path: Path, *, model_ids: list[str], output: Path, device: str = "auto",
               seeds: list[int] | None = None, max_new_tokens: int = 96,
               do_sample: bool = False, temperature: float = 0.8, top_p: float = 0.95) -> dict[str, Any]:
    """Run a model × seed × case paired matrix and write JSON and CSV outputs."""
    if not model_ids or any(not model.strip() for model in model_ids):
        raise ValueError("At least one non-empty model ID is required")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model IDs must be unique")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if do_sample and temperature <= 0:
        raise ValueError("temperature must be > 0 when sampling")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    seeds = seeds or [17, 29, 43, 71, 101]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    spec, spec_hash = load_and_validate_spec(spec_path)
    # Import heavyweight runtime only after validation, so invalid specs fail fast.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        )
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.monotonic()
    observations: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    metrics = spec["metrics"]
    for model_id in model_ids:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(resolved_device).eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        for seed in seeds:
            for case in spec["cases"]:
                arms: dict[str, dict[str, Any]] = {}
                # Reset to the same seed for each arm: paired stochastic comparison.
                for arm in ("baseline", "treatment"):
                    torch.manual_seed(seed)
                    if torch.cuda.is_available() and str(resolved_device).startswith("cuda"):
                        torch.cuda.manual_seed_all(seed)
                    encoded = tokenizer(case[arm], return_tensors="pt").to(resolved_device)
                    generation: dict[str, Any] = {
                        "max_new_tokens": max_new_tokens, "do_sample": do_sample,
                        "pad_token_id": tokenizer.pad_token_id,
                    }
                    if do_sample:
                        generation.update({"temperature": temperature, "top_p": top_p})
                    with torch.inference_mode():
                        generated = model.generate(**encoded, **generation)
                    response = tokenizer.decode(
                        generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True
                    )
                    arms[arm] = {"prompt": case[arm], "response": response,
                                 "response_sha256": hashlib.sha256(response.encode()).hexdigest()}
                for metric in metrics:
                    base_score = score_response(arms["baseline"]["response"], metric)
                    treatment_score = score_response(arms["treatment"]["response"], metric)
                    observations.append({
                        "model_id": model_id, "seed": seed, "case_id": case["id"],
                        "metric_id": metric["id"], "metric_kind": metric["kind"],
                        "baseline_score": base_score, "treatment_score": treatment_score,
                        "delta_treatment_minus_baseline": treatment_score - base_score,
                        "baseline_response_sha256": arms["baseline"]["response_sha256"],
                        "treatment_response_sha256": arms["treatment"]["response_sha256"],
                        "baseline_response": arms["baseline"]["response"],
                        "treatment_response": arms["treatment"]["response"],
                    })
        del model, tokenizer
        if torch.cuda.is_available() and str(resolved_device).startswith("cuda"):
            torch.cuda.empty_cache()
    for model_id in model_ids:
        for metric in metrics:
            rows = [row for row in observations if row["model_id"] == model_id and row["metric_id"] == metric["id"]]
            # Average within case first: prompts are the independent units, not each repeated seed.
            by_case: dict[str, list[float]] = {}
            for row in rows:
                by_case.setdefault(row["case_id"], []).append(row["delta_treatment_minus_baseline"])
            case_deltas = [statistics.fmean(deltas) for deltas in by_case.values()]
            summary = summarize_pairs(case_deltas, seed=(seeds[0] + len(metric["id"])))
            summaries.append({
                "model_id": model_id, "metric_id": metric["id"], "metric_kind": metric["kind"],
                "direction": metric.get("direction", "higher"),
                "cases": len(case_deltas), "seeds_per_case": len(seeds),
                "observations": len(rows), **summary,
            })
    completed = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": "experiment-scientist.matrix.v1",
        "experiment": {"title": spec["title"], "hypothesis": spec["hypothesis"],
                       "primary_metric": spec["primary_metric"], "spec_sha256": spec_hash,
                       "spec_path": str(spec_path), "pre_registered_at_utc": started},
        "models": model_ids, "device": resolved_device, "seeds": seeds,
        "generation": {"max_new_tokens": max_new_tokens, "do_sample": do_sample,
                       "temperature": temperature if do_sample else None,
                       "top_p": top_p if do_sample else None},
        "summaries": summaries, "observations": observations,
        "runtime_seconds": round(time.monotonic() - start_clock, 3),
        "completed_at_utc": completed,
        "interpretation_limits": [
            "Rule-based metrics measure only their declared property and are not semantic-quality or safety judgments.",
            "The bootstrap resamples case-level paired deltas after averaging repeated seeds; with few cases, intervals are unstable.",
            "Repeated seeds do not replace a diverse, representative case sample or independent model replications.",
            "A pre-registration hash records the exact spec bytes; it does not prove the spec was created before seeing any prior results.",
            "Greedy decoding is used by default. Sampling makes paired seed control useful but does not guarantee identical random-number consumption across prompts."
        ],
    }
    json_path = output / "matrix-results.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = output / "matrix-observations.csv"
    fields = ["model_id", "seed", "case_id", "metric_id", "metric_kind", "baseline_score",
              "treatment_score", "delta_treatment_minus_baseline", "baseline_response_sha256",
              "treatment_response_sha256", "baseline_response", "treatment_response"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(observations)
    return {"json": str(json_path), "csv": str(csv_path), "models": len(model_ids),
            "seeds": len(seeds), "cases": len(spec["cases"]), "summaries": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--model-id", action="append", required=True, dest="models",
                        help="Model ID; repeat this flag to compare multiple models")
    parser.add_argument("--output", type=Path, default=Path("outputs/matrix"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seeds", default="17,29,43,71,101", help="Comma-separated unique integer seeds")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--sample", action="store_true", help="Use paired seeded sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    try:
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        result = run_matrix(args.spec, model_ids=args.models, output=args.output, device=args.device,
                            seeds=seeds, max_new_tokens=args.max_new_tokens, do_sample=args.sample,
                            temperature=args.temperature, top_p=args.top_p)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
