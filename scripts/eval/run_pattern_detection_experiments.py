#!/usr/bin/env python3
"""Run VLM pattern-detection experiments over prompt variants, k values, and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_K_VALUES = (0, 1, 2, 4, 8, 12, 16, 24, 32)
DEFAULT_SEEDS = (42, 123, 2026)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Experiment:
    name: str
    task: str
    eval_jsonl: str
    few_shot_jsonl: str
    metric_script: str
    prompt_file: str = ""
    binary_prompt_dir: str = ""


EXPERIMENTS = {
    "multilabel_label_only": Experiment(
        name="multilabel_label_only",
        task="multilabel",
        eval_jsonl="data/synthetic/vlm/eval/multilabel/test.jsonl",
        few_shot_jsonl="data/synthetic/vlm/eval/multilabel/train.jsonl",
        metric_script="scripts/eval/evaluate_multilabel_predictions.py",
        prompt_file="prompts/multilabel/label_only.md",
    ),
    "multilabel_morphology_described": Experiment(
        name="multilabel_morphology_described",
        task="multilabel",
        eval_jsonl="data/synthetic/vlm/eval/multilabel/test.jsonl",
        few_shot_jsonl="data/synthetic/vlm/eval/multilabel/train.jsonl",
        metric_script="scripts/eval/evaluate_multilabel_predictions.py",
        prompt_file="prompts/multilabel/morphology_described.md",
    ),
    "binary_morphology_described": Experiment(
        name="binary_morphology_described",
        task="binary",
        eval_jsonl="data/synthetic/vlm/eval/binary/test.jsonl",
        few_shot_jsonl="data/synthetic/vlm/eval/binary/train.jsonl",
        metric_script="scripts/eval/evaluate_binary_predictions.py",
        binary_prompt_dir="prompts/binary",
    ),
}


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM pattern-detection experiments.")
    parser.add_argument("--dataset-root", default="data/synthetic", help="Dataset root directory.")
    parser.add_argument("--output-root", default="outputs/vlm_outputs/openai/icl_sweep")
    parser.add_argument("--report-dir", default="reports/pattern_detection/openai")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--provider",
        choices=("openai", "vllm"),
        default="openai",
        help="Backend provider to use.",
    )
    parser.add_argument(
        "--runner",
        choices=("openai", "vllm"),
        help="Deprecated alias for --provider.",
    )
    parser.add_argument("--api-base", default="http://localhost:8000/v1", help="vLLM API base URL.")
    parser.add_argument(
        "--response-format",
        choices=("json_schema", "json_object", "none"),
        default="json_schema",
    )
    parser.add_argument(
        "--vllm-response-format",
        choices=("json_schema", "json_object", "none"),
        help="Deprecated alias for --response-format.",
    )
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument(
        "--few-shot-controls",
        default="normal",
        help=(
            "Comma-separated few-shot controls to run. Available: normal, "
            "shuffled_answers, text_only_examples."
        ),
    )
    parser.add_argument(
        "--experiments",
        default=",".join(EXPERIMENTS),
        help=f"Comma-separated experiment names. Available: {', '.join(EXPERIMENTS)}",
    )
    parser.add_argument("--system-prompt-file", default="prompts/system/default.md")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="low",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit records per run.")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Overwrite existing prediction files.",
    )
    parser.add_argument(
        "--skip-binary",
        action="store_true",
        help="Skip the binary diagnostic experiment.",
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def output_stem(
    experiment: Experiment,
    k: int,
    seed: int,
    reasoning_effort: str,
    few_shot_control: str,
) -> str:
    control_part = "" if few_shot_control == "normal" else f"_control-{few_shot_control}"
    if k == 0:
        return f"{experiment.name}{control_part}_k0_reasoning-{reasoning_effort}"
    return f"{experiment.name}{control_part}_k{k}_seed{seed}_reasoning-{reasoning_effort}"


def resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.as_posix()
    return (PROJECT_ROOT / candidate).as_posix()


def run_eval(
    experiment: Experiment,
    *,
    args: argparse.Namespace,
    k: int,
    seed: int,
    few_shot_control: str,
    output_jsonl: Path,
) -> None:
    mode = "zero-shot" if k == 0 else "few-shot"
    cmd = [
        sys.executable,
        resolve_path("scripts/eval/run_vlm_eval.py"),
        "--provider",
        args.provider,
        "--dataset-root",
        resolve_path(args.dataset_root),
        "--eval-jsonl",
        resolve_path(experiment.eval_jsonl),
        "--few-shot-jsonl",
        resolve_path(experiment.few_shot_jsonl),
        "--output",
        output_jsonl.as_posix(),
        "--model",
        args.model,
        "--mode",
        mode,
        "--few-shot-k",
        str(k),
        "--few-shot-control",
        few_shot_control,
        "--seed",
        str(seed),
        "--task",
        experiment.task,
        "--prompt-name",
        experiment.name,
        "--system-prompt-file",
        resolve_path(args.system_prompt_file),
        "--timeout",
        str(args.timeout),
    ]
    if args.provider == "openai":
        cmd.extend(["--reasoning-effort", args.reasoning_effort])
    if args.provider == "vllm":
        cmd.extend(
            [
                "--api-base",
                args.api_base,
                "--response-format",
                args.response_format,
            ]
        )
    if experiment.prompt_file:
        cmd.extend(["--prompt-file", resolve_path(experiment.prompt_file)])
    if experiment.binary_prompt_dir:
        cmd.extend(["--binary-prompt-dir", resolve_path(experiment.binary_prompt_dir)])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.resume:
        cmd.append("--resume")
    run_command(cmd)


def run_metrics(experiment: Experiment, predictions: Path, metrics_path: Path) -> dict[str, object]:
    cmd = [
        sys.executable,
        resolve_path(experiment.metric_script),
        "--predictions",
        predictions.as_posix(),
        "--output",
        metrics_path.as_posix(),
    ]
    run_command(cmd)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def summarize_row(
    experiment: Experiment,
    *,
    k: int,
    seed: int,
    predictions: Path,
    metrics_payload: dict[str, object] | None,
    args: argparse.Namespace,
    few_shot_control: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment": experiment.name,
        "task": experiment.task,
        "provider": args.provider,
        "model": args.model,
        "k": k,
        "seed": "" if k == 0 else seed,
        "reasoning_effort": args.reasoning_effort,
        "few_shot_control": few_shot_control,
        "predictions": predictions.as_posix(),
    }
    if metrics_payload is None:
        return row

    metrics = metrics_payload["metrics"]
    macro = metrics.get("macro", {})
    row["n"] = metrics_payload["n_evaluated"]
    row["macro_accuracy"] = macro.get("accuracy")
    row["macro_f1"] = macro.get("f1")
    row["macro_sensitivity"] = macro.get("sensitivity")
    row["macro_specificity"] = macro.get("specificity")
    if "exact_match_accuracy" in metrics:
        row["exact_match_accuracy"] = metrics["exact_match_accuracy"]
    for label_name, label_metrics in metrics.get("per_label", {}).items():
        row[f"{label_name}_f1"] = label_metrics.get("f1")
        row[f"{label_name}_sensitivity"] = label_metrics.get("sensitivity")
        row[f"{label_name}_specificity"] = label_metrics.get("specificity")
    return row


def write_reports(rows: list[dict[str, object]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "icl_sweep_summary.json"
    csv_path = report_dir / "icl_sweep_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] Wrote report JSON: {json_path}")
    print(f"[OK] Wrote report CSV:  {csv_path}")


def main() -> None:
    args = parse_args()
    if args.runner:
        args.provider = args.runner
    if args.vllm_response_format:
        args.response_format = args.vllm_response_format
    k_values = parse_int_list(args.k_values)
    seeds = parse_int_list(args.seeds)
    few_shot_controls = parse_str_list(args.few_shot_controls)
    experiment_names = [name.strip() for name in args.experiments.split(",") if name.strip()]
    if args.skip_binary:
        experiment_names = [name for name in experiment_names if not name.startswith("binary_")]

    unknown = [name for name in experiment_names if name not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}. Available: {sorted(EXPERIMENTS)}")

    valid_controls = {"normal", "shuffled_answers", "text_only_examples"}
    unknown_controls = [control for control in few_shot_controls if control not in valid_controls]
    if unknown_controls:
        raise ValueError(
            f"Unknown few-shot controls: {unknown_controls}. Available: {sorted(valid_controls)}"
        )

    output_root = Path(args.output_root)
    rows: list[dict[str, object]] = []
    for experiment_name in experiment_names:
        experiment = EXPERIMENTS[experiment_name]
        for k in k_values:
            seeds_for_k = [seeds[0]] if k == 0 else seeds
            controls_for_k = ["normal"] if k == 0 else few_shot_controls
            for few_shot_control in controls_for_k:
                for seed in seeds_for_k:
                    stem = output_stem(
                        experiment,
                        k,
                        seed,
                        args.reasoning_effort,
                        few_shot_control,
                    )
                    output_dir = output_root / experiment.name
                    predictions = output_dir / f"{stem}.jsonl"
                    metrics_path = output_dir / f"{stem}_metrics.json"
                    run_eval(
                        experiment,
                        args=args,
                        k=k,
                        seed=seed,
                        few_shot_control=few_shot_control,
                        output_jsonl=predictions,
                    )
                    metrics_payload = run_metrics(experiment, predictions, metrics_path)
                    rows.append(
                        summarize_row(
                            experiment,
                            k=k,
                            seed=seed,
                            predictions=predictions,
                            metrics_payload=metrics_payload,
                            args=args,
                            few_shot_control=few_shot_control,
                        )
                    )

    write_reports(rows, Path(args.report_dir))


if __name__ == "__main__":
    main()
