#!/usr/bin/env python3
"""
Run ECG visual QA evaluation.

Supports OpenAI or OpenAI-compatible backends.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from ecg_few.prompts import (
    DEFAULT_SYSTEM_INSTRUCTIONS,
    load_markdown_prompt,
    multilabel_answer_text,
    multilabel_json_schema,
)
from ecg_few.simulator.constants import LABEL_NAMES
from openai import APIConnectionError, APIStatusError, OpenAI
BINARY_PROMPT_FILES = {
    "RBBB": "rbbb.md",
    "ST_ELEVATION": "st_elevation.md",
    "T_WAVE_INVERSION": "t_wave_inversion.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a VLM on ECG beat images.")
    parser.add_argument("--provider", choices=("openai", "vllm"), default="openai")
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--eval-jsonl", default="data/vlm/eval/multilabel/test.jsonl")
    parser.add_argument("--few-shot-jsonl", default="data/vlm/eval/multilabel/train.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--task", choices=("auto", "multilabel", "binary"), default="auto")
    parser.add_argument("--system-prompt-file", default="prompts/system/default.md")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--binary-prompt-dir", default="")
    parser.add_argument("--prompt-name", default="")
    parser.add_argument("--mode", choices=("zero-shot", "few-shot"), default="zero-shot")
    parser.add_argument("--few-shot-k", type=int, default=4)
    parser.add_argument(
        "--few-shot-control",
        choices=("normal", "shuffled_answers", "text_only_examples"),
        default="normal",
        help="Counterfactual control to apply to few-shot examples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="low",
    )
    parser.add_argument(
        "--response-format",
        choices=("json_schema", "json_object", "none"),
        default="json_schema",
        help="Structured output mode for the Responses API. OpenAI always uses JSON schema.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def data_url_for_image(path: Path) -> str:
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"Unsupported image type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def task_type(record: dict[str, Any], explicit_task: str) -> str:
    if explicit_task != "auto":
        return explicit_task
    answer = record["expected_answer"]
    if isinstance(answer, dict) and "finding" in answer and "present" in answer:
        return "binary"
    return "multilabel"


def schema_for_task(kind: str) -> dict[str, Any]:
    if kind == "binary":
        return {
            "type": "object",
            "properties": {
                "finding": {"type": "string", "enum": list(LABEL_NAMES)},
                "present": {"type": "boolean"},
            },
            "required": ["finding", "present"],
            "additionalProperties": False,
        }
    return multilabel_json_schema()


def answer_text(answer: dict[str, Any], kind: str) -> str:
    if kind == "binary":
        return json.dumps(
            {"finding": str(answer["finding"]), "present": bool(answer["present"])},
            sort_keys=True,
        )
    return multilabel_answer_text(answer)


def binary_prompt_path(prompt_dir: Path, finding: str) -> Path:
    return prompt_dir / BINARY_PROMPT_FILES[finding]


def prompt_for_record(
    record: dict[str, Any],
    *,
    prompt_file: str,
    binary_prompt_dir: str,
    explicit_task: str,
) -> str:
    kind = task_type(record, explicit_task)
    if kind == "binary" and binary_prompt_dir:
        finding = str(record["expected_answer"]["finding"])
        return load_markdown_prompt(binary_prompt_path(Path(binary_prompt_dir), finding))
    if prompt_file:
        return load_markdown_prompt(prompt_file)
    return str(record["prompt"])


def select_few_shot_records(
    records: list[dict[str, Any]],
    k: int,
    seed: int,
) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    rng = random.Random(seed)
    by_combo: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        answer = record["expected_answer"]
        if "finding" in answer:
            key = f"{answer['finding']}={answer['present']}"
        else:
            key = ",".join(label for label in LABEL_NAMES if answer[label]) or "NORMAL"
        by_combo.setdefault(key, []).append(record)

    selected: list[dict[str, Any]] = []
    for key in sorted(by_combo):
        if len(selected) >= k:
            break
        selected.append(rng.choice(by_combo[key]))

    remaining = [record for record in records if record not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, k - len(selected))])
    return selected[:k]


def apply_few_shot_control(
    records: list[dict[str, Any]],
    *,
    control: str,
    seed: int,
    explicit_task: str,
) -> list[dict[str, Any]]:
    if control in {"normal", "text_only_examples"}:
        return records
    if control != "shuffled_answers":
        raise ValueError(f"Unknown few-shot control: {control}")
    if len(records) <= 1:
        return records

    rng = random.Random(seed + 10_000)
    controlled = copy.deepcopy(records)
    answers = [copy.deepcopy(record["expected_answer"]) for record in records]
    rng.shuffle(answers)

    for record, shuffled_answer in zip(controlled, answers, strict=True):
        kind = task_type(record, explicit_task)
        if kind == "binary":
            record["expected_answer"]["present"] = bool(shuffled_answer["present"])
        else:
            record["expected_answer"] = shuffled_answer
    return controlled


def responses_user_message(prompt: str, image_path: Path | None) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": prompt},
    ]
    if image_path is not None:
        content.append(
            {
                "type": "input_image",
                "image_url": data_url_for_image(image_path),
                "detail": "auto",
            }
        )
    return {
        "type": "message",
        "role": "user",
        "content": content,
    }


def responses_assistant_message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
            }
        ],
    }


def text_format_for_task(kind: str, args: argparse.Namespace) -> dict[str, Any] | None:
    if args.provider == "openai" or args.response_format == "json_schema":
        return {
            "type": "json_schema",
            "name": "ecg_binary_finding" if kind == "binary" else "ecg_findings",
            "schema": schema_for_task(kind),
            "strict": True,
        }
    if args.response_format == "json_object":
        return {"type": "json_object"}
    return None


def build_responses_payload(
    record: dict[str, Any],
    *,
    dataset_root: Path,
    args: argparse.Namespace,
    few_shot_records: list[dict[str, Any]],
    system_prompt: str,
) -> dict[str, Any]:
    kind = task_type(record, args.task)
    messages: list[dict[str, Any]] = []
    if args.mode == "few-shot":
        for example in few_shot_records:
            example_kind = task_type(example, args.task)
            example_prompt = prompt_for_record(
                example,
                prompt_file=args.prompt_file,
                binary_prompt_dir=args.binary_prompt_dir,
                explicit_task=args.task,
            )
            example_image = None
            if args.few_shot_control != "text_only_examples":
                example_image = dataset_root / example["image_path"]
            messages.append(
                responses_user_message(example_prompt, example_image)
            )
            messages.append(
                responses_assistant_message(
                    answer_text(example["expected_answer"], example_kind)
                )
            )

    prompt = prompt_for_record(
        record,
        prompt_file=args.prompt_file,
        binary_prompt_dir=args.binary_prompt_dir,
        explicit_task=args.task,
    )
    messages.append(responses_user_message(prompt, dataset_root / record["image_path"]))

    payload: dict[str, Any] = {
        "model": args.model,
        "instructions": system_prompt,
        "input": messages,
        "max_output_tokens": args.max_output_tokens,
    }
    text_format = text_format_for_task(kind, args)
    if text_format is not None:
        payload["text"] = {"format": text_format}
    if args.temperature != 0:
        payload["temperature"] = args.temperature
    if args.provider == "openai" and args.reasoning_effort != "none":
        payload["reasoning"] = {"effort": args.reasoning_effort}
    return payload


def truncate_error_body(body: str, *, limit: int = 4000) -> str:
    body = body.strip()
    if len(body) <= limit:
        return body
    return body[:limit] + f"... [truncated {len(body) - limit} chars]"


def provider_name(provider: str) -> str:
    return "vLLM" if provider == "vllm" else "OpenAI"


def call_responses_api(
    payload: dict[str, Any],
    *,
    provider: str,
    api_base: str | None,
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    client_kwargs: dict[str, Any] = {
        "api_key": api_key or "EMPTY",
        "timeout": timeout,
    }
    if api_base:
        client_kwargs["base_url"] = api_base.rstrip("/") + "/"

    client = OpenAI(**client_kwargs)
    try:
        response = client.responses.create(**payload)
        return response.model_dump(mode="json")
    except APIStatusError as exc:
        body = ""
        if exc.response is not None:
            try:
                body = exc.response.text
            except Exception:  # noqa: BLE001
                body = str(exc.response)
        body = truncate_error_body(body or str(exc))
        raise RuntimeError(
            f"{provider_name(provider)} Responses API HTTP {exc.status_code}: {body}"
        ) from exc
    except APIConnectionError as exc:
        raise RuntimeError(
            f"{provider_name(provider)} Responses API connection failed: {exc}"
        ) from exc


def extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and "text" in content:
                chunks.append(content["text"])
    return "".join(chunks).strip()


def parse_prediction(text: str, kind: str) -> dict[str, Any]:
    payload = json.loads(text)
    if kind == "binary":
        if payload.get("finding") not in LABEL_NAMES:
            raise ValueError("Missing or invalid field: finding")
        if type(payload.get("present")) is not bool:
            raise ValueError("Missing or non-boolean field: present")
        return {"finding": payload["finding"], "present": payload["present"]}

    prediction: dict[str, bool] = {}
    for label in LABEL_NAMES:
        if type(payload.get(label)) is not bool:
            raise ValueError(f"Missing or non-boolean field: {label}")
        prediction[label] = payload[label]
    return prediction


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_system_prompt(path: str) -> str:
    if path:
        return load_markdown_prompt(path)
    return DEFAULT_SYSTEM_INSTRUCTIONS


def result_record(
    record: dict[str, Any],
    *,
    args: argparse.Namespace,
    prompt: str,
    kind: str,
    prediction: dict[str, Any] | None,
    raw_output_text: str,
    response: dict[str, Any] | None,
    latency_seconds: float | None,
    error: str | None,
    few_shot_ids: list[str],
) -> dict[str, Any]:
    return {
        "id": record["id"],
        "provider": args.provider,
        "model": args.model,
        "api_base": args.api_base if args.provider == "vllm" else None,
        "mode": args.mode,
        "task_type": kind,
        "prompt_name": args.prompt_name,
        "prompt": prompt,
        "image_path": record["image_path"],
        "expected_answer": record["expected_answer"],
        "prediction": prediction,
        "raw_output_text": raw_output_text,
        "response_id": response.get("id") if response else None,
        "usage": response.get("usage") if response else None,
        "latency_seconds": latency_seconds,
        "error": error,
        "few_shot_control": args.few_shot_control,
        "few_shot_ids": few_shot_ids,
        "metadata": record["metadata"],
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    eval_records = read_jsonl(Path(args.eval_jsonl))
    if args.limit:
        eval_records = eval_records[: args.limit]

    few_shot_records: list[dict[str, Any]] = []
    if args.mode == "few-shot":
        few_shot_records = select_few_shot_records(
            read_jsonl(Path(args.few_shot_jsonl)),
            k=args.few_shot_k,
            seed=args.seed,
        )
        few_shot_records = apply_few_shot_control(
            few_shot_records,
            control=args.few_shot_control,
            seed=args.seed,
            explicit_task=args.task,
        )
    few_shot_ids = [record["id"] for record in few_shot_records]

    output_path = Path(args.output)
    done = existing_ids(output_path) if args.resume else set()
    if output_path.exists() and not args.resume:
        output_path.unlink()

    api_key = args.api_key
    if args.provider == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required.")
    else:
        api_key = api_key or os.environ.get("VLLM_API_KEY", "")

    system_prompt = load_system_prompt(args.system_prompt_file)
    for index, record in enumerate(eval_records, start=1):
        if record["id"] in done:
            continue
        image_path = dataset_root / record["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        kind = task_type(record, args.task)
        prompt = prompt_for_record(
            record,
            prompt_file=args.prompt_file,
            binary_prompt_dir=args.binary_prompt_dir,
            explicit_task=args.task,
        )
        payload = build_responses_payload(
            record,
            dataset_root=dataset_root,
            args=args,
            few_shot_records=few_shot_records,
            system_prompt=system_prompt,
        )

        started = time.monotonic()
        try:
            response = call_responses_api(
                payload,
                provider=args.provider,
                api_base=args.api_base if args.provider == "vllm" else None,
                api_key=api_key,
                timeout=args.timeout,
            )
            latency_seconds = time.monotonic() - started
            raw_output_text = extract_output_text(response)
            prediction = parse_prediction(raw_output_text, kind)
            error = None
        except Exception as exc:  # noqa: BLE001
            response = None
            latency_seconds = time.monotonic() - started
            raw_output_text = ""
            prediction = None
            error = str(exc)

        write_jsonl_line(
            output_path,
            result_record(
                record,
                args=args,
                prompt=prompt,
                kind=kind,
                prediction=prediction,
                raw_output_text=raw_output_text,
                response=response,
                latency_seconds=latency_seconds,
                error=error,
                few_shot_ids=few_shot_ids,
            ),
        )
        print(f"[{index}/{len(eval_records)}] {record['id']}", flush=True)


if __name__ == "__main__":
    main()
