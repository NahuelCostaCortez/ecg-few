#!/usr/bin/env python3
"""Run ECG visual QA evaluation against OpenAI or OpenAI-compatible VLM backends."""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ecg_vlm.prompts import (
    DEFAULT_SYSTEM_INSTRUCTIONS,
    load_markdown_prompt,
    multilabel_answer_text,
    multilabel_json_schema,
)
from ecg_vlm.simulator.constants import LABEL_NAMES

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
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
        help="Structured output mode for vLLM. OpenAI always uses JSON schema.",
    )
    parser.add_argument("--dry-run", action="store_true")
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


def openai_user_message(prompt: str, image_path: Path) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": data_url_for_image(image_path)},
        ],
    }


def vllm_user_message(prompt: str, image_path: Path) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url_for_image(image_path)}},
        ],
    }


def build_openai_payload(
    record: dict[str, Any],
    *,
    dataset_root: Path,
    args: argparse.Namespace,
    few_shot_records: list[dict[str, Any]],
    system_prompt: str,
) -> dict[str, Any]:
    kind = task_type(record, args.task)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]}
    ]
    if args.mode == "few-shot":
        for example in few_shot_records:
            example_kind = task_type(example, args.task)
            example_prompt = prompt_for_record(
                example,
                prompt_file=args.prompt_file,
                binary_prompt_dir=args.binary_prompt_dir,
                explicit_task=args.task,
            )
            messages.append(
                openai_user_message(example_prompt, dataset_root / example["image_path"])
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": answer_text(example["expected_answer"], example_kind),
                }
            )

    prompt = prompt_for_record(
        record,
        prompt_file=args.prompt_file,
        binary_prompt_dir=args.binary_prompt_dir,
        explicit_task=args.task,
    )
    messages.append(openai_user_message(prompt, dataset_root / record["image_path"]))

    payload: dict[str, Any] = {
        "model": args.model,
        "input": messages,
        "max_output_tokens": args.max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ecg_binary_finding" if kind == "binary" else "ecg_findings",
                "schema": schema_for_task(kind),
                "strict": True,
            }
        },
    }
    if args.temperature != 0:
        payload["temperature"] = args.temperature
    if args.reasoning_effort != "none":
        payload["reasoning"] = {"effort": args.reasoning_effort}
    return payload


def build_vllm_payload(
    record: dict[str, Any],
    *,
    dataset_root: Path,
    args: argparse.Namespace,
    few_shot_records: list[dict[str, Any]],
    system_prompt: str,
) -> dict[str, Any]:
    kind = task_type(record, args.task)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if args.mode == "few-shot":
        for example in few_shot_records:
            example_kind = task_type(example, args.task)
            example_prompt = prompt_for_record(
                example,
                prompt_file=args.prompt_file,
                binary_prompt_dir=args.binary_prompt_dir,
                explicit_task=args.task,
            )
            messages.append(
                vllm_user_message(example_prompt, dataset_root / example["image_path"])
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": answer_text(example["expected_answer"], example_kind),
                }
            )

    prompt = prompt_for_record(
        record,
        prompt_file=args.prompt_file,
        binary_prompt_dir=args.binary_prompt_dir,
        explicit_task=args.task,
    )
    messages.append(vllm_user_message(prompt, dataset_root / record["image_path"]))

    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_output_tokens,
    }
    if args.response_format == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ecg_binary_finding" if kind == "binary" else "ecg_findings",
                "schema": schema_for_task(kind),
                "strict": True,
            },
        }
    elif args.response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    return payload


def call_openai_responses(
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API connection failed: {exc.reason}") from exc


def call_vllm_chat_completions(
    payload: dict[str, Any],
    *,
    api_base: str,
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"vLLM API connection failed: {exc.reason}") from exc


def extract_output_text(response: dict[str, Any], provider: str) -> str:
    if provider == "vllm":
        return str(response["choices"][0]["message"]["content"]).strip()
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
        "dry_run": args.dry_run,
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
    few_shot_ids = [record["id"] for record in few_shot_records]

    output_path = Path(args.output)
    done = existing_ids(output_path) if args.resume else set()
    if output_path.exists() and not args.resume:
        output_path.unlink()

    api_key = args.api_key
    if args.provider == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not args.dry_run and not api_key:
            raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is set.")
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
        if args.provider == "openai":
            payload = build_openai_payload(
                record,
                dataset_root=dataset_root,
                args=args,
                few_shot_records=few_shot_records,
                system_prompt=system_prompt,
            )
        else:
            payload = build_vllm_payload(
                record,
                dataset_root=dataset_root,
                args=args,
                few_shot_records=few_shot_records,
                system_prompt=system_prompt,
            )

        if args.dry_run:
            response = None
            latency_seconds = None
            raw_output_text = ""
            prediction = None
            error = None
        else:
            started = time.monotonic()
            try:
                if args.provider == "openai":
                    response = call_openai_responses(
                        payload,
                        api_key=api_key,
                        timeout=args.timeout,
                    )
                else:
                    response = call_vllm_chat_completions(
                        payload,
                        api_base=args.api_base,
                        api_key=api_key,
                        timeout=args.timeout,
                    )
                latency_seconds = time.monotonic() - started
                raw_output_text = extract_output_text(response, args.provider)
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
