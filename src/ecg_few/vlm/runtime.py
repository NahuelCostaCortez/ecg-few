"""Runtime selection and local GPU generation helpers for VLM experiments."""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_MEDGEMMA_MODEL = "google/medgemma-4b-it"
REMOTE_RUNTIME = "remote_api"
LOCAL_RUNTIME = "local_gpu"
OPENAI_RUNTIME = "openai"


@dataclass(frozen=True)
class LocalBatchResult:
    texts: list[str]
    batch_size: int
    offload_used: bool
    latency_seconds: float
    peak_cuda_memory_mb: float | None


def resolve_vlm_runtime(provider: str, vlm_runtime: str | None) -> str:
    """Resolve the new runtime flag while preserving old provider-based scripts."""
    if vlm_runtime:
        return vlm_runtime
    if provider == "vllm":
        return REMOTE_RUNTIME
    return OPENAI_RUNTIME


def resolve_model_name(model: str | None, runtime: str) -> str:
    if model:
        return model
    env_model = os.environ.get("VLM_MODEL", "").strip()
    if env_model:
        return env_model
    if runtime in {REMOTE_RUNTIME, LOCAL_RUNTIME}:
        return DEFAULT_MEDGEMMA_MODEL
    return os.environ.get("OPENAI_MODEL", "gpt-5.5")


def resolve_api_base(api_base: str | None, runtime: str) -> str | None:
    if runtime != REMOTE_RUNTIME:
        return api_base
    resolved = (api_base or "").strip() or os.environ.get("VLM_API_BASE", "").strip()
    if not resolved:
        raise RuntimeError("VLM_API_BASE or --api-base is required for --vlm-runtime remote_api.")
    return resolved


def resolve_api_key(explicit_key: str | None, runtime: str) -> str:
    if explicit_key:
        return explicit_key
    if runtime == OPENAI_RUNTIME:
        return os.environ.get("OPENAI_API_KEY", "")
    if runtime == REMOTE_RUNTIME:
        return os.environ.get("VLM_API_KEY", "")
    return os.environ.get("HF_TOKEN", "")


def image_for_local(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text and "out of memory" in text


def _cuda_peak_memory_mb(torch_module: Any) -> float | None:
    if not torch_module.cuda.is_available():
        return None
    return float(torch_module.cuda.max_memory_allocated() / (1024 * 1024))


class LocalGPUGenerator:
    """Batched Transformers generation with adaptive CUDA OOM handling."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MEDGEMMA_MODEL,
        device: str = "cuda",
        dtype: str = "float16",
        attn_implementation: str = "sdpa",
        offload_dir: Path | None = None,
        token: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation
        self.offload_dir = offload_dir or Path("outputs/vlm_outputs/offload")
        self.token = token or os.environ.get("HF_TOKEN", "")
        self.processor: Any | None = None
        self.model: Any | None = None
        self.torch: Any | None = None
        self.offload_used = False

    def load(self, *, offload: bool = False) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        dtype = getattr(torch, self.dtype_name)

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            token=self.token or None,
        )
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "token": self.token or None,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        if offload:
            self.offload_dir.mkdir(parents=True, exist_ok=True)
            kwargs.update(
                {
                    "device_map": "auto",
                    "offload_folder": self.offload_dir.as_posix(),
                }
            )
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
            self.offload_used = True
            return

        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
        if self.device != "auto":
            self.model = self.model.to(self.device)
        self.offload_used = False

    def reload_with_offload(self) -> None:
        self.close()
        self.load(offload=True)

    def close(self) -> None:
        self.model = None
        self.processor = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        gc.collect()

    def generate_batch(
        self,
        messages_batch: Sequence[list[dict[str, Any]]],
        *,
        max_new_tokens: int,
        temperature: float,
        batch_size: int,
    ) -> LocalBatchResult:
        if self.model is None or self.processor is None:
            self.load()
        assert self.model is not None
        assert self.processor is not None
        assert self.torch is not None
        torch = self.torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        dtype = getattr(torch, self.dtype_name)
        inputs = self.processor.apply_chat_template(
            list(messages_batch),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        target_device = self.model.device if hasattr(self.model, "device") else self.device
        inputs = inputs.to(target_device, dtype=dtype)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            if getattr(tokenizer, "eos_token_id", None) is not None:
                generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
            if getattr(tokenizer, "pad_token_id", None) is not None:
                generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
                generation_kwargs["bad_words_ids"] = [[tokenizer.pad_token_id]]
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        input_len = int(inputs["input_ids"].shape[-1])
        generated = outputs[:, input_len:] if int(outputs.shape[-1]) > input_len else outputs
        texts = [
            text.strip()
            for text in self.processor.batch_decode(generated, skip_special_tokens=True)
        ]
        return LocalBatchResult(
            texts=texts,
            batch_size=batch_size,
            offload_used=self.offload_used,
            latency_seconds=time.monotonic() - started,
            peak_cuda_memory_mb=_cuda_peak_memory_mb(torch),
        )

    def generate_adaptive(
        self,
        messages: Sequence[list[dict[str, Any]]],
        *,
        initial_batch_size: int,
        max_new_tokens: int,
        temperature: float,
    ) -> list[LocalBatchResult]:
        results: list[LocalBatchResult] = []
        index = 0
        batch_size = max(1, initial_batch_size)
        while index < len(messages):
            current = list(messages[index : index + batch_size])
            try:
                result = self.generate_batch(
                    current,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    batch_size=batch_size,
                )
            except RuntimeError as exc:
                if _is_cuda_oom(exc):
                    if self.torch is not None and self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        continue
                    if not self.offload_used:
                        self.reload_with_offload()
                        continue
                raise
            results.append(result)
            index += len(current)
        return results
