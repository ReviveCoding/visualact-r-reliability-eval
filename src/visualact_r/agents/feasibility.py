"""One-image local QLoRA feasibility gate for a VRAM-constrained CUDA workstation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from visualact_r.agents.qwen_adapter import QwenVLMConfig, build_qlora_model
from visualact_r.utils import ensure_dir, utc_now, write_json


@dataclass(frozen=True)
class LocalQLoRAFeasibilityConfig:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    revision: str = "main"
    image_path: str = ""
    max_image_pixels: int = 262144
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    memory_budget_fraction: float = 0.95
    run_backward: bool = True

    def validate(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if self.max_image_pixels <= 0:
            raise ValueError("max_image_pixels must be positive")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("lora_rank and lora_alpha must be positive")
        if not 0.1 <= self.memory_budget_fraction <= 1.0:
            raise ValueError("memory_budget_fraction must be in [0.1, 1.0]")
        if not Path(self.image_path).is_file():
            raise FileNotFoundError(f"Feasibility image does not exist: {self.image_path}")


def _bytes_to_gib(value: int | float) -> float:
    return round(float(value) / (1024**3), 4)


def _memory_snapshot(torch_module: Any) -> dict[str, Any]:
    device = torch_module.cuda.current_device()
    properties = torch_module.cuda.get_device_properties(device)
    stats = torch_module.cuda.memory_stats(device)
    return {
        "device_index": int(device),
        "device_name": properties.name,
        "device_total_gib": _bytes_to_gib(properties.total_memory),
        "allocated_gib": _bytes_to_gib(torch_module.cuda.memory_allocated(device)),
        "reserved_gib": _bytes_to_gib(torch_module.cuda.memory_reserved(device)),
        "peak_allocated_gib": _bytes_to_gib(torch_module.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": _bytes_to_gib(torch_module.cuda.max_memory_reserved(device)),
        "inactive_split_bytes_gib": _bytes_to_gib(stats.get("inactive_split_bytes.all.current", 0)),
    }


def _is_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def run_local_qlora_feasibility(
    config: LocalQLoRAFeasibilityConfig, output_dir: str | Path
) -> dict[str, Any]:
    """Load 4-bit QLoRA, execute one image-conditioned forward/backward, and record VRAM peaks.

    The command does not write optimizer checkpoints, does not train on BDD labels, and does
    not touch frozen-final data. It is a capacity gate, not a training result.
    """
    config.validate()
    root = ensure_dir(output_dir)
    report_path = root / "local_qlora_feasibility_report.json"
    report: dict[str, Any] = {
        "status": "HOLD",
        "started_at": utc_now(),
        "config": asdict(config),
        "claim_boundary": "A one-batch QLoRA capacity gate is not a training or performance result.",
    }
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Local QLoRA feasibility requires CUDA; refusing CPU fallback")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        report["memory_before"] = _memory_snapshot(torch)
        qwen_config = QwenVLMConfig(
            model_id=config.model_id,
            revision=config.revision,
            load_in_4bit=True,
            quant_type="nf4",
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            max_images=1,
            max_image_pixels=config.max_image_pixels,
        )
        model, processor = build_qlora_model(qwen_config)
        model.train()
        report["memory_after_model_load"] = _memory_snapshot(torch)
        report["trainable_parameter_count"] = int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        )
        report["total_parameter_count"] = int(
            sum(parameter.numel() for parameter in model.parameters())
        )
        messages = [
            {"role": "system", "content": "Return only a compact JSON object."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{Path(config.image_path).resolve()}"},
                    {
                        "type": "text",
                        "text": "Inspect the road scene. Return a JSON decision with one of AUTO_CLEAR, REVIEW, ESCALATE.",
                    },
                ],
            },
            {"role": "assistant", "content": '{"decision":"REVIEW"}'},
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        device = torch.cuda.current_device()
        inputs = inputs.to(device)
        labels = inputs["input_ids"].clone()
        output = model(**inputs, labels=labels)
        loss = output.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("QLoRA feasibility forward pass did not return a finite loss")
        report["loss"] = float(loss.detach().float().cpu())
        if config.run_backward:
            loss.backward()
        torch.cuda.synchronize()
        report["memory_after_forward_backward"] = _memory_snapshot(torch)
        total_gib = report["memory_after_forward_backward"]["device_total_gib"]
        peak_reserved_gib = report["memory_after_forward_backward"]["peak_reserved_gib"]
        report["memory_budget_gib"] = round(total_gib * config.memory_budget_fraction, 4)
        report["status"] = (
            "PASS" if peak_reserved_gib <= report["memory_budget_gib"] else "HOLD_MEMORY_HEADROOM"
        )
        report["reason"] = (
            "One-image 4-bit NF4 QLoRA forward/backward completed within the configured memory budget."
            if report["status"] == "PASS"
            else "Forward/backward completed, but peak reserved memory exceeded the configured budget."
        )
        del output, inputs, labels, model, processor
        torch.cuda.empty_cache()
    except Exception as exc:
        report["status"] = "OOM_HOLD" if _is_oom(exc) else "ERROR_HOLD"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        try:
            import torch

            if torch.cuda.is_available():
                report["memory_at_failure"] = _memory_snapshot(torch)
                torch.cuda.empty_cache()
        except Exception:
            pass
    finally:
        report["finished_at"] = utc_now()
        write_json(report_path, report)
    return report
