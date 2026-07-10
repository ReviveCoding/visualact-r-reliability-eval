from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

from visualact_r.agents.core import parse_typed_output


@dataclass
class QwenVLMConfig:
    model_id: str
    revision: str = "main"
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    max_new_tokens: int = 192
    max_images: int = 4
    max_image_pixels: int | None = 786432
    adapter_dir: str | None = None


class QwenVLMAdapter:
    """Optional local Qwen3-VL adapter with a strict, fail-closed JSON boundary.

    CI deliberately does not import optional CUDA dependencies. The public inference interface accepts a full
    scene plus a bounded evidence set and uses local ``file://`` image URIs, matching Qwen's documented path.
    """

    def __init__(self, config: QwenVLMConfig):
        self.config = config
        self.model: Any | None = None
        self.processor: Any | None = None

    @staticmethod
    def _model_class(transformers_module: Any) -> Any:
        """Resolve Qwen's official image-text model class without eager fallback evaluation."""
        preferred = getattr(transformers_module, "AutoModelForImageTextToText", None)
        if preferred is not None:
            return preferred
        compatibility = getattr(transformers_module, "AutoModelForMultimodalLM", None)
        if compatibility is not None:
            return compatibility
        raise RuntimeError(
            "Installed transformers exposes neither AutoModelForImageTextToText nor "
            "AutoModelForMultimodalLM; Qwen3-VL requires transformers>=4.57.0"
        )

    @staticmethod
    def _configure_image_budget(processor: Any, max_image_pixels: int | None) -> None:
        """Apply a bounded Qwen image-token budget when configured.

        Qwen's processor maps image_processor.size['longest_edge'] to a maximum
        image-pixel budget. Keep its existing minimum so small evidence crops are not
        artificially upsampled.
        """
        if max_image_pixels is None:
            return
        if max_image_pixels <= 0:
            raise ValueError("max_image_pixels must be positive when configured")
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError("Qwen processor lacks image_processor; cannot enforce image budget")
        current = getattr(image_processor, "size", None)
        if not isinstance(current, dict):
            current = {}
        # Qwen2.5-VL processors expose a pixel budget as ``max_pixels``;
        # Qwen3-VL and generic image processors commonly expose ``longest_edge``.
        # Preserve all unrelated sizing fields, including the minimum image budget.
        budget_key = "max_pixels" if "max_pixels" in current else "longest_edge"
        image_processor.size = {**current, budget_key: int(max_image_pixels)}

    def load_inference(self) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install visualact-r[gpu] for Qwen3-VL inference") from exc
        quantization = None
        if self.config.load_in_4bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.quant_type,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model_class = self._model_class(transformers)
        self.model = model_class.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            device_map="auto",
            quantization_config=quantization,
            low_cpu_mem_usage=True,
        )
        if self.config.adapter_dir:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "Install visualact-r[gpu] to load a trained QLoRA adapter"
                ) from exc
            self.model = PeftModel.from_pretrained(self.model, self.config.adapter_dir)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id, revision=self.config.revision
        )
        self._configure_image_budget(self.processor, self.config.max_image_pixels)

    def _input_device(self) -> Any:
        assert self.model is not None
        try:
            device = self.model.device
            if str(device) != "meta":
                return device
        except AttributeError:
            pass
        return next(self.model.parameters()).device

    def generate_typed_action(self, image_paths: list[str], prompt: str) -> dict[str, Any]:
        if not image_paths:
            raise ValueError("Qwen rollout requires at least the full-scene image")
        if len(image_paths) > self.config.max_images:
            raise ValueError(f"Qwen rollout allows at most {self.config.max_images} images")
        missing = [path for path in image_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Qwen rollout image input is missing: {missing[:3]}")
        if self.model is None or self.processor is None:
            self.load_inference()
        assert self.model is not None and self.processor is not None
        content = [
            {"type": "image", "image": f"file://{Path(path).resolve()}"} for path in image_paths
        ]
        content.append(
            {
                "type": "text",
                "text": prompt
                + " Return a compact JSON object only. No Markdown, prose, comments, or extra keys beyond the typed action schema.",
            }
        )
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._input_device())
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )
        output = self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        return parse_typed_output(output)


def build_qlora_model(config: QwenVLMConfig) -> tuple[Any, Any]:
    """Construct an adapter-tuned single-GPU 4-bit Qwen3-VL model for local training.

    The training path intentionally requires CUDA and 4-bit loading. A silent full-precision
    fallback would invalidate the documented memory plan and can OOM a laptop GPU.
    """
    if not config.load_in_4bit:
        raise ValueError("The QLoRA training path requires qwen.load_in_4bit=true")
    try:
        import torch
        import transformers
        from transformers import AutoProcessor, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("Install visualact-r[gpu] for QLoRA") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA model construction requires CUDA; refusing CPU fallback")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = QwenVLMAdapter._model_class(transformers).from_pretrained(
        config.model_id,
        revision=config.revision,
        quantization_config=quantization,
        device_map={"": torch.cuda.current_device()},
        low_cpu_mem_usage=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    processor = AutoProcessor.from_pretrained(config.model_id, revision=config.revision)
    QwenVLMAdapter._configure_image_budget(processor, config.max_image_pixels)
    return model, processor


def _sft_messages(example: Any, include_target: bool) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [
        {"type": "image", "image": str(Path(path).resolve())} for path in example.image_paths
    ]
    content.append({"type": "text", "text": example.prompt})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a precise assistant that follows the JSON schema."},
        {"role": "user", "content": content},
    ]
    if include_target:
        messages.append({"role": "assistant", "content": example.target})
    return messages


def run_qlora_sft(
    train_examples_path: str,
    dev_examples_path: str | None,
    output_dir: str,
    config: QwenVLMConfig,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_train_epochs: float,
    max_steps: int | None,
    gradient_checkpointing: bool,
) -> dict[str, Any]:
    """Run one local QLoRA supervised-fine-tuning job.

    This code is intentionally not used by CPU CI. It requires an installed CUDA-enabled PyTorch, the optional
    project GPU dependencies, a compatible Qwen3-VL revision, and train/dev-only JSONL examples generated by
    `prepare-sft`. It writes adapter-only artifacts and a run manifest; it does not read frozen-final labels.
    """
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError(
            "Install visualact-r[gpu] with CUDA-enabled PyTorch for QLoRA training"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU; refusing CPU fallback")
    from visualact_r.agents.sft import load_sft_examples
    from visualact_r.utils import ensure_dir, sha256_directory, stable_hash, utc_now, write_json

    train_examples = load_sft_examples(train_examples_path)
    dev_examples = load_sft_examples(dev_examples_path) if dev_examples_path else []
    if not train_examples:
        raise ValueError("QLoRA training requires at least one train SFT example")
    if any(example.split != "train" for example in train_examples):
        raise ValueError("QLoRA train input contains a non-train example")
    if any(example.split != "dev" for example in dev_examples):
        raise ValueError("QLoRA dev input contains a non-dev example")

    class _ExampleDataset(Dataset[Any]):
        def __init__(self, examples: list[Any]):
            self.examples = examples

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> Any:
            return self.examples[index]

    model, processor = build_qlora_model(config)
    if gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False

    class _BatchOneCollator:
        def __call__(self, features: list[Any]) -> dict[str, Any]:
            if len(features) != 1:
                raise RuntimeError(
                    "This release trains Qwen3-VL with batch_size=1; use gradient accumulation for larger effective batches"
                )
            example = features[0]
            full = processor.apply_chat_template(
                _sft_messages(example, include_target=True),
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            prefix = processor.apply_chat_template(
                _sft_messages(example, include_target=False),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            labels = full["input_ids"].clone()
            prefix_length = int(prefix["input_ids"].shape[-1])
            if prefix_length >= int(labels.shape[-1]):
                raise RuntimeError(
                    "SFT target was not appended after the prompt; refusing invalid labels"
                )
            labels[:, :prefix_length] = -100
            if "attention_mask" in full:
                labels[full["attention_mask"] == 0] = -100
            return {**full, "labels": labels}

    root = ensure_dir(output_dir)
    set_seed(20260626)
    training_kwargs: dict[str, Any] = {
        "output_dir": str(root / "checkpoints"),
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "max_steps": max_steps if max_steps is not None else -1,
        "bf16": True,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
    }
    if dev_examples:
        parameter_names = inspect.signature(TrainingArguments.__init__).parameters
        strategy_key = (
            "eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"
        )
        training_kwargs[strategy_key] = "epoch"
    args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_ExampleDataset(train_examples),
        eval_dataset=_ExampleDataset(dev_examples) if dev_examples else None,
        data_collator=_BatchOneCollator(),
    )
    train_result = trainer.train()
    adapter_dir = ensure_dir(root / "adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(root / "processor")
    metrics = {
        key: float(value)
        for key, value in train_result.metrics.items()
        if isinstance(value, (int, float))
    }
    if dev_examples:
        metrics.update(
            {
                f"eval_{key}": float(value)
                for key, value in trainer.evaluate().items()
                if isinstance(value, (int, float))
            }
        )
    manifest = {
        "status": "TRAINED_LOCAL_GPU",
        "created_at_utc": utc_now(),
        "base_model": config.model_id,
        "revision": config.revision,
        "train_example_count": len(train_examples),
        "dev_example_count": len(dev_examples),
        "training_protocol": (
            "shared_full_scene_only"
            if {example.candidate_id for example in train_examples + dev_examples} == {"C2"}
            and all(len(example.image_paths) == 1 for example in train_examples + dev_examples)
            else "candidate_specific_or_multi_image"
        ),
        "train_input": str(train_examples_path),
        "dev_input": str(dev_examples_path) if dev_examples_path else None,
        "adapter_dir": str(adapter_dir),
        "adapter_fingerprint": sha256_directory(adapter_dir),
        "quantization": {"load_in_4bit": config.load_in_4bit, "quant_type": config.quant_type},
        "max_image_pixels": config.max_image_pixels,
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.target_modules,
        },
        "training": {
            "batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "max_steps": max_steps,
            "gradient_checkpointing": gradient_checkpointing,
        },
        "metrics": metrics,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_peak_memory_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
        "input_fingerprint": stable_hash(
            {
                "train": [x.to_dict() for x in train_examples],
                "dev": [x.to_dict() for x in dev_examples],
            }
        ),
        "claim_boundary": "This artifact confirms local adapter training only. BDD-derived performance claims require the frozen evaluation protocol.",
    }
    write_json(root / "qwen_qlora_training_manifest.json", manifest)
    return manifest
