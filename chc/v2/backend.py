"""Memory-bounded, cached real generation for structured observable claims.

Extends the frozen-model approach used by CHC generation.py. A configurable
instruction model uses its official chat template; historical FLAN uses a plain
prompt. Invalid JSON is never repaired with hand-written benchmark content.
"""

from __future__ import annotations
import gc, hashlib, json, time
from pathlib import Path
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from ..resources import choose_device, seed_everything
from .traces import parse_trace


class TraceBackend:
    """Frozen local model with exact prompt/config caching and explicit OOM logs.

    Args:
        cache: Shared weight and response cache root.
        model_id: Model to assess; no automatic download of a larger replacement.
        max_new_tokens: Output cap for compact structured artifacts.
        device: CUDA if available, otherwise CPU. CUDA uses FP16 to bound memory.
    Attributes:
        fallbacks: Actual OOM recovery events, not assumed hardware failures.
        actual_generations/cache_hits: Process-local execution counters.
    """

    def __init__(
        self,
        cache: Path,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 280,
        device: str = "auto",
        constrained: bool = False,
    ) -> None:
        """Load a cached snapshot, with a single explicit CPU fallback on OOM.

        A 1.5B FP16 model needs roughly 3.1 GB for weights, plus activations/KV
        cache. Actual CUDA availability is checked, and the run records allocator
        peaks. Four-bit quantization is not required when FP16 fits; callers can
        choose another model without changing benchmark or scoring code.
        """
        torch.set_num_threads(4)
        self.cache = cache
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.constrained = constrained
        self.device = choose_device(device)
        self.fallbacks = []
        self.actual_generations = 0
        self.cache_hits = 0
        self.weights = cache / "weights"
        (cache / "responses").mkdir(parents=True, exist_ok=True)
        snapshots = self.weights / ("models--" + model_id.replace("/", "--")) / "snapshots"
        local = any(snapshots.glob("*/config.json"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=self.weights, local_files_only=local
        )
        self.causal = not model_id.startswith("google/flan")
        cls = AutoModelForCausalLM if self.causal else AutoModelForSeq2SeqLM
        dtype = torch.float16 if self.device.type == "cuda" and self.causal else torch.float32
        try:
            self.model = (
                cls.from_pretrained(
                    model_id, cache_dir=self.weights, local_files_only=local, torch_dtype=dtype
                )
                .to(self.device)
                .eval()
            )
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            self.device = torch.device("cpu")
            self.fallbacks.append("load_oom_cpu")
            self.model = cls.from_pretrained(
                model_id, cache_dir=self.weights, local_files_only=True, torch_dtype=torch.float32
            ).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)
        # Resolve the actual local snapshot even when Transformers omits revision.
        self.snapshot_ids = sorted(p.name for p in snapshots.glob("*") if p.is_dir())
        self.grammar_tokenizer_data = None

    def generate(
        self,
        question: str,
        evidence: dict[str, dict[str, str]],
        prefix: list[dict[str, Any]] | None = None,
        seed: int = 101,
    ) -> dict[str, Any]:
        """Generate an actual structured trace or continuation without gold labels.

        Args:
            question/evidence: Fixed serving inputs, identical across interventions.
            prefix: Visible prior claims; corruption/correction changes only these.
            seed: Paired sampling seed. Greedy development mode is not a replicate.
        Returns:
            Raw output, validated trace or validation error, tokens, latency,
            truncation, model identity and cache key. Invalid outputs stay invalid.
        Failure cases:
            Truncated inputs are rejected rather than silently losing citations.
            CUDA OOM retries once on CPU and records the change of execution path.
        """
        prefix = prefix or []
        first = len(prefix) + 1
        instruction = (
            f"Use only the supplied evidence to answer the question. Write exactly 2 short atomic factual claims, "
            "each with one fact. Use the prior claims as working notes, but correct them if the evidence contradicts them. "
            'Return only valid JSON with keys "claims" and "final_answer". '
            'Each claim has "id", "text", "depends_on", "evidence_ids". Use compact JSON without indentation. Each text is one short factual assertion, never a question. '
            f"New IDs start at C{first}. Dependencies may reference only earlier IDs. "
            'EVERY claim MUST have a NONEMPTY evidence_ids list containing a supplied ID such as ["E1"] or ["E2"]. Empty citations are invalid. '
            "Use depends_on: [] when there is no earlier dependency. Never cite a current or later claim as a dependency. "
            "final_answer must contain ONLY the short entity, number, date, yes or no that answers the question, not a full sentence. "
            'Example structure: {"claims":[{"id":"C'
            + str(first)
            + '","text":"One factual statement.","depends_on":[],"evidence_ids":["E1"]}],"final_answer":"short answer"}.\n'
            "Question: "
            + question
            + "\nEvidence:\n"
            + json.dumps(evidence, ensure_ascii=False)
            + "\nPrior claims:\n"
            + json.dumps(prefix, ensure_ascii=False)
            + "\nCheck before responding: every claim cites at least one source from "
            + str(list(evidence))
            + ". No empty evidence_ids. No forward or self dependencies."
        )
        messages = [
            {
                "role": "system",
                "content": "You produce concise source-grounded factual JSON. Do not include commentary or Markdown.",
            },
            {"role": "user", "content": instruction},
        ]
        prompt = (
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if self.causal
            else instruction
        )
        settings = {
            "model": self.model_id,
            "revision": self.revision,
            "snapshots": self.snapshot_ids,
            "prompt": prompt,
            "seed": seed,
            "max_new_tokens": self.max_new_tokens,
            "sample": True,
            "temperature": 0.6,
            "input_limit": 2048,
            "constrained_json": self.constrained,
        }
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        path = self.cache / "responses" / f"{key}.json"
        if not self.constrained:
            # Older raw development caches predate the explicit grammar flag.
            # Only the exactly equivalent raw prompt/settings key is eligible.
            legacy = dict(settings)
            legacy.pop("constrained_json")
            previous = (
                self.cache
                / "responses"
                / (
                    hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
                    + ".json"
                )
            )
            if not path.exists() and previous.exists():
                saved = json.loads(previous.read_text(encoding="utf-8"))
                saved["cache_alias_key"] = key
                path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        if path.exists():
            self.cache_hits += 1
            return dict(json.loads(path.read_text(encoding="utf-8")), cache_hit=True)
        seed_everything(seed)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        n = int(encoded["input_ids"].shape[1])
        start = time.perf_counter()
        if n > 2048:
            result = {
                "raw": "",
                "trace": None,
                "validation_error": "Input exceeds fixed 2048-token limit",
                "truncated": True,
                "output_tokens": 0,
            }
        else:
            kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if self.constrained:
                from .grammar import tokenizer_data, prefix_filter

                allowed_ids = [c["id"] for c in prefix] + [f"C{i}" for i in range(first, first + 2)]
                schema = {
                    "type": "object",
                    "properties": {
                        "claims": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string", "enum": allowed_ids[len(prefix) :]},
                                    "text": {"type": "string", "minLength": 3, "maxLength": 240},
                                    "depends_on": {
                                        "type": "array",
                                        "items": {"type": "string", "enum": allowed_ids},
                                    },
                                    "evidence_ids": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 2,
                                        "items": {"type": "string", "enum": list(evidence)},
                                    },
                                },
                                "required": ["id", "text", "depends_on", "evidence_ids"],
                                "additionalProperties": False,
                            },
                        },
                        "final_answer": {"type": "string", "minLength": 1, "maxLength": 100},
                    },
                    "required": ["claims", "final_answer"],
                    "additionalProperties": False,
                }
                # Grammar filtering changes token eligibility, not factual content
                # or benchmark labels. All text is still generated by the model.
                if self.grammar_tokenizer_data is None:
                    self.grammar_tokenizer_data = tokenizer_data(self.tokenizer)
                kwargs["prefix_allowed_tokens_fn"] = prefix_filter(
                    self.grammar_tokenizer_data, schema
                )
            try:
                with torch.inference_mode():
                    output = self.model.generate(**encoded, **kwargs)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                self.model = self.model.float().to(self.device)
                encoded = encoded.to(self.device)
                self.fallbacks.append("generation_oom_cpu")
                seed_everything(seed)
                with torch.inference_mode():
                    output = self.model.generate(**encoded, **kwargs)
            ids = output[0, n:] if self.causal else output[0]
            raw = self.tokenizer.decode(ids, skip_special_tokens=True)
            try:
                trace = parse_trace(raw, evidence, question, tuple(c["id"] for c in prefix))
                error = None
            except ValueError as exc:
                trace = None
                error = str(exc)
            result = {
                "raw": raw,
                "trace": trace,
                "validation_error": error,
                "truncated": False,
                "output_tokens": len(ids),
                "output_limit_reached": len(ids) >= self.max_new_tokens,
            }
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        result.update(
            model_id=self.model_id,
            revision=self.revision,
            snapshots=self.snapshot_ids,
            seed=seed,
            input_tokens=n,
            latency_ms=(time.perf_counter() - start) * 1000,
            cache_key=key,
            cache_hit=False,
            device=str(self.device),
            constrained_json=self.constrained,
        )
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        self.actual_generations += 1
        return result
