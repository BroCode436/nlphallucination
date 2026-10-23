"""Real small-language-model continuation with content-addressed caching.

The model answers from fixed evidence and mutable working notes. Cached outputs
are reused transparently and never counted as independent stochastic trials.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .resources import choose_device, seed_everything
from .schema import DecisionState


class SmallReasoner:
    """Frozen FLAN-T5 continuation backend with bounded memory and actual outputs.

    Args:
        cache: Directory for downloaded weights and generation JSON records.
        model_id: Small encoder-decoder checkpoint; no large model is automatic.
        device: Single execution device or CPU fallback.
        max_length: Maximum input tokens; truncation is detected and reported.
        max_new_tokens: Upper bound on generated final-answer length.
        sample: Enable stochastic sampling; otherwise different seeds repeat greedy
            decoding and must not be interpreted as independent model samples.
    """
    def __init__(self, cache: Path, model_id: str = 'google/flan-t5-small', device: str = 'auto',
                 max_length: int = 512, max_new_tokens: int = 32, sample: bool = False) -> None:
        """Load a bounded real model after the caller has estimated memory.

        Args:
            cache: Writable cache root for weights and generated responses.
            model_id: Approved model identifier; passed to Hugging Face loaders.
            device: Requested CUDA/CPU execution target.
            max_length: Input token budget; fixed across paired interventions.
            max_new_tokens: Generated-token cap, fixed across paired interventions.
            sample: Whether sampling is enabled instead of greedy decoding.
        Returns:
            None; loaded model remains frozen in evaluation mode.
        """
        self.device = choose_device(device)
        self.cache = cache
        (cache/'generations').mkdir(parents=True, exist_ok=True)
        model_cache = cache/'weights'/('models--'+model_id.replace('/','--'))
        # Reuse an already downloaded snapshot without network metadata requests.
        # This keeps the local demo usable offline and avoids refreshing weights
        # halfway through a controlled experiment with a persistent cache.
        local = any((model_cache/'snapshots').glob('*/model.safetensors')) or any((model_cache/'snapshots').glob('*/pytorch_model.bin'))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache/'weights', local_files_only=local)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=cache/'weights', local_files_only=local).to(self.device).eval()
        self.model_id, self.max_length, self.max_new_tokens, self.sample = model_id, max_length, max_new_tokens, sample
        self.actual_generations = 0
        self.cache_hits = 0
        self.fallback_events: list[str] = []

    def prompt(self, state: DecisionState) -> str:
        """Build a label-free prompt with notes first, then fixed source evidence.

        Args:
            state: Current pre/post-action visible state without gold answer.
        Returns:
            Prompt text. Truncation happens later and is recorded, not hidden.
        Design:
            The supplied question/evidence remain fixed across matched runs. Only
            the controlled working-note text changes. This is an artificial
            note-intervention environment, not a model's private reasoning state.
        """
        notes = '\n'.join(c.text for c in state.claims if not c.is_final)
        evidence = '\n'.join(f"{e['title']}: {e['text']}" for e in state.evidence.values())
        return (f'Answer the question using the evidence and working notes. The notes may contain errors. '
                f'Return only the short answer.\nQuestion: {state.question}\nWorking notes: {notes}\nEvidence: {evidence}\nAnswer:')

    def generate(self, state: DecisionState, seed: int) -> dict[str, Any]:
        """Generate or reuse one actual continuation with a fully specified cache key.

        Args:
            state: Current graph/notes; policy outcomes and references are absent.
            seed: Matched seed for corrupt and intervention trajectories.
        Returns:
            Generated answer, model/settings, input/output tokens, actual latency,
            truncation flag, cache status, seed, and model revision information.
        OOM handling:
            Retry once on CPU and record the fallback; other failures propagate.
        """
        prompt = self.prompt(state)
        revision = getattr(self.model.config, '_commit_hash', None)
        settings = {'model': self.model_id, 'revision': revision, 'seed': seed, 'sample': self.sample,
                    'max_length': self.max_length, 'max_new_tokens': self.max_new_tokens,
                    'temperature': .7 if self.sample else None, 'prompt': prompt}
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        path = self.cache/'generations'/f'{key}.json'
        if path.exists():
            self.cache_hits += 1
            return dict(json.loads(path.read_text(encoding='utf-8')), cache_hit=True)
        seed_everything(seed)
        full_length = len(self.tokenizer.encode(prompt, truncation=False))
        encoded = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.max_length).to(self.device)
        start = time.perf_counter()
        kwargs: dict[str, Any] = {'max_new_tokens': self.max_new_tokens, 'do_sample': self.sample}
        if self.sample:
            kwargs['temperature'] = .7
        try:
            with torch.inference_mode():
                output = self.model.generate(**encoded, **kwargs)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
        except torch.cuda.OutOfMemoryError:
            self.fallback_events.append('reasoner_oom_cpu_retry')
            torch.cuda.empty_cache()
            self.device = torch.device('cpu')
            self.model.to(self.device)
            seed_everything(seed)
            with torch.inference_mode():
                output = self.model.generate(**encoded.to(self.device), **kwargs)
        result = {'answer': self.tokenizer.decode(output[0], skip_special_tokens=True), 'seed': seed,
                  'model_id': self.model_id, 'revision': revision, 'sample': self.sample,
                  'input_tokens': int(encoded['input_ids'].shape[1]), 'untruncated_input_tokens': full_length,
                  'truncated': full_length > self.max_length, 'output_tokens': int(output.shape[1]),
                  'latency_ms': (time.perf_counter()-start)*1000, 'device': str(self.device),
                  'cache_hit': False, 'cache_key': key}
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')
        self.actual_generations += 1
        return result
