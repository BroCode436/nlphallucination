"""Download reusable official model snapshots without private keys or duplicate runs."""

from pathlib import Path
import argparse, json, os
from huggingface_hub import snapshot_download


def main() -> None:
    """Download only tokenizer/config/weight files into the selected cache root.

    Existing complete downloads are reused; interrupted Hugging Face downloads
    resume through its normal cache. This command does not train a model, call a
    paid inference service, or manufacture benchmark outputs.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_XET_CACHE", str(args.cache / "xet"))
    for model in ("Qwen/Qwen2.5-1.5B-Instruct", "cross-encoder/nli-deberta-v3-xsmall"):
        path = snapshot_download(
            model,
            cache_dir=args.cache / "weights",
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"],
            ignore_patterns=["onnx/*", "openvino/*"],
        )
        print(json.dumps({"model": model, "snapshot": path}), flush=True)


if __name__ == "__main__":
    main()
