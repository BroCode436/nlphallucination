"""Small Transformers-5 adapter to LM Format Enforcer's public token API.

LMFE 0.11.3's optional Transformers adapter imports a tokenizer class from an old
module path. The core token/JSON parser works independently. This adapter uses
that core, avoiding global monkey patches or modifying installed packages.
"""

from __future__ import annotations
from typing import Any
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData


def tokenizer_data(tokenizer: Any) -> Any:
    """Build a reusable decoded-token trie, preserving leading-space behavior.

    Each ordinary vocabulary token is decoded after a sentinel, matching LMFE's
    official integration. EOS is handled by the grammar. Unicode replacement
    fragments are removed only from incremental decoding, not generated facts.
    Cost is paid once per backend, not once per continuation.
    """
    sentinel = tokenizer.encode("0")[-1]
    special = set(tokenizer.all_special_ids)
    tokens = []
    for index in range(len(tokenizer)):
        if index in special:
            continue
        after = tokenizer.decode([sentinel, index])[1:]
        regular = tokenizer.decode([index])
        tokens.append((index, after, len(after) > len(regular)))

    def decode(ids: list[int]) -> str:
        """Decode an incremental token prefix as required by the parser core."""
        return tokenizer.decode(ids).rstrip("�")

    return TokenEnforcerTokenizerData(tokens, decode, tokenizer.eos_token_id, False, len(tokenizer))


def prefix_filter(data: Any, schema: dict[str, Any]) -> Any:
    """Return Transformers' public prefix_allowed_tokens_fn callable.

    A fresh parser is used for every generation so paired runs do not share
    mutable parser state. The resulting token filter constrains format only.
    """
    enforcer = TokenEnforcer(data, JsonSchemaParser(schema))

    def allowed(batch_id: int, token_ids: Any) -> list[int]:
        """Return grammar-legal next token IDs for the current generated prefix."""
        return enforcer.get_allowed_tokens(token_ids.tolist()).allowed_tokens

    return allowed
