"""Run nanoquant.main under transformers 4.51.3, the version Appendix C states.

The repo's load_model passes `dtype=` to from_pretrained, an alias that only
exists from transformers 4.56. This shim translates it to `torch_dtype=` so the
published code runs on its own pinned version. Nothing else is changed.
"""
import sys
import transformers
from transformers import AutoModelForCausalLM

_orig = AutoModelForCausalLM.from_pretrained.__func__


def _patched(cls, *args, **kwargs):
    if "dtype" in kwargs and "torch_dtype" not in kwargs:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
    return _orig(cls, *args, **kwargs)


AutoModelForCausalLM.from_pretrained = classmethod(_patched)
print(f"[shim] transformers {transformers.__version__}: dtype -> torch_dtype", flush=True)

from nanoquant.main import main  # noqa: E402
sys.argv[0] = "nanoquant.main"
main()
