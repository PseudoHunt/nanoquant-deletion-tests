"""Instrumentation for NanoQuant deletion tests.

Nothing here modifies the upstream repo: everything is a patch applied at run
time by nqx/driver.py, so the stock pipeline stays byte-identical for Step 0.

Provides
  * collect_hin(...)      full input covariance H_in per linear layer, from
                          NanoQuant's own calibration pass
  * prep_hin(...)         shrinkage (matrix generalisation of theirs) + 1% damping
  * make_heldout(...)     32 held-out sequences (wikitext2 *validation* split)
  * block_err(...)        FP-vs-quantised block-output error on the held-out split
  * compress_block_recon_instr(...)  instrumented copy of their block loop with
                          `pre` / `post` measurement points around Step 3
"""
import json
import os
import time
from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn

from nanoquant.core.compress_block import factorize_and_replace, tune_fact, tune_nonfact
from nanoquant.modules.linear import NanoQuantLinear
from nanoquant.utils.eval_utils import evaluate_ppl_after_block
from nanoquant.utils.load_utils import cache_inputs_and_kwargs, load_tokenizer
from nanoquant.utils.utils import (calculate_ranks, cleanup_memory, find_layers, get_decoder_layers,
                                   get_layers_to_factorize, set_seed)
from tqdm import trange

PERCENTILE = 0.999


# ---------------------------------------------------------------------------
# 1. Full input covariance H_in
# ---------------------------------------------------------------------------
def _hin_hook(module, inputs, outputs, layer_name, acc, run_states):
    """Accumulate clipped X^T X. Clipping mirrors _online_clipping_hook exactly so
    that diag(H_in) reproduces their i_norm; only the off-diagonals are new."""
    x = inputs[0].detach().flatten(0, -2).float()
    norms = torch.norm(x, dim=1, keepdim=True)
    n = norms.numel()
    k = max(1, int(n * (1.0 - PERCENTILE)))
    tau = torch.topk(norms.reshape(-1), k).values[-1]

    st = run_states[layer_name]
    gmax = st["global_max"]
    if gmax is None:
        gmax = tau
    elif tau > gmax:
        acc[layer_name].mul_((tau / (gmax + 1e-8)).square())
        gmax = tau
    st["global_max"] = gmax

    xc = x * torch.clamp(gmax / (norms + 1e-8), max=1.0)
    acc[layer_name].add_(xc.T @ xc / xc.shape[0])
    st["n"] += 1


@torch.no_grad()
def _register_hin_hooks(model, dev):
    linear_layers = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and "lm_head" not in n}
    acc, run_states, handles = {}, defaultdict(lambda: {"global_max": None, "n": 0}), []
    for n, m in linear_layers.items():
        acc[n] = torch.zeros(m.weight.shape[1], m.weight.shape[1], dtype=torch.float32, device=dev)
        handles.append(m.register_forward_hook(partial(_hin_hook, layer_name=n, acc=acc, run_states=run_states)))
    return acc, run_states, handles


def collect_hin(model, dataloader, dev, out_dir, strategy="online"):
    """Runs NanoQuant's calibration pass once, collecting i_norm/o_norm exactly as
    upstream does *and* the full H_in. Returns their raw_stats unchanged."""
    from nanoquant.core import importance as imp

    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, "_complete.json")
    if os.path.exists(done):
        print(f"[INSTR] H_in already on disk at {out_dir}, skipping collection")
        return None

    acc, run_states, handles = _register_hin_hooks(model, dev)
    orig = imp._run_calibration_loop
    state = {"n_calls": 0}

    def wrapped(*a, **kw):
        state["n_calls"] += 1
        return orig(*a, **kw)

    imp._run_calibration_loop = wrapped
    try:
        raw_stats = imp.collect_stats(model, dataloader, dev, strategy=strategy)
    finally:
        imp._run_calibration_loop = orig
        for h in handles:
            h.remove()

    meta = {}
    for name, H in acc.items():
        n_upd = run_states[name]["n"]
        H = H / max(n_upd, 1)
        H = 0.5 * (H + H.T)
        d = H.diagonal().mean().item()
        torch.save({"H": H.cpu(), "n_updates": n_upd, "mean_diag": d}, os.path.join(out_dir, f"{name}.pt"))
        meta[name] = {"n_updates": n_upd, "mean_diag": d, "shape": list(H.shape)}
    del acc
    cleanup_memory()
    json.dump(meta, open(done, "w"), indent=1)
    print(f"[INSTR] saved H_in for {len(meta)} layers to {out_dir}")
    return raw_stats


def prep_hin(name, hin_dir, shrinkage=0.4, damp=0.01, device="cuda", dtype=torch.float32):
    """Load H_in and apply (a) the matrix generalisation of NanoQuant's diagonal
    shrinkage -- H <- (1-s)H + s*mean(diag H)*I, which reduces to their formula
    when H is diagonal -- then (b) 1% of mean-diag damping."""
    blob = torch.load(os.path.join(hin_dir, f"{name}.pt"), map_location="cpu")
    H = blob["H"].to(device=device, dtype=dtype)
    md = H.diagonal().mean()
    if 0.0 < shrinkage < 1.0:
        H.mul_(1.0 - shrinkage)
        H.diagonal().add_(shrinkage * md)
    H.diagonal().add_(damp * H.diagonal().mean())
    return H


# ---------------------------------------------------------------------------
# 2. Held-out split + block error
# ---------------------------------------------------------------------------
def make_heldout(model_id, seqlen=2048, n_seq=32, cache="/home/work/exp/artifacts/heldout32.pt"):
    """32 sequences from the wikitext2 *validation* split: disjoint from the
    calibration source (train) and from the PPL source (test)."""
    if os.path.exists(cache):
        return torch.load(cache)
    import datasets
    tok = load_tokenizer(model_id)
    val = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    enc = tok("\n\n".join(val["text"]), return_tensors="pt").input_ids
    assert enc.shape[1] >= n_seq * seqlen, f"validation split too short: {enc.shape}"
    ids = torch.stack([enc[0, i * seqlen:(i + 1) * seqlen] for i in range(n_seq)]).long()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(ids, cache)
    return ids


@torch.no_grad()
def block_err(block, inputs, targets, kwargs, chunk=1):
    """Relative FP-vs-quantised block-output error on the held-out split.
    inputs/targets are the FP block's own inputs and outputs, so this measures
    the block's own error and not accumulated upstream drift."""
    num = 0.0
    den = 0.0
    for j in range(0, inputs.shape[0], chunk):
        x = inputs[j:j + chunk].to("cuda", non_blocking=True)
        t = targets[j:j + chunk].to("cuda", non_blocking=True)
        y = block(x, **kwargs)[0]
        num += (y.float() - t.float()).square().sum().item()
        den += t.float().square().sum().item()
    return num / max(den, 1e-12)


# ---------------------------------------------------------------------------
# 3. Instrumented copy of core/compress_model.py::compress_block_recon
#    Only lines marked [INSTR] differ from upstream; the algorithm is untouched
#    unless ctx supplies a replacement factorize function.
# ---------------------------------------------------------------------------
class Ctx:
    """Carries instrumentation state and the variant hooks for Tests A/B/C."""
    def __init__(self, tag, model_id, hin_dir=None, factorize_fn=None, do_step3=True,
                 out_json=None, ppl_after_block=True):
        self.tag = tag
        self.model_id = model_id
        self.hin_dir = hin_dir
        self.factorize_fn = factorize_fn or factorize_and_replace
        self.do_step3 = do_step3
        self.out_json = out_json
        self.ppl_after_block = ppl_after_block
        self.rec = {"tag": tag, "layers": {}, "blocks": {}, "timing": defaultdict(float)}

    def dump(self):
        if self.out_json:
            os.makedirs(os.path.dirname(self.out_json), exist_ok=True)
            r = dict(self.rec)
            r["timing"] = dict(self.rec["timing"])
            json.dump(r, open(self.out_json, "w"), indent=1)


@torch.no_grad()
def compress_block_recon_instr(model, fp_model, dataloader, quant_config, ctx):
    set_seed(quant_config['seed'])
    dev = "cuda"
    model.cpu()
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = False
    fp_model.gradient_checkpointing_disable()
    fp_model.eval()
    fp_model.config.use_cache = False
    q_blocks = get_decoder_layers(model)
    fp_blocks = get_decoder_layers(fp_model)
    layers_to_factorize = get_layers_to_factorize(model.config.model_type)
    admm_ranks = calculate_ranks(model, layers_to_factorize, quant_config)
    original_inputs, kwargs = cache_inputs_and_kwargs(fp_model, dataloader, dev)
    kwargs = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    kwargs['use_cache'] = False
    if 'past_key_value' in kwargs:
        kwargs['past_key_value'] = None
    compressed_inputs = original_inputs.clone().detach().cpu()

    # [INSTR] held-out split, propagated down the FP stack in parallel
    ho_ids = make_heldout(ctx.model_id, seqlen=quant_config['seqlen'])
    ho_in, _ = cache_inputs_and_kwargs(fp_model, ho_ids, dev)
    ho_in = ho_in.cpu()

    t_total = time.time()
    for i in trange(len(q_blocks), desc="Compressing Layers"):
        cleanup_memory()
        q_block = q_blocks[i].to(dev)
        fp_block = fp_blocks[i].to(dev)
        with torch.no_grad():
            target_outputs = torch.zeros_like(original_inputs)
            for j in range(quant_config['num_calib_samples']):
                target_outputs[j:j + 1] = fp_block(original_inputs[j:j + 1].to(dev), **kwargs)[0].cpu().detach()
        # [INSTR] FP targets for the held-out split, same block, same kwargs
        ho_out = torch.zeros_like(ho_in)
        for j in range(ho_in.shape[0]):
            ho_out[j:j + 1] = fp_block(ho_in[j:j + 1].to(dev), **kwargs)[0].cpu().detach()

        tuning_inputs = compressed_inputs.clone().detach()
        sublayers = find_layers(q_block)
        importance_layer = sublayers.get('mlp.down_proj', sublayers.get('fc2', None))
        if importance_layer is None or not hasattr(importance_layer, 'o_norm'):
            importance = torch.ones(model.config.hidden_size, device=dev)
        else:
            importance = importance_layer.o_norm.to(dev)
        tuning_inputs = tuning_inputs.to(dev)
        target_outputs = target_outputs.to(dev)

        for name in layers_to_factorize:
            if name not in sublayers:
                continue
            key = f"{i}.{name}"
            if quant_config['tune_nonfact']:
                print(f"\t(1/3) Block {i+1}/{len(q_blocks)}, {name} | Tuning Non-Factorized Weights...")
                t = time.time()
                tune_nonfact(q_block, tuning_inputs, target_outputs, importance, kwargs, quant_config)
                ctx.rec["timing"]["tune_nonfact_s"] += time.time() - t
                cleanup_memory()
            print(f"\t(2/3) Block {i+1}/{len(q_blocks)}, {name} | Initialization via ADMM...")
            curr_rank = admm_ranks.get(key)
            t = time.time()
            nano_linear, final_factor_results = ctx.factorize_fn(q_block, name, curr_rank, quant_config,
                                                                 ctx=ctx, key=key)
            ctx.rec["timing"]["admm_stage_s"] += time.time() - t
            del final_factor_results
            cleanup_memory()

            # [INSTR] measurement point `pre`: after ADMM, before Step 3
            e_pre = block_err(q_block, ho_in, ho_out, kwargs)
            ent = ctx.rec["layers"].setdefault(key, {})
            ent["rank"] = curr_rank
            ent["pre"] = e_pre
            print(f"\t\t[INSTR] {key} block_err pre = {e_pre:.6e}")

            if quant_config['tune_fact'] and ctx.do_step3:
                print(f"\t(3/3) Block {i+1}/{len(q_blocks)}, {name} | Tuning Factorized Weights...")
                t = time.time()
                tune_fact(q_block, nano_linear, tuning_inputs, target_outputs, importance, kwargs, quant_config)
                ctx.rec["timing"]["block_refine_s"] += time.time() - t
                cleanup_memory()
            else:
                nano_linear.finalize()  # [INSTR] harden without Step 3 (ADMM-only arms)

            # [INSTR] measurement point `post`: after Step 3
            e_post = block_err(q_block, ho_in, ho_out, kwargs)
            ent["post"] = e_post
            print(f"\t\t[INSTR] {key} block_err post = {e_post:.6e}")
            cleanup_memory()

        ctx.rec["blocks"][str(i)] = {
            "pre": ctx.rec["layers"][f"{i}.{layers_to_factorize[-1]}"]["pre"],
            "post": ctx.rec["layers"][f"{i}.{layers_to_factorize[-1]}"]["post"],
        }
        fp_blocks[i] = fp_block.cpu()
        original_inputs = target_outputs.clone().detach().cpu()
        ho_in = ho_out  # [INSTR] advance the held-out FP stream
        with torch.no_grad():
            for j in range(quant_config['num_calib_samples']):
                compressed_inputs[j:j + 1] = q_block(compressed_inputs[j:j + 1].to(dev), **kwargs)[0].cpu().detach()
        q_blocks[i] = q_block.cpu()
        del q_block, fp_block, target_outputs
        cleanup_memory()

        if ctx.ppl_after_block:
            test_ppl = evaluate_ppl_after_block(model, model_name=quant_config['model_id'], dev=dev)
            print(f"\t\tBlock {i}: Test Data PPL        = {test_ppl:.3f}")
            ctx.rec["blocks"][str(i)]["ppl_wiki2"] = test_ppl
        ctx.dump()

    ctx.rec["timing"]["block_stage_total_s"] = time.time() - t_total
    ctx.dump()
    return model


def factorize_stock(layer, name, rank, quant_config, ctx=None, key=None):
    """Upstream factorize_and_replace, signature-adapted for the Ctx hook."""
    return factorize_and_replace(layer, name, rank, quant_config)
