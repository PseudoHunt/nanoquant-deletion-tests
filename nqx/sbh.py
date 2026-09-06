"""Single-block development harness for Tests A and B.

Caches, once: the FP inputs and FP outputs of blocks 0 and N/2 for the 128
calibration sequences and the 32 held-out sequences, plus NanoQuant's own
i_norm/o_norm and the full input covariance H_in for those two blocks.

A run then executes NanoQuant's block pipeline on one block only -- everything
else stays FP -- and reports:
  primary   : held-out block-output error `pre` (after ADMM, before Step 3)
              and `post` (after Step 3)
  secondary : end-to-end wikitext2 PPL and KL-to-FP with only that block quantised
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  import-order guard
from nanoquant.core.compress_block import tune_fact, tune_nonfact
from nanoquant.core.importance import collect_stats, get_shrunk_stats, register_stats
from nanoquant.utils.data_utils import get_calib_loader, prepare_dataset
from nanoquant.utils.eval_utils import evaluate_ppl
from nanoquant.utils.load_utils import cache_inputs_and_kwargs, load_model, load_tokenizer
from nanoquant.utils.utils import (calculate_ranks, cleanup_memory, find_layers, get_decoder_layers,
                                   get_layers_to_factorize, set_seed)

import instrument as I

CACHE = "/home/work/exp/artifacts/sbh"
N_HO = 32


# Appendix C: gamma = 0.2 for Llama/Qwen. The repo default is 0.4, which matches
# neither of the paper's two stated regimes (0.2 Llama/Qwen, 0.6 Gemma).
PAPER_GAMMA = 0.2


def _qc(seed=0, **kw):
    kw.setdefault("calib_shrinkage", PAPER_GAMMA)
    c = NanoQuantConfigDataclass(seed=seed, **kw)
    return c.to_dict()


# ---------------------------------------------------------------------------
@torch.no_grad()
def do_cache(args):
    os.makedirs(CACHE, exist_ok=True)
    qd = _qc(seed=args.calib_seed, model_id=args.model_id)
    model = load_model(args.model_id, qd['seqlen'], device_map="cpu")
    blocks = get_decoder_layers(model)
    targets = [0, len(blocks) // 2]
    json.dump({"target_blocks": targets, "n_blocks": len(blocks), "calib_seed": args.calib_seed},
              open(f"{CACHE}/meta.json", "w"))

    data = prepare_dataset(args.model_id, qd)
    tok = load_tokenizer(args.model_id)
    calib = get_calib_loader(data, tok, qd['num_calib_samples'], qd['seed'], qd['seqlen'])
    ho = I.make_heldout(args.model_id, seqlen=qd['seqlen'], n_seq=N_HO)
    allseq = torch.cat([calib, ho], dim=0)                       # 128 calib + 32 held-out
    torch.save({"calib": calib, "heldout": ho}, f"{CACHE}/sequences.pt")

    # NanoQuant's own calibration statistics + full H_in, on the calibration set only
    keep = [f"model.layers.{b}." for b in targets]
    raw = I.collect_hin(model, calib, "cuda", f"{CACHE}/hin", strategy=qd['calib_strategy'],
                        name_filter=lambda n: any(n.startswith(k) for k in keep))
    if raw is None:
        raw = collect_stats(model, calib, "cuda", strategy=qd['calib_strategy'])
    shrunk = get_shrunk_stats(raw, shrinkage=qd['calib_shrinkage'])
    torch.save({k: {n: v.cpu() for n, v in shrunk[k].items()} for k in ['i_norm', 'o_norm']},
               f"{CACHE}/stats.pt")
    model = register_stats(model, shrunk)
    cleanup_memory()

    # FP hidden states: block-0 input, then walk the FP stack
    model.cpu()
    model.eval()
    model.config.use_cache = False
    x, kwargs = cache_inputs_and_kwargs(model, allseq, "cuda")
    kwargs = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
    kwargs['use_cache'] = False
    if 'past_key_value' in kwargs:
        kwargs['past_key_value'] = None
    torch.save({k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()},
               f"{CACHE}/kwargs.pt")

    x = x.detach().cpu().requires_grad_(False)
    for b in range(max(targets) + 1):
        blk = blocks[b].to("cuda")
        y = torch.zeros_like(x)
        with torch.no_grad():
            for j in range(x.shape[0]):
                y[j:j + 1] = blk(x[j:j + 1].to("cuda"), **kwargs)[0].cpu()
        if b in targets:
            torch.save(x, f"{CACHE}/in_b{b}.pt")
            torch.save(y, f"{CACHE}/out_b{b}.pt")
            print(f"[cache] saved FP in/out for block {b}", flush=True)
        blocks[b] = blk.cpu()
        x = y
        cleanup_memory()
    print("[cache] done")


# ---------------------------------------------------------------------------
@torch.no_grad()
def kl_to_fp(qmodel, fpmodel, seqs, dev="cuda"):
    """Mean token-level KL(FP || quantised) on the held-out sequences."""
    tot, n = 0.0, 0
    for j in range(seqs.shape[0]):
        b = seqs[j:j + 1].to(dev)
        lt = fpmodel(b, use_cache=False).logits.float()
        ls = qmodel(b, use_cache=False).logits.float()
        p = F.log_softmax(lt, dim=-1)
        q = F.log_softmax(ls, dim=-1)
        tot += (p.exp() * (p - q)).sum(-1).mean().item()
        n += 1
    return tot / max(n, 1)


def do_run(args):
    meta = json.load(open(f"{CACHE}/meta.json"))
    b = args.block
    qd = _qc(seed=args.seed, model_id=args.model_id, nonfact_epochs=args.nonfact_epochs,
             fact_epochs=args.fact_epochs, admm_outer_iters=args.admm_outer_iters,
             admm_penalty_scheduler="linear", tune_fact=not args.no_step3)
    dev = "cuda"
    t0 = time.time()

    model = load_model(args.model_id, qd['seqlen'], device_map="cpu")
    stats = torch.load(f"{CACHE}/stats.pt")
    stats['stats_device'] = 'cpu'
    model = register_stats(model, stats)
    model.eval()
    model.config.use_cache = False
    blocks = get_decoder_layers(model)
    names = get_layers_to_factorize(model.config.model_type)
    ranks = calculate_ranks(model, names, qd)

    kwargs = torch.load(f"{CACHE}/kwargs.pt")
    kwargs = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
    x = torch.load(f"{CACHE}/in_b{b}.pt").detach().requires_grad_(False)
    y = torch.load(f"{CACHE}/out_b{b}.pt").detach().requires_grad_(False)
    n_cal = qd['num_calib_samples']
    cal_in, cal_out = x[:n_cal], y[:n_cal]
    ho_in, ho_out = x[n_cal:], y[n_cal:]

    ctx = I.Ctx(tag=args.tag, model_id=args.model_id, hin_dir=f"{CACHE}/hin",
                out_json=f"/home/work/exp/artifacts/sbh_runs/{args.tag}.json", ppl_after_block=False)
    ctx.rec["args"] = vars(args)
    ctx.rec["block"] = b
    vopts = json.loads(args.variant_opts)
    ctx.rec["variant_opts"] = vopts
    if args.variant == "stock":
        factorize_fn = I.factorize_stock
    else:
        import variants
        factorize_fn = variants.get_factorize_fn(args.variant, vopts)

    blk = blocks[b].to(dev)
    sub = find_layers(blk)
    imp_layer = sub.get('mlp.down_proj')
    importance = imp_layer.o_norm.to(dev) if hasattr(imp_layer, 'o_norm') \
        else torch.ones(model.config.hidden_size, device=dev)
    cal_in_d, cal_out_d = cal_in.to(dev), cal_out.to(dev)

    set_seed(qd['seed'])
    for name in names:
        if name not in sub:
            continue
        key = f"{b}.{name}"
        if qd['tune_nonfact']:
            t = time.time()
            tune_nonfact(blk, cal_in_d, cal_out_d, importance, kwargs, qd)
            ctx.rec["timing"]["tune_nonfact_s"] += time.time() - t
            cleanup_memory()
        hin_key = f"model.layers.{b}.{name}"
        W_ref = sub[name].weight.data.clone()          # FP weight entering the ADMM
        t = time.time()
        nano_linear, _ = factorize_fn(blk, name, ranks[key], qd, ctx=ctx, key=key, hin_key=hin_key)
        ctx.rec["timing"]["admm_stage_s"] += time.time() - t
        cleanup_memory()

        H_obj = I.prep_hin(hin_key, f"{CACHE}/hin", shrinkage=qd['calib_shrinkage'], device=dev)
        e_pre = I.block_err(blk, ho_in, ho_out, kwargs)
        j_pre, f_pre = I.recon_objectives(W_ref, nano_linear, H_obj)
        ent = ctx.rec["layers"].setdefault(key, {})
        ent["rank"] = ranks[key]
        ent["pre"] = e_pre
        ent["J_pre"] = j_pre
        ent["fro_pre"] = f_pre
        print(f"\t\t[SBH] {key} pre = {e_pre:.6e}  J_pre = {j_pre:.6e}  fro_pre = {f_pre:.6e}", flush=True)

        if qd['tune_fact'] and not args.no_step3:
            t = time.time()
            tune_fact(blk, nano_linear, cal_in_d, cal_out_d, importance, kwargs, qd)
            ctx.rec["timing"]["block_refine_s"] += time.time() - t
            cleanup_memory()
        else:
            nano_linear.finalize()
        e_post = I.block_err(blk, ho_in, ho_out, kwargs)
        j_post, f_post = I.recon_objectives(W_ref, nano_linear, H_obj)
        ent["post"] = e_post
        ent["J_post"] = j_post
        ent["fro_post"] = f_post
        print(f"\t\t[SBH] {key} post = {e_post:.6e}  J_post = {j_post:.6e}  fro_post = {f_post:.6e}", flush=True)
        del W_ref, H_obj
        cleanup_memory()
        ctx.dump()

    last = f"{b}.{names[-1]}"
    ctx.rec["block_pre"] = ctx.rec["layers"][last]["pre"]
    ctx.rec["block_post"] = ctx.rec["layers"][last]["post"]
    del cal_in_d, cal_out_d
    cleanup_memory()

    # secondary: end-to-end PPL and KL-to-FP, only this block quantised
    if not args.no_eval:
        blocks[b] = blk
        model = model.to(dev).eval()
        from nanoquant.utils.data_utils import get_test_loaders
        _, testenc = get_test_loaders("wikitext2", model_name=args.model_id, seqlen=model.seqlen)
        ctx.rec["ppl_wiki2"] = evaluate_ppl(model, testenc, dev, "wikitext2", None, verbose=False)
        fp = load_model(args.model_id, qd['seqlen'], device_map="cpu").to(dev).eval()
        fp.config.use_cache = False
        seqs = torch.load(f"{CACHE}/sequences.pt")["heldout"]
        ctx.rec["kl_to_fp"] = kl_to_fp(model, fp, seqs, dev)
        del fp
        cleanup_memory()

    ctx.rec["timing"]["total_s"] = time.time() - t0
    ctx.dump()
    print("[SBH] FINAL " + json.dumps({k: ctx.rec.get(k) for k in
                                       ["tag", "block", "block_pre", "block_post", "ppl_wiki2", "kl_to_fp"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cache", "run"])
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib_seed", type=int, default=0)
    ap.add_argument("--variant", default="stock")
    ap.add_argument("--variant_opts", default="{}")
    ap.add_argument("--nonfact_epochs", type=int, default=8)
    ap.add_argument("--fact_epochs", type=int, default=8)
    ap.add_argument("--admm_outer_iters", type=int, default=400)
    ap.add_argument("--no_step3", action="store_true")
    ap.add_argument("--no_eval", action="store_true")
    a = ap.parse_args()
    os.makedirs("/home/work/exp/artifacts/sbh_runs", exist_ok=True)
    (do_cache if a.mode == "cache" else do_run)(a)
