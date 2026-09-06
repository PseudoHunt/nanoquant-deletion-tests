"""Driver mirroring NanoQuantModel.quantize_model, with instrumentation patched in.

Reproduces hub.quantize_model step for step; the only differences are the
instrumented block loop and the optional variant hooks for Tests A/B/C.
"""
import argparse
import json
import os
import time

import torch

# import order matters: nanoquant.core alone is a circular import upstream
from nanoquant.modules.hub import NanoQuantConfigDataclass  # noqa: F401  (import first)
from nanoquant.core.compress_model import compress_model_recon
from nanoquant.core.importance import collect_stats, get_shrunk_stats, register_stats
from nanoquant.utils.data_utils import get_calib_loader, prepare_dataset
from nanoquant.utils.eval_utils import evaluate_model
from nanoquant.utils.load_utils import load_model, load_tokenizer
from nanoquant.utils.utils import cleanup_memory, find_layers, get_decoder_layers, get_layers_to_factorize

import instrument as I


def bpw_report(model, quant_config):
    """Exact bits-per-weight of the factorised layers, counting every stored scale."""
    from nanoquant.modules.linear import NanoQuantLinear
    tot_bits, tot_w = 0, 0
    per = {}
    for name, m in model.named_modules():
        if not isinstance(m, NanoQuantLinear):
            continue
        d_out, d_in, r = m.out_features, m.in_features, m.rank
        bits = r * (d_in + d_out)                      # binary U and V
        bits += 16 * d_in + 16 * d_out                 # scale_pre, scale_post
        if getattr(m, "scale_mid", None) is not None:
            bits += 16 * r                             # scale_mid, when the arm uses one
        extra = getattr(m, "_extra_bits", 0)           # FP16 residual (Test C arms)
        bits += extra
        tot_bits += bits
        tot_w += d_in * d_out
        per[name] = bits / (d_in * d_out)
    return {"bpw_factorized": tot_bits / max(tot_w, 1), "per_layer": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="stock")
    ap.add_argument("--bits", type=float, default=1.0)
    ap.add_argument("--num_calib_samples", type=int, default=128)
    ap.add_argument("--nonfact_epochs", type=int, default=8)
    ap.add_argument("--fact_epochs", type=int, default=8)
    ap.add_argument("--admm_outer_iters", type=int, default=400)
    ap.add_argument("--no_step3", action="store_true", help="ADMM-only arms (skip Step 3)")
    ap.add_argument("--no_kd", action="store_true", help="skip the global KD stage")
    ap.add_argument("--no_ppl_after_block", action="store_true")
    ap.add_argument("--collect_hin", action="store_true")
    ap.add_argument("--hin_dir", default=None)
    ap.add_argument("--outdir", default="/home/work/exp/artifacts")
    ap.add_argument("--ppl_task", default="wikitext2,c4")
    ap.add_argument("--variant_opts", default="{}")
    args = ap.parse_args()

    t0 = time.time()
    qc = NanoQuantConfigDataclass(
        model_id=args.model_id, bits=args.bits, seed=args.seed,
        num_calib_samples=args.num_calib_samples, calib_dataset="wikitext2",
        nonfact_epochs=args.nonfact_epochs, fact_epochs=args.fact_epochs,
        admm_outer_iters=args.admm_outer_iters, admm_penalty_scheduler="linear",
        tune_model=not args.no_kd,
    )
    qd = qc.to_dict()

    model = load_model(args.model_id, qd['seqlen'], device_map="cpu")
    fp_model = load_model(args.model_id, qd['seqlen'], device_map="cpu")

    data = prepare_dataset(args.model_id, qd)
    tokenizer = load_tokenizer(args.model_id)
    dataloader = get_calib_loader(data, tokenizer, qd['num_calib_samples'], qd['seed'], qd['seqlen'])

    t_cal = time.time()
    if args.collect_hin:
        raw_stats = I.collect_hin(model, dataloader, "cuda", args.hin_dir, strategy=qd['calib_strategy'])
        if raw_stats is None:  # H_in already cached; still need their stats
            raw_stats = collect_stats(model, dataloader, "cuda", strategy=qd['calib_strategy'])
    else:
        raw_stats = collect_stats(model, dataloader, "cuda", strategy=qd['calib_strategy'])
    shrunk_stats = get_shrunk_stats(raw_stats, shrinkage=qd['calib_shrinkage'])
    model = register_stats(model, shrunk_stats)
    t_cal = time.time() - t_cal

    vopts = json.loads(args.variant_opts)
    factorize_fn = I.factorize_stock
    if args.variant != "stock":
        import variants
        factorize_fn = variants.get_factorize_fn(args.variant, vopts)

    ctx = I.Ctx(tag=args.tag, model_id=args.model_id, hin_dir=args.hin_dir,
                factorize_fn=factorize_fn, do_step3=not args.no_step3,
                out_json=os.path.join(args.outdir, f"{args.tag}.json"),
                ppl_after_block=not args.no_ppl_after_block)
    ctx.rec["args"] = vars(args)
    ctx.rec["variant_opts"] = vopts
    ctx.rec["timing"]["calibration_s"] = t_cal

    model = I.compress_block_recon_instr(model, fp_model, dataloader, qd, ctx)

    if qc.tune_model:
        t = time.time()
        model = compress_model_recon(model, fp_model, dataloader, qd)
        ctx.rec["timing"]["global_kd_s"] = time.time() - t

    ctx.rec["bpw"] = bpw_report(model, qd)
    cleanup_memory()
    model.eval().cuda()
    res = evaluate_model(model=model, tokenizer=tokenizer, tasks_str="",
                         eval_ppl=args.ppl_task, batch_size=1)
    ctx.rec["ppl"] = {k: v["ppl"] for k, v in res.items()}
    ctx.rec["timing"]["total_s"] = time.time() - t0
    ctx.dump()
    print("[INSTR] FINAL " + json.dumps({"tag": args.tag, "seed": args.seed,
                                         "ppl": ctx.rec["ppl"],
                                         "bpw": ctx.rec["bpw"]["bpw_factorized"],
                                         "timing": dict(ctx.rec["timing"])}))


if __name__ == "__main__":
    main()
