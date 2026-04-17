"""CLI entry points: sfp <model_path> [options]"""
from __future__ import annotations

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(
        prog="sfp",
        description="Spectral Flow Probe v2 — 7-band Phased Array Radar for any Transformer",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ─── scan ──────────────────────────────────────────────────
    p_scan = sub.add_parser("scan", help="Run a 7-band radar scan on a single model")
    p_scan.add_argument("model", help="HF repo ID or local path")
    p_scan.add_argument("-o", "--output", default=None, help="Save JSON to this path")
    p_scan.add_argument("--plot", default=None, help="Save radar PNG to this path")
    p_scan.add_argument("--depth", action="store_true",
                        help="Also measure per-layer depth profiles (slower)")
    p_scan.add_argument("--dtype", default="bf16",
                        choices=["bf16", "fp16", "fp32"])

    # ─── compare ───────────────────────────────────────────────
    p_cmp = sub.add_parser("compare", help="Compare two model fingerprints")
    p_cmp.add_argument("model_a", help="Base model path or HF ID")
    p_cmp.add_argument("model_b", help="Instruct model path or HF ID")
    p_cmp.add_argument("-o", "--output", default=None)
    p_cmp.add_argument("--plot", default=None)
    p_cmp.add_argument("--dtype", default="bf16",
                       choices=["bf16", "fp16", "fp32"])

    # ─── rotate ────────────────────────────────────────────────
    p_rot = sub.add_parser("rotate",
                            help="Test isovolumetric rotation between two checkpoints")
    p_rot.add_argument("model_a", help="Base model path")
    p_rot.add_argument("model_b", help="Instruct model path")
    p_rot.add_argument("--gpu", type=int, default=0, help="GPU id (or -1 for CPU)")
    p_rot.add_argument("-o", "--output", default=None)

    # ─── profile ───────────────────────────────────────────────
    p_prof = sub.add_parser("profile",
                             help="Single-model spectrum profile (no pair needed)")
    p_prof.add_argument("model", help="Model path or HF ID")
    p_prof.add_argument("--gpu", type=int, default=0)
    p_prof.add_argument("-o", "--output", default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [SFP] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.cmd == "scan":
        _cmd_scan(args)
    elif args.cmd == "compare":
        _cmd_compare(args)
    elif args.cmd == "rotate":
        _cmd_rotate(args)
    elif args.cmd == "profile":
        _cmd_profile(args)


def _dtype(name):
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _cmd_scan(args):
    from .probe import SpectralProbe
    from .plot import plot_radar

    probe = SpectralProbe(args.model, dtype=_dtype(args.dtype))
    fp = probe.scan(depth_profile=args.depth)
    print("\n" + "=" * 70)
    print(fp.summary())
    print("=" * 70)

    if args.output:
        fp.to_json(args.output)
        print(f"\nJSON saved to {args.output}")
    if args.plot:
        plot_radar(fp, save=args.plot)
        print(f"Plot saved to {args.plot}")


def _cmd_compare(args):
    from .probe import SpectralProbe
    from .fingerprint import BandwidthComparison
    from .plot import plot_comparison

    probe_a = SpectralProbe(args.model_a, dtype=_dtype(args.dtype))
    fp_a = probe_a.scan()
    del probe_a

    probe_b = SpectralProbe(args.model_b, dtype=_dtype(args.dtype))
    fp_b = probe_b.scan()
    del probe_b

    cmp = BandwidthComparison(
        fp_a=fp_a, fp_b=fp_b,
        label_a=args.model_a.split("/")[-1],
        label_b=args.model_b.split("/")[-1],
    )
    print("\n" + "=" * 70)
    print(cmp.summary())
    print("=" * 70)

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(cmp.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\nJSON saved to {args.output}")
    if args.plot:
        plot_comparison(cmp, save=args.plot)
        print(f"Plot saved to {args.plot}")


def _cmd_rotate(args):
    from .rotation import RotationAnalyzer
    ra = RotationAnalyzer()
    gpu_id = args.gpu if args.gpu >= 0 else "cpu"
    report = ra.compare(args.model_a, args.model_b, gpu_id=gpu_id)
    print("\n" + "=" * 70)
    print(report.summary())
    print("=" * 70)
    if args.output:
        report.to_json(args.output)
        print(f"\nJSON saved to {args.output}")


def _cmd_profile(args):
    from .rotation import RotationAnalyzer
    ra = RotationAnalyzer()
    gpu_id = args.gpu if args.gpu >= 0 else "cpu"
    profile = ra.profile(args.model, gpu_id=gpu_id)
    print("\n" + "=" * 70)
    print(profile.summary())
    print("=" * 70)
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\nJSON saved to {args.output}")


if __name__ == "__main__":
    main()
