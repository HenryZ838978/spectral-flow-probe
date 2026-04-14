"""CLI entry point: sfp <model_path> [options]"""
from __future__ import annotations

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="sfp",
        description="Spectral Flow Probe — 20-minute geometric diagnostic for any Transformer",
    )
    parser.add_argument("model", help="HuggingFace model path or local directory")
    parser.add_argument("-n", "--n-prompts", type=int, default=50,
                        help="Number of prompts (default: 50 built-in)")
    parser.add_argument("-o", "--output", default=None,
                        help="Save JSON report to this path")
    parser.add_argument("--plot", default=None,
                        help="Save diagnostic plot to this path (png/pdf)")
    parser.add_argument("--no-moe", action="store_true",
                        help="Skip MoE routing analysis")
    parser.add_argument("--dtype", default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Model dtype (default: bf16)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [SFP] %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch
    from .probe import SpectralProbe
    from .prompts import DEFAULT_PROMPTS

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

    probe = SpectralProbe(args.model, dtype=dtype_map[args.dtype])
    prompts = DEFAULT_PROMPTS[:args.n_prompts]
    report = probe.run(prompts=prompts, check_moe=not args.no_moe)

    print("\n" + "=" * 60)
    print(report.summary())
    print("=" * 60)

    dx = report.diagnose()
    print(f"\nDiagnosis:")
    for k, v in dx.items():
        if not k.startswith("_"):
            print(f"  {k}: {v}")

    if args.output:
        report.to_json(args.output)
        print(f"\nJSON saved to {args.output}")

    if args.plot:
        from .plot import plot_diagnosis
        plot_diagnosis(report, save=args.plot)
        print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    main()
