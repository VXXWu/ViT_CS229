"""Launch all 6 attention mechanism variants on Modal.

Usage:
  # Launch all 6 variants (100 epochs each)
  python launch_zoo.py

  # Launch specific variants
  python launch_zoo.py --variants softmax sigmoid relu

  # Extend to 400 epochs
  python launch_zoo.py --epochs 400

  # Resume interrupted runs
  python launch_zoo.py --resume
"""

import argparse
import subprocess
import sys

VARIANTS = [
    # (attn_type, norm_type, description)
    ('softmax',  'layernorm', 'Standard softmax attention (baseline)'),
    ('sigmoid',  'layernorm', 'Sigmoid attention (Ramapuram et al. ICLR 2025)'),
    ('relu',     'layernorm', 'ReLU attention (Wortsman et al. 2023)'),
    ('gated',    'layernorm', 'Gated SDPA (Qiu et al. NeurIPS 2025 Best Paper)'),
    ('linear',   'layernorm', 'Linear attention (Katharopoulos et al. 2020)'),
    ('softmax',  'dyt',       'DyT normalization (Zhu et al. CVPR 2025)'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Subset of variants to launch (e.g., softmax sigmoid)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    args = parser.parse_args()

    selected = VARIANTS
    if args.variants:
        # Match by attn_type or by "attn_norm" combo
        selected = []
        for v in VARIANTS:
            key = v[0] if v[1] == 'layernorm' else f"{v[0]}_{v[1]}"
            if v[0] in args.variants or key in args.variants:
                selected.append(v)

    print(f"Launching {len(selected)} variants for {args.epochs} epochs:\n")
    for attn, norm, desc in selected:
        key = attn if norm == 'layernorm' else f"{attn}_{norm}"
        print(f"  {key:20s} — {desc}")
    print()

    for attn, norm, desc in selected:
        cmd = [
            sys.executable, '-m', 'modal', 'run', 'train_modal.py::train',
            '--attn-type', attn,
            '--norm-type', norm,
            '--epochs', str(args.epochs),
        ]
        if args.resume:
            cmd.extend(['--resume', 'True'])

        key = attn if norm == 'layernorm' else f"{attn}_{norm}"
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Launching: {key}")
        print(f"  {' '.join(cmd)}")

        if not args.dry_run:
            subprocess.run(cmd)


if __name__ == '__main__':
    main()
