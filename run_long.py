from exp.exp_ETT import Exp_ETT
import argparse
import numpy as np
import torch
import yaml

# Pre-parse --config only, so we can set YAML values as argparse defaults
# before the full parse runs.  CLI flags still override YAML.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--config', type=str, default=None)
_pre_args, _ = _pre.parse_known_args()

parser = argparse.ArgumentParser(description='SpikingTSF — long-term forecasting')

# Config file (YAML).  All keys are treated as parser defaults; explicit CLI
# flags always win.
parser.add_argument('--config', type=str, default=None,
                    help='path to YAML config (e.g. configs/SpikF/ETTh1.yaml)')

# ── Model ──────────────────────────────────────────────────────────────────
parser.add_argument('--model', type=str, default='SpikF',
                    choices=[
                        # TS-LIF two-compartment models
                        'TSGRU', 'TSTCN', 'TSFormer',
                        # SeqSNN-RPE adapted models
                        'iSpikformer', 'SpikeRNN', 'SpikTCN', 'SpikGRU',
                        'Spikformer', 'Spikingformer', 'QKFormer',
                        # Clock-driven SNN model
                        'SpikF',
                        # ANN baselines
                        'DLinear', 'ITransformer',
                    ],
                    help='model name')

# ── Dataset ────────────────────────────────────────────────────────────────
parser.add_argument('--data', type=str, default='ETTh1',
                    choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'traffic',
                             'weather', 'electricity', 'metr-la', 'pems-bay',
                             'solar-energy', 'exchange',
                             'solar-txt',   # solar_AL.txt via Dataset_TXT
                             'elec-txt',    # electricity.txt via Dataset_TXT (SeqSNN format)
                             ])
parser.add_argument('--root_path', type=str, default='./datasets/long/')
parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--features', type=str, default='M', choices=['S', 'M', 'MS'])
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--checkpoints', type=str, default='exp/run_ETT/')

# ── Forecasting task ───────────────────────────────────────────────────────
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=336)
parser.add_argument('--label_len', type=int, default=48)

# ── SNN core hyperparameters ──────────────────────────────────────────────
parser.add_argument('--T', type=int, default=16, help='SNN time steps')
parser.add_argument('--tau', type=float, default=2.0, help='LIF membrane time constant')
parser.add_argument('--levels', type=int, default=2,
                    help='number of blocks / layers (maps to "blocks" in models)')

# ── Dimension / capacity hyperparameters ─────────────────────────────────
# alpha: main capacity knob — maps to d_model for transformer-family models
#        and to hidden_dim for RNN/TCN-family models.
# SpikF uses it as a float multiplier; all others expect an integer dimension.
parser.add_argument('--alpha', type=float, default=256.0,
                    help='main capacity param: d_model (transformer) or '
                         'hidden_dim (RNN/TCN); integer for most models, '
                         'float multiplier for SpikF')
parser.add_argument('--patch_num', type=int, default=48,
                    help='number of patches (SpikF)')
parser.add_argument('--patch_dim', type=int, default=32,
                    help='patch embedding dim (SpikF)')
parser.add_argument('--hidden_dim', type=int, default=720,
                    help='dense/decoder hidden size (SpikF)')

# ── Transformer / attention hyperparameters ───────────────────────────────
parser.add_argument('--n_heads', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d_ff', type=int, default=None,
                    help='feedforward hidden dim (default: 4 × d_model inside model)')
parser.add_argument('--common_thr', type=float, default=1.0,
                    help='LIF spike threshold used in attention blocks')
parser.add_argument('--qk_scale', type=float, default=0.125,
                    help='Q·K attention score scaling factor')
parser.add_argument('--attn_type', type=str, default='standard',
                    choices=['standard', 'xnor_gray', 'xnor_log'],
                    help='attention variant (Spikformer / Spikingformer / QKFormer)')
parser.add_argument('--gray_bits', type=int, default=10,
                    help='bit-width for Gray-code attention (xnor_gray / xnor_log)')

# ── Spike encoder ─────────────────────────────────────────────────────────
parser.add_argument('--encoder_type', type=str, default='conv',
                    choices=['conv', 'delta', 'repeat'],
                    help='spike encoder type')

# ── Positional encoding ───────────────────────────────────────────────────
parser.add_argument('--pe_type', type=str, default='none',
                    choices=['none', 'learn', 'static', 'conv', 'neuron', 'random'],
                    help='positional encoding type (SeqSNN-family models)')
parser.add_argument('--pe_mode', type=str, default='add',
                    choices=['add', 'concat'],
                    help='how PE is combined with features (add or concat)')
parser.add_argument('--num_pe_neuron', type=int, default=10,
                    help='number of PE neurons for neuron/random PE type')
parser.add_argument('--neuron_pe_scale', type=float, default=1000.0,
                    help='frequency scale for neuron PE')

# ── ANN / general architecture hyperparameters ────────────────────────────
parser.add_argument('--kernel_size', type=int, default=3,
                    help='dilated conv kernel size (SpikTCN / TSTCN)')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--moving_avg', type=int, default=25,
                    help='trend decomposition kernel size (DLinear)')
parser.add_argument('--individual', action='store_true', default=False,
                    help='per-variate independent linear layers (DLinear)')
# ITransformer new-style config passthrough
parser.add_argument('--d_model', type=int, default=256,
                    help='model embedding dim for new-style Model(args) models '
                         '(ITransformer)')
parser.add_argument('--e_layers', type=int, default=2,
                    help='encoder layers (ITransformer new-style)')
parser.add_argument('--d_layers', type=int, default=1,
                    help='decoder layers (new-style models)')
parser.add_argument('--factor', type=int, default=1,
                    help='attention factor (new-style models)')

# ── Optimizer ─────────────────────────────────────────────────────────────
parser.add_argument('--optimizer', type=str, default='Adam',
                    choices=['Adam', 'AdamW', 'SGD', 'RMSprop'],
                    help='optimizer class (torch.optim.*)')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='optimizer weight decay (L2 regularisation)')
parser.add_argument('--grad_clip', type=float, default=0.0,
                    help='max gradient norm for clipping (0 = disabled)')

# ── LR scheduler ──────────────────────────────────────────────────────────
parser.add_argument('--scheduler', type=str, default='cosine',
                    choices=['cosine', 'step', 'none'],
                    help='learning rate scheduler')
parser.add_argument('--scheduler_T_max', type=int, default=20,
                    help='T_max for CosineAnnealingLR')
parser.add_argument('--scheduler_step', type=int, default=1,
                    help='step_size for StepLR (epochs between decays)')
parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                    help='multiplicative decay factor for StepLR')

# ── Training ───────────────────────────────────────────────────────────────
parser.add_argument('--cols', type=str, nargs='+',
                    help='specific columns to select from dataset')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=10,
                    help='early stopping patience (epochs without improvement)')
parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
parser.add_argument('--loss', type=str, default='mae', choices=['mae', 'mse'],
                    help='training loss function')
parser.add_argument('--random_seed', type=int, default=42,
                    help='seed for single-run mode (ignored when --itr > 1)')
parser.add_argument('--itr', type=int, default=3,
                    help='number of independent runs (1 = single run, no averaging)')
parser.add_argument('--save', type=int, default=0,
                    help='save prediction arrays to disk (1=yes)')

# ── Hardware ───────────────────────────────────────────────────────────────
parser.add_argument('--gpu', type=int, default=0, help='CUDA device index')

# Load YAML config and use as defaults (CLI args still override)
if _pre_args.config is not None:
    with open(_pre_args.config) as _f:
        _cfg = yaml.safe_load(_f)
    parser.set_defaults(**_cfg)

args = parser.parse_args()
args.rank = args.gpu   # alias used internally by Exp_Basic / exp_ETT


# Seeds chosen for good statistical independence: 42 (de-facto ML default),
# 1234 (common baseline), 3407 ("torch.manual_seed(3407) is all you need"),
# 100, 999 (spread far from the first three for 5-run mode).
_SEEDS = [42, 1234, 3407]

_METRIC_KEYS = ['mae', 'mse', 'rmse', 'rse', 'r2', 'mape', 'mspe', 'corr']


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def main(rank):
    args.rank = rank
    num_runs = args.itr

    # For a single run, respect --random_seed; for multi-run use the preset list.
    if num_runs == 1:
        seeds = [args.random_seed]
    else:
        seeds = _SEEDS[:num_runs]

    base_setting = (
        f'{args.model}_{args.data}_{args.features}'
        f'_sl{args.seq_len}_pl{args.pred_len}'
        f'_T{args.T}_lv{args.levels}_a{int(args.alpha)}'
        f'_lr{args.lr}_bs{args.batch_size}'
        f'_opt{args.optimizer}_wd{args.weight_decay}'
        f'_sch{args.scheduler}'
    )

    all_metrics = {k: [] for k in _METRIC_KEYS}

    for run_idx in range(num_runs):
        seed = seeds[run_idx]
        set_seed(seed)

        setting = f'{base_setting}_seed{seed}'
        print(f'\n{"="*60}')
        print(f'  Run {run_idx + 1}/{num_runs}  |  seed={seed}')
        print(f'{"="*60}')

        exp = Exp_ETT(args)

        print(f'>>>>>>>  training : {setting}  >>>>>>>>>>>>>>>')
        exp.train(setting)

        print(f'>>>>>>>  testing  : {setting}  <<<<<<<<<<<<<<<')
        m = exp.test(setting)

        print(
            f'\nRun {run_idx + 1} results:\n'
            f'  MAE={m["mae"]:.4f}  MSE={m["mse"]:.4f}  RMSE={m["rmse"]:.4f}\n'
            f'  RSE={m["rse"]:.4f}  R²={m["r2"]:.4f}\n'
            f'  MAPE={m["mape"]:.4f}  MSPE={m["mspe"]:.4f}  CORR={m["corr"]:.4f}'
        )

        for k in _METRIC_KEYS:
            all_metrics[k].append(m[k])

    # ── Final summary ──────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  Final Results  ({num_runs} run{"s" if num_runs > 1 else ""}, '
          f'seeds={seeds})')
    print(f'{"="*60}')
    print(f'{"Metric":<8}  {"Mean":>10}  {"Std":>10}')
    print(f'{"-"*32}')
    for k in _METRIC_KEYS:
        vals = np.array(all_metrics[k])
        print(f'{k.upper():<8}  {vals.mean():>10.4f}  {vals.std():>10.4f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main(args.rank)
