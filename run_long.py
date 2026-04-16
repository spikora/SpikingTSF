from exp.exp_ETT import Exp_ETT
import argparse
import torch

parser = argparse.ArgumentParser(description='SpikingTSF — long-term forecasting')

# ── Model ──────────────────────────────────────────────────────────────────
parser.add_argument('--model', type=str, default='SpikF',
                    choices=['SpikF', 'iSpikformer', 'SpikeRNN',
                             'SpikTCN', 'SpikGRU', 'TSLIF', 'DLinear',
                             'Spikformer', 'Spikingformer', 'ITransformer'],
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

# ── SNN hyperparameters ────────────────────────────────────────────────────
parser.add_argument('--T', type=int, default=16, help='SNN time steps')
parser.add_argument('--tau', type=float, default=2.0, help='LIF time constant')
parser.add_argument('--levels', type=int, default=2, help='number of blocks/levels')
parser.add_argument('--patch_num', type=int, default=48, help='number of patches (SpikF/TSLIF)')
parser.add_argument('--patch_dim', type=int, default=32, help='patch/TCN hidden dim')
parser.add_argument('--alpha', type=float, default=2.0, help='alpha (SpikF/iSpikformer)')
parser.add_argument('--hidden_dim', type=int, default=720, help='dense hidden size')

# ── Transformer hyperparameters (Spikformer / Spikingformer / ITransformer) ─
parser.add_argument('--d_model', type=int, default=256,
                    help='attention / embedding dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d_ff', type=int, default=1024,
                    help='feedforward hidden dim (default 4 × d_model)')
parser.add_argument('--common_thr', type=float, default=1.0,
                    help='LIF spike threshold (Spikformer / Spikingformer)')
parser.add_argument('--qk_scale', type=float, default=0.125,
                    help='attention score scale (Spikformer / Spikingformer)')
parser.add_argument('--encoder_type', type=str, default='conv',
                    choices=['conv', 'delta', 'repeat'],
                    help='spike encoder type (Spikformer / Spikingformer)')

# ── ANN / general hyperparameters ─────────────────────────────────────────
parser.add_argument('--kernel_size', type=int, default=3, help='Conv kernel size (SpikTCN)')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--moving_avg', type=int, default=25, help='trend kernel size (DLinear)')
parser.add_argument('--individual', action='store_true', default=False,
                    help='per-variate linear layers (DLinear)')

# ── Training ───────────────────────────────────────────────────────────────
parser.add_argument('--cols', type=str, nargs='+')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--loss', type=str, default='mae', choices=['mae', 'mse'])
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--model_name', type=str, default='SpikF')
parser.add_argument('--save', type=int, default=0)

# ── Hardware ───────────────────────────────────────────────────────────────
parser.add_argument('--gpu', type=int, default=0, help='CUDA device index')

args = parser.parse_args()
args.rank = args.gpu   # alias used internally by Exp_Basic / exp_ETT


def main(rank):
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    args.rank = rank

    setting = '{}_{}_{}_sl{}_pl{}_lr{}_bs{}'.format(
        args.model, args.data, args.features,
        args.seq_len, args.pred_len, args.lr, args.batch_size,
    )

    exp = Exp_ETT(args)

    print(f'>>>>>>>  training : {setting}  >>>>>>>>>>>>>>>')
    exp.train(setting)

    print(f'>>>>>>>  testing  : {setting}  <<<<<<<<<<<<<<<')
    mse, mae = exp.test(setting)
    print(f'Final | mse: {mse:.4f} | mae: {mae:.4f}')


if __name__ == '__main__':
    main(args.rank)
