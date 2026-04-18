import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import warnings

# --- SNN models (spikingjelly clock_driven backend) ---
from models.SpikF import SpikF
from models.iSpikformer import iSpikformer
from models.SpikeRNN import SpikeRNN
from models.SpikTCN import SpikeTCN
from models.SpikGRU import SpikeGRU
from models.TSGRU import TSGRU
from models.TSTCN import TSTCN
from models.TSFormer import TSFormer
from models.TSLIF import Model as TSLIF

# --- SNN models (spikingjelly activation_based backend, adapted from SeqSNN) ---
from models.Spikformer import Spikformer
from models.Spikingformer import Spikingformer
from models.QKFormer import QKFormer

# --- ANN baselines ---
from models.DLinear import Model as DLinear
from models.ITransformer import Model as ITransformer

from utils.metrics import metric, metric_
from utils.tools import EarlyStopping
warnings.filterwarnings('ignore')
from data_provider.ETT_data_loader import (
    Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute, Solar
)
from data_provider.traffic_data_loader import Dataset_H5, Dataset_TXT
from exp.exp_basic import Exp_Basic

# Both reset APIs needed: clock_driven models are reset by cd_functional;
# activation_based models (Spikformer, Spikingformer) also self-reset inside
# forward(), but we call both here defensively so external callers are safe.
from spikingjelly.clock_driven import functional as cd_functional
try:
    from spikingjelly.activation_based import functional as ab_functional
    _HAS_AB = True
except ImportError:
    _HAS_AB = False

torch.autograd.set_detect_anomaly(True)
from torch.optim.lr_scheduler import CosineAnnealingLR


# Models that use the Model(configs) convention — enc_in injected before init
_NEW_STYLE_MODELS = {
    'TSLIF', 'DLinear', 'ITransformer',
}

# Models that are pure ANN (no spikingjelly reset needed)
_ANN_MODELS = {'DLinear', 'ITransformer'}

_NEW_STYLE_CLASS = {
    'TSLIF': TSLIF,
    'DLinear': DLinear,
    'ITransformer': ITransformer,
}


class Exp_ETT(Exp_Basic):
    def __init__(self, args):
        super(Exp_ETT, self).__init__(args)
        self.test_loader = self._get_data(flag='test')

    def _build_model(self):
        # --- resolve input dimension from dataset name ---
        if self.args.features == 'S':
            self.input_dim = 1
        elif self.args.features == 'M':
            _dim_map = {
                'ETTh1': 7, 'ETTh2': 7, 'ETTm1': 7, 'ETTm2': 7,
                'ECL': 321, 'electricity': 321,
                'elec-txt': 321,              # electricity.txt (SeqSNN / LSTNET format)
                'exchange': 8,
                'traffic': 862,
                'weather': 21,
                'illness': 7,
                'metr-la': 207,
                'pems-bay': 325,
                'solar-energy': 137, 'solar_AL': 137,
                'Solar': 137,
                'solar-txt': 137,             # solar_AL.txt via Dataset_TXT
            }
            self.input_dim = _dim_map.get(self.args.data, 1)
        else:
            self.input_dim = 1

        # Inject enc_in/c_out so Model(configs) convention works
        self.args.enc_in = self.input_dim
        self.args.c_out = self.input_dim

        name = self.args.model

        if name in _NEW_STYLE_MODELS:
            model = _NEW_STYLE_CLASS[name](self.args)

        elif name == 'iSpikformer':
            model = iSpikformer(
                input_len=self.args.seq_len,
                T=self.args.T,
                blocks=self.args.levels,
                D=self.input_dim,
                pred_len=self.args.pred_len,
                tau=self.args.tau,
                d_model=self.args.alpha,
                d_ff=getattr(self.args, 'd_ff', None),
                heads=getattr(self.args, 'heads', 8),
                common_thr=getattr(self.args, 'common_thr', 1.0),
                qk_scale=getattr(self.args, 'qk_scale', 0.125),
                encoder_type=getattr(self.args, 'encoder_type', 'conv'),
            )
        elif name == 'SpikeRNN':
            model = SpikeRNN(
                input_len=self.args.seq_len,
                T=self.args.T,
                blocks=self.args.levels,
                D=self.input_dim,
                pred_len=self.args.pred_len,
                tau=self.args.tau,
                hidden_dim=self.args.alpha,
                encoder_type=getattr(self.args, 'encoder_type', 'conv'),
                pe_type=getattr(self.args, 'pe_type', 'none'),
                pe_mode=getattr(self.args, 'pe_mode', 'add'),
                num_pe_neuron=getattr(self.args, 'num_pe_neuron', 10),
                neuron_pe_scale=getattr(self.args, 'neuron_pe_scale', 1000.0),
            )
        elif name == 'SpikeTCN':
            model = SpikeTCN(
                input_len=self.args.seq_len,
                T=self.args.T,
                blocks=self.args.levels,
                D=self.input_dim,
                pred_len=self.args.pred_len,
                tau=self.args.tau,
                hidden_dim=self.args.alpha,
                kernel_size=getattr(self.args, 'kernel_size', 3),
                encoder_type=getattr(self.args, 'encoder_type', 'conv'),
                pe_type=getattr(self.args, 'pe_type', 'none'),
                num_pe_neuron=getattr(self.args, 'num_pe_neuron', 10),
                neuron_pe_scale=getattr(self.args, 'neuron_pe_scale', 1000.0),
            )
        elif name == 'SpikeGRU':
            model = SpikeGRU(
                input_len=self.args.seq_len,
                T=self.args.T,
                blocks=self.args.levels,
                D=self.input_dim,
                pred_len=self.args.pred_len,
                tau=self.args.tau,
                hidden_dim=self.args.alpha,
                encoder_type=getattr(self.args, 'encoder_type', 'conv'),
                pe_type=getattr(self.args, 'pe_type', 'none'),
                pe_mode=getattr(self.args, 'pe_mode', 'add'),
                num_pe_neuron=getattr(self.args, 'num_pe_neuron', 10),
                neuron_pe_scale=getattr(self.args, 'neuron_pe_scale', 1000.0),
            )
        elif name == 'Spikformer':
            model = Spikformer(
                input_len=self.args.seq_len,
                T=self.args.T,
                blocks=self.args.levels,
                D=self.input_dim,
                pred_len=self.args.pred_len,
                tau=self.args.tau,
                d_model=self.args.alpha,
                d_ff=getattr(self.args, 'd_ff', None),
                heads=getattr(self.args, 'heads', 8),
                common_thr=getattr(self.args, 'common_thr', 1.0),
                qk_scale=getattr(self.args, 'qk_scale', 0.125),
                encoder_type=getattr(self.args, 'encoder_type', 'conv'),
                pe_type=getattr(self.args, 'pe_type', 'none'),
                pe_mode=getattr(self.args, 'pe_mode', 'add'),
                attn_type=getattr(self.args, 'attn_type', 'standard'),
                gray_bits=getattr(self.args, 'gray_bits', 10),
                num_pe_neuron=getattr(self.args, 'num_pe_neuron', 10),
                neuron_pe_scale=getattr(self.args, 'neuron_pe_scale', 1000.0),
                dropout=getattr(self.args, 'dropout', 0.1),
            )
        elif name == 'Spikingformer':
            model = Spikingformer(
                input_len=self.args.seq_len,
                T=self.args.T,
                blocks=self.args.levels,
                D=self.input_dim,
                pred_len=self.args.pred_len,
                tau=self.args.tau,
                d_model=self.args.alpha,
                d_ff=getattr(self.args, 'd_ff', None),
                heads=getattr(self.args, 'heads', 8),
                common_thr=getattr(self.args, 'common_thr', 1.0),
                qk_scale=getattr(self.args, 'qk_scale', 0.125),
                encoder_type=getattr(self.args, 'encoder_type', 'conv'),
                pe_type=getattr(self.args, 'pe_type', 'conv'),
                attn_type=getattr(self.args, 'attn_type', 'standard'),
                gray_bits=getattr(self.args, 'gray_bits', 10),
                dropout=getattr(self.args, 'dropout', 0.1),
            )
        elif name == 'QKFormer':
            model = QKFormer(
                input_len=self.args.seq_len,
                T=self.args.T,
                blocks=self.args.levels,
                D=self.input_dim,
                pred_len=self.args.pred_len,
                tau=self.args.tau,
                d_model=self.args.alpha,
                d_ff=getattr(self.args, 'd_ff', None),
                heads=getattr(self.args, 'heads', 8),
                common_thr=getattr(self.args, 'common_thr', 1.0),
                qk_scale=getattr(self.args, 'qk_scale', 0.125),
                encoder_type=getattr(self.args, 'encoder_type', 'conv'),
                pe_type=getattr(self.args, 'pe_type', 'conv'),
                attn_type=getattr(self.args, 'attn_type', 'standard'),
                gray_bits=getattr(self.args, 'gray_bits', 10),
                dropout=getattr(self.args, 'dropout', 0.1),
            )
        elif name == 'TSGRU':
            model = TSGRU(
                input_len=self.args.seq_len,
                T=self.args.T,
                blocks=self.args.levels,
                D=self.input_dim,
                pred_len=self.args.pred_len,
                tau=self.args.tau,
                hidden_dim=self.args.alpha,
                encoder_type=getattr(self.args, 'encoder_type', 'conv'),
                pe_type=getattr(self.args, 'pe_type', 'none'),
                pe_mode=getattr(self.args, 'pe_mode', 'add'),
                num_pe_neuron=getattr(self.args, 'num_pe_neuron', 10),
                neuron_pe_scale=getattr(self.args, 'neuron_pe_scale', 1000.0),
            )
        elif name == 'TSTCN':
            model = TSTCN(
                input_len=self.args.seq_len,
                T=self.args.T,
                blocks=self.args.levels,
                D=self.input_dim,
                pred_len=self.args.pred_len,
                tau=self.args.tau,
                hidden_dim=self.args.alpha,
                kernel_size=getattr(self.args, 'kernel_size', 3),
                encoder_type=getattr(self.args, 'encoder_type', 'conv'),
                pe_type=getattr(self.args, 'pe_type', 'none'),
                num_pe_neuron=getattr(self.args, 'num_pe_neuron', 10),
                neuron_pe_scale=getattr(self.args, 'neuron_pe_scale', 1000.0),
            )
        elif name == 'TSFormer':
            model = TSFormer(
                input_len=self.args.seq_len,
                T=self.args.T,
                blocks=self.args.levels,
                D=self.input_dim,
                pred_len=self.args.pred_len,
                tau=self.args.tau,
                d_model=self.args.alpha,
                d_ff=getattr(self.args, 'd_ff', None),
                heads=getattr(self.args, 'heads', 8),
                common_thr=getattr(self.args, 'common_thr', 1.0),
                qk_scale=getattr(self.args, 'qk_scale', 0.125),
                encoder_type=getattr(self.args, 'encoder_type', 'conv'),
            )
        else:
            # Default: SpikF
            model = SpikF(
                self.args.seq_len, self.args.patch_num, self.args.patch_dim,
                self.args.T, self.args.levels, self.input_dim,
                self.args.pred_len, self.args.tau, self.args.alpha,
                self.args.hidden_dim,
            )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {name} | Parameters: {total_params:,}")
        return model

    def _get_data(self, flag):
        args = self.args

        # CSV/custom datasets — 70/10/20 split (TSLib default)
        # H5 traffic datasets (PEMS-BAY, METR-LA) — 60/20/20 split (SeqSNN)
        # TXT sensor datasets (solar, electricity-SeqSNN) — 60/20/20 split (SeqSNN)
        data_dict = {
            'ETTh1':      Dataset_ETT_hour,
            'ETTh2':      Dataset_ETT_hour,
            'ETTm1':      Dataset_ETT_minute,
            'ETTm2':      Dataset_ETT_minute,
            'weather':    Dataset_Custom,
            'ECL':        Dataset_Custom,
            'electricity': Dataset_Custom,    # CSV version (ECL.csv)
            'Solar':      Solar,
            'solar-energy': Solar,            # txt version via fixed Solar class
            'traffic':    Dataset_Custom,
            'exchange':   Dataset_Custom,
            'illness':    Dataset_Custom,
            # H5 traffic datasets — adapted from SeqSNN
            'metr-la':    Dataset_H5,
            'pems-bay':   Dataset_H5,
            # Plain-txt sensor datasets — adapted from SeqSNN
            'solar-txt':  Dataset_TXT,        # solar_AL.txt direct
            'elec-txt':   Dataset_TXT,        # electricity.txt (SeqSNN LSTNET format)
        }
        Data = data_dict[self.args.data]

        if flag in ('test', 'val'):
            shuffle_flag, drop_last, batch_size = False, False, 1
        else:
            shuffle_flag, drop_last, batch_size = True, True, args.batch_size

        # Dataset_H5 and Dataset_TXT don't use cols/target in the same way —
        # pass them as kwargs but the classes will ignore unsupported ones safely.
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            cols=args.cols,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999))

    def _select_criterion(self, losstype):
        return nn.L1Loss() if losstype == 'mae' else nn.MSELoss()

    def train(self, setting):
        torch.cuda.empty_cache()
        train_loader = self._get_data(flag='train')
        valid_loader = self._get_data(flag='val')
        self.test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        print(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=0.0002)
        model_optim = self._select_optimizer()
        scheduler = CosineAnnealingLR(model_optim, T_max=20)
        criterion = self._select_criterion(self.args.loss)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                pred, true = self._process_one_batch(batch_x, batch_y)

                if len(pred.shape) == 4:
                    # SpikF returns (T, B, pred_len, D) — average over T before loss
                    pred = pred.mean(dim=0)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    avg_loss = np.mean(train_loss[-100:])
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {avg_loss:.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left:.1f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.1f}s")
            train_loss = np.average(train_loss)
            valid_loss = self.valid(valid_loader, criterion, flag='valid')
            test_loss = self.valid(self.test_loader, criterion, flag='test')
            print(f"Epoch {epoch+1} | Train: {train_loss:.7f} | Valid: {valid_loss:.7f} | Test: {test_loss:.7f}")

            scheduler.step()
            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return path + '/checkpoint.pth'

    def valid(self, valid_loader, criterion, flag):
        torch.cuda.empty_cache()
        self.model.eval()
        total_loss, mses, maes, weights = [], [], [], []

        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                pred, true = self._process_one_batch(batch_x, batch_y)
                weights.append(true.shape[0] / self.args.batch_size)
                if len(pred.shape) == 4:
                    pred = pred.mean(dim=0)
                mae, mse = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
                mses.append(mse)
                maes.append(mae)
                total_loss.append(criterion(pred.detach().cpu(), true.detach().cpu()).item())

        total_loss = np.average(total_loss)
        mse = sum(w * m for w, m in zip(weights, mses)) / sum(weights)
        mae = sum(w * m for w, m in zip(weights, maes)) / sum(weights)
        print(f'  [{flag}] mse: {mse:.6f} | mae: {mae:.6f}')
        return total_loss

    def test(self, setting, evaluate=0):
        torch.cuda.empty_cache()
        self.model.eval()

        path = os.path.join(self.args.checkpoints, setting)
        self.model.load_state_dict(torch.load(path + '/checkpoint.pth'))

        preds, trues = [], []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                pred, true = self._process_one_batch(batch_x, batch_y)
                if len(pred.shape) == 4:
                    pred = pred.mean(dim=0)
                preds.append(pred.detach().cpu())
                trues.append(true.detach().cpu())

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print(preds.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, RSE, R2 = metric_(preds, trues)
        print(f'|  Normed  | mse: {mse:.7f} | mae: {mae:.7f} | RSE: {RSE:.7f} | R²: {R2:.7f} |')

        if self.args.save:
            folder_path = 'exp/ETT_results/' + setting + '/'
            os.makedirs(folder_path, exist_ok=True)
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return mse, mae

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.args.rank)
        batch_y = batch_y.float()

        # Reset SNN state before each forward pass.
        # cd_functional covers clock_driven neurons (SpikF, TSLIF, etc.)
        # ab_functional covers activation_based neurons (Spikformer, Spikingformer)
        # Each call is a no-op for neuron types it doesn't own.
        if self.args.model not in _ANN_MODELS:
            cd_functional.reset_net(self.model)
            if _HAS_AB:
                ab_functional.reset_net(self.model)

        outputs = self.model(batch_x)

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.rank)
        return outputs, batch_y
