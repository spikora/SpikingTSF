import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        try:
            self.model = self._build_model().to(args.rank)
        except Exception:
            self.model = self._build_model().cuda()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def valid(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
