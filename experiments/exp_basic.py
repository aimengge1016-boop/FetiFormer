import os
import torch
from model import iTransformer, iTransformer_IRON


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'iTransformer': iTransformer,
            'iTransformer_IRON': iTransformer_IRON,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # Do NOT blindly override CUDA_VISIBLE_DEVICES for single-GPU runs.
            # Many scripts/users select the physical GPU via CUDA_VISIBLE_DEVICES (e.g. "2"),
            # and then expect `--gpu 0` (within the visible set) to work.
            if self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
            else:
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.args.gpu))
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
