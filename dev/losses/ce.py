import torch.nn.functional as F


class CE:
    def __call__(self, action_preds, action_targs, **kwargs):
        return F.cross_entropy(action_preds, action_targs)