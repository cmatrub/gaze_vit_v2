import torch.nn.functional as F
from einops import rearrange, reduce

class CEGazePostAvg_DS: 
    '''u mean pool the attn logits across the heads, then u calculate cross entropy. this is for the downsampling scheme'''
    def __init__(self, reg_lambda=1.0):
        self.reg_lambda = reg_lambda


    def _preprocess_gaze_preds_and_targs(self, gaze_preds, gaze_targs):
        # mean pool across the num_heads dimension
        gaze_preds = reduce(gaze_preds, 'b h p -> b p', reduction='mean')

        return gaze_preds, gaze_targs


    def __call__(self, action_preds, action_targs, gaze_preds, gaze_targs, **kwargs):
        ''' averages over batch_size and num_heads in one go. if u wanna see the intermediary ce values for each head separately u need to define a separate loss function'''
        ce = F.cross_entropy(action_preds, action_targs)

        gaze_preds, gaze_targs = self._preprocess_gaze_preds_and_targs(gaze_preds, gaze_targs)

        reg = self.reg_lambda * F.cross_entropy(gaze_preds, gaze_targs)

        return ce + reg