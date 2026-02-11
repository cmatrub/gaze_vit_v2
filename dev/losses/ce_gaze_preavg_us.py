import torch.nn.functional as F
from einops import rearrange, repeat
from math import sqrt

class CEBeforeAvgUpsampleSoftmaxed:
    ''' upsamples the cls token's attention softmax values. take mean across heads afterwards. 
    Note: the code as it's written now assumes square patches (it infers patch size from the shapes of gaze_preds and gaze_targs). 
    if you want nonsquare patches you have refactor this code to accept the patch size tuple as an arg.'''

    def __init__(self, reg_lambda=1.0):
        self.reg_lambda = reg_lambda


    def _preprocess_gaze_preds_and_targs(self, gaze_preds, gaze_targs):

        h = gaze_preds.shape[1] # (b, h, p) where b = batch_size, h = number of heads, p = number of patches
        p = gaze_preds.shape[2]
        N = gaze_targs.shape[1] # N = 84^2
        patch_size = sqrt(N)/sqrt(p)
        l = sqrt(p)
         
        
        # broadcast across heads
        gaze_targs = repeat(gaze_targs, 'b N -> b heads N', h=h) 

        # upsample
        # for one patch it'd be 'b h 1 -> b h patch_size patch_size'
        # p patches means height = w width = sqrt(p). im denoting width and height as 'l' for square side length 
        # so u should go from b h p -> b h l*patch_size l*patch_size
        gaze_preds = repeat(gaze_preds, 'b h  -> b h (l patch_size) (l patch_size)', l=l, patch_size=patch_size)
        gaze_preds = rearrange(gaze_preds, 'b h l patch_size l patch_size -> b h (patch_size l patch_size)') # flatten


        #  collapse the batch_size and num_heads dim into one that cross_entropy will average over to return u a scalar (this averages across the batch and heads so u dont need to do 2 separate averages)
        gaze_preds = rearrange(gaze_preds, 'b h p -> (b h) p')
        gaze_targs = rearrange(gaze_targs, 'b h p -> (b h) p')

        return gaze_preds, gaze_targs


    def __call__(self, action_preds, action_targs, gaze_preds, gaze_targs, **kwargs):
        ''' averages over batch_size and num_heads in one go. if u wanna see the intermediary ce values for each head separately u need to define a separate loss function'''
        ce = F.cross_entropy(action_preds, action_targs)

        gaze_preds, gaze_targs = self._preprocess_gaze_preds_and_targs(gaze_preds, gaze_targs)

        reg = self.reg_lambda * F.cross_entropy(gaze_preds, gaze_targs)

        return ce + reg