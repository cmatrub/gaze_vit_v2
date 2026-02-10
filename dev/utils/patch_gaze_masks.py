from einops import reduce, rearrange
import torch.nn.functional as F


def patch_gaze_masks(gaze_masks, patch_size=(14,14), **kwargs):
    '''
    returns a (b x f x n) tensor of "attention" values mean-pooled and softmax-ed 
    input: b x f x h x w
    h = h_out * patch_height
    w = w_out * patch_width
    n = h_out*w_out
    '''
    patch_height = patch_size[0]
    patch_width = patch_size[1]

    gaze_masks = reduce(gaze_masks,
        "b f (h_out ph) (w_out pw) -> b f (h_out w_out)", # note the () around h_out w_out flattens the 2d grid into 1d. why: u can only softmax over 1 dim, so a 2d grid wont work
        "mean", 
        ph=patch_height, 
        pw=patch_width)
    
    gaze_masks = F.softmax(gaze_masks, dim=-1)
    
    gaze_masks = rearrange(gaze_masks, 'b 1 n -> b n') # squeezing out the frames dim which is 1 (for now)
    
    return gaze_masks

# if not applying downsampling patching scheme use this
def do_nothing(gaze_masks, **kwargs):
    return gaze_masks