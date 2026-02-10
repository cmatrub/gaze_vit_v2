class AttentionExtractor:
    def __init__(self):
        self.cls_qkt_logits = None 
    
    def hook_fn(self, layer, input, output):
        # im leaving out the processing logic and keeping the hook dumb- softmaxing is part of the regularization computation, the hook is only supposed to return what it gets from the model
        attn_weights = input[0] # input comes in as a tuple wrapping the input tensor so u need to index into it

        # attn_weights shape: [batch_size, num_heads, num_patches, num_patches]. here num_patches includes CLS token.
        self.cls_qkt_logits = attn_weights[:, :, 0, 1:] # leave 1st and 2 dims alone, grab only 1st row of third (cls dot prods), exclude 1st element of 4th dim (bc that's cls self attn)