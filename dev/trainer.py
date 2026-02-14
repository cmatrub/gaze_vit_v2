import torch
from utils.hook import AttentionExtractor
from utils.patch_gaze_masks import patch_gaze_masks
import torch.nn.functional as F
import losses
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from vit_pytorch import ViT
from utils.hook import AttentionExtractor
import numpy as np

class Trainer:
    def __init__(self, config, train_loader, val_loader):
        self.cfg = config.trainer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # not in yaml bc its not gna change

        self._construct_components(train_loader, val_loader)
        

    def _construct_components(self, train_loader, val_loader):
        self.loss_fn = instantiate(self.cfg.loss)
        self.model = instantiate(self.cfg.model).to(self.device)
        self.attn_extractor = instantiate(self.cfg.attention_extractor)
        self.handle = self.model.transformer.layers[-1][0].attend.register_forward_hook(self.attn_extractor.hook_fn) # type: ignore
        self.optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())
        self.train_loader = train_loader
        self.val_loader = val_loader


    def _train_step(self, batch):
        observations, action_targs, gaze_targs = batch
        observations = observations.to(self.device)
        action_targs = action_targs.to(self.device)
        gaze_targs = gaze_targs.to(self.device)

        # forward pass
        self.optimizer.zero_grad()
        action_preds = self.model(observations)

        # extract out attention weights
        gaze_preds = self.attn_extractor.cls_qkt_logits
        
        # calculate loss w regularization
        loss = self.loss_fn(action_preds, action_targs, gaze_preds, gaze_targs)
        
        # backward pass
        loss.backward()

        # update model weights in opposite direction of gradient
        self.optimizer.step()
        
        return loss.item() # so train() can log it


    def train(self):
        for epoch in range(self.cfg.training.num_epochs):
            losses = []
            
            for batch in self.train_loader:
                loss_dict = self._train_step(batch)
                losses.append(loss_dict['total_loss'])
            
            print(f"Epoch {epoch}: {np.mean(losses):.4f}")