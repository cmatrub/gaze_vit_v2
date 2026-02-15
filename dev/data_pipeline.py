from torch.utils.data import DataLoader, TensorDataset
from GABRIL_utils.utils import load_dataset
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
from einops import rearrange
import torch

class DataPipeline:
    # having this as a class allows me to train multiple models for one dataset load if i later find dataset loading to be a major time drain

    def __init__(self, config):
        self.cfg = config.data_pipeline

        observations, actions, gaze_masks, gaze_coordinates = instantiate(self.cfg.load_dataset)

        observations = torch.as_tensor(observations, dtype=torch.float32)/255.0
        actions = torch.as_tensor(actions, dtype=torch.long)
        gaze_masks = torch.as_tensor(gaze_masks, dtype=torch.float32)

        self.preprocess_gaze = instantiate(self.cfg.preprocess_gaze) 
        gaze_masks = self.preprocess_gaze(gaze_masks=gaze_masks)

        train_obs, val_obs, train_act, val_act, train_gaze, val_gaze = train_test_split(observations, actions, gaze_masks, **self.cfg.train_test_split)
        
        # bundle em tog
        train_dataset = TensorDataset(train_obs, train_act, train_gaze) 
        val_dataset = TensorDataset(val_obs, val_act, val_gaze)

        self.train_loader = DataLoader(train_dataset, **self.cfg.dataloader, shuffle=True)
        self.val_loader = DataLoader(val_dataset, **self.cfg.dataloader, shuffle=False)
        

        
        

        