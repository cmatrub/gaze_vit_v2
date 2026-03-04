from trainer import Trainer
from data_pipeline import DataPipeline
import hydra
from omegaconf import DictConfig
import torch


@hydra.main(config_path="config", config_name="base", version_base=None) # version_base=None tells Hydra to use the latest default behaviors and suppress deprecation warnings
def main(config: DictConfig):
    torch.manual_seed(config.seed)

    data_pipeline = DataPipeline(config)

    trainer = Trainer(config=config, train_loader=data_pipeline.train_loader, val_loader=data_pipeline.val_loader)
    trainer.train()


if __name__ == "__main__":
    main()
