from trainer import Trainer

def main():
    trainer = Trainer(config='/scr/nityakas/gaze_vit_v2/config/vit_s_14_train_args.yaml')
    trainer.train()


if __name__ == "__main__":
    main()
