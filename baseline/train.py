from configs import _C as cfg
import argparse
import torch
import os


from trainer import MTLTrainer, STLTrainer
from models import  build_model
from utils import CCCLoss, CrossEntropyLoss, FocalLoss, setup_exp
from logger import TensorboardLogger, CSVLogger, WandbLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default='baseline/config.yml')
    parser.add_argument('--opts', nargs='*', default=[])
    args = parser.parse_args()
    return args

def main(cfg):

    logger = WandbLogger(cfg)

    criterion = {'t1': CrossEntropyLoss()}

    trainer = STLTrainer(cfg, criterion=criterion, logger=logger, metrics=['f1'])
    
    trainer.train()

    trainer.load_model()
    # trainer.test(trainer.val_loader, name='val')
    # trainer.test(trainer.train_loader, name='train')
    trainer.test()



if __name__ == "__main__":

    args = parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    setup_exp(cfg)

    main(cfg)