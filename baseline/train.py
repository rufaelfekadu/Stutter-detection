from configs import _C as cfg
import argparse
import torch
import os


from trainer import build_trainer
from utils import  setup_exp
from logger import TensorboardLogger, CSVLogger, WandbLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data_config', type=str, default='baseline/configs/data/fluencybank.yml')
    parser.add_argument('--model_config', type=str, default='baseline/configs/model/lstm.yml')
    parser.add_argument('--opts', nargs='*', default=[])
    args = parser.parse_args()
    return args

def main(cfg):

    logger = WandbLogger(cfg)

    trainer = build_trainer(cfg, logger, metrics=['f1', 'wacc'])
    
    trainer.train()
    trainer.load_model()
    trainer.test()



if __name__ == "__main__":

    args = parse_args()

    cfg.merge_from_file(args.data_config)
    cfg.merge_from_file(args.model_config)
    cfg.merge_from_list(args.opts)

    setup_exp(cfg)

    main(cfg)