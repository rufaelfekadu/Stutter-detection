import argparse
import torch
import os


from stutter.utils.misc import setup_exp
from stutter.config import cfg
from stutter.utils.logger import WandbLogger
from stutter.trainer import build_trainer
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--data_config', type=str, default='baseline/configs/data/fluencybankyoho.yml')
    parser.add_argument('--model_config', type=str, default='baseline/configs/model/whisperyoho.yml')
    parser.add_argument('--logger', action='store_true')
    parser.add_argument('--opts', nargs='*', default=[])
    args = parser.parse_args()
    return args

def main(cfg):

    logger = WandbLogger(cfg) if args.logger else None
    metrics = ['video_metrics']

    trainer = build_trainer(cfg, logger, metrics)
    
    trainer.train()
    # trainer.load_model()
    trainer.test()



if __name__ == "__main__":

    args = parse_args()

    cfg.merge_from_file(args.data_config)
    cfg.merge_from_file(args.model_config)
    cfg.merge_from_list(args.opts)

    setup_exp(cfg)

    main(cfg)