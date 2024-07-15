from configs import _C as cfg
import argparse
import torch

from trainer import MTLTrainer, STLTrainer
from models import  build_model
from utils import CCCLoss, CrossEntropyLoss, FocalLoss
from logger import TensorboardLogger, CSVLogger, WandbLogger

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default='baseline/config.yml')
    parser.add_argument('--opts', nargs='*', default=[])
    args = parser.parse_args()
    return args

def main(cfg):

    # logger = TensorboardLogger(log_dir=cfg.output.log_dir)
    # logger = CSVLogger(log_dir=cfg.log_dir)
    logger = WandbLogger(cfg)

    criterion = {'t2': CrossEntropyLoss()}

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
    cfg.freeze()

    set_seed(cfg.seed)
    
    print(cfg)
    
    main(cfg)