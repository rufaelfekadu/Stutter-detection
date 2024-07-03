from configs import _C as cfg
import argparse
import torch

from trainer import Trainer
from models import LSTMModel
from utils import CCCLoss
from logger import TensorboardLogger, CSVLogger, WandbLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default='baseline/config.yml')
    parser.add_argument('--opts', nargs='*', default=[])
    args = parser.parse_args()
    return args

def main(cfg):

    # logger = TensorboardLogger(log_dir=cfg.log_dir)
    # logger = CSVLogger(log_dir=cfg.log_dir)
    logger = WandbLogger(cfg)
    model = LSTMModel(input_size=40, hidden_size=64, num_layers=1,output_size=6)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = {'t1': torch.nn.CrossEntropyLoss(), 't2': CCCLoss()}

    trainer = Trainer(cfg, model=model,optimizer=optimiser, criterion=criterion, logger=logger)
    trainer.train()
    trainer.test()



if __name__ == "__main__":
    args = parse_args()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    print(cfg)
    
    main(cfg)