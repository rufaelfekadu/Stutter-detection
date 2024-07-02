from torch.utils.tensorboard import SummaryWriter
from logging import getLogger
import logging
import wandb


class baselogger(object):
    def __init__(self) -> None:
        pass

    def add_scalar(self, name, value, epoch):
        pass

class TensorboardLogger(baselogger):
    def __init__(self, log_dir) -> None:
        super(TensorboardLogger, self).__init__()
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, name, value, epoch):
        self.writer.add_scalar(name, value, epoch)

    def close(self):
        self.writer.close()

class ConsoleLogger(baselogger):
    def __init__(self) -> None:
        super(ConsoleLogger, self).__init__()

    def add_scalar(self, name, value, epoch):
        print(f'{name}: {value}')

class FileLogger(baselogger):
    def __init__(self, log_dir) -> None:
        super(FileLogger, self).__init__()
        self.log_dir = log_dir
        self.logger = getLogger(__name__)
        self.logger.setLevel('INFO')
        self.file_handler = logging.FileHandler(log_dir)
        self.logger.addHandler(self.file_handler)

    def add_scalar(self, name, value, epoch):
        self.logger.info(f'{name}: {value}')

    def close(self):
        self.file_handler.close()

class CSVLogger(baselogger):
    def __init__(self, log_dir) -> None:
        super(CSVLogger, self).__init__()
        self.log_dir = log_dir
        self.log_file = open(log_dir, 'w')
        self.log_file.write('epoch,loss\n')

    def add_scalar(self, name, value, epoch):
        self.log_file.write(f'{epoch},{value}\n')

    def close(self):
        self.log_file.close()

class WandbLogger(baselogger):
    def __init__(self, log_dir) -> None:
        super(WandbLogger, self).__init__()
        wandb.init(project=log_dir)

    def add_scalar(self, name, value, epoch):
        wandb.log({name: value})

    def close(self):
        pass