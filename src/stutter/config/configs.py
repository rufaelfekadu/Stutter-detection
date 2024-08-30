from yacs.config import CfgNode as CN


_C = CN()

_C.name = 'stutter-detection'
_C.seed = 42
_C.tasks = ['t1']
_C.setting = 'yoho'
_C.cache_dir = '/tmp/'



# logging and saving
_C.output = CN()
_C.output.save_dir = 'outputs/'
_C.output.log_dir = 'logs'
_C.output.checkpoint_dir = 'checkpoints' 

# dataset configs
_C.data = CN()

_C.data.name = 'sep28k'
_C.data.root = 'datasets/sep28k/clips'
_C.data.label_path = 'datasets/sep28k/SEP-28k_labels_new.csv'
_C.data.ckpt = 'dataset'

_C.data.n_mels = 40
_C.data.win_length = 400
_C.data.hop_length = 160
_C.data.n_fft = 512
_C.data.sr = 16000
_C.data.n_frames = 3
_C.data.annotation = 'secondary_event'
_C.data.aggregate = True
_C.data.annotator = "A3"
_C.data.split_strategy = "ds_5"


# data loader configs


# model configs
_C.model = CN()
_C.model.name = 'lstm'
_C.model.input_size = 40
_C.model.hidden_size = 64
_C.model.num_layers = 1
_C.model.output_size = 6
_C.model.dropout = 0.5
_C.model.seq_len = 301

# video model configs
_C.model.vivit = CN()
_C.model.vivit.num_frames = 10
_C.model.vivit.video_size = [10, 224, 224]




# convlstm
_C.model.emb_dim = 64

# loss configs
_C.loss = CN()
_C.loss.gamma = 2
_C.loss.alpha = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
_C.loss.reduction = 'mean'
_C.loss.weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


# training configs
_C.solver = CN()
_C.solver.losses = ['ce']
_C.solver.device = 'cuda'
_C.solver.epochs = 10

# optimizer
_C.solver.optimizer = 'adam'
_C.solver.lr = 0.001
_C.solver.momentum = 0.9


# early stopping
_C.solver.es_patience = 5

#  dataloader
_C.solver.batch_size = 32
_C.solver.num_workers = 4

_C.solver.log_interval = 10

# scheduler
_C.solver.scheduler = 'plateau'
_C.solver.factor = 0.1
_C.solver.patience = 3
_C.solver.lr_step = 10
_C.solver.gamma = 0.5
_C.solver.lr_decay = 0.1
_C.solver.T_max = 10
_C.solver.eta_min = 0.0001
_C.solver.min_lr = 0.001
_C.solver.milestones = [50, 100, 150]

# logging
_C.solver.eval_steps = 10
_C.solver.log_steps = 10


