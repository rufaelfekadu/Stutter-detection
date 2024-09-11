
from .loss import loss_registery

def build_loss(cfg):
    criterion = {}
    for key, loss in zip(cfg.tasks, cfg.solver.losses):
        criterion[key] = loss_registery[loss](device=cfg.solver.device, **cfg.loss ).to(cfg.solver.device)
    return criterion