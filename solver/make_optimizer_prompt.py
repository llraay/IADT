import torch


def make_optimizer(cfg, model):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        if "text_encoder" in key:
            value.requires_grad_(False)
        if "image_encoder" in key:
            lr = cfg.SOLVER.STAGE2.IMS_LR
            # lr= 0.1*cfg.SOLVER.STAGE2.BASE_LR
        if "classifier" in key:
            lr = 3e-4
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.IMS_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
      
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]


    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    
    return optimizer
