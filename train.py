import os
from utils.logger import setup_logger
from datasets.make_dataloader_clip import make_dataloader
from models.make_model_clip import make_model
from solver.make_optimizer_prompt import make_optimizer
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor import do_train
import random
import torch
import numpy as np


import argparse
from config import cfg

#import ipdb



def set_seed(seed):
    torch.manual_seed(seed)  #设置 PyTorch 的随机数种子为输入的 seed 值；
    torch.cuda.manual_seed(seed) #设置 PyTorch 在 CUDA 上的随机数种子为输入的 seed 值；
    torch.cuda.manual_seed_all(seed) #设置 PyTorch 在所有的 CUDA 设备上的随机数种子为输入的 seed 值；
    np.random.seed(seed) #设置 NumPy 的随机数种子为输入的 seed 值；
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    #torch.backends.cudnn.benchmark = True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PromptSG Training")  # 定义命令行解析器对象
    parser.add_argument(
        "--config_file", default="configs/vit_clipreid.yml", help="path to config file", type=str
    )  # 添加命令行参数 没有值时为default

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)  # 添加 'opts' 参数，它将捕获所有剩余的命令行参数
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()  # 从命令行中结构化解析参数

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)   #从文件中合并配置信息到当前的配置对象cfg中。
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
   
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    loss_func, arc_criterion= make_loss(cfg, num_classes=num_classes)


    optimizer = make_optimizer(cfg, model)
   
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA, cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                  cfg.SOLVER.STAGE2.WARMUP_ITERS, cfg.SOLVER.STAGE2.WARMUP_METHOD)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler, 
        num_classes,
        num_query, args.local_rank
    )

    