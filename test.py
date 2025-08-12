import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import os
from config import cfg
import argparse
import torch
from datasets.make_dataloader_clip import make_dataloader
from models.make_model_clip import make_model
from processor.processor import do_inference
from utils.logger import setup_logger
from torchstat import stat
import thop
from torchinfo import summary
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PromptSG", output_dir, if_train=False)
    logger.info(args)

    # if args.config_file != "":
    #     logger.info("Loaded configuration file {}".format(args.config_file))
    #     with open(args.config_file, 'r') as cf:
    #         config_str = "\n" + cf.read()
    #         logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    train_loader, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    # checkpoint = torch.load(cfg.TEST.WEIGHT)
    # model.load_state_dict(checkpoint,strict=False)

    model = torch.load(cfg.TEST.WEIGHT)


    summary(model,(1,3,256,128))

    do_inference(cfg,
                model,
                val_loader,
                num_query)


