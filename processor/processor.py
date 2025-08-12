# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda.amp import autocast
import torch.distributed as dist
from tqdm import tqdm
from torchvision.utils import save_image
import logging
import torch.nn.functional as F
from models.clip.clip import tokenize, _transform
from models.clip.simple_tokenizer import SimpleTokenizer
from models.make_model_clip import PromptLearner,TextEncoder
from solver.scheduler_factory import create_scheduler
from loss.supcontrast import SupConLoss
from loss.triplet_loss import TripletLoss,PlasticityLoss


    
def is_master(args):
    return (not args.distributed) or args.gpu == 0 or args.dp



# def get_loss(model, images, target, camid, target_view, xent, id_loss, triplet=None):
#     img_feature_last, img_feature, img_feature_proj, text_feature, cross_x, cls_score = model(x=images, label=target)
#     loss_i2t = xent(img_feature_proj, text_feature, target, target)
#     loss_t2i = xent(text_feature, img_feature_proj, target, target)
#     loss = loss_i2t + loss_t2i
#     I2TLOSS = id_loss(cls_score, target)
#
#     total_loss = loss+I2TLOSS
#     # cls_score = 0.0
#     return total_loss,cls_score

def get_loss(model, images, target, camid, target_view, xent, id_loss, triplet=None):
    img_feature_last, img_feature, img_feature_proj, text_feature, cross_x, cls_score ,img_local_attr_feat= model(x=images, label=target)

    loss_i2t = xent(img_feature_proj, text_feature, target, target)
    loss_t2i = xent(text_feature, img_feature_proj, target, target)
    loss =  loss_t2i + loss_i2t
    idLOSS = id_loss(cls_score, target)
    triloss, _, _ = triplet(img_feature_proj, target)

    #fti4ci中的ortho损失函数
    img_salient_local_feats=img_local_attr_feat
    ortho_loss = torch.nn.MSELoss()
    batch_size, length, dim = img_salient_local_feats.size()
    img_salient_local_feats = F.normalize(img_salient_local_feats, p=2, dim=-1)
    cosine_score = torch.matmul(img_salient_local_feats, img_salient_local_feats.permute(0, 2, 1))
    eye_matrix = torch.eye(length).unsqueeze(0).repeat(batch_size, 1, 1).to(img_salient_local_feats.device)
    loss_ortho = ortho_loss(cosine_score, eye_matrix)

    total_loss = idLOSS + triloss  + 0.9*loss + loss_ortho
    # print(loss,idLOSS,triloss,loss_ortho)
    # cls_score = 0.0
    return total_loss,cls_score

def do_train(cfg,
             model,
             train_loader,
             val_loader,
             optimizer,
             scheduler,
             num_class,
             num_query, local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    # eval_period = 5
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')

    if device:
        model.to(local_rank)
    triplet = TripletLoss(cfg.SOLVER.MARGIN)
    xent = SupConLoss(device)
    id_loss = nn.CrossEntropyLoss().to(device)
    loss_meter = AverageMeter()
    loss_t_meter = AverageMeter()
    id_loss_meter = AverageMeter()
    id1_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()


    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_t_meter.reset()
        id_loss_meter.reset()
        id1_loss_meter.reset()
        evaluator.reset()
        acc_meter.reset()

        scheduler.step()
        model.train()

        for n_iter,(img, vid, target_cam, target_view) in enumerate(train_loader):
        
            optimizer.zero_grad()

            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
           
            with amp.autocast(enabled=True): 
                loss,logits= get_loss(model,img,target,target_cam,target_view, xent,id_loss,triplet)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f},  Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), 250,
                                    loss_meter.avg, acc_meter.avg , scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, 64 / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model,
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model,
                           os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.NAMES, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        if epoch % eval_period == 0 :
            model.eval()
            for n_iter, (img, vid, camid, camids, target_view, img_path) in enumerate(val_loader):

                with torch.no_grad():
                    img = img.to(device)
                    features = model(img)
                    evaluator.update((features,vid, camid,img_path))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)
    with open('/data3/shangrui/newsg2/models/make_model_clip.py', "r") as f:
        code1 = f.read()
    with open('/data3/shangrui/newsg2/processor/processor.py', "r") as f:
        code2 = f.read()
    with open('/data3/shangrui/newsg2/configs/vit_clipreid.yml', "r") as f:
        code3 = f.read()
    with open('/data3/shangrui/newsg2/output/train_log.txt', "a") as f:
        f.write(code1)
        f.write(code2)
        f.write(code3)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("PromptSG.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, vid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            features = model(img)
            evaluator.update((features,vid, camid,imgpath))
        img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

