import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .clip.model import Transformer, LayerNorm

_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from loss.arcface import ArcFace
import ipdb

from models import MHTransformer
import torch.nn.functional as F
from .utils import get_img_patch_feats

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class IMG2TEXTLOCAL(nn.Module):
# class GETATTRIBUTE(nn.Module):

    def __init__(self, clip_model,topk, img_patch_dim, token_feat, num_k, tf_layer, tf_head, epsilon):
        super().__init__()
        self.num_k = num_k  # 局部特征的数量

        self.topk = topk  # 选择前topk个局部特征

        self.epsilon = epsilon  # 注意力权重的阈值

        self.local_atte_fc = nn.Sequential(nn.Linear(img_patch_dim, token_feat), nn.Sigmoid())  # 局部注意力的全连接层和Sigmoid激活函数

        self.transformer = MHTransformer.Transformer(dim_self=img_patch_dim, num_heads=tf_head, dim_ref=img_patch_dim,
                                                     num_layers=tf_layer)  # 多头自注意力Transformer
        self.templates = nn.Parameter(torch.randn(1, num_k, img_patch_dim))  # 可学习的模板，用于初始化局部特征的生成

        self.clip_model = clip_model

    def get_latent_local_attributes_feats(self, featuremap):
        # 根据特征图获取潜在的局部属性特征
        batch_size = featuremap.shape[0]
        feature_dim = featuremap.shape[2]

        initial_templates = self.templates.expand(batch_size, self.num_k, feature_dim)  # 扩展模板的维度以匹配批大小和局部特征数量
        cat_feature = torch.cat([initial_templates, featuremap], dim=1)  # 将模板和特征图拼接起来
        latent_local_feats = self.transformer(cat_feature, mask=None)[:, :self.num_k, :]  # 通过Transformer获取潜在的局部特征
        latent_local_feats = self.local_atte_fc(latent_local_feats)  # 应用局部注意力的全连接层和Sigmoid激活函数

        return latent_local_feats

    def get_img_local_attr_feats(self, img_feature_proj, image_patch_feats):
        # 根据全局特征和图像块特征获取图像的局部属性特征
        bs = image_patch_feats.shape[0]
        latent_local_feats = self.get_latent_local_attributes_feats(image_patch_feats)  # 获取潜在的局部特征

        # 根据注意力分数进行初步筛选
        attention_weights = torch.matmul(latent_local_feats, img_feature_proj.unsqueeze(dim=2)).squeeze(dim=2)
        attention_weights = F.softmax(attention_weights, dim=1)

        local_attr_num = []
        sorted_indices = torch.argsort(attention_weights, dim=1, descending=True)
        sorted_indices = sorted_indices[:, :self.topk]
        selected_local_feats = []

        for i in range(bs):
            mask = attention_weights[i] > self.epsilon
            non_indices = torch.nonzero(mask).squeeze()
            num_r = non_indices.numel() if non_indices.numel() < self.topk else self.topk
            if num_r < 1:
                num_r = 1

            # 确保属性特征的顺序
            select_indices = sorted_indices[i][:num_r]
            select_indices = torch.sort(select_indices, dim=0).values
            select_id = torch.cat((select_indices, sorted_indices[i][num_r:]), dim=0)
            local_attr_num.append(num_r)
            selected_local_feats.append(latent_local_feats[i, select_id, :])

        selected_local_feats = torch.stack(selected_local_feats, dim=0)

        return F.normalize(selected_local_feats, dim=-1), local_attr_num  # 归一化后的局部特征及其数量

    def forward(self, img_feature_proj, img_patch_feats):
        # 将图像转换为文本描述
        img_local_attr_feat, local_attr_num = self.get_img_local_attr_feats(img_feature_proj,img_patch_feats)  # 获取图像的局部属性特征及其数量
        return img_local_attr_feat, local_attr_num  # (64,12,512)

class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x  = self.bottleneck(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # ipdb.set_trace()
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_feature = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return text_feature

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512

        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024

        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)  # 16
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)   # 8
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]   # 16
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        self.cross_attn = nn.MultiheadAttention(self.in_planes_proj,
                                                self.in_planes_proj // 64)
        self.cross_modal_transformer = Transformer(width=self.in_planes_proj,
                                                   layers=2,
                                                   heads=self.in_planes_proj //
                                                         64)
        scale = self.cross_modal_transformer.width ** -0.5
        self.ln_pre_t = LayerNorm(self.in_planes_proj)
        self.ln_pre_i = LayerNorm(self.in_planes_proj)
        self.ln_post = LayerNorm(self.in_planes_proj)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        self.img2text = IM2TEXT(embed_dim=512,
                                middle_dim=512,
                                output_dim=512,
                                n_layer=2)

        # self.img2att_text = IM2TEXT(embed_dim=512,
        #                         middle_dim=512,
        #                         output_dim=512,
        #                         n_layer=2)

        top_k = 16
        #
        # self.get_attributes = GETATTRIBUTE(clip_model,top_k, img_patch_dim=self.in_planes, token_feat=self.in_planes_proj,
        #                                    num_k=64, tf_layer=3, tf_head=1, epsilon=0.05)

        self.img2localtext= IMG2TEXTLOCAL(clip_model,top_k, img_patch_dim=self.in_planes, token_feat=self.in_planes_proj,
                                           num_k=64, tf_layer=3, tf_head=1, epsilon=0.05)
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding,top_k)
        self.text_encoder = TextEncoder(clip_model)
        # for param in self.text_encoder.parameters():
        #     param.requires_grad = False


    def forward(self, x=None, get_image=False, get_text=False, label=None, cam_label=None, view_label=None):

        if self.model_name == 'ViT-B-16':
            img_patch_feats, img_feature_last, img_feature, image_features_proj = self.image_encoder(x)
            img_feature_proj = image_features_proj[:, 0]

        if self.model_name == 'RN50':
            img_patch_feats, image_features_last, image_features, image_features_proj = self.image_encoder(x)  # x3 ,x4, xproj
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]
            image_features_proj = image_features_proj.permute(1, 0, 2)

        if get_image:
            feat_proj = self.bottleneck_proj(img_feature_proj)
            return img_feature_proj

        token_features = self.img2text(img_feature_proj)  # 提取到的S

        # img_local_attr_feat, local_attr_num = self.get_attributes(img_feature_proj, img_patch_feats )  # 得到局部属性特征和其数量
        img_local_attr_feat, local_attr_num = self.img2localtext(img_feature_proj, img_patch_feats)

        img_attr_text = self.img2text(img_local_attr_feat)

        prompts = self.prompt_learner(token_features, img_attr_text)
        # prompts = self.prompt_learner(token_features)
        text_feature = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)

        cross_x = self.cross_former(text_feature.unsqueeze(1), image_features_proj, image_features_proj)
        # cross_x = self.cross_former(image_features_proj, image_features_proj, image_features_proj)

        cross_x_bn = self.bottleneck_proj(cross_x)

        if self.training:
            cls_score = self.classifier_proj(cross_x_bn)
            return img_feature_last, img_feature, img_feature_proj, text_feature, cross_x, cls_score,img_local_attr_feat
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return cross_x_bn
            else:
                return cross_x

    def cross_former(self, q, k, v):
        q = q.transpose(1, 0)
        k = k.transpose(1, 0)
        v_t = v.transpose(1, 0)
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v_t),
            need_weights=False)[0]
        cross_x = x.permute(1, 0, 2)  # NLD -> LND
        cross_x = self.cross_modal_transformer(cross_x)
        cross_x = cross_x.permute(1, 0, 2)  # LND -> NLD
        cross_x = self.ln_post(cross_x)
        return cross_x[0]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding, top_k):
        super().__init__()

        ctx_init = "A photo of a X person with " + '* '*top_k  # a X X X X person with different viewpoints.
        # -1 0    1   2 3 4   5      6  78910 11 12 13 14 15 16 17 18
        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + 1:8, :])
        self.register_buffer("att_suf", embedding[:, 8+top_k:, :])
        self.num_class = num_class

    def forward(self, bias, img_attr_text):
        # cls_ctx = self.cls_ctx
        b = bias.shape[0]

        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        att_suf = self.att_suf.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                bias,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
                img_attr_text,
                att_suf
            ],
            dim=1,
        )

        return prompts

# class PromptLearner(nn.Module):
#     def __init__(self, num_class, dataset_name, dtype, token_embedding,top_k):
#         super().__init__()
#
#         ctx_init = "A photo of a person with " + "* "*top_k  # a X X X X person with different viewpoints.
#                # -1 0   1   2  3   4  567890123456
#         ctx_dim = 512
#         # use given words to initialize context vectors
#         ctx_init = ctx_init.replace("_", " ")
#         n_ctx = 4
#
#         tokenized_prompts = clip.tokenize(ctx_init).cuda()
#
#         with torch.no_grad():
#             embedding = token_embedding(tokenized_prompts).type(dtype)
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.register_buffer("token_prefix", embedding[:, :7 , :])
#
#
#         self.register_buffer("att_suf", embedding[:, 7+top_k:, :])
#
#         self.num_class = num_class
#
#     def forward(self, bias, img_local_attr_feat):
#         # cls_ctx = self.cls_ctx
#         b = bias.shape[0]
#
#         prefix = self.token_prefix.expand(b, -1, -1)  # 64 5 512
#         att_suf = self.att_suf.expand(b, -1, -1)
#
#         prompts = torch.cat(
#             [
#                 prefix,  # (n_cls, 1, dim)
#                 img_local_attr_feat,
#                 att_suf
#             ],
#             dim=1,
#         )
#
#         return prompts



# class PromptLearner(nn.Module):
#     def __init__(self, num_class, dataset_name, dtype, token_embedding,topk):
#         super().__init__()
#
#         ctx_init = "A photo of a X person"  # a X X X X person with different viewpoints.
#
#         ctx_dim = 512
#         # use given words to initialize context vectors
#         ctx_init = ctx_init.replace("_", " ")
#         n_ctx = 4
#
#         tokenized_prompts = clip.tokenize(ctx_init).cuda()
#
#         with torch.no_grad():
#             embedding = token_embedding(tokenized_prompts).type(dtype)
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#
#         self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
#         self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + 1:, :])
#         self.num_class = num_class
#
#     def forward(self, bias,x):
#         # cls_ctx = self.cls_ctx
#         b = bias.shape[0]
#
#         prefix = self.token_prefix.expand(b, -1, -1)
#         suffix = self.token_suffix.expand(b, -1, -1)
#         bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
#
#         prompts = torch.cat(
#             [
#                 prefix,  # (n_cls, 1, dim)
#                 bias,  # (n_cls, n_ctx, dim)
#                 suffix,  # (n_cls, *, dim)
#
#             ],
#             dim=1,
#         )
#
#         return prompts


# class PromptLearner(nn.Module):
#     def __init__(self, num_class, dataset_name, dtype, token_embedding,_):
#         super().__init__()
#
#         ctx_init = "A photo of a person"  # a X X X X person with different viewpoints.
#
#         ctx_dim = 512
#         # use given words to initialize context vectors
#         ctx_init = ctx_init.replace("_", " ")
#         n_ctx = 4
#
#         tokenized_prompts = clip.tokenize(ctx_init).cuda()
#
#         with torch.no_grad():
#             embedding = token_embedding(tokenized_prompts).type(dtype)
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#
#         self.register_buffer("prompt", embedding)
#
#
#     def forward(self, bias):
#         # cls_ctx = self.cls_ctx
#         b = bias.shape[0]
#
#         prompts = self.prompt.expand(b,-1,-1)
#
#         return prompts

