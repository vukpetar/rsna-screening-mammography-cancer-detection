import sys
sys.path.append('./classification')

import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)
from timm.models.layers import trunc_normal_

from nextvit import NCB, ConvBNReLU, NTB

def to_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str): return eval(x)

class Q1Net(nn.Module):

    def __init__(self, stem_chs, depths, path_dropout, attn_drop=0, drop=0,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75,
                 use_checkpoint=False):
        super(Q1Net, self).__init__()
        self.use_checkpoint = use_checkpoint

        self.stage_out_channels = [[96] * (depths[0]),
                                   [192] * (depths[1] - 1) + [256],
                                   [384, 384, 384, 384, 512] * (depths[2] // 5),
                                   [768] * (depths[3] - 1) + [1024]]

        # Next Hybrid Strategy
        self.stage_block_types = [[NCB] * depths[0],
                                  [NCB] * (depths[1] - 1) + [NTB],
                                  [NCB, NCB, NCB, NCB, NTB] * (depths[2] // 5),
                                  [NCB] * (depths[3] - 1) + [NTB]]

        self.stem = nn.Sequential(
            ConvBNReLU(1, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is NCB:
                    layer = NCB(input_channel, output_channel, stride=stride, path_dropout=dpr[idx + block_id],
                                drop=drop, head_dim=head_dim)
                    features.append(layer)
                elif block_type is NTB:
                    layer = NTB(input_channel, output_channel, path_dropout=dpr[idx + block_id], stride=stride,
                                sr_ratio=sr_ratios[stage_id], head_dim=head_dim, mix_block_ratio=mix_block_ratio,
                                attn_drop=attn_drop, drop=drop)
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)

        self.norm = nn.BatchNorm2d(output_channel, eps=1e-5)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.stage_out_idx = [sum(depths[:idx + 1]) - 1 for idx in range(len(depths))]
        print('initialize_weights...')
        self._initialize_weights()

    def merge_bn(self):
        self.eval()
        for idx, module in self.named_modules():
            if isinstance(module, NCB) or isinstance(module, NTB):
                module.merge_bn()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for idx, layer in enumerate(self.features):
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class Q2Net(nn.Module):

    def __init__(self, transformer_dim = 1024):
        super(Q2Net, self).__init__()

        self.transformer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            dim_feedforward=2*transformer_dim,
            nhead=4,
            dropout=0.25,
            batch_first=True,
        ) 
        self.head = nn.Linear(transformer_dim, 2)

    def forward(self, batch, src_key_padding_mask = None):
        x = self.transformer(batch, src_key_padding_mask=src_key_padding_mask)
        output = self.head(x[:, 0, :])
        return output

def load_q1_pretrained(path, q1_model):

    nextvitb_model = torch.load(path)
    nextvitb_model_params = nextvitb_model["model"].copy()
    del nextvitb_model_params["stem.0.conv.weight"]
    del nextvitb_model_params["proj_head.0.weight"]
    del nextvitb_model_params["proj_head.0.bias"]
    q1_model.load_state_dict(nextvitb_model_params, strict=False)

    return q1_model

def load_image(df, index, patch_size):
    row = df.iloc[index]
    img_path = f"./dataset/positive_images/{row.patient_id}/{row.image_id}.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255
    h, w = img.shape[:2]

    xmin, ymin, xmax, ymax = (np.array(to_list(row.pad_breast_box)) * h).astype(int)
    crop = img[ymin:ymax, xmin:xmax]
    
    crop_h, crop_w = ymax - ymin, xmax - xmin
    crop_h_full = math.ceil(crop_h / patch_size) * patch_size
    crop_w_full = math.ceil(crop_w / patch_size) * patch_size
    crop = np.pad(crop, ((0, crop_h_full - crop_h), (0, crop_w_full - crop_w)), 'constant')
    
    return crop

def patch_generator(image, patch_size):
    h, w = image.shape[:2]
    image = image[np.newaxis]
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[:, i:i+patch_size, j:j+patch_size]
            yield patch
            
            
def z_filling(df, patient_ids, q1_model, patch_size, device, batch_size=32, max_num_patches = 32):

    z_matrix, key_padding_masks = [], []
    for patient_id in patient_ids:
        patches = []
        patient_z_matrix = []
        rows = df[df.id == patient_id]
        for row in rows.iterrows():
            img = load_image(df, row[0], patch_size)
            for patch in patch_generator(img, patch_size):
                patches.append(patch)
        if len(patches) > max_num_patches:
            patches = patches[:max_num_patches]
        num_image = len(patches)
        with torch.no_grad():
            for b in range(0, num_image, batch_size):
                b_patches = patches[b:b+batch_size]
                torch_image = torch.from_numpy(np.stack(b_patches, axis=0)).float().to(device)
                z = q1_model(torch_image)
                patient_z_matrix.append(z)
                
        patient_z_matrix = torch.cat(patient_z_matrix, 0)
        number_of_patches = patient_z_matrix.shape[0]
        patient_z_matrix = F.pad(patient_z_matrix, (0, 0, 0, max_num_patches-number_of_patches), "constant", 0)
        z_matrix.append(patient_z_matrix)
        key_padding_mask = torch.cat((torch.ones((number_of_patches, )), torch.zeros((max_num_patches-number_of_patches, )))).to(device)
        key_padding_masks.append(key_padding_mask)
    patches = None
    patient_z_matrix = None

    return torch.stack(z_matrix, 0), torch.stack(key_padding_masks, 0)

def run_iteration(
    df,
    batch_patient_ids,
    labels,
    patch_size,
    patches_per_in_inter,
    q1_model,
    q2_model,
    criterion,
    q1_optimizer,
    q2_optimizer,
    inner_iterations,
    grad_acc_steps,
    device
):
    if type(labels) == list:
        labels = np.array(labels, dtype=np.int64)
    
    if type(labels) == np.ndarray:
        labels = torch.from_numpy(labels).long().to(device)

    z_matrix, key_padding_mask = z_filling(df, batch_patient_ids, q1_model, patch_size, device)
    q1_optimizer.zero_grad()
    q2_optimizer.zero_grad()

    for j in range(inner_iterations):
        z_index = 0
        for patient_ind, patient_id in enumerate(batch_patient_ids):
            patches = []
            rows = df[df.id == patient_id]
            for row in rows.iterrows():
                img = load_image(df, row[0], patch_size)
                for patch in patch_generator(img, patch_size):
                    patches.append(patch)

                    if len(patches) == patches_per_in_inter:
                        torch_image = torch.from_numpy(np.stack(patches, axis=0)).float().to(device)
                        z = q1_model(torch_image)
                        z_matrix[patient_ind, z_index:z_index+len(z)] = z

                        y_pred = q2_model(z_matrix, key_padding_mask)
                        loss = criterion(y_pred, labels)
                        z_index += len(z)
                        patches = []
            if j % grad_acc_steps:
                q1_optimizer.step()
                q1_optimizer.zero_grad()
                q2_optimizer.step()
                q2_optimizer.zero_grad()