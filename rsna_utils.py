import sys
sys.path.append('./classification')

import time
import datetime
import math
from collections import defaultdict, deque
import numpy as np
import cv2
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from timm.utils import accuracy
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

def load_image(df, img_path, index, patch_size):
    row = df.iloc[index]
    img_path = f"{img_path}/{row.patient_id}/{row.image_id}.png"
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255
        h, w = img.shape[:2]

        xmin, ymin, xmax, ymax = (np.array(to_list(row.pad_breast_box)) * h).astype(int)
        crop = img[ymin:ymax, xmin:xmax]
        
        crop_h, crop_w = ymax - ymin, xmax - xmin
        crop_h_full = math.ceil(crop_h / patch_size) * patch_size
        crop_w_full = math.ceil(crop_w / patch_size) * patch_size
        crop = np.pad(crop, ((0, crop_h_full - crop_h), (0, crop_w_full - crop_w)), 'constant')
    except:
        crop = np.random.random((384*3, 384*2))
    
    
    return crop

def patch_generator(image, patch_size):
    h, w = image.shape[:2]
    image = image[np.newaxis]
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[:, i:i+patch_size, j:j+patch_size]
            yield patch
            
            
def z_filling(df, img_path, patient_ids, q1_model, patch_size, device, batch_size=32, max_num_patches = 32):

    z_matrix, key_padding_masks = [], []
    for patient_id in patient_ids:
        patches = []
        patient_z_matrix = []
        rows = df[df.id == patient_id]
        for row in rows.iterrows():
            img = load_image(df, img_path, row[0], patch_size)
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
    img_path,
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

    z_matrix, key_padding_mask = z_filling(df, img_path, batch_patient_ids, q1_model, patch_size, device)
    q1_optimizer.zero_grad()
    q2_optimizer.zero_grad()

    for j in range(inner_iterations):
        for patient_ind, patient_id in enumerate(batch_patient_ids):
            z_index = 0
            patches = []
            rows = df[df.id == patient_id]
            for row in rows.iterrows():
                img = load_image(df, img_path, row[0], patch_size)
                for patch in patch_generator(img, patch_size):
                    patches.append(patch)

                for p in range(0, len(patches), patches_per_in_inter):
                    torch_image = torch.from_numpy(np.stack(patches[p:p+patches_per_in_inter], axis=0)).float().to(device)
                    z = q1_model(torch_image)
                    z_matrix[patient_ind, z_index:z_index+len(z)] = z.detach()

                    y_pred = q2_model(z_matrix, key_padding_mask)
                    loss = criterion(y_pred, labels) / grad_acc_steps
                    z_index += len(z)
                patches = []

        if (j+1) % grad_acc_steps == 0:
            q1_optimizer.step()
            q1_optimizer.zero_grad()
            q2_optimizer.step()
            q2_optimizer.zero_grad()

    return loss.item()

        

class RsnaDataset(Dataset):
    def __init__(self, patient_ids, labels, positive_ratio = "1in8",is_train=False):
        self.patient_ids = patient_ids
        self.labels = labels
        self.is_train = is_train
        
        if not is_train:
            self.length = len(patient_ids)
        else:
            p_num, all_num = positive_ratio.split("in")
            self.initial_pos_loc = [1]*int(p_num) + [0]*int(all_num)
            self.positive_indexes = np.where(labels == 1)[0]
            self.negative_indexes = np.where(labels == 0)[0]
            self.positive_index_locations = self.set_positive_index_locations(self.initial_pos_loc)
            self.length = len(self.positive_index_locations) / np.sum(self.positive_index_locations) * len(self.positive_indexes)
            self.positive_index = 0
            self.negative_index = 0
        self.length = int(self.length)
            

    def __len__(self):
        return self.length

    def set_positive_index_locations(self, initial_pos_loc):
        self.positive_index_locations = initial_pos_loc.copy()
        shuffle(self.positive_index_locations)
        return self.positive_index_locations

    def __getitem__(self, index):
        
        if not self.is_train:
            p_id = self.patient_ids[index]
            target = self.labels[index]
        else:
            if len(self.positive_index_locations):
                class_ind = self.positive_index_locations.pop()
            else:
                self.positive_index_locations = self.set_positive_index_locations(self.initial_pos_loc)
                class_ind = self.positive_index_locations.pop()
                
            if class_ind:
                self.positive_index = self.positive_index % len(self.positive_indexes)
                if self.positive_index == 0:
                    shuffle(self.positive_indexes)
                new_index = self.positive_indexes[self.positive_index]
                self.positive_index += 1
            else:
                self.negative_index = self.negative_index % len(self.negative_indexes)
                if self.negative_index == 0:
                    shuffle(self.negative_indexes)
                new_index = self.negative_indexes[self.negative_index]
                self.negative_index += 1
            p_id = self.patient_ids[new_index]
            target = self.labels[new_index]
                
        return p_id, target

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

@torch.no_grad()
def evaluate(df, img_path, data_loader, q1_model, q2_model, patch_size, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    for p_ids, target in metric_logger.log_every(data_loader, 10, header):
        target = target.to(device, non_blocking=True)

        z_matrix, key_padding_mask = z_filling(df, img_path, p_ids, q1_model, patch_size, device)
        output = q2_model(z_matrix, key_padding_mask)
        loss = criterion(output, target)

        acc1,  = accuracy(output, target, topk=(1,))

        batch_size = len(p_ids)
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch(
    df, img_path, patch_size, patches_per_in_inter, grad_acc_steps,
    q1_model, q2_model, criterion, data_loader,
    q1_optimizer, q2_optimizer, inner_iterations, 
    device, epoch
):
    q1_model.train(True)
    q2_model.train(True)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('q1_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('q2_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for p_ids, targets in metric_logger.log_every(data_loader, print_freq, header):

        targets = targets.to(device, non_blocking=True)
        loss_value = run_iteration(
            df,
            img_path,
            p_ids,
            targets,
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
        )

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(q1_lr=q1_optimizer.param_groups[0]["lr"])
        metric_logger.update(q2_lr=q2_optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}