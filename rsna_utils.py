from pathlib import Path
import time
import datetime
import math
from collections import defaultdict, deque
import numpy as np
import cv2
from random import shuffle
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.dataset import Dataset

def to_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str): return eval(x)

def load_image(df, img_path, index, patch_size):
    row = df.iloc[index]
    
    img_path = (Path(img_path) / f"{row.machine_id}/{row.patient_id}/{row.image_id}.png").as_posix()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255
    h, _ = img.shape[:2]

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

class RsnaDataset(Dataset):
    def __init__(self, patient_ids, labels, positive_ratio = "1in8",is_train=False):
        self.patient_ids = patient_ids
        self.labels = labels
        self.is_train = is_train
        
        if not is_train:
            self.length = len(patient_ids)
        else:
            p_num, all_num = positive_ratio.split("in")
            self.initial_pos_loc = [1]*int(p_num) + [0]*(int(all_num)-int(p_num))
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
