import sys
import math
import numpy as np
import torch
from torch.autograd import Variable
from timm.utils import accuracy
from rsna_utils import (
    load_image,
    z_filling,
    patch_generator,
    MetricLogger,
    SmoothedValue
)

def run_iteration(
    df,
    img_path,
    batch_patient_ids,
    labels,
    patch_size,
    patches_per_in_iter,
    q1_model,
    q2_model,
    criterion,
    q1_optimizer,
    q2_optimizer,
    inner_iterations,
    grad_acc_steps,
    device,
    max_num_patches=32,
    img_mean=0,
    img_std=1
):
    if type(labels) == list:
        labels = np.array(labels, dtype=np.int64)
    
    if type(labels) == np.ndarray:
        labels = torch.from_numpy(labels).long().to(device)

    z_matrix, key_padding_mask = z_filling(
        df,
        img_path,
        batch_patient_ids,
        q1_model,
        patch_size,
        device,
        max_num_patches=max_num_patches,
        img_mean=img_mean,
        img_std=img_std
    )
    q1_optimizer.zero_grad()
    q2_optimizer.zero_grad()

    for j in range(inner_iterations):
        for patient_ind, patient_id in enumerate(batch_patient_ids):
            patches = []
            rows = df[df.id == patient_id]
            for row in rows.iterrows():
                img = load_image(df, img_path, row[0], patch_size, img_mean, img_std)
                for patch in patch_generator(img, patch_size):
                    patches.append(patch)

            patches = patches[:max_num_patches]
            for p in range(0, len(patches), patches_per_in_iter):
                b_patches = patches[p:p+patches_per_in_iter]
                torch_image = torch.from_numpy(np.stack(b_patches, axis=0)).float().to(device)
                z = q1_model(torch_image)
                z_matrix_variable = Variable(z_matrix)
                z_matrix_variable[patient_ind, p:p+len(b_patches)] = z

                y_pred = q2_model(z_matrix_variable, key_padding_mask)
                loss = criterion(y_pred, labels) / grad_acc_steps
                loss.backward(retain_graph=False)
            patches = []

        if (j+1) % grad_acc_steps == 0:
            q1_optimizer.step()
            q1_optimizer.zero_grad()
            q2_optimizer.step()
            q2_optimizer.zero_grad()

    return loss.item()

def train_one_epoch(
    df, img_path, patch_size, patches_per_in_iter, grad_acc_steps,
    q1_model, q2_model, criterion, data_loader,
    q1_optimizer, q2_optimizer, inner_iterations, 
    device, epoch, max_num_patches, img_mean, img_std
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
            patches_per_in_iter,
            q1_model,
            q2_model,
            criterion,
            q1_optimizer,
            q2_optimizer,
            inner_iterations,
            grad_acc_steps,
            device,
            max_num_patches,
            img_mean,
            img_std
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
