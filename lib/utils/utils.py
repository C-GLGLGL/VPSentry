import torch
from torch.optim.lr_scheduler import _LRScheduler

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr
        lr = param_group['lr']
    return lr

def adjust_lr_poly(optimizer, base_lr, epoch, total_epoch, power):
    if len(optimizer.param_groups) > 1:
        lr = base_lr * (1 - epoch / total_epoch) ** power
        for i in range(len(optimizer.param_groups)):
            param_group = optimizer.param_groups[i]
            if i == 0:
                param_group['lr'] = lr * 0.1
            else:
                param_group['lr'] = lr
    else:
        lr = base_lr * (1 - epoch / total_epoch) ** power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr

class WarmupConsineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr_ratio=0.1, last_epoch=-1, anneal_strategy='linear'):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr_ratio = min_lr_ratio
        self.anneal_strategy = anneal_strategy
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        current_epoch = self.last_epoch + 1
        if current_epoch <= self.warmup_epochs:
            progress = current_epoch / self.warmup_epochs
            return [base_lr * progress for base_lr in self.base_lrs]
        else:
            anneal_epochs = self.max_epochs - self.warmup_epochs
            progress = (current_epoch - self.warmup_epochs) / anneal_epochs
            # progress = min(progress, 1.0)
            if self.anneal_strategy == 'linear':
                decay_factor = 1 - progress
            else:
                decay_factor = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress)))
            
            return [base_lr * self.min_lr_ratio + (base_lr - base_lr * self.min_lr_ratio) * decay_factor for base_lr in self.base_lrs]