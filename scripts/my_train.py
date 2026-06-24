import os
import logging
import time
from datetime import datetime
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
from sklearn import metrics

import sys
current_path = os.path.abspath(__file__)
sys_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(sys_path)

from config import config
from lib.sentry.sentrynet import SentryNet
from lib.sentry.sentry_discriminator import Discriminator
from lib.dataloader.dataloader import get_video_dataset, VideoDataset
from lib.utils.utils import clip_gradient, WarmupConsineLR
from eval.evaluator import Evaluator
from eval.dice_score import dice_coeff, auto_data_convert

def cofficent_calculate(preds,gts,threshold=0.5):
    eps = 1e-5
    preds = preds > threshold
    intersection = (preds * gts).sum()
    union =(preds + gts).sum()
    dice = 2 * intersection  / (union + eps)
    iou = intersection/(union - intersection + eps)
    return dice, iou

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def dice_loss(self, input, target):
        input = torch.sigmoid(input)
        # target = torch.sigmoid(target)
        N = target.size(0)
        smooth = 1e-5
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss
    
    def structure_loss(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        # wbce = F.binary_cross_entropy(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(-2, -1)) / weit.sum(dim=(-2, -1))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(-2, -1))
        union = ((pred + mask)*weit).sum(dim=(-2, -1))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()

    def forward(self, *inputs):
        pred, target = tuple(inputs)
        # total_loss = self.structure_loss(pred.squeeze(), target.squeeze().float()) + self.dice_loss(pred.squeeze(), target.squeeze().float())
        total_loss = self.structure_loss(pred, target.float())
        return total_loss

@torch.no_grad()
def start_eval(eval_loader, model, sentry):
    mean_dice = 0.
    ## Evaluation ##
    if config.eval_on:
        logging.info("Start Eval...")
        model = model.eval()
        sentry = sentry.eval()
        evaluator = Evaluator(config.metric_list, config.batchsize*(config.video_time_clips+1))
        mean_case_score_list, max_case_score_list = [], []
        # my dice #
        tot_dice = 0.
        tot_dice2 = 0.
        tot_iou = 0.
        tot_mae = 0.
        nums = 0
        model.memory_p = None
        for i, (images, gts, _) in enumerate(eval_loader, start=1):
            images = images.cuda(device=device_ids[0])
            gts = gts.reshape(-1, gts.shape[3], gts.shape[4])

            prototypes, features, _ = model.forward_generate(images)
            preds = model(prototypes, features, (gts.shape[-2], gts.shape[-1]), mode="eval", sentry=sentry)

            # my dice #
            if not config.tf_img_only:
                tot_dice += dice_coeff(preds.squeeze(1), gts.to(preds.device), preds.device).item()
                temp_dice, temp_iou = cofficent_calculate(preds.squeeze(1), gts.to(preds.device))
                tot_dice2 += temp_dice.item()
                tot_iou += temp_iou.item()
                m = auto_data_convert(preds.squeeze()).astype(int)
                t = auto_data_convert(gts).astype(int)
                tot_mae += metrics.mean_absolute_error(t, m)
            else:
                tot_dice += dice_coeff(preds.squeeze(), gts.float().to(preds.device), preds.device).item()
            nums += 1

            preds = preds.squeeze().cpu().detach().numpy()
            gts = gts.squeeze().cpu().numpy()
            evaluator.eval(preds, gts, config.tf_img_only)
        
        mean_dice = tot_dice / nums
        mean_dice2 = tot_dice2 / nums
        mean_iou = tot_iou / nums
        mean_mae = tot_mae / nums
        print("Dice coeff: ", mean_dice, " Dice2: ", mean_dice2)
        logging.info("Dice coeff: " + str(mean_dice) + " Dice2: " + str(mean_dice2))
        print("mIoU: ", mean_iou)
        logging.info("mIoU: " + str(mean_iou))
        print("MAE: ", mean_mae)
        logging.info("MAE: " + str(mean_mae))

        result = evaluator.get_result()
        mean_score_ind, max_score_ind = [], []
        mean_score_list, max_score_list = [], []
        for i, (name, value) in enumerate(result.items()):
            if 'max' in name or 'mean' in name:
                if 'max' in name:
                    max_score_list.append(value)
                    max_score_ind.append(i)
                else:
                    mean_score_list.append(value)
                    mean_score_ind.append(i)
            else:
                mean_score_list.append([value]*256)
                mean_score_ind.append(i)

        # calculate all the metrics at frame-level
        max_case_score_list.append(max_score_list)
        mean_case_score_list.append(mean_score_list)
        max_case_score_list = np.mean(np.array(max_case_score_list), axis=0)
        mean_case_score_list = np.mean(np.array(mean_case_score_list), axis=0)
        case_score_list = []
        for index in range(len(config.metric_list)):
            real_max_index = np.where(np.array(max_score_ind) == index)
            real_mean_index = np.where(np.array(mean_score_ind) == index)
            if len(real_max_index[0]) > 0:
                case_score_list.append(max_case_score_list[real_max_index[0]].max().round(5))
            else:
                case_score_list.append(mean_case_score_list[real_mean_index[0]].mean().round(5))
        final_score_list = ['{:.5f}'.format(case) for case in case_score_list]
        print([config.metric_list[i] + ": " + final_score_list[i] for i in range(len(final_score_list))])
        logging.info([config.metric_list[i] + ": " + final_score_list[i] for i in range(len(final_score_list))])
    
    return mean_dice

def train(train_loader, eval_loader, model, sentry, optimizer, optimizer_sentry, epoch, save_path, loss_func, loss_sentry, max_Dice):
    global step
    model.cuda(device=device_ids[0]).train()
    sentry.cuda(device=device_ids[0]).train()
    loss_all = 0
    epoch_step = 0

    try:
        # mean_dice = start_eval(eval_loader, model, sentry)
        ## Training ##
        for i, (images, gts, _) in enumerate(train_loader, start=1):
            images = images.cuda(device=device_ids[0])
            gts = gts.cuda(device=device_ids[0])
            if gts.shape[0] == 1:
                continue
            
            ### First: Training Sentry ###
            sentry = sentry.train()
            prototypes, features, initial_out = model.forward_generate(images)
            neg_pairs, pos_pairs = model.forward_train_sentry(prototypes.clone().detach())
            neg_out = sentry(neg_pairs)
            pos_out = sentry(pos_pairs)
            fake_loss = loss_sentry(neg_out, torch.zeros_like(neg_out))
            real_loss = loss_sentry(pos_out, torch.ones_like(pos_out))
            sentry_loss = fake_loss + real_loss
            optimizer_sentry.zero_grad()
            sentry_loss.backward()
            # clip_gradient(optimizer_sentry, config.clip)
            optimizer_sentry.step()

            ### Second: Training Model ###
            sentry = sentry.eval()
            preds = model(prototypes, features, (gts.shape[-2], gts.shape[-1]), mode="train", sentry=sentry)
            preds = preds.view(gts.shape)
            initial_out = initial_out.view(gts.shape)
            loss_1 = loss_func(preds[:, 0].squeeze(1).contiguous(), gts[:, 0].contiguous().view(-1, *(gts.shape[2:])).squeeze(1)) + \
                    loss_func(initial_out[:, 0].squeeze(1).contiguous(), gts[:, 0].contiguous().view(-1, *(gts.shape[2:])).squeeze(1))
            loss_2 = loss_func(preds[:, -1].squeeze(1).contiguous(), gts[:, -1].contiguous().view(-1, *(gts.shape[2:])).squeeze(1)) + \
                    loss_func(initial_out[:, -1].squeeze(1).contiguous(), gts[:, -1].contiguous().view(-1, *(gts.shape[2:])).squeeze(1))
            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            # clip_gradient(optimizer, config.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 50 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}, Sentry_loss: {:.4f}'.
                    format(datetime.now(), epoch, config.epoches, i, total_step, loss.data, sentry_loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f}, Sentry_loss: {:.4f}'.
                    format(epoch, config.epoches, i, total_step, loss.data, sentry_loss.data))
        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, config.epoches, loss_all))
        
        ## Evaluation ##
        if config.eval_on:
            mean_dice = start_eval(eval_loader, model, sentry)
            if float(mean_dice) > max_Dice:
                max_Dice = float(mean_dice)
                logging.info("meanDice: " + str(mean_dice) + ", saving epoch: " + str(epoch))
                torch.save(model.state_dict(), os.path.join(save_path, "ckpt_epoch_%d.pth"%(epoch)))
                torch.save(sentry.state_dict(), os.path.join(save_path, "sentry_epoch_%d.pth"%(epoch)))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, "ckpt_epoch_%d.pth"%(epoch)))
            torch.save(sentry.state_dict(), os.path.join(save_path, "sentry_epoch_%d.pth"%(epoch)))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + '/Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise
    
    return max_Dice

gpu_id = config.gpu_id
if ',' in gpu_id:
    device_ids = gpu_id.split(',')
    device_ids = [int(idx) for idx in device_ids]
else:
    device_ids = [int(gpu_id)]
device = torch.device('cuda:{}'.format(device_ids[0]) if torch.cuda.is_available() else 'cpu')
print('USE GPU: ', gpu_id)

if __name__ == '__main__':
    
    current_time = time.strftime('%Y-%m%d-%H%M%S', time.localtime()) + '-' + "SentryNet"
    config_path = "/media/cgl/VPSentry/scripts/config.py"
    save_path = os.path.join(config.save_path, current_time)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copyfile(
        config_path, 
        os.path.join(save_path, os.path.basename(config_path))
    )
    # logging
    logging.basicConfig(filename=os.path.join(save_path,'log.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.getLogger(__name__)

    model = SentryNet(f_num=config.video_time_clips, img_size=config.size, mlp_ratio=2.0)
    sentry = Discriminator(in_channels=1, pool_size=11)

    print("model success loaded!")

    cudnn.benchmark = True

    base_params = [params for name, params in model.named_parameters() if ("feature_extractor" in name)]
    finetune_params = [params for name, params in model.named_parameters() if ("feature_extractor" not in name)]
    optimizer = torch.optim.AdamW([{'params':base_params, 'lr':0.1*config.base_lr}, {'params':finetune_params, 'lr':config.base_lr}], weight_decay=config.weight_decay, amsgrad=False)
    optimizer_sentry = torch.optim.AdamW(sentry.parameters(), lr=config.sentry_lr, weight_decay=config.weight_decay, amsgrad=False)

    scheduler_dis = WarmupConsineLR(
        optimizer_sentry,
        warmup_epochs=2,
        max_epochs=config.epoches,
        min_lr_ratio=config.sentry_power,
        anneal_strategy="anneal",
    )
    scheduler = WarmupConsineLR(
        optimizer,
        warmup_epochs=2,
        max_epochs=config.epoches,
        min_lr_ratio=config.base_power,
        anneal_strategy="anneal",
    )

    loss_func = CrossEntropyLoss()
    loss_sentry = torch.nn.BCELoss()

    # load data
    print('load data...')
    train_dataset = get_video_dataset(config.dataset, 'train')
    eval_dataset = get_video_dataset(config.evaldataset, 'eval')
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=config.batchsize,
                                   shuffle=True,
                                   num_workers=config.num_workers,
                                   pin_memory=False
                                   )
    eval_loader = data.DataLoader(dataset=eval_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=config.num_workers,
                                    pin_memory=False)

    logging.info('Train on {}'.format(config.dataset))
    logging.info('Eval on {}'.format(config.evaldataset))
    print('Train on {}'.format(config.dataset))
    print('Eval on {}'.format(config.evaldataset))

    total_step = len(train_loader)
    
    logging.info("Network-Train")
    print("Network-Train")
    
    logging.info(
        f"Config: epoch: {config.epoches}; lr: {config.base_lr}; batchsize: {config.batchsize}; "
        f"video_time_clips: {config.video_time_clips}, trainsize: {config.size}; clip: {config.clip}; "
        f"save_path: {config.save_path}"
    )
    print(
        f"Config: epoch: {config.epoches}; lr: {config.base_lr}; batchsize: {config.batchsize}; "
        f"trainsize: {config.size}; clip: {config.clip}; save_path: {config.save_path}"
    )
    step = 0

    print("Start train...")
    max_Dice = 0.

    for epoch in range(config.epoches):
        logging.info("Current LR:" + str(optimizer.state_dict()['param_groups'][1]['lr']))
        logging.info("Current Sentry LR:" + str(optimizer_sentry.state_dict()['param_groups'][0]['lr']))
        # if epoch == (config.epoches-1):
        #     import ipdb; ipdb.set_trace()
        model.memory_p = None
        model.memory_neg_p = None
        max_Dice = train(train_loader, eval_loader, model, sentry, optimizer, optimizer_sentry, epoch, save_path, loss_func, loss_sentry, max_Dice)
        scheduler.step()
        scheduler_dis.step()
