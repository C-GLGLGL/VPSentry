import os
import sys
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Resize

current_path = os.path.abspath(__file__)
sys_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
sys.path.append(sys_path)

from scripts.config import config
from lib.dataloader.preprocess import *

USE_A = False

class VideoDataset(Dataset):
    def __init__(self, video_dataset, transform=None, time_interval=1, ttype='train', img_only=False):
        super(VideoDataset, self).__init__()

        self.time_clips = config.video_time_clips
        self.video_train_list = []
        self.ttype = ttype
        self.img_only = img_only

        video_root = os.path.join(config.dataset_root, video_dataset)
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')

        cls_list = os.listdir(img_root)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)

            tmp_list = os.listdir(cls_img_path)
            
            if 'SUN-SEG' in config.dataset_root:
                tmp_list = list(filter(lambda x: 'case' in x, tmp_list))
                if 'train' in ttype:
                    tmp_list.sort(key=lambda name: (
                        int(name.split('-')[0].split('_')[-1]),
                        int(name.split('_a')[1].split('_')[0]),
                        int(name.split('_image')[1].split('.jpg')[0])))
                else:
                    tmp_list.sort(key=lambda name: (
                        int(name.split('-')[0].split('_')[-1]),
                        int(name.split('_a')[1].split('_')[0]),
                        int(name.split('_image')[1].split('.jpg')[0])))
            elif 'ClinicDB' in config.dataset_root:
                tmp_list.sort(key=lambda name:(
                    int(name.split('.')[0])
                ))
            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png"))
                ))
        # ensemble
        for cls in cls_list:
            li = self.video_filelist[cls]
            if 'train' in ttype:
                for begin in range(0, len(li) - (self.time_clips - 1) * time_interval - 1, self.time_clips):
                    batch_clips = []
                    # batch_clips.append(li[begin])
                    for t in range(self.time_clips):
                        batch_clips.append(li[begin + time_interval * t])
                    self.video_train_list.append(batch_clips)
            else:
                begin = 0  # change for inference from first frame
                while begin < len(li):
                    if len(li) - begin - 1 < self.time_clips:
                        begin = len(li) - self.time_clips
                    batch_clips = []
                    # batch_clips.append(li[begin])
                    for t in range(self.time_clips):
                        batch_clips.append(li[begin + time_interval * t])
                    begin += self.time_clips
                    self.video_train_list.append(batch_clips)

        self.img_label_transform = transform
        self.totensor = ToTensor()

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        for idx, (img_path, label_path) in enumerate(img_label_li):
            if not USE_A:
                img = Image.open(img_path).convert('RGB')
            else:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if not self.img_only:
                if not USE_A:
                    label = Image.open(label_path).convert('L')
                else:
                    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            else:
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            if USE_A:
                img_label = self.img_label_transform(image=img, mask=label)
                img = self.totensor(img_label['image'])
                if self.ttype == 'train':
                    label = self.totensor(img_label['mask']).unsqueeze(0)
                else:
                    label = self.totensor(img_label['mask'])

            img_li.append(img)
            label_li.append(label)
        if not USE_A:
            img_li, label_li = self.img_label_transform(img_li, label_li)
        for idx, (img, label) in enumerate(zip(img_li, label_li)):
            if idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                if not self.img_only:
                    LABEL = torch.zeros(len(img_li), *(label.shape))
                else:
                    LABEL = np.zeros((len(img_li), 1, *(label.shape)), dtype=np.uint8)
                IMG[idx, :, :, :] = img
            else:
                IMG[idx, :, :, :] = img
            LABEL[idx, :, :, :] = label
        return IMG, LABEL, img_label_li

    def __len__(self):
        return len(self.video_train_list)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        return img

class Test_Dataset(Dataset):
    def __init__(self, root, testset):
        time_interval = 1

        self.time_clips = config.video_time_clips
        self.video_test_list = []

        video_root = os.path.join(root, testset, 'Frame')
        cls_list = os.listdir(video_root)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []
            cls_path = os.path.join(video_root, cls)
            tmp_list = os.listdir(cls_path)
            tmp_list = list(filter(lambda x: 'case' in x, tmp_list))

            tmp_list.sort(key=lambda name: (
                #int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))

            for filename in tmp_list:
                self.video_filelist[cls].append(os.path.join(cls_path, filename))

        # ensemble
        for cls in cls_list:
            li = self.video_filelist[cls]
            begin = 0
            while begin < len(li):
                if len(li) - begin - 1 < self.time_clips:
                    begin = len(li) - self.time_clips
                batch_clips = []
                batch_clips.append(li[0])
                for t in range(self.time_clips):
                    batch_clips.append(li[begin + time_interval * t])
                begin += self.time_clips
                self.video_test_list.append(batch_clips)

        self.img_transform = Compose([
            Resize((config.size[0], config.size[1]), Image.BILINEAR),
            ToTensor(),
            Normalize([0.4732661, 0.44874457, 0.3948762],
                      [0.22674961, 0.22012031, 0.2238305])
        ])

    def __getitem__(self, idx):
        img_path_li = self.video_test_list[idx]
        IMG = None
        img_li = []
        for idx, img_path in enumerate(img_path_li):
            img = Image.open(img_path).convert('RGB')
            img_li.append(self.img_transform(img))
        for idx, img in enumerate(img_li):
            if IMG is not None:
                IMG[idx, :, :, :] = img
            else:
                IMG = torch.zeros(len(img_li), *(img.shape))
                IMG[idx, :, :, :] = img
        return IMG, img_path_li

    def __len__(self):
        return len(self.video_test_list)


def get_video_dataset(dataset_name=None, ttype='train', time_interval=1):
    """
        In statistics:
            'mean': array([0.4732661 , 0.44874457, 0.3948762]
            'std': array([0.22674961, 0.22012031, 0.2238305]
    """
    statistics = torch.load(config.data_statistics)
    tf_img_only = config.tf_img_only

    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        # Random_crop_Resize_Video(15),
        Random_horizontal_flip_video(0.5),
        Random_vertical_flip_video(0.5),
        Random_zooming(0.5, (config.size[0], config.size[1])),
        Random_rotation90(0.5),
        # Random_rotation(0.5, angle=25),
        toTensor_video(),
        Normalize_video(statistics["mean"], statistics["std"])
    ])
    trsf_eval = Compose_imglabel([
        Resize_video(config.size[0], config.size[1], img_only=tf_img_only),
        toTensor_video(img_only=tf_img_only),
        Normalize_video(statistics["mean"], statistics["std"])
        # Normalize_video([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    

    if 'train' in ttype:
        train_loader = VideoDataset(dataset_name, transform=trsf_main, time_interval=time_interval, ttype=ttype)
    else:
        train_loader = VideoDataset(dataset_name, transform=trsf_eval, time_interval=1, ttype=ttype, img_only=tf_img_only)

    return train_loader


if __name__ == "__main__":
    statistics = torch.load(config.data_statistics)
    trsf_main = Compose_imglabel([
        Resize_video(config.size[0], config.size[1]),
        Random_crop_Resize_Video(7),
        Random_horizontal_flip_video(0.5),
        toTensor_video(),
        Normalize_video(statistics["mean"], statistics["std"])
    ])
    train_loader = VideoDataset(config.dataset, transform=trsf_main, time_interval=1)
