import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--lr_mode', type=str, default="poly")
parser.add_argument('--base_lr', type=float, default=2e-4)
parser.add_argument('--sentry_lr', type=float, default=4e-4)
parser.add_argument('--base_power', type=float, default=0.05)
parser.add_argument('--sentry_power', type=float, default=0.05)
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--clip', type=float, default=3.0, help='gradient clipping margin')
parser.add_argument('--epoches', type=int, default=30)

# data
parser.add_argument('--data_statistics', type=str,
                    default="lib/dataloader/statistics.pth", help='The normalization statistics.')
parser.add_argument('--dataset', type=str, default="TrainDataset", help="Training Dataset")
parser.add_argument('--evaldataset', type=str,
                    default="TestHardDataset/Unseen", help="TestHardDataset/Unseen")
parser.add_argument('--dataset_root', type=str,
                    default="/media/cgl/SUN-SEG/", help="/media/cgl/SUN-SEG/")
parser.add_argument('--size', type=tuple, default=(352, 352))
parser.add_argument('--batchsize', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--video_time_clips', type=int, default=20)
parser.add_argument('--save_path', type=str, default='/media/cgl/VPSentry/experiments/')

# eval
parser.add_argument('--eval_on', type=bool, default=True)
parser.add_argument('--tf_img_only', type=bool, default=False)
parser.add_argument(
    '--metric_list', type=list, help='set the evaluation metrics',
    default=['Smeasure', 'meanEm', 'wFmeasure', 'MAE'],
    choices=["Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm", "meanFm", "maxFm",
                "meanSen", "maxSen", "meanSpe", "maxSpe", "meanDice", "maxDice", "meanIoU", "maxIoU"])

config = parser.parse_args()