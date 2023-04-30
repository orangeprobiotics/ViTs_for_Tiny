import os
import math
import logging
import argparse
import torch.utils.data
import torch
import torch.optim as optim
from timm.data import create_transform
from lr_scheduler import build_scheduler
from timm.data import Mixup
import albumentations as albu
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms  # 图像预处理包
import random
import numpy as np
from my_dataset import MyDataSet
from svt_oo import svt_small as create_model
from utils import read_split_data, train_one_epoch, evaluate
from pcpvt import svt_small as create_teachermodel
from DKD import dkd_loss
from drloc import cal_selfsupervised_loss
from timm.data.transforms import str_to_pil_interp

logger = logging.getLogger(__name__)


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter if args.color_jitter > 0 else None,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            auto_augment=args.aa if args.aa != 'none' else None,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            interpolation=args.train_interpolation,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=str_to_pil_interp(args.train_interpolation)),
            # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    return transforms.Compose(t)


def _weight_decay(init_weight, epoch, warmup_epochs=20, total_epoch=300):
    if epoch <= warmup_epochs:
        cur_weight = min(init_weight / warmup_epochs * epoch, init_weight)
    else:
        cur_weight = init_weight * (1.0 - (epoch - warmup_epochs) / (total_epoch - warmup_epochs))
    return cur_weight


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


cut = Cutout(n_holes=1, length=16)


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # cut,
            transforms.Normalize([0.48439634, 0.50444293, 0.45058095], [0.18411824, 0.1838756, 0.19508828])]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.48439634, 0.50444293, 0.45058095], [0.18411824, 0.1838756, 0.19508828])])}
    # 验证时的图像处理
    # data_transform = {
    #     "train": build_transform(is_train=True, args=args),
    #     "val": build_transform(is_train=False, args=args)}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    if args.mix_up:
        mixup_fun = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=args.num_classes)
    else:
        mixup_fun = None

    criterion_ssup = cal_selfsupervised_loss
    batch_size = args.batch_size  # 定义batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes, ).to(device)
    num_params = count_parameters(model)
    print("Total Parameter: \t%2.6fM" % num_params)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # optimizer = optim.SGD(pg, momentum=0.9, nesterov=True,
    #                       lr=5e-4, weight_decay=0.05)
    # optimizer = optim.AdamW(pg, eps=1e-8, betas=(0.9, 0.999),
    #                         lr=5e-4, weight_decay=0.05)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (
            1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # lr_scheduler = build_scheduler(args, optimizer, len(train_loader))
    best_val_acc = 0.0
    set_seed(args)
    init_lambda_drloc = 0.0
    for epoch in range(args.epochs):
        # train
        if args.use_drloc:
            init_lambda_drloc = _weight_decay(
                args.lambda_drloc,
                epoch,
                args.ssl_warmup_epochs,
                args.epochs)
        train_loss, train_acc = train_one_epoch(student_model=model,
                                                # teacher_model=teacher_model,
                                                criterion_ssup=criterion_ssup,
                                                lambda_drloc=init_lambda_drloc,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                args=args,
                                                mixup_fun=mixup_fun,
                                                # dkd=dkd_loss
                                                )

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(args.Smodel_name))
    tb_writer.close()
    logger.info("Best Accuracy: \t%f" % best_val_acc)
    logger.info("End Training!")
    print("Best Accuracy:", best_val_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=72)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument("--use_drloc", type=bool, default=False, help="Use Dense Relative localization loss")
    parser.add_argument("--sample_size", type=int, default=64)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--mix_up', type=bool, default=False)
    parser.add_argument("--drloc_mode", type=str, default="l1", choices=["l1", "ce", "cbr"])
    parser.add_argument("--lambda_drloc", type=float, default=0.5, help="weight of Dense Relative localization loss")
    parser.add_argument("--use_abs", type=bool, default=True)
    parser.add_argument("--ssl_warmup_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--decay_epochs", type=int, default=15)
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--scheduler_name', type=str,
                        default="cosine")
    parser.add_argument('--min_lr', type=float, default=5e-6)
    parser.add_argument('--warmup_lr', type=float, default=5e-7)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--data-path', type=str,
                        default="../data_set/birds_data/birds_photos")
    parser.add_argument('--Smodel_name', default='my_model', help='create model name')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()

    main(args)
