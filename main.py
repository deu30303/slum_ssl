from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus, DeepLabV3Plus_Aux
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map
from augmentation import *

import argparse
import torch.nn as nn
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
# from tqdm.notebook import tqdm
import random
import copy
import torch.nn.functional as F


MODE = None


def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plusaux', 'deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)

    parser.add_argument('--save-path', type=str, required=True)

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', dest='plus', default=True, action='store_true',
                        help='whether to use ST++')
    
    parser.add_argument('--class_name', default='label', type=str, help='landcover class name')
    parser.add_argument('--labeled_num', default=1000, type=int, help='the number of labeled sample')

    args = parser.parse_args()
    return args

class Metrics:
    def __init__(self, num_classes, ignore_label):
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes)

    def update(self, pred, target):
        # pred = pred.argmax(dim=1)
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self):
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self):
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self):
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)
    

        
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

SEEDS = [42, 43, 44]
set_seed(42)


def get_train_augmentation(size, seg_fill):
    return Compose([
#         ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # RandomAdjustSharpness(sharpness_factor=0.1, p=0.5),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        # RandomGaussianBlur((3, 3), p=0.5),
#         RandomGrayscale(p=0.2),
        RandomRotation(degrees=10, p=0.3, seg_fill=seg_fill),
        RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
    ])


def get_normalize():
    return Compose([
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def get_strong_augmentation(size, seg_fill):
    return Compose([
#         ColorJitter(brightness=0.001, contrast=0.001, saturation=0.00, hue=0.000),
        # RandomAdjustSharpness(sharpness_factor=0.1, p=0.5),
        # RandomAutoContrast(p=0.2),
        RandomAutoContrast(p=0.2),
        RandomGaussianBlur((3, 3), p=0.2),
    ])

def get_val_augmentation(size):
    return Compose([
        Resize(size),
    ])


global_step = 0

traintransform = get_train_augmentation([256, 256], 255)
strongtransform = get_strong_augmentation([256, 256], 255)
valtransform = get_val_augmentation([256, 256])
normalize = get_normalize()


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    def forward(self, pred, target):
        CE = F.cross_entropy(pred, target, reduction='none', ignore_index=255)
        pt = torch.exp(-CE)
        loss = ((1 - pt) ** 2) * CE # gamma
        alpha = torch.Tensor([0.1, 0.9]) # alpha(bigger for 1(pos), MNG only)
        alpha = (target==0) * alpha[0] + (target==1) * alpha[1]
        return torch.mean(alpha * loss)


device = 'cuda' 
class_num = 2

def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None, 
                         transform=traintransform, s_transform=strongtransform, normalize=normalize)
    valloader = DataLoader(valset, batch_size=4 if args.dataset == 'cityscapes' else 1,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    
    model, optimizer = init_basic_elems(args)
    t_model, t_optimizer = init_basic_elems(args)
     
    

    #<====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (6 if args.plus else 3))

    global MODE
    MODE = 'train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path,
                           transform=traintransform, s_transform=strongtransform, normalize=normalize)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=4, drop_last=True)

    model, optimizer = init_basic_elems(args)
    print('\nParams: %.1fM' % count_params(model))

    t_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args)

    """
        ST++ framework with selective re-training
    """
#     # <===================================== Select Reliable IDs =====================================>
#     print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')

#     dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path,
#                           transform=traintransform, s_transform=strongtransform, normalize=normalize)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2, drop_last=False)

#     select_reliable(checkpoints, dataloader, args)

#     # <================================ Pseudo label reliable images =================================>
#     print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')

#     cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
#     dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path,
#                           transform=traintransform, s_transform=strongtransform, normalize=normalize)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2, drop_last=False)

#     label(t_model, dataloader, args)

#     # <================================== The 1st stage re-training ==================================>
#     print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')

#     MODE = 'train'

#     labelset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
#                            args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path,
#                            transform=traintransform, s_transform=strongtransform, normalize=normalize)
#     labelloader = DataLoader(labelset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
    
#     MODE = 'semi_train_un'

#     unlabelset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
#                            args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path,
#                            transform=traintransform, s_transform=strongtransform, normalize=normalize)
#     unlabelloader = DataLoader(unlabelset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)

#     model, optimizer = init_basic_elems(args)

#     t_model = train_un(model, teacher_model, labelloader, unlabelloader, valloader, criterion, optimizer, args)

#     # <=============================== Pseudo label unreliable images ================================>
#     print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')

#     cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
#     dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path,
#     transform=traintransform, s_transform=strongtransform, normalize=normalize)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2, drop_last=False)

#     label(t_model, dataloader, args)

#     # <================================== The 2nd stage re-training ==================================>
#     print('\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images')

#     MODE = 'train'

#     labelset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
#                            args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path,
#                            transform=traintransform, s_transform=strongtransform, normalize=normalize)
#     labelloader = DataLoader(labelset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
    
#     MODE = 'semi_train_un'

#     unlabelset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
#                            args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path,
#                            transform=traintransform, s_transform=strongtransform, normalize=normalize)
#     unlabelloader = DataLoader(unlabelset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)

#     model, optimizer = init_basic_elems(args)

#     t_model = train_un(model, t_model, labelloader, unlabelloader, valloader, criterion, optimizer, t_optimizer,  args)


# def init_basic_elems(args):
#     model_zoo = {'deeplabv3plusaux':DeepLabV3Plus_Aux, 'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
#     if args.model == 'deeplabv3plusaux':
#         model = model_zoo[args.model](args.backbone, dilations=[6, 12, 18], nclass=2)
#     else:
#         model = model_zoo[args.model](args.backbone, nclass=2)

#     head_lr_multiple = 10.0
#     if args.model == 'deeplabv2':
#         assert args.backbone == 'resnet101'
#         model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
#         head_lr_multiple = 1.0

#     optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
#                      {'params': [param for name, param in model.named_parameters()
#                                  if 'backbone' not in name],
#                       'lr': args.lr * head_lr_multiple}],
#                     lr=args.lr, momentum=0.9, weight_decay=1e-4)

#     model = DataParallel(model).cuda()

#     return model, optimizer


def train(model, trainloader, valloader, criterion, optimizer, args):
    iters = 0
    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0
    best_loss = 10000000000

    global MODE

    if MODE == 'train':
        checkpoints = []

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        total_loss = 0.0

        try:
            for i, (img, img_w, mask) in enumerate(trainloader):
                img, mask = img.cuda(), mask.cuda()

                pred = model(img)
                pred_w = model(img_w)
                if args.model == 'deeplabv3plusaux':
                    pred_, gram = pred[0], pred[1]
                    pred_w_, gram_w = pred_w[0], pred_w[1]
                    
                
                loss = criterion(pred_, mask.to(torch.int64))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                iters += 1
                lr = args.lr * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0
        except ValueError as e:
            print(e)
        metrics = Metrics(2, 255) 
        # metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)

        model.eval()

        with torch.no_grad():
            for img, mask, _ in valloader:
                img = img.cuda()
                pred = model(img)
                if args.model == 'deeplabv3plusaux':
                    pred = pred[0]
                pred = torch.argmax(pred, dim=1)

                metrics.update(pred.cpu(), mask.to(torch.int64))
                # metric.add_batch(pred.cpu().numpy(), mask.numpy())
                # mIOU = metric.evaluate()[-1]
                _, mIOU = metrics.compute_iou()

        
        ious, mIOU = metrics.compute_iou()
        acc, macc = metrics.compute_pixel_acc()
        f1, mf1 = metrics.compute_f1()

        print("f1 : ", f1)
        print("miou : ", ious)
        print("Pixel Acc : ", acc)
        print("best_mIOU : ", previous_best)

        mIOU *= 100.0
        if total_loss < best_loss:
            print('Saving...')
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            best_loss = total_loss
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

            best_model = deepcopy(model)

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints

    return best_model


def train_un(model, t_model, labelloader, unlabelloader, valloader, criterion, optimizer, optimizer_t, args):
    iters = 0
    total_iters = len(labelloader) * args.epochs
    
    fc_criterion = FocalLoss().cuda()
    
    unlabeled_train_iter = iter(unlabelloader)

    previous_best = 0.0
    best_loss = 10000000000

    global MODE

    if MODE == 'train':
        checkpoints = []

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        t_model.train()
        total_loss = 0.0

        try:
            for i, (img, img_w, mask) in enumerate(labelloader):
                img, mask = img.cuda(), mask.cuda()
                
                batch_size = img.shape[0]
                try:
                    img_u_s, img_u_w, _ = unlabeled_train_iter.next()
                except:
                    unlabeled_train_iter = iter(unlabelloader)
                    img_u_s, img_u_w, _  = unlabeled_train_iter.next()
                    
                img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
                   
                    
                t_images = torch.cat((img, img_u_w, img_u_s))                    
                t_logits, t_gram =  t_model(t_images)
                t_logits_l = t_logits[:batch_size]
                
                t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
                t_gram_uw, t_gram_us = t_gram[batch_size:].chunk(2)
                

                
                t_loss_l = criterion(t_logits_l, mask.to(torch.int64))
                
                soft_pseudo_label = torch.softmax(t_logits_uw.detach(), dim=1)
                max_probs, hard_pseudo_label = torch.max(t_logits_uw, dim=1)
                

                mask = max_probs.ge(0.95).float()
                t_loss_u = torch.mean(-(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=1)).sum(dim=1) * mask) \
                            + 1e-3*torch.mean((t_gram_uw.detach() - t_gram_us)**2)
                
                t_loss_uda = t_loss_l +  t_loss_u
                
                s_images = torch.cat((img, img_u_s))
                s_logits = model(s_images)[0]
                s_logits_l = s_logits[:batch_size]
                s_logits_us = s_logits[batch_size:]
                del s_logits

                s_loss_l_old = fc_criterion(s_logits_l.detach(), mask.to(torch.int64))
                s_loss = fc_criterion(s_logits_us, hard_pseudo_label)
                
                optimizer.zero_grad()
                s_loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    s_logits_l = model(img)[0]
                    s_loss_l_new = criterion(s_logits_l.detach(), mask.to(torch.int64))
                    
                dot_product = s_loss_l_new - s_loss_l_old
                
                _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=1)
                t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
                t_loss = t_loss_uda + t_loss_mpl
                
                optimizer_t.zero_grad()
                t_loss.backward()
                optimizer_t.step()

               
                iters += 1
                lr = args.lr * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0
                optimizer_t.param_groups[0]["lr"] = lr
                optimizer_t.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0
        except ValueError as e:
            print(e)
        metrics = Metrics(2, 255) 

        model.eval()
        t_model.eval()

        with torch.no_grad():
            for img, mask, _ in valloader:
                img = img.cuda()
                pred = t_model(img)
                if args.model == 'deeplabv3plusaux':
                    pred = pred[0]
                pred = torch.argmax(pred, dim=1)

                metrics.update(pred.cpu(), mask.to(torch.int64))
                _, mIOU = metrics.compute_iou()

        
        ious, mIOU = metrics.compute_iou()
        acc, macc = metrics.compute_pixel_acc()
        f1, mf1 = metrics.compute_f1()

        print("f1 : ", f1)
        print("miou : ", ious)
        print("Pixel Acc : ", acc)
        print("best_mIOU : ", previous_best)

        mIOU *= 100.0
        if total_loss < best_loss:
            print('Saving...')
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            best_loss = total_loss
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

            best_model = deepcopy(model)

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints

    return best_model


def select_reliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()

    id_to_reliability = []

    with torch.no_grad():
        for img, mask, id in dataloader:
            id = id[0]
            img = img.cuda().float()

            preds = []
            for model in models:
                if args.model == 'deeplabv3plusaux':
                    preds.append(torch.argmax(model(img)[0], dim=1).cpu())
                else:
                    preds.append(torch.argmax(model(img), dim=1).cpu())

            mIOU = []
            for i in range(len(preds) - 1):
                metric = Metrics(2, 255)
                metric.update(preds[i], preds[-1])
                mIOU.append(metric.compute_iou()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id, reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')


def label(model, dataloader, args):
    model.eval()

    metric = Metrics(2, 255)
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in dataloader:
            mask, id = mask[0], id[0]
            img = img.cuda()
            pred = model(img, True)
            pred = torch.argmax(pred[0], dim=1).cpu()
            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)

            pred.save(r'%s/%s' % (args.pseudo_mask_path, os.path.basename(id+'.tif')))

#             tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))


if __name__ == '__main__':
    args = parse_args()

    print()
    print(args)

    main(args)