import argparse
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.data_loading import BasicDataset, SyntheticDataset
from unet import UNet


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    #for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
    for batch in dataloader:
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches

def evaluate_high(nets, dataloader, device, channels_sensors):
    for net in nets:
        net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    #for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
    for batch in dataloader:
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = []
            for i, net in enumerate(nets):
                if i == 0:
                    img = image[:,0:channels_sensors[0]]
                else:
                    img = image[:,channels_sensors[0]:]
                mask_pred += [torch.unsqueeze(net(img), dim=-1)]
            mask_pred = torch.cat(mask_pred,-1)
            #mask_pred = torch.max(mask_pred, -1)[0]
            #mask_pred = torch.mean(mask_pred, -1)
            mask_pred = 0.25 * mask_pred[:,:,:,:,0] + 0.75 * mask_pred[:,:,:,:,1]
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
    for net in nets:
        net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the UNet on images and target masks')
    parser.add_argument('--epoch', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--checkpoint', type=str, default="checkpoints/", help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--channels_sensors', type=list, default=[1, 3], help='Number of channels for every sensors')

    return parser.parse_args()

if __name__ == "__main__":
    dir_img = '../SyntheticDummyDataset_LightAndColor/data/test/'
    dir_mask ='../SyntheticDummyDataset_LightAndColor/data/train/labels/'

    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    logging.info(f'Using device {device}')

    n_channels_net = sum(args.channels_sensors)
    model_path = f"{args.checkpoint}/checkpoint_epoch{args.epoch}.pth"
    net = UNet(n_channels=n_channels_net, n_classes=args.classes, bilinear=args.bilinear)
    net.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f'Model loaded from {model_path}')
    net.to(device=device)
    dataset = SyntheticDataset(dir_img, dir_mask, args.scale, args.channels_sensors)

    loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)

    val_score = evaluate(net, eval_loader, device)

    print(val_score)
    logging.info('Validation Dice score: {}'.format(val_score))
