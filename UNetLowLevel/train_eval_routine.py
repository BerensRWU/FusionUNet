import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split


from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

def train_net(net,
              device,
              dir_img: str,
              dir_mask: str,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              channels_sensors: list = [1, 3],
              type_disruptions: list = [None, None],
              prop_disruptions: list = [0, 0],
              level_disruptions: list = [0, 0],
              sensor_ids_list: list = [0, 1]
              ):
    # 1. Create dataset
    try:
        dataset = SyntheticDataset(dir_img, dir_mask, img_scale, channels_sensors,sensor_ids_list=sensor_ids_list,
                                   type_disruptions=type_disruptions, prop_disruptions=prop_disruptions, level_disruptions=level_disruptions)
    except (AssertionError, RuntimeError):
        raise NotImplementedError

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])  # , generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        #with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
        for batch in train_loader:
            images = batch['image']
            true_masks = batch['mask']
            assert images.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = net(images)
                loss = criterion(masks_pred, true_masks) \
                       + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                   F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                   multiclass=True)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            #pbar.update(images.shape[0])
            global_step += 1
            epoch_loss += loss.item()

            #pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round
            division_step = (n_train // (10 * batch_size))
            if division_step > 0:
                if global_step % division_step == 0:
                    val_score = evaluate(net, val_loader, device)
                    scheduler.step(val_score)

                    #logging.info('Validation Dice score: {}'.format(val_score))
    return net


def eval_network(net, dir_img: str, dir_mask: str,
                sensor_ids_list: list = [0, 1],
                type_disruptions: list = [None, None],
                prop_disruptions: list = [0, 0],
                level_disruptions: list = [0, 0], ):
    """

    Returns
    -------
    object
    """
    dataset = SyntheticDataset(dir_img,
                               dir_mask,
                               args.scale,
                               args.channels_sensors,
                               sensor_ids_list=sensor_ids_list,
                               type_disruptions=type_disruptions,
                               prop_disruptions=prop_disruptions,
                               level_disruptions=level_disruptions)

    loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)

    val_score = evaluate(net, eval_loader, device)
    
    return val_score.cpu()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--det', action='store_true', default=False, help='Determistic or not')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--channels_sensors', type=str, default='1 3', help='Number of channels for every sensors')
    parser.add_argument('--initial_seed', type=int, default=2022, help='Sets the initial seed.')
    parser.add_argument('--data_root', type=str, default="./data/")
    parser.add_argument('--disturb_type_training', type=str, default="None None")
    parser.add_argument('--disturb_prop_training', type=str, default="0 0")
    parser.add_argument('--disturb_level_training', type=str, default="0 0")
    parser.add_argument('--saved_fn', type=str, default="trained_missing_none")
    parser.add_argument('--sensor_ids_list', type=str, default="0 1")
    parser.add_argument('--repeats', type=str, default="0")

    args = parser.parse_args()

    args.sensor_ids_list = args.sensor_ids_list.split(" ")
    args.sensor_ids_list = [int(x) for x in args.sensor_ids_list]

    args.disturb_prop_training = args.disturb_prop_training.split(" ")
    args.disturb_prop_training = [float(x) for x in args.disturb_prop_training]
    
    args.disturb_level_training = args.disturb_level_training.split(" ")
    args.disturb_level_training = [int(x) for x in args.disturb_level_training]

    args.disturb_type_training = args.disturb_type_training.split(" ")

    args.channels_sensors = args.channels_sensors.split(" ")
    args.channels_sensors = [int(x) for x in args.channels_sensors]

    args.repeats = args.repeats.split(" ")
    args.repeats = [int(x) for x in args.repeats]

    return args


if __name__ == '__main__':
    args = get_args()
    
    if args.det:
        from utils.data_loading_det import SyntheticDataset
    else:
        from utils.data_loading import SyntheticDataset
    
    torch.manual_seed(args.initial_seed)

    n_sensors = len(args.channels_sensors)
    n_channels_net = sum(args.channels_sensors)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    os.makedirs(args.saved_fn, exist_ok = True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel

    logging.info(f'Network:\n'
                 f'\t{n_channels_net} input channels\n'
                 f'\t{args.classes} output channels (classes)\n'
                 f'\t{args.channels_sensors} channels of sensors\n'
                 # f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'
                 f'Missing:\n'
                 f'\t{args.disturb_prop_training} disturb_prop_training\n'
                 f'\t{args.disturb_type_training} disturb_type_training\n'
                 f'\t{args.disturb_level_training} disturb_level_training\n'
                 )
    try:
        val_score_missing_none = []

        for rep in args.repeats:
            
            dir_img_train = Path(f'{args.data_root}/rep_{rep}/train/')
            dir_mask_train = Path(f'{args.data_root}/rep_{rep}/train/labels/')
            dir_img_test = Path(f'{args.data_root}/rep_{rep}/test/')
            dir_mask_test = Path(f'{args.data_root}/rep_{rep}/test/labels/')
            
            net = UNet(n_channels=n_channels_net, n_classes=args.classes, bilinear=args.bilinear)
            net.to(device=device)
            print(f"Network generated, repeat: {rep}")
            net = train_net(net=net,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            learning_rate=args.lr,
                            device=device,
                            img_scale=args.scale,
                            val_percent=args.val / 100,
                            amp=args.amp,
                            channels_sensors=args.channels_sensors,
                            prop_disruptions=args.disturb_prop_training,
                            type_disruptions=args.disturb_type_training,
                            level_disruptions=args.disturb_level_training,
                            dir_img=dir_img_train,
                            dir_mask=dir_mask_train,
                            sensor_ids_list=args.sensor_ids_list)
                            
            print(f"Finished Training, repeat: {rep}")
            
            for type_sens in ["none", "blur", "snp_delta", "delete_pixel"]:
                eval_score = np.zeros((1,6))
                for level_sens in range(6):
                    print("Training:",args.saved_fn)
                    print("Eval: ",type_sens, type_sens, level_sens, "Rep", rep)
                    eval_score[0,level_sens] = eval_network(net, dir_img_test, dir_mask_test, type_disruptions=[type_sens, type_sens],
                                            prop_disruptions=[1, 1], level_disruptions=[level_sens, level_sens], 
                                            sensor_ids_list=args.sensor_ids_list)
                filename = f"{args.saved_fn}/eval_disturbances_sens1_{type_sens}_sens2_{type_sens}.npy"
                if os.path.exists(filename):
                    loaded_array = np.load(filename)
                    appended_array = np.concatenate((loaded_array,eval_score))
                    np.save(filename, appended_array)
                else:
                    np.save(filename, eval_score)
                    
            for type_sens1 in ["blur", "snp_delta", "delete_pixel"]:
                type_sens2 = "none"
                eval_score = np.zeros((1,6))
                for level_sens in range(6):
                    print("Training:",args.saved_fn)
                    print("Eval: ",type_sens1, type_sens2, level_sens, "Rep", rep)
                    eval_score[0,level_sens] = eval_network(net, dir_img_test, dir_mask_test, type_disruptions=[type_sens1, type_sens2],
                                            prop_disruptions=[1, 1], level_disruptions=[level_sens, level_sens], 
                                            sensor_ids_list=args.sensor_ids_list)
                filename = f"{args.saved_fn}/eval_disturbances_sens1_{type_sens1}_sens2_{type_sens2}.npy"
                if os.path.exists(filename):
                    loaded_array = np.load(filename)
                    appended_array = np.concatenate((loaded_array,eval_score))
                    np.save(filename, appended_array)
                else:
                    np.save(filename, eval_score)
                    
            for type_sens2 in ["blur", "snp_delta", "delete_pixel"]:
                type_sens1 = "none"
                eval_score = np.zeros((1,6))
                for level_sens in range(6):
                    print("Training:",args.saved_fn)
                    print("Eval: ",type_sens1, type_sens2, level_sens, "Rep", rep)
                    eval_score[0,level_sens] = eval_network(net, dir_img_test, dir_mask_test, type_disruptions=[type_sens1, type_sens2],
                                            prop_disruptions=[1, 1], level_disruptions=[level_sens, level_sens], 
                                            sensor_ids_list=args.sensor_ids_list)
                filename = f"{args.saved_fn}/eval_disturbances_sens1_{type_sens1}_sens2_{type_sens2}.npy"
                if os.path.exists(filename):
                    loaded_array = np.load(filename)
                    appended_array = np.concatenate((loaded_array,eval_score))
                    np.save(filename, appended_array)
                else:
                    np.save(filename, eval_score)
                
    except KeyboardInterrupt:

        raise
