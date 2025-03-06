# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics

import matplotlib.pyplot as plt
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
import lime
from lime import lime_image

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt



def todevice(batch_data, device):
    if isinstance(batch_data, torch.Tensor):
        return batch_data.to(device)
    elif isinstance(batch_data,list):
        return [todevice(data,device) for data in batch_data]
    elif isinstance(batch_data,dict):
        return {k:todevice(k) for k,v in batch_data.items()}
    else:
        raise NotImplementedError("Not support data type %s" % type(batch_data))


def predict(input,model):
    model.eval()
    ouput_dict = model(input)
    return todevice(ouput_dict)


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)
    print('-----------------Dataset Building------------------')
    opencood_validate_dataset = build_dataset(hypes, visualize=True, train=False)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=0,
                            collate_fn=opencood_validate_dataset.collate_batch_train,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model


    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(val_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('explain start')
    epoches = hypes['train_params']['epoches']
    print("load epoch:",epoches)
    # used to help schedule learning rate
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            model.eval()

            batch_data = train_utils.to_device(batch_data, device)
            explainer = lime_image.LimeImageExplainer()
            
            explaination = explainer.explain_instance(image=batch_data["ego"][""],classifier_fn=predict,)
            lime_img, mask = explaination.get_image_and_mask(
                label=label.item(),
                positive_only=False,
                hide_rest=False,
                num_features=11,
                min_weight=0.05
            )
            plt.plot(lime_img)
            plt.show()


if __name__ == '__main__':
    main()
