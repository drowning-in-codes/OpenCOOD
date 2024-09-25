# -*- coding: utf-8 -*-
# Modified by proanimer
# OriginaL Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import numpy as np
import datetime
import statistics
from types import SimpleNamespace
import logging
import torch
import tqdm
import wandb
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.version import __PROJECT__


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument(
        "--hypes_yaml",
        type=str,
        required=True,
        help="data generation yaml file needed ",
    )
    parser.add_argument("--model_dir", default="", help="Continued training path")
    parser.add_argument("--last_epoch", type=int,default=-1,required=False, help="specify the last epoch to continue training")
    parser.add_argument(
        "--half", action="store_true", help="whether train with half precision."
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    opt = parser.parse_args()
    return opt


def create_logger(log_file):
    # 创建一个logger
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    # 创建一个文件处理器，并将日志输出到文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    # 创建一个格式化器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # 将格式化器添加到文件处理器
    file_handler.setFormatter(formatter)
    # 将文件处理器添加到logger
    logger.addHandler(file_handler)

    return logger


def main():
    logger = create_logger("training.log")
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print("-----------------Dataset Building------------------")
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes["train_params"]["batch_size"], drop_last=True
        )

        train_loader = DataLoader(
            opencood_train_dataset,
            batch_sampler=batch_sampler_train,
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            sampler=sampler_val,
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            opencood_train_dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
        )

    print("---------------Creating Model------------------")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model,last_epoch=opt.last_epoch)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[opt.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    # init wandb
    # run = wandb.init(project=__PROJECT__,config=opt,job_type="train")
    logger.info(
        f"{opt.hypes_yaml} Training start {opt.model_dir if opt.model_dir else ''} \n"
    )
    epoches = hypes["train_params"]["epoches"]
    # used to help schedule learning rate
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        avg_loss = []
        start_time = datetime.datetime.now()
        if hypes["lr_scheduler"]["core_method"] != "cosineannealwarm":
            scheduler.step(epoch)
        if hypes["lr_scheduler"]["core_method"] == "cosineannealwarm":
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print("learning rate %.7f" % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data["ego"])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict, batch_data["ego"]["label_dict"])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data["ego"])
                    final_loss = criterion(ouput_dict, batch_data["ego"]["label_dict"])
            avg_loss.append(final_loss.detach().cpu().numpy())
            # run.log({"train/final_loss":final_loss,"train/epoch":epoch})
            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes["lr_scheduler"]["core_method"] == "cosineannealwarm":
                scheduler.step_update(epoch * num_steps + i)

        # logger info
        end_time = datetime.datetime.now()
        training_time = end_time - start_time
        logger.info(
            f'Epoch: {epoch}, Learning Rate: {param_group["lr"]},Training time: {training_time},Loss: {np.mean(avg_loss)}\n'
        )

        # after training 30+ epochs,results are saved every 5 epochs
        # this is because I found that the model is progressively stable here
        if (
            (epoch + 1 < 30 and (epoch + 1) % hypes["train_params"]["save_freq"] == 0)
            or (epoch + 1 >= 30 and (epoch + 1) % 5 == 0)
        ):
            if "wild_setting" in hypes and "loc_err" in hypes["wild_setting"] and hypes["wild_setting"]['loc_err'] and ( hypes["wild_setting"]["xyz_std"] != 0 or hypes["wild_setting"]["ryp_std"] != 0):
                file_name = "net_epoch%d_loc_%f_head_%f.pth" % (
                    epoch + 1,
                    hypes["wild_setting"]["xyz_std"],
                    hypes["wild_setting"]["ryp_std"],
                )
                torch.save(
                    model_without_ddp.state_dict(),
                    os.path.join(saved_path, file_name),
                )
            else:
                file_name = "net_epoch%d.pth" % (epoch + 1)
                torch.save(
                    model_without_ddp.state_dict(),
                    os.path.join(saved_path, file_name),
                )
        if ((epoch + 1) % hypes["train_params"]["eval_freq"] == 0):
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data["ego"])

                    final_loss = criterion(ouput_dict, batch_data["ego"]["label_dict"])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            # run.log({"train/final_loss":valid_ave_loss,"train/epoch":epoch})
            print("At epoch %d, the validation loss is %f" % (epoch, valid_ave_loss))
            writer.add_scalar("Validate_Loss", valid_ave_loss, epoch)

    print("Training Finished, checkpoints saved to %s" % saved_path)

    logger.info(
        f"{opt.hypes_yaml} Training finish {opt.model_dir if opt.model_dir else ''} \n"
    )


if __name__ == "__main__":
    main()
