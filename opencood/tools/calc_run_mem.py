from torchvision.models import resnet50
import torch
import torch.nn as nn
from thop import profile
from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import clever_format

import argparse
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import open3d as o3d
from torch.utils.data import DataLoader
from collections import OrderedDict
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt
from torchstat import stat
import torchvision.models as models
from ptflops import get_model_complexity_info
from torch.profiler import profile as tprofile, record_function, ProfilerActivity


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Continued training path"
    )
    parser.add_argument(
        "--hypes_yaml", type=str, required=False, help="Continued training path"
    )
    parser.add_argument("--epoch", "-e", type=int, required=False, help="specify epoch")

    opt = parser.parse_args()
    return opt




def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)
    print("Dataset Building")
    opencood_dataset = build_dataset(hypes, visualize=True, train=False, isSim=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    print("Creating Model")
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Model from checkpoint")
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()
    cav_content = None
    mem_before_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    mem_before_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    print("Mem reserved (before):", mem_before_reserved, " MB")
    print("Mem allocated (before):", mem_before_allocated, " MB")
    for i, batch_data in tqdm(enumerate(data_loader)):
        batch_data = train_utils.to_device(batch_data, device)
        cav_content = batch_data["ego"]
        break
    with torch.no_grad():
        model(cav_content)
    mem_after_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    mem_after_allocated = torch.cuda.memory_allocated() / (1024 ** 2)

    print("Mem reserved (after):", mem_after_reserved, " MB")
    print("Mem allocated (after):", mem_after_allocated, " MB")

    with torch.no_grad():
        with tprofile(activities=[ProfilerActivity.CUDA,ProfilerActivity.CPU], record_shapes=True, use_cuda=True) as prof:
            with record_function("model_inference"):
                model(cav_content)
    print("GPU time sorted operators:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print("CPU time sorted operators:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


if __name__ == "__main__":
    main()
