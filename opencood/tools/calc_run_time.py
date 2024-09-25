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
from torch.profiler import profile, record_function, ProfilerActivity


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


def inference_throughput_naive(model, data):
    print("start inference throughput performance test")
    run_num = 50
    print("warm up ...\n")
    with torch.no_grad():  # warm up
        for i in range(run_num):
            output = model(data)
    print("warm up done.")
    run_num = 200
    with torch.no_grad():
        start_time = time.time()
        for i in range(run_num):
            output = model(data)
        end_time = time.time()
        infer_thro = run_num / (end_time - start_time)
        print("inference throughput (naive): ", infer_thro)

    return infer_thro


def inference_throughput_cuda_event(model, data):
    print("start inference throughput performance test")
    run_num = 100
    print("warm up ...\n")
    with torch.no_grad():  # warm up
        for i in range(run_num):
            output = model(data)
    print("warm up done.")

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    # 初始化一个时间容器
    run_num = 200
    timings = np.zeros((run_num,))

    print("start testing ...\n")
    with torch.no_grad():
        for i in range(run_num):
            starter.record()
            output = model(data)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成
            curr_time = starter.elapsed_time(
                ender
            )  # 从 starter 到 ender 之间用时,单位为毫秒
            timings[i] = curr_time / 1000

    infer_thro = run_num / timings.sum()
    print("inference throughput (cuda event): ", infer_thro)

    return infer_thro


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
    for i, batch_data in tqdm(enumerate(data_loader)):
        batch_data = train_utils.to_device(batch_data, device)
        cav_content = batch_data["ego"]
        break

    infer_throu_cuda_event = inference_throughput_cuda_event(model, cav_content)
    print(
        "inference throughput (by cuda.Event):", infer_throu_cuda_event, " sample/sec."
    )
    torch.cuda.empty_cache()

    infer_throu_naive_event = inference_throughput_naive(model, cav_content)
    print(
        "inference throughput (by naive time):", infer_throu_naive_event, " sample/sec."
    )
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
