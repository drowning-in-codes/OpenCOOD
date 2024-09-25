from torchvision.models import resnet50
import torch
import torch.nn as nn
from thop import profile
from fvcore.nn import FlopCountAnalysis,flop_count_table
from thop import clever_format

import argparse
import os
import time
from tqdm import tqdm

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

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--hypes_yaml', type=str, required=False,
                        help='Continued training path')
    parser.add_argument('--epoch', '-e',type=int,required=False,
                        help='specify epoch')

    opt = parser.parse_args()
    return opt

def calc_param(model):
    """
    Calculate the number of parameters in the model.
    :param model: model
    :return: number of parameters
    """
    param_num =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(param_num*4/1024/1024, 'MB') # assume dtype float32

macs_list = []
params_list = []
def main():
    opt = test_parser()
    hypes = yaml_utils.load_yaml(None, opt)
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False,isSim=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=0,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()
    for i, batch_data in tqdm(enumerate(data_loader)):
        # print(i)
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            output_dict = OrderedDict()
            cav_content = batch_data['ego']
            output_dict['ego'] = model(cav_content)
            # pred_box_tensor, pred_score, gt_box_tensor = \
            #     opencood_dataset.post_process(batch_data,
            #                          output_dict)
            if method == "thop":
                macs, params = profile(model, inputs=(cav_content,))
                macs, params = clever_format([macs, params], "%.3f")
                print(macs, params)
                macs_list.append(macs)
                params_list.append(params)
            elif method == "stat":
                stat(model, cav_content.shape)
            elif method == "fvcore":
                print(flop_count_table(FlopCountAnalysis(model, (cav_content,))))

            else:
                flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                          print_per_layer_stat=True)
                macs_list.append(flops)
                params_list.append(params)

        if i == calc_avg_num-1 and method != "stat":
            avg_macs = sum([float(macs[:-4]) for macs in macs_list])/calc_avg_num
            avg_params = sum([float(params[:-4]) for params in params_list])/calc_avg_num
            print("Average MACs: ", avg_macs,"Average Params: ", avg_params)
            break
        elif i == calc_avg_num-1:
            break
if __name__ == '__main__':
    calc_avg_num = 50
    method = "fvcore"
    main()



