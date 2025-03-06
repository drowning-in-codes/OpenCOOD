import os
import argparse
from torch.utils.data import DataLoader

from opencood.hypes_yaml.v2_yaml_utils import load_yaml
from opencood.visualization import vis_utils_v2
from opencood.data_utils.datasets.v2_early_fusion_dataset  import \
    V2EarlyFusionDataset


def vis_parser():
    parser = argparse.ArgumentParser(description="data visualization")
    parser.add_argument('--color_mode', type=str, default="intensity",
                        help='lidar color rendering mode, e.g. intensity,'
                             'z-value or constant.')
    parser.add_argument('--isSim', action='store_true')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    opt = vis_parser()
    params = load_yaml(os.path.join(current_path,
                                    '../hypes_yaml/v2v4_vis.yaml'))

    opencda_dataset = V2EarlyFusionDataset(params, visualize=True,
                                            train=False)
    data_loader = DataLoader(opencda_dataset, batch_size=1,
                             collate_fn=opencda_dataset.collate_batch_train,
                             shuffle=False,
                             pin_memory=False)

    vis_utils_v2.visualize_sequence_dataloader(data_loader,
                                            params['postprocess']['order'],
                                               save_path='./v2v4_vis.png',
                                               color_mode=opt.color_mode,)