# MIT License

# Copyright (c) 2020-2021 Yantao Lu and Xuetao Hao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" Script for attentional 2D convolution layer.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Union, Tuple
import numpy as np
from typing import OrderedDict

_ENCODED_FEATURES_NUM = 3  # x, y, r
_CONV_ATT_COMBINE = True
is_norm = True
scale_val = 1.0


# conv2d + bn + relu
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, activation=True,
                 batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


class Conv2dFactory:
    @classmethod
    def conv(cls, in_channels, out_channels, k, s, p, activation=True,
             batch_norm=True, bias=False, flag=True):
        if flag:
            return cls.AttentionConv2d(in_channels, out_channels, k, s, p, bias=bias)
        else:
            return cls.Conv2d(in_channels, out_channels, k, s, p, activation, batch_norm)

    @classmethod
    def ConvTranspose(cls, in_channels, out_channels, k, s, p, activation=True,
                        batch_norm=True, bias=False, flag=True):
        if flag:
            return cls.AttentionConvTranspose2d(in_channels, out_channels, k, s, p, bias=bias)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, k, s, p, activation, batch_norm)

    @classmethod
    def Conv2d(cls, in_channels, out_channels, k, s, p, activation=True,
               batch_norm=True):
        return Conv2d(in_channels, out_channels, k, s, p, activation, batch_norm)

    @classmethod
    def AttentionConv2d(cls, in_channels, out_channels, k, s, p, bias=False, **_):
        return AttentionConv2D(in_channels, out_channels, k, s, p, bias=bias)

    @classmethod
    def AttentionConvTranspose2d(cls, in_channels, out_channels, k, s, p, bias=False, **_):
        return AttentionConvTranspose2D(in_channels, out_channels, k, s, p, bias=bias)


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class Sequential(torch.nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, given is a small example::

        # Example of using Sequential
        model = Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))

        # Example of using Sequential with kwargs(python 3.6+)
        model = Sequential(
                  conv1=nn.Conv2d(1,20,5),
                  relu1=nn.ReLU(),
                  conv2=nn.Conv2d(20,64,5),
                  relu2=nn.ReLU()
                )
    """

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        # i = 0
        for module in self._modules.values():
            # print(i)
            input = module(input)
            # i += 1
        return input


norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2d),
    "BN1d": ("bn1d", nn.BatchNorm1d),
    "GN": ("gn", nn.GroupNorm),
}


def build_norm_layer(cfg, num_features, postfix=""):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        # if layer_type == 'SyncBN':
        #     layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def kaiming_init(
        module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _linear_map(x, range=[-1.0, 1.0]):
    span = range[1] - range[0]
    k = span / (torch.max(x) - torch.min(x))
    b = range[1] - torch.max(x) * k
    return (k * x + b)


def _generate_position_info(x):
    """

    :param x:
    :return:
    """
    h, w = x.shape[2], x.shape[3]

    lin_h = torch.linspace(0, h - 1, steps=h).cuda()
    lin_w = torch.linspace(0, w - 1, steps=w).cuda()
    grid_h, grid_w = torch.meshgrid(lin_h, lin_w)

    grid_h = torch.abs(grid_h - h // 2 + 0.5) if h % 2 == 0 \
        else torch.abs(grid_h - h // 2)

    grid_w = torch.abs(grid_w - w // 2 + 0.5) if w % 2 == 0 \
        else torch.abs(grid_w - w // 2)
    grid_h = grid_h / float(h // 2)
    grid_w = grid_w / float(w // 2)
    polar_r = torch.sqrt(grid_h ** 2 + grid_w ** 2)
    if is_norm:
        # grid_h = _linear_map(grid_h)
        # grid_w = _linear_map(grid_w)
        polar_r = _linear_map(polar_r)

    grid_h_batch = grid_h.unsqueeze(0).unsqueeze(0).repeat(
        x.shape[0], 1, 1, 1)
    grid_w_batch = grid_w.unsqueeze(0).unsqueeze(0).repeat(
        x.shape[0], 1, 1, 1)
    polar_r_batch = polar_r.unsqueeze(0).unsqueeze(0).repeat(
        x.shape[0], 1, 1, 1)
    return grid_h_batch, grid_w_batch, polar_r_batch


class AttentionConv2D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 input_size: Union[int, Tuple[int, int]] = None,
                 concat_mesh: bool = True,
                 occlusion_flag: bool = False,
                 return_attention: bool = False,
                 is_header: bool = False,
                 last_attention: bool = False) -> None:
        super(AttentionConv2D, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups
        self._bias = bias
        self._padding_mode = padding_mode
        self._input_size = input_size
        self._concat_mesh = concat_mesh
        self._occlusion_flag = occlusion_flag
        self._return_attention = return_attention

        self._att_map_a = None
        self._att_map_b = None
        self._last_attention = last_attention
        assert (self._out_channels % 2 == 0)
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        # hyper-parameters
        branch_out_channels = self._out_channels // 2
        # only x, y
        conv_att_in_channels = branch_out_channels + 2 if _CONV_ATT_COMBINE \
            else branch_out_channels
        # 3 encoder features (mean, max, 1x1 conv) from input with range
        conv_attention_in_channels = 4

        # conv a and b
        self._conv_a = Sequential(
            nn.Conv2d(
                self._in_channels, branch_out_channels, self._kernel_size,
                stride=self._stride, padding=self._padding, dilation=self._dilation,
                groups=self._groups, bias=self._bias, padding_mode=self._padding_mode))
        if is_header:
            self._conv_a.add(nn.BatchNorm2d(branch_out_channels))
        else:
            self._conv_a.add(build_norm_layer(norm_cfg, branch_out_channels)[1])

        self._conv_b = Sequential(
            nn.Conv2d(
                self._in_channels, branch_out_channels, self._kernel_size,
                stride=self._stride, padding=self._padding, dilation=self._dilation,
                groups=self._groups, bias=self._bias, padding_mode=self._padding_mode))
        if is_header:
            self._conv_b.add(nn.BatchNorm2d(branch_out_channels))
        else:
            self._conv_b.add(build_norm_layer(norm_cfg, branch_out_channels)[1])
        # attention a and b
        self._conv_att_a = Sequential(
            nn.Conv2d(conv_att_in_channels, 1, 1, bias=self._bias),
            nn.BatchNorm2d(1))
        self._conv_att_b = Sequential(
            nn.Conv2d(conv_att_in_channels, 1, 1, bias=self._bias),
            nn.BatchNorm2d(1))

        self._conv_attention_a = nn.Conv2d(
            conv_attention_in_channels, 1, 3,
            stride=1, padding=1, bias=False,
            dilation=self._dilation)

        self._conv_attention_b = nn.Conv2d(
            conv_attention_in_channels, 1, 3,
            stride=1, padding=1, bias=False,
            dilation=self._dilation)
        # init attention weights.
        self.init_attention_weight()
        self._sigmoid = nn.Sigmoid()

        self.scale_1 = Scale(scale=scale_val)
        self.scale_2 = Scale(scale=scale_val)

    @property
    def bias(self):
        return (self._conv_a[0].bias.data, self._conv_b[0].bias.data)

    def init_attention_weight(self):
        # [c_out, c_in, kernel_size[0],kernel_size[1]]
        # TODO(Xuetao): hard code 3 for conv 1*1 + mean + max of input.
        with torch.no_grad():
            self._conv_attention_a.weight.data[:, 3:, :, :] = torch.abs(
                self._conv_attention_a.weight.data[:, 3:, :, :])
            self._conv_attention_b.weight.data[:, 3:, :, :] = torch.abs(
                self._conv_attention_b.weight.data[:, 3:, :, :])

    def init_attention_weight_guassian(self):
        # [c_out, c_in, kernel_size[0],kernel_size[1]]
        # TODO(Xuetao): hard code 3 for conv 1*1 + mean + max of input.
        with torch.no_grad():
            shape = self._conv_attention_a.weight.shape[-2:]
            c_out = self._conv_attention_a.weight.shape[0]
            m, n = [(ss - 1) // 2 for ss in shape]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            guassian_k = np.exp(-(x * x + y * y) / (2 * m * n)) * (1 / (2 * shape[0]))
            guassian_k = torch.from_numpy(guassian_k).view(
                1, 1, shape[0], shape[1]).repeat(c_out, 1, 1, 1)

            self._conv_attention_a.weight.data[:, 3:, :, :] = guassian_k
            self._conv_attention_b.weight.data[:, 3:, :, :] = guassian_k

    @property
    def att_map(self):
        return self._att_map_a, self._att_map_b

    def fill_bias(self, bias):
        if isinstance(bias, float):
            biases = (bias, bias)
        elif isinstance(bias, tuple) or isinstance(bias, list):
            biases = bias
        else:
            raise ValueError("Invalid bias type.")
        if self._bias:
            self._conv_a[0].bias.data.fill_(biases[0])
            self._conv_b[0].bias.data.fill_(biases[1])

    def forward(self, x):
        """
          Forard function for Conv2D.
          Parameters:
          ----------
          occlusion_map: bool
            Introduce occlusion_map as an attention attribute.
        """
        x_a = self._conv_a(x)
        x_b = self._conv_b(x)

        grid_h_batch, grid_w_batch, polar_r_batch = _generate_position_info(x_a)
        attr_a = self._conv_att_a(torch.cat(
            [x_a, grid_h_batch, grid_w_batch], dim=1))
        attr_b = self._conv_att_b(torch.cat(
            [x_b, 1.0 - grid_h_batch, 1.0 - grid_w_batch], dim=1))

        attention_maps_a = self._conv_attention_a(torch.cat(
            [attr_a, torch.max(x_a, 1)[0].unsqueeze(1), torch.mean(x_a, 1).unsqueeze(1),
             polar_r_batch], dim=1))
        attention_maps_b = self._conv_attention_b(torch.cat(
            [attr_b, torch.max(x_b, 1)[0].unsqueeze(1), torch.mean(x_b, 1).unsqueeze(1),
             -1.0 * polar_r_batch], dim=1))
        # self._att_map_a = self._sigmoid(attention_maps_a)
        # self._att_map_b = self._sigmoid(attention_maps_b)
        if self._last_attention:
            x_a = self._sigmoid(attention_maps_a) * x_a
            x_b = self._sigmoid(attention_maps_b) * x_b
        else:
            # avoid grad vanish.
            x_a = (1 + self.scale_1(self._sigmoid(attention_maps_a))) * x_a
            x_b = (1 + self.scale_2(self._sigmoid(attention_maps_b))) * x_b
        out = torch.cat((x_a, x_b), dim=1)
        return out

    def get_output_size(self):
        """ Get size of output tensor.
        """
        if type(self._input_size) is not tuple:
            input_size = (self._input_size, self._input_size)
        else:
            input_size = self._input_size
        return self._get_output_size(input_size)

    def _get_output_size(self, input_size):
        if type(self._kernel_size) is not tuple:
            kernel_size = (self._kernel_size, self._kernel_size)
        else:
            kernel_size = self._kernel_size

        if type(self._stride) is not tuple:
            stride = (self._stride, self._stride)
        else:
            stride = self._stride

        if type(self._padding) is not tuple:
            pad = (self._padding, self._padding)
        else:
            pad = self._pad

        if type(self._dilation) is not tuple:
            dilation = (self._dilation, self._dilation)
        else:
            dilation = self._dilation

        out_size_h = (input_size[0] + (2 * pad[0]) - \
                      (dilation[0] * (kernel_size[0] - 1)) - 1) // stride[0] + 1
        out_size_w = (input_size[1] + (2 * pad[1]) - \
                      (dilation[1] * (kernel_size[1] - 1)) - 1) // stride[1] + 1
        if type(input_size) is int:
            return out_size_h
        else:
            return (out_size_h, out_size_w)


class AttentionConvTranspose2D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 output_padding: Union[int, Tuple[int, int]] = 0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 padding_mode: str = 'zeros',
                 concat_mesh: bool = True,
                 occlusion_flag: bool = False,
                 return_attention: bool = False,
                 is_header: bool = False,
                 last_attention: bool = False) -> None:
        super(AttentionConvTranspose2D, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._output_padding = output_padding
        self._groups = groups
        self._bias = bias
        self._dilation = dilation
        self._padding_mode = padding_mode
        self._concat_mesh = concat_mesh
        self._occlusion_flag = occlusion_flag
        self._return_attention = return_attention
        self._last_attention = last_attention
        self._att_map_a = None
        self._att_map_b = None
        assert (self._out_channels % 2 == 0)
        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

        # hyper-parameters
        branch_out_channels = self._out_channels // 2
        # only x, y
        conv_att_in_channels = branch_out_channels + 2 if _CONV_ATT_COMBINE \
            else branch_out_channels
        # 3 encoder features (mean, max, 1x1 conv) from input with range
        conv_attention_in_channels = 4

        self._transconv_a = Sequential(
            nn.ConvTranspose2d(
                self._in_channels, branch_out_channels, self._kernel_size,
                stride=self._stride, padding=self._padding, output_padding=self._output_padding,
                groups=self._groups, bias=self._bias, dilation=self._dilation,
                padding_mode=self._padding_mode))
        if is_header:
            self._transconv_a.add(nn.BatchNorm2d(branch_out_channels))
        else:
            self._transconv_a.add(build_norm_layer(norm_cfg, branch_out_channels)[1])

        self._transconv_b = Sequential(
            nn.ConvTranspose2d(
                self._in_channels, branch_out_channels, self._kernel_size,
                stride=self._stride, padding=self._padding, output_padding=self._output_padding,
                groups=self._groups, bias=self._bias, dilation=self._dilation,
                padding_mode=self._padding_mode))
        if is_header:
            self._transconv_b.add(nn.BatchNorm2d(branch_out_channels))
        else:
            self._transconv_b.add(build_norm_layer(norm_cfg, branch_out_channels)[1])

        self._conv_att_a = Sequential(
            nn.Conv2d(conv_att_in_channels, 1, 1, bias=self._bias),
            nn.BatchNorm2d(1))
        self._conv_att_b = Sequential(
            nn.Conv2d(conv_att_in_channels, 1, 1, bias=self._bias),
            nn.BatchNorm2d(1))

        self._conv_attention_a = nn.Conv2d(
            conv_attention_in_channels, 1, 3,
            stride=1, padding=1, bias=False,
            dilation=self._dilation)

        self._conv_attention_b = nn.Conv2d(
            conv_attention_in_channels, 1, 3,
            stride=1, padding=1, bias=False,
            dilation=self._dilation)
        # init attention weights.
        self.init_attention_weight()
        self._sigmoid = nn.Sigmoid()

        self.scale_1 = Scale(scale=scale_val)
        self.scale_2 = Scale(scale=scale_val)

    @property
    def bias(self):
        return (self._transconv_a[0].bias.data, self._transconv_b[0].bias.data)

    def init_attention_weight(self):
        # [c_out, c_in, kernel_size[0],kernel_size[1]]
        # TODO(Xuetao): hard code 3 for conv 1*1 + mean + max of input.
        with torch.no_grad():
            self._conv_attention_a.weight.data[:, 3:, :, :] = torch.abs(
                self._conv_attention_a.weight.data[:, 3:, :, :])
            self._conv_attention_b.weight.data[:, 3:, :, :] = torch.abs(
                self._conv_attention_b.weight.data[:, 3:, :, :])

    def init_attention_weight_guassian(self):
        # [c_out, c_in, kernel_size[0],kernel_size[1]]
        # TODO(Xuetao): hard code 3 for conv 1*1 + mean + max of input.
        with torch.no_grad():
            shape = self._conv_attention_a.weight.shape[-2:]
            c_out = self._conv_attention_a.weight.shape[0]
            m, n = [(ss - 1) // 2 for ss in shape]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            guassian_k = np.exp(-(x * x + y * y) / (2 * m * n)) * (1 / (2 * shape[0]))
            guassian_k = torch.from_numpy(guassian_k).view(
                1, 1, shape[0], shape[1]).repeat(c_out, 1, 1, 1)

            self._conv_attention_a.weight.data[:, 3:, :, :] = guassian_k
            self._conv_attention_b.weight.data[:, 3:, :, :] = guassian_k

    @property
    def att_map(self):
        return self._att_map_a, self._att_map_b

    def fill_bias(self, bias):
        if isinstance(bias, float):
            biases = (bias, bias)
        elif isinstance(bias, tuple) or isinstance(bias, list):
            biases = bias
        else:
            raise ValueError("Invalid bias type.")
        if self._bias:
            self._transconv_a[0].bias.data.fill_(biases[0])
            self._transconv_b[0].bias.data.fill_(biases[1])

    def forward(self, x):
        """
          Forard function for AttentionConvTranspose2D.
          Parameters:
          ----------
          occlusion_map: bool
            Introduce occlusion_map as an attention attribute.
        """
        x_a = self._transconv_a(x)
        x_b = self._transconv_b(x)

        grid_h_batch, grid_w_batch, polar_r_batch = _generate_position_info(x_a)

        attr_a = self._conv_att_a(torch.cat(
            [x_a, grid_h_batch, grid_w_batch], dim=1))
        attr_b = self._conv_att_b(torch.cat(
            [x_b, 1.0 - grid_h_batch, 1.0 - grid_w_batch], dim=1))

        attention_maps_a = self._conv_attention_a(torch.cat(
            [attr_a, torch.max(x_a, 1)[0].unsqueeze(1), torch.mean(x_a, 1).unsqueeze(1),
             polar_r_batch], dim=1))
        attention_maps_b = self._conv_attention_b(torch.cat(
            [attr_b, torch.max(x_b, 1)[0].unsqueeze(1), torch.mean(x_b, 1).unsqueeze(1),
             -1.0 * polar_r_batch], dim=1))
        # self._att_map_a = self._sigmoid(attention_maps_a)
        # self._att_map_b = self._sigmoid(attention_maps_b)
        if self._last_attention:
            x_a = self._sigmoid(attention_maps_a) * x_a
            x_b = self._sigmoid(attention_maps_b) * x_b
        else:
            # avoid grad vanish.
            x_a = (1 + self.scale_1(self._sigmoid(attention_maps_a))) * x_a
            x_b = (1 + self.scale_2(self._sigmoid(attention_maps_b))) * x_b
        out = torch.cat((x_a, x_b), dim=1)
        return out
