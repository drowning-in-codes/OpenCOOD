import os
from pathlib import Path
import torch
from PIL import Image
import yaml


def crop_img(dir_name,file, w=None, h=None, vis=False):
    img = Image.open(os.path.join(dir_name, file))
    img_size = img.size
    width, height = img_size
    if w is None:
        w = width // 2.5
        h = height // 1.5
    left = (width - w) / 2
    top = (height - h) / 2
    right = (width + w) / 2
    bottom = (height + h) / 2
    img = img.crop((left, top, right, bottom))
    if vis:
        img.show()
    img.save(os.path.join(dir_name, file[:-4] + "_crop.jpg"))
def traverse_directory(w=None,h=None,vis=False,dir_name=None,file_name=None):
    assert dir_name is not None or file_name is not None
    if dir_name is not None:
        for file in os.listdir(dir_name):
            path = os.path.join(dir_name, file)
            if os.path.isfile(path) and file.endswith(".jpg") and not file.endswith("_crop.jpg"):
                crop_img(dir_name,file,w,h,vis)
            elif os.path.isdir(path):
                traverse_directory(w,h,vis,dir_name=path)  # 递归调用自身
    if file_name is not None:
           crop_img(os.getcwd(),file_name,w,h,vis)


def clean_file(dir_name):
    """
    clean all files end with _crop.jpg
    :param dir_name:
    :return:
    """
    for file in os.listdir(dir_name):
        path = os.path.join(dir_name, file)
        if os.path.isfile(path) and file.endswith("_crop.jpg"):
            os.remove(path)
        elif os.path.isdir(path):
            clean_file(path)
import torch.nn.functional as F


def manual_iter():
    with open("requirements.txt") as f:
        try:
            while True:
                line = next(f)
                print(line,end='')
        except StopIteration:
            pass

def frange(start,stop,increment):
    x = start
    while x < stop:
        yield x
        x += increment

from collections import deque
class linehistory:
    def __init__(self,lines,histlen=3):
        self.lines = lines
        self.history = deque(maxlen=histlen)

class Node_2:
    def __init__(self,value):
        self._value = value
        self._children = []

    def __repr__(self):
        return 'Node({!r})'.format(self._value)

    def add_child(self,node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)

    def depth_first(self):
        yield self
        for c in self:
            yield from c.depth_first()
class Node2:
    def __init__(self,value):
        self._value = value
        self._children = []

    def __repr__(self):
        return f"Node({self._value})"

    def add_child(self,node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)

    def depth_first(self):
        return DepthFirstIterator(self)
class DepthFirstIterator:
    def __init__(self,start_node):
        self._node = start_node
        self._children_iter = None
        self._child_iter = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._children_iter is None:
            self._children_iter = iter(self._node)
            return self._node
        elif self._child_iter:
            try:
                nextchild = next(self._child_iter)
                return nextchild
            except StopIteration:
                self._child_iter = None
                return next(self)
        else:
            self._child_iter = next(self._children_iter).depth_first()
            return next(self)

class myTuple:
    def __init__(self):
        self.values = [1,2,3,]
    def __next__(self):
        if self.values:
            return self.values.pop()
        else:
            raise StopIteration
class Node:
    def __init__(self,value):
        self._value = value
        self._children = []

    def __repr__(self):
        return f"Node({self._value})"

    def add_child(self,node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)

class Countdown:
    def __init__(self,start):
        self.start = start

    def __next__(self):
        return 1
    def __iter__(self):
        n = self.start
        while n > 0:
            yield n
            n -= 1
    def __reversed__(self):
        n = 1
        while n <= self.start:
            yield n
            n += 1
import torch
import torch.nn as nn
import torch.nn.functional as F


def w8_a16_forward(weight, input, scales, bias=None):
    casted_weights = weight.to(input.dtype)
    output = F.linear(input, casted_weights) * scales

    if bias is not None:
        output = output + bias

    return output
class W8A16LinearLayer(nn.Module):

    def __init__(self,in_features,out_features,bias=True,dtype=torch.float32):
        super().__init__()

        self.register_buffer(
            "int8_weights",
            torch.randint(
                -128, 127, (out_features, in_features), dtype=torch.int8
            )
        )
        self.register_buffer("scales",
                             torch.randn((out_features), dtype=dtype))

        if bias:
            self.register_buffer("bias",
                                 torch.randn((1, out_features),
                                             dtype=dtype))

        else:
            self.bias = None

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights
                                   / scales.unsqueeze(1)).to(torch.int8)
        self.int8_weights = int8_weights
        self.scales = scales

    def forward(self, input):
        return w8_a16_forward(self.int8_weights,
                              input, self.scales, self.bias)

def replace_linear_with_target(module,
                               target_class, module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
                any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias

            new_module = target_class(child.in_features,
                                      child.out_features,
                                      old_bias is not None,
                                      child.weight.dtype)
            setattr(module, name, new_module)
            if old_bias is not None:
                getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target(
                child, target_class, module_name_to_exclude)

class DummyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = torch.nn.Embedding(1, 1)
    # Try with bias
    self.linear_1 = nn.Linear(1, 1)
    # Try without bias
    self.linear_2 = nn.Linear(1, 1, bias=False)
    # Lm prediction head
    self.lm_head = nn.Linear(1, 1, bias=False)

if __name__ == '__main__':
    pass