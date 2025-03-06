import os
from pathlib import Path

import cv2
import torch
from PIL import Image
import yaml


def crop_img(file_path,dst_dir=None,w=None, h=None, vis=False):
    img = Image.open(file_path)
    img_size = img.size
    width, height = img_size
    if w is None:
        w = width // 2
        h = height // 2.6
    left = (width - w) / 2 + 120
    top = (height - h) / 2 - 100
    right = (width + w) / 2 + 400
    bottom = (height + h) / 2 + 120
    img = img.crop((left, top, right, bottom))
    if vis:
        img.show()
    # if "cobevt" in file_path.lower() and w is not None and h is not None:
    #     reshape_image(file_path,w=w,h=h)
    img.save(os.path.join(dst_dir, Path(file_path).stem + "_crop"+Path(file_path).suffix))

def traverse_directory(src_dir=None,dst_dir=None,**kwargs):
    assert src_dir is not None,"src_dir must be specified"
    dst_dir = src_dir if dst_dir is None else dst_dir
    for dirpath,dirnames,filenames in os.walk(src_dir):
        # for dirname in dirnames:
        #     path = os.path.join(dirpath,dirname)
        #     if os.path.isdir(path):
        #         traverse_directory(src_dir=path,dst_dir=dst_dir,**kwargs)

        for file in filenames:
            file_path = os.path.join(dirpath,file)
            if (os.path.isfile(file_path) and (file.endswith(".jpg") or file.endswith(".png")) \
                                                and not file.endswith("_crop.jpg")):
                # print(file_path)
                # make target dir
                # print(file)
                # print(dirpath,src_dir)
                rel_dir = os.path.relpath(dirpath,src_dir)
                # print(rel_dir)
                save_dir = os.path.join(dst_dir,rel_dir)
                os.makedirs(save_dir,exist_ok=True)
                crop_img(file_path,dst_dir=save_dir,**kwargs)

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

def reshape_image(tgt_file,w=None,h=None):
    """
    clean all files end with _crop.jpg
    :param dir_name:
    :return:
    """
    assert w is not None and h is not None,"w and h must be specified"
    tgt_image = cv2.imread(tgt_file)
    save_image = cv2.resize(tgt_image,(w,h))
    cv2.imwrite(tgt_file,save_image)

if __name__ == '__main__':
    src_dir = "./raw_images"
    dst_dir = "./crop_results"
    traverse_directory(src_dir=src_dir,dst_dir=dst_dir)