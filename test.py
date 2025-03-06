import open3d as o3d
import numpy as np
from PIL import Image
import cv2
import torch

mq = torch.randn(1, 2, 4, 100, 224, )
k = v = torch.randn(1, 4, 100, 224, )
attn = torch.einsum('b g h n d, b h s d -> b g h n s', mq, k)
print(attn.shape)
# crop the image
# image = Image.open('./vis_result.png')
# image = np.array(image)
# H,W,_ = image.shape
# print(H,W)
# image = image[H//4+160:3*H//4-165,W//4+100:3*W//4-400]
# print(image.shape)
# H,W,_ = image.shape
# # define bbox
# bbox = [H//4+30,W//4+20,H//4+105,W//4+55]
# # draw bbox
# # cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
# rect = (((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2), (bbox[2]-bbox[0], bbox[3]-bbox[1]), 20)
# bbox = cv2.boxPoints(rect)
# bbox = np.intp(bbox)
# cv2.drawContours(image,[bbox],0,(255,0,0),2)
# bbox= bbox.astype(np.float32)
# (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox.reshape(4, 2)
# widthA = np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
# widthB = np.sqrt(((x4 - x3) ** 2) + ((y4 - y3) ** 2))
# maxWidth = max(int(widthA), int(widthB))
#
# heightA = np.sqrt(((x1 - x4) ** 2) + ((y1 - y4) ** 2))
# heightB = np.sqrt(((x2 - x3) ** 2) + ((y2 - y3) ** 2))
# maxHeight = max(int(heightA), int(heightB))
#
# # 构建新图像的四个顶点坐标
# dst = np.array([
#     [0, 0],
#     [maxWidth - 1, 0],
#     [maxWidth - 1, maxHeight - 1],
#     [0, maxHeight - 1]], dtype="float32")
# # 计算透视变换矩阵
# M = cv2.getPerspectiveTransform(bbox, dst)
#
# # 对原图像应用透视变换
# warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
#
# # 现在 warped 图像包含了旋转矩形区域，可以使用切片操作进一步处理或保存
# cropped_image = warped.copy()
# cropped_image = Image.fromarray(cropped_image)
# cropped_image.save('./vis_result_crop_1.png')
#
# bbox = [H//4+200,W//4-30,H//4+275,W//4+5]
# rect = (((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2), (bbox[2]-bbox[0], bbox[3]-bbox[1]), -14)
# bbox = cv2.boxPoints(rect)
# bbox = np.intp(bbox)
# cv2.drawContours(image,[bbox],0,(255,0,0),2)
#
# bbox = bbox.astype(np.float32)
# #save the bbox content
# # 计算四个顶点的最小和最大 x 和 y 值，用于确定直角矩形的尺寸
# (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox.reshape(4, 2)
# widthA = np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
# widthB = np.sqrt(((x4 - x3) ** 2) + ((y4 - y3) ** 2))
# maxWidth = max(int(widthA), int(widthB))
#
# heightA = np.sqrt(((x1 - x4) ** 2) + ((y1 - y4) ** 2))
# heightB = np.sqrt(((x2 - x3) ** 2) + ((y2 - y3) ** 2))
# maxHeight = max(int(heightA), int(heightB))
#
# # 构建新图像的四个顶点坐标
# dst = np.array([
#     [0, 0],
#     [maxWidth - 1, 0],
#     [maxWidth - 1, maxHeight - 1],
#     [0, maxHeight - 1]], dtype="float32")
# # 计算透视变换矩阵
# M = cv2.getPerspectiveTransform(bbox, dst)
#
# # 对原图像应用透视变换
# warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
#
# # 现在 warped 图像包含了旋转矩形区域，可以使用切片操作进一步处理或保存
# cropped_image = warped.copy()
# cropped_image = Image.fromarray(cropped_image)
# cropped_image.save('./vis_result_crop_2.png')
#
# image = Image.fromarray(image)
# image.save('./vis_result_crop.png')
