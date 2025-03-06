import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import scienceplots
from matplotlib import rcParams

plt.style.use(['science'])
plt.rcParams.update({
    "font.family": "serif",  # specify font family here
    "font.serif": ["Times"],  # specify font here

    "font.size": 11})
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# 创建图形和子图
xlabel_font = {
    # 'fontsize': rcParams['axes.titlesize'], # 设置成和轴刻度标签一样的大小
    'fontsize': 12,
    # 'fontweight': rcParams['axes.titleweight'], # 设置成和轴刻度标签一样的粗细
    'fontweight': 'light',
    # 'color': rcParams['axes.titlecolor'], # 设置成和轴刻度标签一样的颜色
    'color': 'gray',
    'family': 'serif',
}

ax_1 = axes[0]
for ax in ax_1:
    ax.tick_params(labelsize=8,axis='both', which='both', bottom=True, top=False, left=True, right=False)
    ax.set_xlabel('Communication Volume(log2)', fontdict=xlabel_font,labelpad=1)
    # ax.set_ylabel('AP@0.7', fontsize=14, fontweight='light')
ax_1[0].set_ylabel('AP@0.7 on Default Towns', fontdict=xlabel_font)
ax_1[1].set_ylabel('AP@0.7 on Culver City',fontdict=xlabel_font)
ax_1[2].set_ylabel('AP@0.7 on V2XSet',fontdict=xlabel_font)
# 第一个子图
# axes[0].set_title('Default OPV2V Towns', fontdict=title_font)
# axes[0].set_xlim(10, 2-
# 数据点和线段样式
line_styles = [
    {'color': '#1f77b4', 'linestyle': '--',  'markersize': 5, 'linewidth': 1.5,"marker":'.'},  # 蓝色实线，圆圈标记
    {'color': '#ff7f0e', 'linestyle': '-.',  'markersize': 5, 'linewidth': 1.5,"marker":'2'},  # 橙色虚线，正方形标记
    {'color': '#2ca02c', 'linestyle': '-',  'markersize': 5, 'linewidth': 1.5,"marker":'1'},  # 绿色点划线，三角形标记
]
marker_shape = ['o','D','p','*','x','v','+']
marker_color = ['#bcbd22','#d62728','#9467bd','#8c564b','#e377c2','#bcbd22','#7f7f7f']
labels = ["DiscoNet","V2VNet","AdaFusion","Attentive Fusion","V2X-ViT","CoBEVT","Late Fusion"]

v2xvit_x = 23.6882503
v2xvit_y = 82.81

late_fusion_x = 17.91
late_fusion_y = 73.83
# 100*352*7

v2vnet_x = 23.6882503
v2vnet_y = 80.89

attentive_fusion_x = 23.6882503
attentive_fusion_y = 81.62

cobevt_x = 23.6882503
cobevt_y = 84.98

adafusion_x = 23.6882503
adafusion_y = 85.47

disconet_x = 21.6882503
disconet_y = 82.27
for i in range(1,len(labels)+1):
    ax_1[0].scatter(eval(f"{labels[i-1].lower().replace('-','').replace(' ','_')}_x"), eval(f"{labels[i-1].lower().replace('-','').replace(' ','_')}_y"), c=marker_color[i-1],marker=marker_shape[i-1],s=17)

# 数据点和线段数据 18.2016
# where2_comm_x = list(map(lambda x:x-5,[11.0588,15.1462,17.7313191,18.82702,19.59804,21.35,22.6901,23.19,24.6177543,24.87305955,25.10328781]))
# where2_comm_y = [65.3,67.231,70.689,74.692,76.789,79.672,80.489,80.657,80.665,80.672,80.68]
# assert len(where2_comm_x) == len(where2_comm_y), "wrong"
# axes[0].plot(where2_comm_x, where2_comm_y, label="Where2comm", **line_styles[0])

where2_comm_x = list(map(lambda x:x ,
                         [3.72,7.12,10.9657,13.228,14.391 , 15.50313191,   20.35,23.688]))
where2_comm_y = [48.2,53.312,65.448,72.982, 76.6502,   80.212, 82.545 ,82.62]
assert len(where2_comm_x) == len(where2_comm_y), "wrong"
ax_1[0].plot(where2_comm_x, where2_comm_y, **line_styles[0])

# corange_x = list(map(lambda x:x-5,[12.1,14.70,15.3218,16.39227,17.27612,18.1472,19.55362,20.36303682 ,21.44837 ,23.63535536 ,23.9755,24.32680,24.99159]))
# corange_y = [70.0121,70.486,70.52,70.675,70.885,71.286,72.44,74.168,80.8086,87.5126,87.566,87.4426,87.423]

codefilling_x = list(map(lambda x: x,
                         [3.6543,7.31,9.68,14.82702, 16.59804, 23.688,]))
codefilling_y = [  51.0312,56.31,65.452,   80.17,  82.745,  83.36]
assert len(codefilling_x) == len(codefilling_y), "wrong"
ax_1[0].plot(codefilling_x, codefilling_y,  **line_styles[1])


QTF_x = list(map(lambda x: x,[2.681,7.523,12.752,15.0732,16.13 ,23.688]))
QTF_y = [54.672,62.312,81.572,84.23,85.821,86.72]
assert len(QTF_x) == len(QTF_y), "wrong"
ax_1[0].plot(QTF_x, QTF_y,   **line_styles[2])

# ax_1[0].legend()
ax_1[0].grid(True)


# 第二个子图 - Culver
# axes[1].set_title('Culver City', fontdict=title_font)
# 添加子图内容...

v2xvit_x = 23.6882503
v2xvit_y = 73.26

late_fusion_x = 17.91064
late_fusion_y = 64.92

v2vnet_x = 23.6882503
v2vnet_y = 75.42

attentive_fusion_x = 23.6882503
attentive_fusion_y = 73.58

cobevt_x = 23.6882503
cobevt_y = 74.87

adafusion_x = 23.6882503
adafusion_y = 78.42

disconet_x = 21.6882503
disconet_y = 72.94
for i in range(1, len(labels) + 1):
    ax_1[1].scatter(eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_x"),
                    eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_y"), c=marker_color[i - 1],
                    marker=marker_shape[i - 1],s=17)



# 数据点和线段数据
# where2_comm_x = list(map(lambda x:x-0.58,[13.1,14.23,16.14469903,17.380,17.89177107,18.5824 ,19.15692078,20.10872517,20.56668829,20.81040968,21.69006504,22.89888455,23.19460283,24.27198118,25.10328781]))
# where2_comm_y = [56.12,56.4115,57.371	,57.791,59.8302,61.623,64.155,68.99,70.963,71.548,72.209,72.2665,	72.271,72.29,72.32]
# assert len(where2_comm_x) == len(where2_comm_y), "wrong"
# axes[1].plot(where2_comm_x, where2_comm_y,  label="Where2comm", **line_styles[0])
where2_comm_x = list(map(lambda x: x ,
                         [3.512,11.813,12.687,19.3123,23.6882503]))
where2_comm_y = [48.797,51.67,55.57,71.2831,72.34]
assert len(where2_comm_x) == len(where2_comm_y), "wrong"
ax_1[1].plot(where2_comm_x, where2_comm_y,  **line_styles[0])

codefilling_x = list(map(lambda x: x ,
                         [3.5143, 9.0588,13.412  ,14.82702,23.6882503]))
codefilling_y = [52.982, 58.13, 73.41 ,74.689, 75.84 ]
assert len(codefilling_x) == len(codefilling_y), "wrong"
ax_1[1].plot(codefilling_x, codefilling_y,  **line_styles[1])

QTF_x = list(map(lambda x:x,[3.632,9.53,12.47,19.0620,23.6882503]))
QTF_y = [55.7679,63.12,76.45,78.67,79.36]
assert len(QTF_x) == len(QTF_y), "wrong"
ax_1[1].plot(QTF_x, QTF_y,    **line_styles[2])


# ax_1[1].legend()
ax_1[1].grid(True)
# 第三个子图 - V2XSet
# axes[2].set_title('V2XSet', fontdict=title_font)
# 添加子图内容...
v2xvit_x = 23.6882503
v2xvit_y = 78.34

late_fusion_x = 17.9106
late_fusion_y = 62.85

v2vnet_x = 23.6882503
v2vnet_y = 80.17

attentive_fusion_x = 23.6882503
attentive_fusion_y = 76.37

cobevt_x = 23.6882503
cobevt_y = 77.32

adafusion_x = 23.6882503
adafusion_y = 76.62

disconet_x = 23.6882503
disconet_y = 74.57
for i in range(1, len(labels) + 1):
    ax_1[2].scatter(eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_x"),
                    eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_y"), c=marker_color[i - 1],
                    marker=marker_shape[i - 1], s=17)

# 数据点和线段数据
where2_comm_x = list(map(lambda x:x,[3.7783,9.1231,12.91,16.313,18.712,23.6882503]))
where2_comm_y = [48.92,56.22,63.31,71.123,75.847,76.96]
assert len(where2_comm_x) == len(where2_comm_y), "wrong"
ax_1[2].plot(where2_comm_x, where2_comm_y,  **line_styles[0])

codefilling_x = list(map(lambda x:x ,
                         [3.68, 7.313,12.0588, 13.542, 17.1462,  18.82702, 23.6882503,
                          ]))
codefilling_y = [50.982,56.31,  66.452, 69.231, 76.689, 77.789,  78.33 ]
assert len(codefilling_x) == len(codefilling_y), "wrong"
ax_1[2].plot(codefilling_x, codefilling_y,  **line_styles[1])

QTF_x = list(map(lambda x:x,[3.2,8.14,11.13,14.331,17.938,23.6882503]))
QTF_y = [54.26,65.31,69.7059,76.31,80.319,81.25]
assert len(QTF_x) == len(QTF_y), "wrong"
ax_1[2].plot(QTF_x, QTF_y,  **line_styles[2])

# ax_1[2].legend()
ax_1[2].grid(True)



ax_2 = axes[1]
for ax in ax_2:
    ax.tick_params(labelsize=8,axis='both', which='both', bottom=True, top=False, left=True, right=False)
    ax.set_xlabel('Communication Volume(log2)', fontdict=xlabel_font, labelpad=1)

    # ax.set_ylabel('AP@0.7', fontsize=14, fontweight='light')
ax_2[0].set_ylabel('AP@0.5 on Default Towns', fontdict=xlabel_font)
ax_2[1].set_ylabel('AP@0.5 on Culver City', fontdict=xlabel_font)
ax_2[2].set_ylabel('AP@0.5 on V2XSet', fontdict=xlabel_font)
# 第一个子图
# axes[0].set_title('Default OPV2V Towns', fontdict=title_font)
# axes[0].set_xlim(10, 26)
# 数据点和线段样式
labels = ["DiscoNet", "V2VNet", "AdaFusion", "Attentive Fusion", "V2X-ViT", "CoBEVT", "Late Fusion"]

v2xvit_x = 23.6882503
v2xvit_y = 90.02

late_fusion_x = 17.91
late_fusion_y = 85.74
# 100*352*7

v2vnet_x = 23.6882503
v2vnet_y = 89.91

attentive_fusion_x = 23.6882503
attentive_fusion_y = 90.52

cobevt_x = 23.6882503
cobevt_y = 92.32

adafusion_x = 23.6882503
adafusion_y = 91.58

disconet_x = 21.6882503
disconet_y = 89.91
for i in range(1, len(labels) + 1):
    ax_2[0].scatter(eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_x"),
                    eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_y"), c=marker_color[i - 1],
                    marker=marker_shape[i - 1], s=17)

# 数据点和线段数据 18.2016
# where2_comm_x = list(map(lambda x:x-5,[11.0588,15.1462,17.7313191,18.82702,19.59804,21.35,22.6901,23.19,24.6177543,24.87305955,25.10328781]))
# where2_comm_y = [65.3,67.231,70.689,74.692,76.789,79.672,80.489,80.657,80.665,80.672,80.68]
# assert len(where2_comm_x) == len(where2_comm_y), "wrong"
# ax_2[0].plot(where2_comm_x, where2_comm_y, label="Where2comm", **line_styles[0])

where2_comm_x = list(map(lambda x: x,
                         [3.72, 6.12, 11.9657,  14.391,  19.35,23.6882503]))
where2_comm_y = [61.2, 68.312, 74.448, 81.6502,  87.87,88.74]
assert len(where2_comm_x) == len(where2_comm_y), "wrong"
ax_2[0].plot(where2_comm_x, where2_comm_y,  **line_styles[0])

# corange_x = list(map(lambda x:x-5,[12.1,14.70,15.3218,16.39227,17.27612,18.1472,19.55362,20.36303682 ,21.44837 ,23.63535536 ,23.9755,24.32680,24.99159]))
# corange_y = [70.0121,70.486,70.52,70.675,70.885,71.286,72.44,74.168,80.8086,87.5126,87.566,87.4426,87.423]

codefilling_x = list(map(lambda x: x,
                         [3.6543, 9.68, 11.82702, 16.59804, 23.688, ]))
codefilling_y = [63.0312, 76.452, 82.17, 88.745, 89.76]
assert len(codefilling_x) == len(codefilling_y), "wrong"
ax_2[0].plot(codefilling_x, codefilling_y, **line_styles[1])

QTF_x = list(map(lambda x: x, [3.681,  9.452,11.13, 15.1032, 23.688]))
QTF_y = [63.672,  84.672,86.311, 91.6121, 93.01]
assert len(QTF_x) == len(QTF_y), "wrong"
ax_2[0].plot(QTF_x, QTF_y,  **line_styles[2])

# ax_2[0].legend()
ax_2[0].grid(True)

# 第二个子图 - Culver
# axes[1].set_title('Culver City', fontdict=title_font)
# 添加子图内容...

v2xvit_x = 23.6882503
v2xvit_y = 87.24

late_fusion_x = 17.91064
late_fusion_y = 79.84

v2vnet_x = 23.6882503
v2vnet_y = 86.16

attentive_fusion_x = 23.6882503
attentive_fusion_y = 85.25

cobevt_x = 23.6882503
cobevt_y = 86.02

adafusion_x = 23.6882503
adafusion_y = 87.63

disconet_x = 21.6882503
disconet_y = 87.14
for i in range(1, len(labels) + 1):
    ax_2[1].scatter(eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_x"),
                    eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_y"), c=marker_color[i - 1],
                    marker=marker_shape[i - 1], s=17)

# 数据点和线段数据
# where2_comm_x = list(map(lambda x:x-0.58,[13.1,14.23,16.14469903,17.380,17.89177107,18.5824 ,19.15692078,20.10872517,20.56668829,20.81040968,21.69006504,22.89888455,23.19460283,24.27198118,25.10328781]))
# where2_comm_y = [56.12,56.4115,57.371	,57.791,59.8302,61.623,64.155,68.99,70.963,71.548,72.209,72.2665,	72.271,72.29,72.32]
# assert len(where2_comm_x) == len(where2_comm_y), "wrong"
# ax_2[1].plot(where2_comm_x, where2_comm_y,  label="Where2comm", **line_styles[0])
where2_comm_x = list(map(lambda x: x,
                         [3.012, 11.813, 17.687, 20.3123, 23.6882503]))
where2_comm_y = [55.21, 65.67, 78.57, 85.2831, 85.94]
assert len(where2_comm_x) == len(where2_comm_y), "wrong"
ax_2[1].plot(where2_comm_x, where2_comm_y,  **line_styles[0])

codefilling_x = list(map(lambda x: x,
                         [3.2143, 9.0588, 16.412, 18.82702, 23.6882503]))
codefilling_y = [56.982,66.41, 78.41, 85.989, 86.84]
assert len(codefilling_x) == len(codefilling_y), "wrong"
ax_2[1].plot(codefilling_x, codefilling_y,  **line_styles[1])

QTF_x = list(map(lambda x: x, [3.432, 9.53, 12.47, 17.9620, 23.6882503]))
QTF_y = [58.7679, 70.12, 79.45, 87.67, 88.76]
assert len(QTF_x) == len(QTF_y), "wrong"
ax_2[1].plot(QTF_x, QTF_y,  **line_styles[2])

# ax_2[1].legend()
# ax_2[1].legend(ncol=5,loc="upper center",bbox_to_anchor=(0.5, -0.2),fontsize=10,shadow=True,frameon=True)
ax_2[1].grid(True)
# 第三个子图 - V2XSet
# ax_2[2].set_title('V2XSet', fontdict=title_font)
# 添加子图内容...
v2xvit_x = 23.6882503
v2xvit_y = 88.65

late_fusion_x = 17.9106
late_fusion_y = 72.77

v2vnet_x = 23.6882503
v2vnet_y = 90.51

attentive_fusion_x = 23.6882503
attentive_fusion_y = 87.74

cobevt_x = 23.6882503
cobevt_y = 88.19

adafusion_x = 23.6882503
adafusion_y = 87.63

disconet_x = 21.6882503
disconet_y = 85.52
for i in range(1, len(labels) + 1):
    ax_2[2].scatter(eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_x"),
                    eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_y"), c=marker_color[i - 1],
                    marker=marker_shape[i - 1], label=labels[i - 1],s=17)

# 数据点和线段数据
where2_comm_x = list(map(lambda x: x, [3.3783, 9.1231, 11.91, 17.313, 20.712, 23.6882503]))
where2_comm_y = [49.92, 58.22, 66.31, 75.123, 86.847, 87.91]
assert len(where2_comm_x) == len(where2_comm_y), "wrong"
ax_2[2].plot(where2_comm_x, where2_comm_y, label="Where2comm", **line_styles[0])

codefilling_x = list(map(lambda x: x,
                         [3.48, 8.313, 12.0588, 16.146 ,19.82702, 23.6882503,
                          ]))
codefilling_y = [50.982,  62.452, 70.231, 78.692, 88.689,88.72]
assert len(codefilling_x) == len(codefilling_y), "wrong"
ax_2[2].plot(codefilling_x, codefilling_y, label="CodeFilling", **line_styles[1])

QTF_x = list(map(lambda x: x, [3.2, 7.41,12.13, 15.331, 19.938, 23.6882503]))
QTF_y = [52.26, 66.32,74.7059, 85.31, 90.319,91.46]
assert len(QTF_x) == len(QTF_y), "wrong"
ax_2[2].plot(QTF_x, QTF_y, label="QCTF(Ours)", **line_styles[2])

ax_2[2].grid(True)
# plt.tight_layout()
fig.legend(ncol=5,loc="lower center",mode="expand",bbox_to_anchor=(0.12,0.05,0.785,0.7),fontsize=10,shadow=True,frameon=True)

# 调整底部边距以防止图例被裁剪
plt.subplots_adjust(bottom=0.2)
# 显示图形
plt.show()





if __name__ == '__main__':
    plt.style.use(['science'])
    plt.rcParams.update({
        "font.family": "serif",  # specify font family here
        "font.serif": ["Times"],  # specify font here
        "font.size": 11})
    # localization_error()
    # heading_error()










