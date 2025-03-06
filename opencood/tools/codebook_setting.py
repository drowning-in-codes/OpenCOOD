import matplotlib.pyplot as plt
import scienceplots
import numpy as np

def plot_codebook_settings():
    # 创建一个包含3个子图的图形对象fig和轴对象axes
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))  # 一行三列布局
    xlabel_font = {
        'fontsize': 12,
        'fontweight': 'light',
        'color': 'gray',
        'family': 'serif',
    }
    ax_1 = axes[0]
    ax_2 = axes[1]

    for ax in ax_1:
        ax.tick_params(labelsize=8,axis='both', which='both', bottom=True, top=False, left=True, right=False)
        ax.set_xlabel('Communication Volume(log2)', fontdict=xlabel_font, labelpad=1)
        ax.set_ylabel('AP@0.7 on OPV2V', fontsize=14,fontdict=xlabel_font)
    for ax in ax_2:
        ax.tick_params(labelsize=8,axis='both', which='both', bottom=True, top=False, left=True, right=False)
        ax.set_xlabel('Communication Volume(log2)', fontdict=xlabel_font, labelpad=1)
        ax.set_ylabel('AP@0.7 on V2XSet', fontsize=14,fontdict=xlabel_font)
    ax_1[-1].set_ylabel('AP@0.7', fontsize=14,fontdict=xlabel_font)
    ax_2[-1].set_ylabel('AP@0.5', fontsize=14,fontdict=xlabel_font)
    # 数据示例
    x1 = [2.881,6.173,12.232,13.81,16.6032, 23.688]
    y1 =  [55.672,61.412,81.872,83.672,85.6321,85.823]

    x2 = [2.681,7.523,12.752,15.0732,17.13 ,23.688]
    y2 =  [54.672,62.312,81.572,83.73,85.021,85.423]

    x3 = [2.781,8.523,11.752,17.8732,18.92 ,23.688]
    y3 = [50.672,61.312,70.572,79.23,79.921,80.423]

    x4 = [2.681,7.123,11.152,17.641,18.6032, 23.688]
    y4 = [49.69,54.812,66.372,76.313,77.9,78.423]
    line_styes = [{'color': 'red', 'linestyle': 'solid'},{ 'color': 'green', 'linestyle': 'solid',},
                  {'linestyle': 'solid','color': 'blue'},{ 'color': 'purple', 'linestyle': 'solid',}]
    # 添加linewidth
    for i in range(len(line_styes)):
        line_styes[i]['linewidth'] = 2
    # 第一个子图 - 多条线及对应的标签
    ax_1[0].plot(x1, y1, **line_styes[0], label='$C_e$=256')
    ax_1[0].plot(x2, y2, **line_styes[1],label='$C_e$=128')  # 添加第二条线
    ax_1[0].plot(x3, y3, **line_styes[2], label='$C_e$=64')  # 添加第三条线
    ax_1[0].plot(x4, y4, **line_styes[3], label='$C_e$=16')  # 添加第三条线

    ax_1[0].grid(True)
    ax_1[0].legend()
    # 第二个子图
    # 数据示例
    x1 = [2.681, 6.573, 10.232, 14.931,17.08, 17.8032, 23.688]
    y1 = [51.672, 62.412, 75.972,83.41 ,85.272, 85.9321, 86.623]

    x2 = [2.681, 8.223,9.641 ,12.752, 15.0732, 17.13, 23.688]
    y2 = [51.072, 69.612,72.41 ,81.172, 83.73, 85.021, 85.423]

    x3 = [2.781, 8.523, 10.752,15.12 ,18.8732, 19.92, 23.688]
    y3 = [50.672, 62.312, 70.572,75.13 ,78.63, 79.121, 79.883]

    x4 = [2.681, 7.523, 10.152, 14.7641, 18.6032, 23.688]
    y4 = [48.39, 52.812, 66.372, 73.631, 76.72, 77.653]

    # 第一个子图 - 多条线及对应的标签
    ax_1[1].plot(x1, y1, **line_styes[0], label='$n_e$=1024')
    ax_1[1].plot(x2, y2, **line_styes[1],label='$n_e$=512')  # 添加第二条线
    ax_1[1].plot(x3, y3, **line_styes[2], label='$n_e$=256')  # 添加第三条线
    ax_1[1].plot(x4, y4, **line_styes[3], label='$n_e$=128')  # 添加第三条线
    # axes[0].set_title('Subfigure 1')
    ax_1[1].grid(True)
    ax_1[1].legend()

    # 第三个子图
    x = np.linspace(0, 1, 5)
    y = [85.412, 85.672, 85.872, 85.875, 85.9321]
    y2 = [77.46,77.85,78.14,79.36,79.67]
    y3 = [78.31,79.342,81.05,81.34,81.423]

    ax_1[2].set_xlabel("Channel selection rates",fontdict=xlabel_font,labelpad=1)
    ax_1[2].plot(x,y, linewidth=3,label="Default Towns")
    ax_1[2].plot(x,y2, linewidth=3,label="Culver City")
    ax_1[2].plot(x,y3, linewidth=3,label="V2XSet")
    ax_1[2].grid(True)
    ax_1[2].legend()

    # 数据示例
    x1 = [2.881, 7.173, 12.232, 15.81, 17.6032, 23.688]
    y1 = [55.672, 62.412, 74.872, 78.672, 80.6321, 82.823]

    x2 = [3.2,8.23,12.54,14.531,17.938,23.688]
    y2 = [54.26,65.52,73.57,75.61,80.319,81.25]

    x3 = [3.281, 7.523, 13.652, 17.8732, 19.82, 23.688]
    y3 = [49.672, 60.312, 66.572, 74.23, 75.921, 77.423]

    x4 = [3.171, 8.123, 12.152, 17.341, 19.6032, 23.688]
    y4 = [47.69, 54.812, 63.152, 71.313, 75.2, 76.423]
    line_styes = [{'color': 'red', 'linestyle': 'solid'}, {'color': 'green', 'linestyle': 'solid', },
                  {'linestyle': 'solid', 'color': 'blue'}, {'color': 'purple', 'linestyle': 'solid', }]
    ax_2[0].grid(True)
    # 添加linewidth
    for i in range(len(line_styes)):
        line_styes[i]['linewidth'] = 2
    # 第一个子图 - 多条线及对应的标签
    ax_2[0].plot(x1, y1, **line_styes[0], label='$C_e$=256')
    ax_2[0].plot(x2, y2, **line_styes[1], label='$C_e$=128')  # 添加第二条线
    ax_2[0].plot(x3, y3, **line_styes[2], label='$C_e$=64')  # 添加第三条线
    ax_2[0].plot(x4, y4, **line_styes[3], label='$C_e$=16')  # 添加第三条线
    ax_2[0].text(0.5, -0.25, '(a)', transform=ax_2[0].transAxes, ha='center', fontsize=15)
    ax_2[1].text(0.5, -0.25, '(b)', transform=ax_2[1].transAxes, ha='center', fontsize=15)
    ax_2[2].text(0.5, -0.25, '(c)', transform=ax_2[2].transAxes, ha='center', fontsize=15)
    ax_2[0].legend()

    # 第二个子图
    # 数据示例
    x1 = [3.181, 5.573, 10.132, 14.331, 17.48, 18.8032, 23.688]
    y1 = [52.652, 59.412, 66.872, 76.41,79.272, 81.9321, 83.05]

    x2 = [3.2,8.33,11.23,14.531,17.938,23.688]
    y2 = [51.26,64.81,69.7059,74.31,80.319,81.25]

    x3 = [3.281, 10.523, 15.652, 18.4732, 19.12, 23.688]
    y3 = [49.172,60.312, 72.572, 74.63, 76.321, 77.223]

    x4 = [2.881, 10.523, 14.152, 18.341, 19.8032, 23.688]
    y4 = [47.79, 54.812, 66.552, 74.313, 75.2, 75.423]

    # 第一个子图 - 多条线及对应的标签
    ax_2[1].plot(x1, y1, **line_styes[0], label='$n_e$=1024')
    ax_2[1].plot(x2, y2, **line_styes[1], label='$n_e$=512')  # 添加第二条线
    ax_2[1].plot(x3, y3, **line_styes[2], label='$n_e$=256')  # 添加第三条线
    ax_2[1].plot(x4, y4, **line_styes[3], label='$n_e$=128')  # 添加第三条线
    # axes[0].set_title('Subfigure 1')
    ax_2[1].grid(True)
    ax_2[1].legend()

    # 第三个子图
    x = np.linspace(0, 1, 5)
    y = [91.39, 91.48, 92.54, 93.05, 93.12]
    y2 = [87.92, 88.02, 88.14, 88.78, 88.82]
    y3 = [89.91, 91.08, 91.35, 91.46, 91.52]


    ax_2[2].set_xlabel("Channel selection rates", fontdict=xlabel_font, labelpad=1)
    ax_2[2].plot(x, y, linewidth=3, label="Default Towns")
    ax_2[2].plot(x, y2, linewidth=3, label="Culver City")
    ax_2[2].plot(x, y3, linewidth=3, label="V2XSet")
    ax_2[2].grid(True)
    ax_2[2].legend()
    # 调整子图之间的间距
    plt.tight_layout()
    # 调整底部边距以防止图例被裁剪
    plt.subplots_adjust(hspace =.2,bottom=0.12)
    # 显示图形
    plt.show()

if __name__ == '__main__':
    plt.style.use('science')
    plt.rcParams.update({
        "font.family": "serif",  # specify font family here
        "font.serif": ["Times"],  # specify font here
        "font.size": 11})
    plot_codebook_settings()