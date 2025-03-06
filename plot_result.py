import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def comm_vol():
    # 创建图形和子图
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    # 设置标题字体样式
    title_font = {
        'fontsize': 14,
        'fontweight': 'light',
        'color': 'black'
    }
    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.set_xlabel('Communication Volume', fontsize=14, fontweight='light')
        ax.set_ylabel('AP@0.7', fontsize=14, fontweight='light')
    # 第一个子图
    axes[0].set_title('Default OPV2V Towns', fontdict=title_font)
    # axes[0].set_xlim(10, 26)
    # 数据点和线段样式
    line_styles = [
                   {'color': 'orange', 'linestyle': 'solid', 'marker': 'o'},
                   {'color': 'purple', 'linestyle': 'solid', 'marker': 'x'},
                   ]
    marker_shape = ['o','x','s','d','8','p','h']
    marker_size = 10
    marker_color = ['blue','red','green','black','pink']
    labels = ["V2X-ViT","Late Fusion","V2VNet","Attentive Fusion","CoBEVT"]

    v2xvit_x = 25.103287808412 - 5
    v2xvit_y = 82.61

    late_fusion_x = 16.10328781
    late_fusion_y = 73.83

    v2vnet_x = 25.103287808412- 5
    v2vnet_y = 82.22

    attentive_fusion_x = 25.103287808412- 5
    attentive_fusion_y = 81.51

    cobevt_x = 25.103287808412 - 5
    cobevt_y = 86.13
    for i in range(1,len(labels)+1):
        axes[0].scatter(eval(f"{labels[i-1].lower().replace('-','').replace(' ','_')}_x"), eval(f"{labels[i-1].lower().replace('-','').replace(' ','_')}_y"), c=marker_color[i-1],marker=marker_shape[i-1], label=labels[i-1])

    # 数据点和线段数据 18.2016
    # where2_comm_x = list(map(lambda x:x-5,[11.0588,15.1462,17.7313191,18.82702,19.59804,21.35,22.6901,23.19,24.6177543,24.87305955,25.10328781]))
    # where2_comm_y = [65.3,67.231,70.689,74.692,76.789,79.672,80.489,80.657,80.665,80.672,80.68]
    # assert len(where2_comm_x) == len(where2_comm_y), "wrong"
    # axes[0].plot(where2_comm_x, where2_comm_y, label="Where2comm", **line_styles[0])

    where2_comm_x = list(map(lambda x: x -2 ,
                             [9.68,11.0588,13.542 ,15.1462, 17.7313191, 18.82702, 19.59804, 20.35,21.578 ,22.6901,
                             ]))
    where2_comm_y = [64.982,65.3, 66.452,67.231, 70.689, 74.692, 76.789, 79.672,80.245 ,80.489,   ]
    assert len(where2_comm_x) == len(where2_comm_y), "wrong"
    axes[0].plot(where2_comm_x, where2_comm_y, label="Where2comm", **line_styles[0])

    # corange_x = list(map(lambda x:x-5,[12.1,14.70,15.3218,16.39227,17.27612,18.1472,19.55362,20.36303682 ,21.44837 ,23.63535536 ,23.9755,24.32680,24.99159]))
    # corange_y = [70.0121,70.486,70.52,70.675,70.885,71.286,72.44,74.168,80.8086,87.5126,87.566,87.4426,87.423]

    corange_x = list(map(lambda x: x - 5 + 0.58,
                         [12.1, 15.3218, 16.39227,  18.1472, 19.55362, 20.36303682, 21.44837,
                          23.63535536,  24.32680, 24.99159]))
    corange_y = [70.0121, 70.52, 70.675,71.286, 72.44, 74.168, 80.8086, 87.5126, 87.4426,
                 87.423]
    assert len(corange_x) == len(corange_y), "wrong"
    axes[0].plot(corange_x, corange_y,  label="CoRange", **line_styles[1])



    axes[0].legend()
    axes[0].grid(True)


    # 第二个子图 - Culver
    axes[1].set_title('Culver City', fontdict=title_font)
    # 添加子图内容...

    v2xvit_x = 25.103287808412 - 5
    v2xvit_y = 73.65

    late_fusion_x = 16.10328781
    late_fusion_y = 58.82

    v2vnet_x = 25.103287808412- 5
    v2vnet_y = 73.43

    attentive_fusion_x = 25.103287808412- 5
    attentive_fusion_y = 73.57

    cobevt_x = 25.103287808412- 5
    cobevt_y = 77.32
    for i in range(1, len(labels) + 1):
        axes[1].scatter(eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_x"),
                        eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_y"), c=marker_color[i - 1],
                        marker=marker_shape[i - 1], label=labels[i - 1])



    # 数据点和线段数据
    # where2_comm_x = list(map(lambda x:x-0.58,[13.1,14.23,16.14469903,17.380,17.89177107,18.5824 ,19.15692078,20.10872517,20.56668829,20.81040968,21.69006504,22.89888455,23.19460283,24.27198118,25.10328781]))
    # where2_comm_y = [56.12,56.4115,57.371	,57.791,59.8302,61.623,64.155,68.99,70.963,71.548,72.209,72.2665,	72.271,72.29,72.32]
    # assert len(where2_comm_x) == len(where2_comm_y), "wrong"
    # axes[1].plot(where2_comm_x, where2_comm_y,  label="Where2comm", **line_styles[0])
    where2_comm_x = list(map(lambda x: x ,
                             [13.05, 14.23, 16.14469903, 17.380, 17.89177107, 18.5824, 19.15692078, 19.76314,
                              20.26668829,  20.68006504]))
    where2_comm_y = [56.12, 56.4115, 57.371, 57.791, 59.8302, 61.623, 64.155, 68.99, 71.963, 72.32]
    assert len(where2_comm_x) == len(where2_comm_y), "wrong"
    axes[1].plot(where2_comm_x, where2_comm_y, label="Where2comm", **line_styles[0])


    corange_x = list(map(lambda x:x-5+ 0.58,[17.47,20.0620,21.4146,22.169298 ,22.385241,22.766942,23.0639,23.63401,24.2832828,24.885696,25.10271]))
    corange_y = [59.7679,59.828,60.312, 61.17,62.9185,66.775, 69.94,75.365,77.44,79.616,79.668]
    assert len(corange_x) == len(corange_y), "wrong"
    axes[1].plot(corange_x, corange_y,  label="CoRange",  **line_styles[1])


    axes[1].legend()
    axes[1].grid(True)
    # 第三个子图 - V2XSet
    axes[2].set_title('V2XSet', fontdict=title_font)
    # 添加子图内容...
    v2xvit_x = 25.103287808412- 5
    v2xvit_y = 71.25

    late_fusion_x = 16.10328781
    late_fusion_y = 64.84

    v2vnet_x = 25.103287808412- 5
    v2vnet_y = 73.89

    attentive_fusion_x = 25.103287808412- 5
    attentive_fusion_y = 74.03

    cobevt_x = 25.103287808412- 5
    cobevt_y = 66.06
    for i in range(1, len(labels) + 1):
        axes[2].scatter(eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_x"),
                        eval(f"{labels[i - 1].lower().replace('-', '').replace(' ', '_')}_y"), c=marker_color[i - 1],
                        marker=marker_shape[i - 1], label=labels[i - 1])

    # 数据点和线段数据
    where2_comm_x = list(map(lambda x:x-3,[13.523,13.90686899,15.56939,16.434,17.268335,	18.22,		19.2722	,20.931	,22.4477,23.57]))
    where2_comm_y = [46.32,46.62,47.721,49.542,52.106	,56.519	,61.546,65.35,65.366,65.38]
    assert len(where2_comm_x) == len(where2_comm_y), "wrong"
    axes[2].plot(where2_comm_x, where2_comm_y, label="Where2comm", **line_styles[0])

    corange_x = list(map(lambda x:x-5+ 0.58,[14.9259,15.7259,16.58164,18.991519,20.938,22.02845,23.241,24.22700,24.82700,25.096]))
    corange_y = [58.26,58.4659,58.7059,60.454,66.5580,72.6133,77.319,81.395,82.492,82.57184]
    assert len(corange_x) == len(corange_y), "wrong"
    axes[2].plot(corange_x, corange_y, label="CoRange", **line_styles[1])

    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    # 显示图形
    plt.show()


def localization_error():
    # 分成subplots
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    # bold the fonts
    xlabel_font = {
        # 'fontsize': rcParams['axes.titlesize'], # 设置成和轴刻度标签一样的大小
        'fontsize': 14,
        # 'fontweight': rcParams['axes.titleweight'], # 设置成和轴刻度标签一样的粗细
        'fontweight': 'light',
        # 'color': rcParams['axes.titlecolor'], # 设置成和轴刻度标签一样的颜色
        'color': 'black',
    }
    xlables = []
    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.set_xlabel('Localization Error Std(m)', fontdict=xlabel_font)
        ax.set_ylabel('AP@0.7', fontdict=xlabel_font)
        y_interval = 5
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=y_interval))
        # 设置第一个子图的横轴和纵轴标签
    dots_num = 6
    labels = ["Where2comm", "V2VNet", "CoBEVT", "V2X-VIT", "Attentive Fusion", "CoRange"]
    model_num = len(labels)
    axes[0].set_title('Detection Performance on Default Towns')
    x = np.linspace(0, 0.5, dots_num)
    y1 = [80.68	,79.49,	74.29,	68.59	,56.71	,41.926]
    y2 = [82.2	,80.24	,75.66,	68.14	,57.016,47.026]
    y3 = [86.17	,84.37,	79.45	,68.897	,51.1,	37.27]
    y4 = [82.6	,81.3	,76.26	,69.24	,54.616	,46.616]
    y5 = [81.55	,80.2353,75.659	,66.760,	52.93,	42.73]

    y6 = [87.39,86.12,82.77,75.11,61.95	,48.14]
    # where2comm v2xvit v2vnet Nofusion Cobevt opv2v when2comm
    line_styles = [{'color': 'blue', 'linestyle': '--', 'marker': 'o'},
                   {'color': 'red', 'linestyle': '-', 'marker': 'x'},
                   {'color': 'green', 'linestyle': '-', 'marker': '2'},
                   {'color': 'black', 'linestyle': '-', 'marker': '^'},
                   {'color': 'purple', 'linestyle': '-.', 'marker': '8'},
                   {'color': 'orange', 'linestyle': '-', 'marker': 's'},
                   ]
    assert model_num == len(line_styles), "The number of models and line styles should be the same."
    for i in range(1,model_num+1):
        axes[0].plot(x, eval(f"y{i}"), **line_styles[i-1],label=labels[i-1])
    axes[1].set_title('Detection Performance on Culver City')
    y1 = [72.32, 72.29	,67.24	,56.51	,45.96	,36.94]
    y2 = [73.43, 72.1,67.26,59.59,51.09,44.195]
    y3 = [77.32, 72.87,68.75,58.17,46.178,38.21]
    y4 = [73.65, 71.39,66.54,55.91,	45.06,36.14]
    y5 = [73.57, 72.623	,67.84,57.71,46.56,37.14]

    y6 = [79.66, 79.14	,76.02	,67.88	,58.13	,47.54]
    # where2comm v2xvit v2vnet Nofusion Cobevt opv2v when2comm
    for i in range(1, model_num + 1):
        axes[1].plot(x, eval(f"y{i}"), **line_styles[i - 1], label=labels[i - 1])

    # 设置第一个子图的横轴和纵轴标签
    axes[2].set_title('Detection Performance on V2XSet')
    y1 = [65.24,62.62,55.6,45.5,34.89,26.89]
    y2 = [73.89	,71.641	,66.044	,56.5341,44.077,34.48]
    y3 = [66.06	,64.07	,59.3,53.03	,45.81,39.542]
    y4 = [71.25,66.417	,62.166	,58.8252,54.619	,50.55]
    y5 = [74.03,71.47	,65.963	,55.78	,41.34	,31.67]

    y6 = [82.64	,80.69	,74.18	,63.71	,53.67	,49.69]
    for i in range(1, model_num + 1):
        axes[2].plot(x, eval(f"y{i}"), **line_styles[i - 1], label=labels[i - 1])

    for ax in axes:
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    # 显示图形
    plt.show()




def heading_error():
    # 分成subplots
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    # bold the fonts
    xlabel_font = {
        # 'fontsize': rcParams['axes.titlesize'], # 设置成和轴刻度标签一样的大小
        'fontsize': 14,
        # 'fontweight': rcParams['axes.titleweight'], # 设置成和轴刻度标签一样的粗细
        'fontweight': 'light',
        # 'color': rcParams['axes.titlecolor'], # 设置成和轴刻度标签一样的颜色
        'color': 'black',
    }
    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.set_xlabel('Heading Error Std(m)', fontdict=xlabel_font)
        ax.set_ylabel('AP@0.7', fontdict=xlabel_font)
        y_interval = 5
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=y_interval))

    dots_num = 11
    labels = ["Where2comm", "V2VNet", "CoBEVT", "V2X-VIT", "Attentive Fusion", "CoRange"]
    model_num = len(labels)
    axes[0].set_title('Detection Performance on Default Towns')
    x = np.linspace(0, 1, dots_num)
    y1 = [80.68,79.88,78.95,75.82,	70.75	,66.87	,64.18	,58.98	,55.8	,52.93	,51.09]
    y2 = [82.2,81.54,79.67,76.935	,73.4129,	69.331	,65.45	,61.6115,	58.05,	54.89,	52.228]
    y3 = [86.17,85.93,84.796,81.547,77.3483,72.35,67.97,63.7,59.36,55.56,51.94]
    y4 = [82.6,80.3	,78.36,75.94,72.616	,69.89,	65.82	,59.87	,55.77,	52.38,47.13]
    y5 = [81.55, 81.01,	79.35,76.12	,71.85,	67.47,	63.38,	59.58,	56.4,53.33,50.39]
    y6 = [87.39,87.31993,86.0973,83.477,80.0178,75.2174,70.4116,65.578,61.554,57.865,54.2366]
    for i in range(1, model_num + 1):
        assert len(eval(f"y{i}")) == dots_num, f"y{i} should have {dots_num} elements.but have"+str(len(eval(f"y{i}")))
    # where2comm v2xvit v2vnet Nofusion Cobevt opv2v when2comm
    line_styles = [{'color': 'blue', 'linestyle': '--', 'marker': 'o'},
                   {'color': 'red', 'linestyle': '-', 'marker': 'x'},
                   {'color': 'green', 'linestyle': '-', 'marker': '2'},
                   {'color': 'black', 'linestyle': '-', 'marker': '^'},
                   {'color': 'purple', 'linestyle': '-.', 'marker': '8'},
                   {'color': 'orange', 'linestyle': '-', 'marker': 's'},
                   ]
    assert model_num == len(line_styles), "The number of models and line styles should be the same."
    for i in range(1, model_num + 1):
        axes[0].plot(x, eval(f"y{i}"), **line_styles[i - 1], label=labels[i - 1])

    # 第二个子图

    axes[1].set_title('Detection Performance on Culver City')
    x = np.linspace(0, 1, dots_num)
    y1 = [73.32, 73.29,	72.23	,65.82	,62.71	,58.64,	55.32	,52.8,	48.72	,47.672,46.74]
    y2 = [73.43,72.63	,70.57,	63.9,	63.069	,59.34	,56.16,	53.09	,50.62	,48.163	,45.859]
    y3 = [77.32, 72.496,	70.35,	66.949,	63.49,	59.45,	55.87	,52.83,	50.115,	47.49,	45.116]
    y4 = [73.65, 71.25	,68.47	,62.7,	60.06,	58.27	,55.27	,52.17	,48.24,	46.62,45.31]
    y5 = [73.57,72.53,	69.748,	66.02,	62.16,	58.34,	54.748	,51.8,	48.91,	46.54	,46.54]
    y6 = [79.66,79.22	,77.03	,73.14	,68.4	,63.92	,60.19,	56.83	,53.78	,51.05	,48.61]

    for i in range(1, model_num + 1):
        axes[1].plot(x, eval(f"y{i}"), **line_styles[i - 1], label=labels[i - 1])

    # 第三个子图

    axes[2].set_title('Detection Performance on V2Xset')
    x = np.linspace(0, 1, dots_num)
    y1 = [65.24	,64.98	,64.82	,62.19	,58.53	,54.536	,50.82	,46.98	,43.6	,40.45	,37.86]
    y2 = [73.89,72.63	,70.10019,	66.7,	62.771	,59.11,	55.412	,51.78,	48.17,	44.948	,42.06]
    y3 = [66.06	,65.43	,63.964	,61.844,	59.73,	57.13	,54.4	,51.83	,49.46	,47.49,47.04]
    y4 = [71.25,67.362	,66.417	,64.56	,62.4982	,60.46	,58.4442	,56.33	,54.294	,52.41	,50.6674]
    y5 = [74.03,72.77	,70	,66.37	,62.16	,58.07	,54.739	,50.29,	48.91	,46.73	,44.25]

    y6 = [80.32, 78.43, 75.748, 73.02, 69.16, 65.34, 59.748, 56.8, 54.20, 51.54, 49.419]
    for i in range(1, model_num + 1):
        axes[2].plot(x, eval(f"y{i}"), **line_styles[i - 1], label=labels[i - 1])

    for ax in axes:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    # 显示图形
    plt.show()


if __name__ == '__main__':
    localization_error()
    heading_error()
    comm_vol()










