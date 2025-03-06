#!/bin/bash
# 禁止脚本在命令失败时退出
set +e
echo "Start to run the qualitive result"
echo "---------Default OPV2V Towns--------"
# 定义要执行的 make 命令列表
model_config_opv2v=("opencood/logs/point_pillar_where2comm_opv2v" "opencood/logs/v2vnet" "opencood/logs/pointpillar_CoBEVT_nocompression"
"opencood/logs/pointpillar_attentive_fusion/pointpillar_attentive_fusion"
"opencood/logs/v2x-vit" "opencood/logs/point_pillar_range_comm_fusion_opencood"
    )

# 循环执行每个 make 命令
for cmd in "${model_config_opv2v[@]}"; do
    echo "Executing: $cmd"
    eval "$cmd"
    echo "Command completed with exit code $?"
done
#echo "---------V2XSet--------"
#model_config_v2xset=("opencood/logs/point_pillar_where2comm_v2xset" "opencood/logs/v2vnet" "opencood/logs/cobevt_lidar"
#"opencood/logs/pointpillar_attentive_fusion/pointpillar_attentive_fusion"
#"opencood/logs/v2x-vit" "opencood/logs/v2xset_range_comm"
#    )
## 循环执行每个 make 命令
#for cmd in "${model_config_v2xset[@]}"; do
#    echo "Executing: $cmd"
#    eval "$cmd"
#    echo "Command completed with exit code $?"
#done
#echo "---------V2V4Real--------"
#model_config_v2xset=("opencood/logs/point_pillar_where2comm_v2xset" "opencood/logs/v2vnet" "opencood/logs/cobevt_lidar"
#"opencood/logs/pointpillar_attentive_fusion/pointpillar_attentive_fusion"
#"opencood/logs/v2x-vit" "opencood/logs/v2xset_range_comm"
#    )
## 循环执行每个 make 命令
#for cmd in "${model_config_v2xset[@]}"; do
#    echo "Executing: $cmd"
#    eval "$cmd"
#    echo "Command completed with exit code $?"
#done

# 重新开启脚本在命令失败时的退出行为（可选）
set -e

# 可以在这里添加后续的命令或逻辑