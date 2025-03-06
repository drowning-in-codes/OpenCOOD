compiler = python
hypes_yaml = opencood/logs/point_pillar_range_comm_fusion_opencood

# opv2v
model_dir = opencood/logs/point_pillar_range_comm_fusion_opencood
model_dir = opencood/logs/v2vnet
model_dir = opencood/logs/pointpillar_CoBEVT_nocompression
# model_dir = opencood/logs/pointpillar_attentive_fusion/pointpillar_attentive_fusion
# model_dir = opencood/logs/v2x-vit
# model_dir = opencood/logs/point_pillar_where2comm_opv2v

# v2xset
model_dir = opencood/logs/v2xset_range_comm
# model_dir = opencood/logs/v2vnet
# model_dir = opencood/logs/pointpillar_attentive_fusion/pointpillar_attentive_fusion
# model_dir = opencood/logs/v2x-vit
# model_dir = opencood/logs/point_pillar_where2comm_v2xset
# model_dir = opencood/logs/cobevt_lidar

# v2v4real
model_dir = opencood/logs/v2v4_corange
model_dir = opencood/logs/v2v4_v2vnet
model_dir = opencood/logs/v2v4_attfuse
# model_dir = opencood/logs/v2v4_v2xvit
# model_dir = opencood/logs/v2v4_cobevt

PROGRAM = ./opencood/tools/calc_flops_param.py
PROGRAM = ./opencood/tools/calc_run_time.py
PROGRAM = ./opencood/tools/infer.py

VIS_PROGRAM = ./opencood/tools/attn_map_vis.py
vis_model_dir = opencood/logs/v2xset_range_comm
vis_model_dir = opencood/logs/v2xset_range_comm

COM_PROGRAM = ./opencood/tools/explain_communication.py
comm_model_dir = opencood/logs/point_pillar_range_comm_fusion_opencood

SAL_PROGRAM = ./opencood/tools/explain_saliency.py
comm_model_dir = opencood/logs/point_pillar_range_comm_fusion_opencood

.PHONY: calc, weight_vis, comm_vis, explain_saliency

calc:
	${compiler} ${PROGRAM} --model_dir "${model_dir}"

weight_vis:
	${compiler} ${VIS_PROGRAM} --model_dir "${vis_model_dir}"

comm_vis:
	${compiler} ${COM_PROGRAM} --model_dir "${comm_model_dir}"

explain_saliency:
	${compiler} ${SAL_PROGRAM} --model_dir "${comm_model_dir}"

