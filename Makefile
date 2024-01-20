TRAIN_TOOL = './opencood/tools/train.py'
REFERENCE_TOOL = './opencood/tools/inference.py'
HYPES_YAML = './opencood/hypes_yaml/point_pillar_range_fusion/point_pirllar_range_fusion_v3.yaml'
MODEL_DIR = 'opencood/logs/point_pillar_range_fusion_2024_01_19_15_38_23'
ENV_NAME = 'opencood'

env: 
	conda activate ${ENV_NAME}

train: 
	 python ${TRAIN_TOOL} --hypes_yaml ${HYPES_YAML} --model_dir ${MODEL_DIR}


inference: 
	 python ${REFERENCE_TOOL} --model_dir ${MODEL_DIR} --fusion_method "intermediate"

.PHONY:train,inference,env