TRAIN_TOOL = './opencood/tools/train_da.py'
TRAIN_TOOL_DA = './opencood/tools/train_da.py'
REFERENCE_TOOL = './opencood/tools/inference.py'
REFERENCE_TOOL_DA = './opencood/tools/inference_da.py'
HYPES_YAML = 'opencood/hypes_yaml/domain_adaptions/point_pillar_early_fusion.yaml'
MODEL_DIR = 'opencood/logs/point_pillar_early_fusion_da_2024_07_11_19_51_43'
ENV_NAME = 'opencood'
LAST_EPOCH = -1
# 设置 --half 标志
HALF_FLAG =
EXPLAIN_YAML = 'opencood/logs/v2v4_corange'
PROGRAM=opencood/tools/calc_flops_params.py
PROGRAM=opencood/tools/calc_run_time.py
PROGRAM=opencood/tools/calc_run_mem.py

# 如果设置了 --half 变量,则添加 --half 标志
ifneq ($(HALF),)
    HALF_FLAG = --half
endif
.PHONY: train,inference,env,train_da,inference_da

explain:
	python ${PROGRAM} --hypes_yaml ${EXPLAIN_YAML} --model_dir ${EXPLAIN_YAML} 

env: 
	conda activate ${ENV_NAME} 


train: 
	 python ${TRAIN_TOOL} --hypes_yaml ${HYPES_YAML} --model_dir ${MODEL_DIR} $(HALF_FLAG)

inference: 
	 python ${REFERENCE_TOOL} --model_dir ${MODEL_DIR} --fusion_method "intermediate"

train_da: 
	 python ${TRAIN_TOOL_DA} --hypes_yaml ${HYPES_YAML} --model_dir ${MODEL_DIR} $(HALF_FLAG)

inference_da: 
	 python ${REFERENCE_TOOL_DA} --last_epoch ${LAST_EPOCH} --model_dir ${MODEL_DIR} --fusion_method "intermediate"

# TODO train cobevt from 20