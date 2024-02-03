#! /bin/bash

pip install rouge_chinese wandb nltk jieba datasets transformers==4.32.0 deepspeed==0.10.0 accelerate==0.21.0 transformers_stream_generator==0.0.4  tiktoken -i https://pypi.tuna.tsinghua.edu.cn/simple


GPUS_PER_NODE=8
# WORLD_SIZE=1
# MASTER_PORT=6000
# RANK=0
# MASTER_ADDR="localhost"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
is_master=${MASTER-"0"}

if [[ $is_master -eq 1 ]];then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $HOSTNAME --master_port $MASTER_PORT"
else
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
fi
ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

echo $ROOT_DIR

model_name_or_path="pretrained_model_name_or_path"
DATA_ARGS="--data_path data_config.json"
output_dir="checkpoint"

torchrun $DISTRIBUTED_ARGS \
    /train_sft.py \
    --model_name_or_path $model_name_or_path \
    $DATA_ARGS \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 80 \
    --save_total_limit 50 \
    --report_to wandb \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --deepspeed "/default_offload_opt_param.json" \
    --lazy_preprocess True >train_sft.log
