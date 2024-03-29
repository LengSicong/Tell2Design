#! /bin/bash

# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

# OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=bond0 NCCL_IB_GID_INDEX=3 NCCL_NET_GDR_LEVEL=0"
OPTIONS_NCCL="NCCL_DEBUG=info"
HOST_FILE_PATH="hostfile_single"


config_json="$script_dir/ds_config.json"
gpt_options=" \
       --experiment-name cogview-floorplan-2nd_finetune \
       --img-tokenizer-num-tokens 8192 \
       --dataset-type CompactBinaryDataset \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 12 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --save $main_dir/data/checkpoints \
       --train-iters 100000 \
       --resume-dataloader \
       --train-data ./data/merge_2nd_finetune.bin \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .1 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --max-position-embeddings 1537 \
       --max-memory-length 0 \
       --fp16 \
       --txt-loss-scale 5 \
       --finetune \
       --fast-load \
       --load /home/sicong/CogView/data/checkpoints/cogview-floorplan-1strun11-28-22-14/
"

# gpt_options="${gpt_options}
#                --deepspeed \
#                --deepspeed_config ${config_json} \
# "


# run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_gpt2.py $@ ${gpt_options}"
run_cmd="${OPTIONS_NCCL} CUDA_VISIBLE_DEVICES=1 python pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
