export PYTHONWARNINGS="ignore"
CUDA_VISIBLE_DEVICES=0 accelerate launch imagen.py --train \
                            --source ./2nd_finetune_dataset \
                            --epoch 20 \
                            --wandb \
                            --no_patching \
                            --samples_out ./outputs/unet1_cond3.0_sample512_start256_lr1e-5_dim256_nofp16_human_data/samples \
                            --imagen ./outputs/unet1_cond3.0_sample512_start256_lr1e-5_dim256_nofp16_human_data/imagen.pth --self_cond \
                            --cond_scale 3.0 \
                            --test_pkl ./dataset/test_pkl/0.pkl \
                            --text_encoder t5-large \
                            --pretrained t5-large \
                            --batch_size 9 \
                            --sample_steps 512 \
                            --start_size 256 \
                            --num_unets 0 \
                            --train_unet 1 \
                            --sample_unet 1 \
                            --lr 1e-5 \
                            --random_drop_tags 0.5 \
                            --unet_dims 256 \
                            --unet2_dims 128
