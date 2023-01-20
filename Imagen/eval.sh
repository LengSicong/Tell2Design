export PYTHONWARNINGS="ignore"

CUDA_VISIBLE_DEVICES=3 python imagen.py --source ./dataset \
                                                    --wandb --no_patching \
                                                    --output ./outputs/unet1_cond3.0_sample512_start256_lr1e-5_dim256_nofp16/eval_samples/eval.png \
                                                    --imagen ./outputs/unet1_cond3.0_sample512_start256_lr1e-5_dim256_nofp16/imagen.pth \
                                                    --cond_scale 3.0 \
                                                    --test_pkl ./eval_data/annotated_3.pkl \
                                                    --text_encoder t5-large \
                                                    --pretrained t5-large \
                                                    --batch_size 64 \
                                                    --micro_batch_size 32 \
                                                    --sample_steps 256 \
                                                    --start_size 64 \
                                                    --num_unets 1 \
                                                    --train_unet 1 \
                                                    --sample_unet 1 \
                                                    --replace
