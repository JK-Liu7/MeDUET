export OMP_NUM_THREADS=4
export TIMM_FUSED_ATTN=1

torchrun --nnodes=1 --nproc_per_node=4 --master_port=29509 main_pretrain.py --distributed