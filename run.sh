PYTHONPATH="./:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=2,3 taskset -c 50-63 \
torchrun --nproc_per_node=2 --master_port=1111 basicsr/train_mix.py -opt options/train/ESRGAN/train_Real_ESRGAN_EMA_x4_canon.yml --launcher pytorch