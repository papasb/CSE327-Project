#!/bin/bash
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate spike_driven


# Print Python environment details
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Python site-packages: $(python -c 'import sys; print(sys.path)')"

export PYTHONPATH=/home/mjkim2/miniconda3/envs/spike_driven/lib/python3.9/site-packages:$PYTHONPATH

THRESHOLD=12000  # 12GB 이상일 때 실행 (원하면 조정)

echo "GPU 메모리 감시 시작... 최소 ${THRESHOLD} MiB 필요"

while true; do
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    if [ "$FREE_MEM" -ge "$THRESHOLD" ]; then
        echo "충분한 GPU 메모리 확보됨: $FREE_MEM MiB"
        echo "학습을 시작합니다..."
        break
    else
        sleep 30
    fi
done


# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 train.py -c conf/cifar10/2_256_300E_t4.yml --model sdt --spike-mode lif
python ../train.py -c conf/2_256_300E_t4.yml -data-dir "/home/bcl/Documents/mj_weather" --model sdt --classeval True --spike-mode lif --output /home/bcl/Spike_Driven_Transformer/Myeongjun/RESULT
# python train.py -c conf/cifar10/2_512_300E_t4.yml --model sdt --spike-mode lif
# python train.py -c conf/cifar10/2_512_300E_t4_TET.yml --model sdt --spike-mode lif
