#!/bin/bash
#SBATCH --job-name=trainvw_arch1           # 任务名
#SBATCH --nodes=1                   # 这里不用动 多节点脚本请查官方文档
#SBATCH --ntasks=1                  # 这里不用动 多任务脚本请查官方文档
#SBATCH --cpus-per-task=16           # 要几块CPU (一般4块就够用了)
#SBATCH --mem=16GB                  # 最大内存
#SBATCH --time=167:00:00             # 运行时间上限
#SBATCH --mail-type=END             # ALL / END
#SBATCH --mail-user=yw3642@nyu.edu  # 结束之后给哪里发邮件
#SBATCH --output=log/%x%A.out         # 正常输出写入的文件
#SBATCH --error=log/%x%A.err          # 报错信息写入的文件
#SBATCH --gres=gpu:1                # 需要几块GPU (同时最多8块)
#SBATCH -p aquila                   # 有GPU的partition
#SBATCH --nodelist=agpu8            # 3090

module purge                        # 清除所有已加载的模块
module load anaconda3 cuda/11.1.1               # 加载anaconda (load virtual env for training)
nvidia-smi
nvcc --version
cd /gpfsnyu/scratch/yw3642/lang-seg   # 切到程序目录

echo "START"               # 输出起始信息
source deactivate
source /gpfsnyu/packages/anaconda3/5.2.0/bin/activate lseg          # 调用 virtual env
export CUDA_VISIBLE_DEVICES=0
wandb offline
python -u train_lseg.py --dataset vizwiz --data_path ../datasets --batch_size 1 --exp_name lseg_vizwiz_l16_640_512_arch1 \
--base_lr 0.004 --weight_decay 1e-4 --no-scaleinv --max_epochs 50 --widehead --accumulate_grad_batches 2 --backbone clip_vitl16_384 \
--arch_option 1 --block_depth 2
echo "FINISH"                       # 输出起始信息
