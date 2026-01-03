export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="3b8190fcd4e64b10225bdf05797f4326836e6f1f"
# export WANDB_MODE="offline"
export PYTHONPATH=${PWD}
export CUDA_VISIBLE_DEVICES=0

# python3 scripts/train.py --config configs/local_nusc_mini.yaml
python3 scripts/train.py --config configs/local_nusc.yaml
# python3 scripts/train.py --config configs/local_kitti.yaml