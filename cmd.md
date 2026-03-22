## train
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Robotlab-Go2-v0 --headless --num_envs=1024

## eval
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Robotlab-Go2-v0 --num_envs=64
