# CHANGELOG

## 2026.7.20 v1.1

Task `RobotLab-Go2-PPO-Symmetry-v1`:

1. Migrate the training environment of the `RobotLab-Go2-MoeCts-Symmetry-v1` to PPO.
2. Support RoboGauge evaluation for PPO

## 2026.7.6 v1

Task `RobotLab-Go2-MoeCts-Symmetry-v1`:

1. Finished adding symmetry augmentation, use symmetry data for critic, actor, latent loss, instead of only entropy loss.
2. Merge `mdp.hip_pos_penalty_l1` into `mdp.joint_pos_penalty_l1`, change `stand_still_scale: 1.0 -> 10.0`
3. In `Go2RLGymCommandCfg`, change zero command to be `[0,0,0]` instead of `x=y=0`, which make sure the `joint_pos_penalty_l1` can take effect when the robot is standing still.
4. Change `num_envs=10900` for the Go2 PPO and MoE-CTS symmetry tasks.
