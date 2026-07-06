# CHANGELOG

## 2026.7.6 v1

1. Finished adding symmetry augmentation, use symmetry data for critic, actor, latent loss, instead of only entropy loss.
2. Merge `mdp.hip_pos_penalty_l1` into `mdp.joint_pos_penalty_l1`, change `stand_still_scale: 1.0 -> 10.0`
3. In `Go2RLGymCommandCfg`, change zero command to be `[0,0,0]` instead of `x=y=0`, which make sure the `joint_pos_penalty_l1` can take effect when the robot is standing still.
4. Change `num_envs=8192` when using `RobotLab-Go2-Symmetry-v1` task.
