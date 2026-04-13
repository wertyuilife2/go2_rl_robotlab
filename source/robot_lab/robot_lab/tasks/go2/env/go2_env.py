from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VecEnvStepReturn
from robot_lab.tasks.go2.manager.action_manager import ActionManagerGo2, ActionManagerGo2WithDelay

import torch

class Go2Env(ManagerBasedRLEnv):
    cfg: ManagerBasedRLEnvCfg
    
    def load_managers(self):
        super().load_managers()
        # override action manager
        self.action_manager = ActionManagerGo2(self.cfg.actions, self)
        print("[Go2Env-INFO] Overriding action manager with ActionManagerGo2: ", self.action_manager)


class ActionDelayGo2Env(ManagerBasedRLEnv):
    cfg: ManagerBasedRLEnvCfg

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the ActionDelayGo2Env with the given configuration.

        Args:
            cfg: The configuration for the environment.
            render_mode: Rendering mode for the environment, e.g., "human" or "rgb_array". Default is None.
            **kwargs: Additional keyword arguments for customization.
        """
        # Call the parent class initializer
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        print(
            "[ActionDelayGo2Env-WARNING] You are using ActionDelayGo2Env; "
            "make sure all ActionTerms support multiple calls to process_actions() "
            "within a single step()."
        )

    def load_managers(self):
        super().load_managers()
        # override action manager
        self.action_manager = ActionManagerGo2WithDelay(self.cfg.actions, self)
        print("[ActionDelayGo2Env-INFO] Overriding action manager with ActionManagerGo2WithDelay: ", self.action_manager)            
        
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.

        IMPORTANT NOTE:
            We intentionally call action_manager.process_action_with_delay() multiple times inside step
            (contrary to its original intent) to implement action-delay functionality.
            So, we assume that all ActionTerm.process_actions() are allowed to be
            called multiple times within a single step call.
        """
        # call update_action once per step to set action and prev action
        self.action_manager.update_action(action.to(self.device))
        
        # randomly determine when to start applying actions within the decimation steps for each environment
        actions_start_decimation = torch.randint(0, self.cfg.decimation+1, (self.num_envs, 1), device=self.device)

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for i in range(self.cfg.decimation):
            self._sim_step_counter += 1
            
            # determine which envs should apply delayed action at this decimation step
            action_delay_masks = (i < actions_start_decimation)
            self.action_manager.process_action_with_delay(action_delay_masks)
                
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim() 
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras



