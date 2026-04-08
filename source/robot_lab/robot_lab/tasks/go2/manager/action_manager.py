from isaaclab.managers import ActionManager
import torch
from collections.abc import Sequence

# ActionManagerGo2 is a simple custom ActionManager that 
# maintain _prev_prev_action for action smoothness reward computation.
class ActionManagerGo2(ActionManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_prev_action = torch.zeros_like(self._action)
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        super().reset(env_ids)
        if env_ids is None:
            self._prev_prev_action.zero_()
        else:
            self._prev_prev_action[env_ids] = 0.0
        return {}
    
    def process_action(self, action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        # check if action dimension is valid
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prev_prev_action[:] = self._prev_action
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim
    
    @property
    def prev_prev_action(self):
        return self._prev_prev_action
    
# ActionManagerGo2WithDelay is a custom ActionManager that 
# maintain _prev_prev_action for action smoothness reward computation.
# and also do random action delay by process_action_with_delay() function.
class ActionManagerGo2WithDelay(ActionManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_prev_action = torch.zeros_like(self._action)
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        super().reset(env_ids)
        if env_ids is None:
            self._prev_prev_action.zero_()
        else:
            self._prev_prev_action[env_ids] = 0.0
        return {}
    
    def process_action(self, action: torch.Tensor):
        raise NotImplementedError("Use update_action() and process_action_with_delay() instead for ActionManagerWithDelay.")
    
    def process_action_with_delay(self, action_delay_masks: torch.Tensor):
        """Processes the actions sent to the environment.
        
        Important Note:
            This function can be called multiple times within a single step() call to implement action delay.

        Args:
            action_delay_masks: A tensor of shape (num_envs, 1) indicating which actions to apply at this time step.
        """
        # action_delay_masks == True means delay (use prev action)
        action = torch.where(action_delay_masks, self._prev_action, self._action)
        
        # split the actions and apply to each tensor
        # NOTE: we also assume that all term.process_actions can be called multiple times within a single step() call
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def update_action(self, action: torch.Tensor):
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        self._prev_prev_action[:] = self._prev_action
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)
    
    @property
    def prev_prev_action(self):
        return self._prev_prev_action