from isaaclab.managers import ActionManager
import torch
from collections.abc import Sequence

class ActionManagerWithDelay(ActionManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_prev_action = torch.zeros_like(self._action)
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        super().reset(env_ids)
        self._prev_prev_action.zero_()
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