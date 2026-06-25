import torch
from tensordict import TensorDict

obs_dict = TensorDict(
    {
        'a': torch.zeros(1024, 3),
        'b': torch.zeros(1024, 5),
    }
)

# obs = TensorDict(
#     {key: torch.zeros(24, *value.shape) for key, value in obs_dict.items()},
#     batch_size=[24, 1024],
# )

for key, item in obs_dict.items():
    print(key, item.shape)
