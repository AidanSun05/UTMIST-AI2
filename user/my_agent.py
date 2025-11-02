# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
#
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
#
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
#
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# To run the sample TTNN model, you can uncomment the 2 lines below:
import ttnn

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        super(MLPPolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs.to(torch.float32)))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x)).to(torch.bfloat16)

class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0],
            action_dim=10,
            hidden_dim=hidden_dim,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim) #NOTE: features_dim = 10 to match action space output
        )

class TTMLPPolicy(nn.Module):
    def __init__(self, state_dict, mesh_device):
        super(TTMLPPolicy, self).__init__()
        self.mesh_device = mesh_device
        self.fc1 = ttnn.from_torch(
            state_dict["fc1.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.fc2 = ttnn.from_torch(
            state_dict["fc2.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.fc3 = ttnn.from_torch(
            state_dict["fc3.weight"].T,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        self.fc1_b = ttnn.from_torch(state_dict["fc1.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.fc2_b = ttnn.from_torch(state_dict["fc2.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.fc3_b = ttnn.from_torch(state_dict["fc3.bias"], device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # print("Running TT forward pass!")
        obs = obs.to(torch.bfloat16)
        tt_obs = ttnn.from_torch(obs, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)

        x1 = ttnn.linear(tt_obs, self.fc1, bias=self.fc1_b, activation="relu")
        tt_obs.deallocate()

        x2 = ttnn.linear(x1, self.fc2, bias=self.fc2_b, activation="relu")
        x1.deallocate()

        x3 = ttnn.linear(x2, self.fc3, bias=self.fc3_b, activation="relu")
        x2.deallocate()

        tt_out = ttnn.to_torch(x3).flatten().to(torch.float32)

        return tt_out


class SubmittedAgent(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''
    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        super().__init__(file_path)
        self.time = 0

        # To run a TTNN model, you must maintain a pointer to the device and can be done by
        # uncommmenting the line below to use the device pointer
        self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = PPO("MlpPolicy", self.env, verbose=0)
            del self.env
        else:
            self.model = PPO.load(self.file_path, custom_objects={'policy_kwargs': MLPExtractor.get_policy_kwargs()})

        # To run the sample TTNN model during inference, you can uncomment the 5 lines below:
        # This assumes that your self.model.policy has the MLPPolicy architecture defined in `train_agent.py` or `my_agent_tt.py`
        mlp_state_dict = self.model.policy.features_extractor.model.state_dict()
        self.tt_model = TTMLPPolicy(mlp_state_dict, self.mesh_device)
        self.model.policy.features_extractor.model = self.tt_model
        self.model.policy.vf_features_extractor.model = self.tt_model
        self.model.policy.pi_features_extractor.model = self.tt_model

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1zv1zC-VHOiBAFiWjjlXZULuDWhTon90_/view?usp=drive_link"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        # This uses if-statements for recovery and trained PPO model for attack/strategy
        # During recovery, the agent does not attack
        self.time += 1

        # Recovery
        recovery = False
        action = self.act_helper.zeros()
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]

        # If off the edge, come back
        if pos[0] > 10.67 / 2:
            action = self.act_helper.press_keys(['a'])
            if self.time % 2 == 0:
                action = self.act_helper.press_keys(['space'], action)
            recovery = True
        elif pos[0] < -10.67 / 2:
            action = self.act_helper.press_keys(['d'])
            if self.time % 2 == 0:
                action = self.act_helper.press_keys(['space'], action)
            recovery = True

        # Jump if below map
        if (pos[1] > 1.6) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)
            recovery = True

        # Middle gap
        if -3 < pos[0] < 3:
            # Head toward opponent
            if not opp_KO and opp_pos[0] > pos[0]:
                action = self.act_helper.press_keys(['d'])
            else:
                # Go to left (lower) side by default
                action = self.act_helper.press_keys(['a'])

            # Jump as much as possible in the middle gap
            if self.time % 2 == 0:
                action = self.act_helper.press_keys(['space'], action)
            recovery = True

        if recovery:
            return action

        # Normal PPO behavior
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    # If modifying the number of models (or training in general), modify this
    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
