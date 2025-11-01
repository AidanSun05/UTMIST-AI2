'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below.

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

import torch
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.agent import *
from typing import Optional, Type, List, Tuple
import math

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.file_path is None:
            # policy_kwargs = {
            #     'activation_fn': nn.ReLU,
            #     'lstm_hidden_size': 256,
            #     'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
            #     'shared_lstm': False,
            #     'enable_critic_lstm': True,
            #     'share_features_extractor': True,

            # }
            # self.model = RecurrentPPO("MlpLstmPolicy",
            #                           self.env,
            #                           verbose=2,
            #                           learning_rate=1e-4,
            #                           n_steps=512,
            #                           batch_size=256,
            #                           ent_coef=0.003,
            #                           policy_kwargs=policy_kwargs,
            #                           device=device,
            #                           tensorboard_log="tb_log")

            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'share_features_extractor': True,

            }
            self.model = PPO("MlpPolicy",
                             self.env,
                             verbose=2,
                             learning_rate=5e-5,
                             n_steps=512,
                             batch_size=256,
                             ent_coef=0.1,
                             policy_kwargs=policy_kwargs,
                             device=device,
                             tensorboard_log="tb_log")
            del self.env
        else:
            policy_kwargs = {
                'activation_fn': nn.Tanh,
                'net_arch': [dict(pi=[64, 64, 32], vf=[64, 64, 32])],
                'share_features_extractor': False,
            }

            self.model = PPO.load(self.file_path, device=device)
            self.model.learning_rate = lambda p: 1e-6 + (1e-5 - 1e-6) * p
            self.model.ent_coef = 0.01
            self.model.clip_range = lambda _: 0.1
            self.model.batch_size = 512
            self.model.tensorboard_log = "tb_log"

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=2):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        return action

class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1  # Increment step counter
        return action

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        super(MLPPolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # Feature extraction only

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))  # Features for both policy and value heads

class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0],
            action_dim=10,  # Not directly used in extractor
            hidden_dim=hidden_dim,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)

class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'net_arch': [dict(pi=[64, 64], vf=[64, 64])],  # Separate heads after features
                'features_extractor_class': MLPExtractor,
                'features_extractor_kwargs': dict(features_dim=64, hidden_dim=64),
                'share_features_extractor': True,  # Now this makes sense
                'log_std_init': -0.5,
                'ortho_init': True,
            }

            self.model = self.sb3_class("MlpPolicy",
                                        self.env,
                                        policy_kwargs=policy_kwargs,
                                        verbose=2,
                                        learning_rate=2e-5,  # Very conservative for custom architecture
                                        n_steps=1024,
                                        batch_size=128,  # Smaller batches for more updates
                                        n_epochs=8,
                                        ent_coef=0.02,
                                        clip_range=0.15,
                                        vf_coef=0.8,  # Higher value focus for stability
                                        max_grad_norm=0.8,
                                        gamma=0.99,
                                        gae_lambda=0.92,
                                        normalize_advantage=True,
                                        tensorboard_log="tb_log")
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''

def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    return -zone_penalty * env.dt if player.body.position.y >= zone_height else 0.0

def move_to_opponent_reward(env: WarehouseBrawl) -> float:
    """
    Computes the reward based on whether the agent is moving toward the opponent.
    The reward is calculated by taking the dot product of the agent's normalized velocity
    with the normalized direction vector toward the opponent.

    Args:
        env (WarehouseBrawl): The game environment

    Returns:
        float: The computed reward
    """

    player: Player = env.objects["player"]
    opponent = env.objects["opponent"]

    x = player.body.position.x
    if x < -6.5 or x > 6.5:
        return 0.0

    tx = opponent.body.position.x
    v = player.body.velocity.x
    dist = tx - x
    if v * dist > 0:
        return 0.5 * env.dt

    return -0.2 * env.dt

def fall_reward(env: WarehouseBrawl) -> float:
    player = env.objects["player"].body
    platform = env.objects["platform1"].body
    x = player.position.x
    y = player.position.y
    px = platform.position.x
    py = platform.position.y

    # Boundary penalties
    if x < -6.5:
        return -8.0 / (1 + math.exp(3 * (x + 6.5))) * env.dt
    elif x > 6.5:
        return -8.0 / (1 + math.exp(3 * (6.5 - x))) * env.dt

    # Reward successful crossing, penalize risky behavior
    mid_gap = -3 < x < 3

    if mid_gap:
        # With positive Y down:
        # - dy > 0 means falling (BAD)
        # - dy < 0 means jumping up (GOOD)

        # Base penalty for being in the gap (encourage quick crossing)
        reward = -0.2

        # Clamped absolute velocities
        dx = min(1.0, abs(player.velocity.x) / 5)
        dy = min(1.0, abs(player.velocity.y) / 5)

        # Reward horizontal and upward movement across the gap
        if dx > 0 and dy < 0:
            reward += 0.08 * (dx ** 2 + dy ** 2) ** 0.5

        # Penalize being below platform or lower ground
        if y < -2.85 or y < py:
            # Falling
            reward -= 0.1

        # Encourage landing on platform
        if px - 0.8 < x < px + 0.8 and py - 1 < y < py:
            reward += 0.15

        return reward * env.dt

    return 0

def idle_penalty(env: WarehouseBrawl) -> float:
    player = env.objects["player"]
    if abs(player.body.velocity.x) < 0.5 and abs(player.body.velocity.y) < 0.5:
        return -0.1 * env.dt
    return 0

def damage_interaction_reward(env):
    """
    Reward attacking the opponent and penalize taking hits.
    Adds situational awareness for distance and spam prevention.
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]

    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    reward = 0.0

    # --- Offensive reward: hitting enemy ---
    if damage_dealt > 0:
        reward += 5.0 * (damage_dealt / 10.0)

    # --- Defensive penalty: getting hit ---
    if damage_taken > 0:
        reward -= 3.0 * (damage_taken / 10.0)

    # --- Smart attacking behavior ---
    player_x = player.body.position.x
    opponent_x = opponent.body.position.x
    dist = abs(player_x - opponent_x)
    is_attacking = isinstance(player.state, AttackState)

    # Reward close-range attacking (good aim)
    if is_attacking and dist < 2.0:
        reward += 0.5

    # Punish long-range spamming (ineffective)
    if is_attacking and dist >= 3.0:
        reward -= 0.2

    return reward * env.dt

def holding_more_than_3_keys(env: WarehouseBrawl) -> float:
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return -env.dt
    return 0

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 2.0
        elif env.objects["player"].weapon == "Spear":
            return 1.0
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -1.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager():
    reward_functions = {
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.3),
        'move_to_opponent_reward': RewTerm(func=move_to_opponent_reward, weight=5.0),
        'fall_reward': RewTerm(func=fall_reward, weight=0.5),
        'idle_penalty': RewTerm(func=idle_penalty, weight=0.2),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=6.0),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=5)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=3)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=2)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=4)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=4.5))
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':
    # Create agent
    my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)

    # Start here if you want to train from scratch. e.g:
    # my_agent = RecurrentPPOAgent()

    # Start here if you want to train from a specific timestep. e.g:
    # my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_8/rl_model_1100011_steps.zip')

    # Reward manager
    reward_manager = gen_reward_manager()
    # Self-play settings
    selfplay_handler = SelfPlayRandom(
        partial(type(my_agent)), # Agent class and its keyword arguments
                                 # type(my_agent) = Agent class
    )

    # Set save settings here:
    save_handler = SaveHandler(
        agent=my_agent, # Agent to save
        save_freq=100_000, # Save frequency
        max_saved=40, # Maximum number of saved models
        save_path='checkpoints', # Save path
        run_name='experiment_9',
        mode=SaveHandlerMode.FORCE # Save mode, FORCE or RESUME
    )

    # Set opponent settings here:
    opponent_specification = {
                    'self_play': (8, selfplay_handler),
                    'constant_agent': (0.5, partial(ConstantAgent)),
                    'based_agent': (1.5, partial(BasedAgent)),
                }
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    train(my_agent,
        reward_manager,
        save_handler,
        opponent_cfg,
        CameraResolution.LOW,
        train_timesteps=5_000_000,
        train_logging=TrainLogging.PLOT
    )
