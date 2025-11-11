from typing import Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import gymnasium as gym
from gymnasium.core import ActType, ObsType

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderCNN(nn.Module):
    def __init__(self, in_channels, embedding_dim=2048, input_shape=(128, 128)):
        super(EncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        self.fc1 = nn.Linear(self._compute_conv_output((in_channels, input_shape[0], input_shape[1])), embedding_dim)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
    
    def _compute_conv_output(self, shape:Tuple[int, int, int]):
        with torch.no_grad():
            x = torch.randn(1, shape[0], shape[1], shape[2])
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            
            return x.shape[1] * x.shape[2] * x.shape[3]
    
    def forward(self,x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        return x
    
class DecoderCNN(nn.Module):
    def __int__(self,hidden_size:int,state_size:int,embedding_size:int, use_bn:bool = True, output_shape:Tuple[int,int]=(3,128,128)):
        super(DecoderCNN, self).__init__()
        
        self.output_shape = output_shape
        self.embedding_size = embedding_size
        self.use_bn = use_bn

        self.fc1 = nn.Linear(hidden_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 256*(output_shape[1]//16)*(output_shape[2]//16))
        
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(32, output_shape[0], kernel_size=3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        
    def forward(self,h:torch.Tensor, s:torch.Tensor):
        x = torch.cat([h,s], dim=-1)
        x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        
        x = x.view(-1, 256, self.output_shape[1]//16, self.output_shape[2]//16)
        
        if self.use_bn:
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = torch.relu(self.bn3(self.conv3(x)))
        else:
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
        
        x = self.conv4(x)
        
        return x
    
class InitialWrapper(nn.Module):
    def __init__(self, env: gym.Env, no_ops:int = 0, repeat: int = 1):
        super(InitialWrapper, self).__init__(env)
        self.repeat = repeat
        self.no_ops = no_ops
        self.op_counter = 0
        
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        if self.op_counter < self.no_ops:
            self.op_counter += 1
            obs, reward, done, info = self.env.step(0)
        
        tottal_reward = 0.0
        done = False
        for _ in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            tottal_reward += reward
            if done:
                break
        
        return obs, tottal_reward, done, info

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, new_shape: Sequence[int] = (128, 128, 3), grayscale: bool = False):
        super(PreprocessFrame, self).__init__(env)
        self.shape = new_shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)
        self.grayscale = grayscale

        if self.grayscale:
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*self.shape[:-1], 1), dtype=np.float32)

    def observation(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.astype(np.uint8)
        new_frame = cv.resize(obs, self.shape[:-1], interpolation=cv.INTER_AREA)
        if self.grayscale:
            new_frame = cv.cvtColor(new_frame, cv.COLOR_RGB2GRAY)
            new_frame = np.expand_dims(new_frame, -1)

        torch_frame = torch.from_numpy(new_frame).float()
        torch_frame = torch_frame / 255.0

        return torch_frame

def make_env(env_name: str, new_shape: Sequence[int] = (128, 128, 3), grayscale: bool = True, **kwargs):
    env = gym.make(env_name, **kwargs)
    env = PreprocessFrame(env, new_shape, grayscale=grayscale)
    return env


class RewardModel(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(hidden_dim + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        
    def forward(self, h: torch.Tensor, s: torch.Tensor):
        x = torch.cat([h, s], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class DynamicsModel(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, state_dim: int, embedding_dim: int, rnn_layer: int = 1):
        super(DynamicsModel, self).__init__()

        self.hidden_dim = hidden_dim
        
        # Can be any recurrent network
        self.rnn = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(rnn_layer)])
        
        # Projection layer to make efficient use of concatenated inputs
        self.project_state_action = nn.Linear(action_dim + state_dim, hidden_dim)
        
        # Return mean and log-variance of the normal distribution
        self.prior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_action = nn.Linear(hidden_dim + action_dim, hidden_dim)
        
        # Return mean and log-variance of the normal distribution
        self.posterior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_obs = nn.Linear(hidden_dim + embedding_dim, hidden_dim)

        self.state_dim = state_dim

        self.act_fn = nn.ReLU()

    def forward(self, prev_hidden: torch.Tensor, prev_state: torch.Tensor, actions: torch.Tensor,
                obs: torch.Tensor = None, dones: torch.Tensor = None):
        """
        Forward pass of the dynamics model for one time step.
        :param prev_hidden: Previous hidden state of the RNN: (batch_size, hidden_dim)
        :param prev_state: Previous stochastic state: (batch_size, state_dim)
        :param action: One hot encoded actions: (sequence_length, batch_size, action_dim)
        :param obs: This is the encoded observation from the encoder, not the raw observation!: (sequence_length, batch_size, embedding_dim)
        :param dones: Terminal states of the environment
        :return: 
        """
        B, T, _ = actions.size() # They are crucial to to infernece without access to observations

        hiddens_list = []
        posterior_means_list = []
        posterior_logvars_list = []
        prior_means_list = []
        prior_logvars_list = []
        prior_states_list = []
        posterior_states_list = []
        
        # (B, 1, hidden_dim)
        hiddens_list.append(prev_hidden.unsqueeze(1))  
        prior_states_list.append(prev_state.unsqueeze(1))
        posterior_states_list.append(prev_state.unsqueeze(1))

        for t in range(T - 1):
            ### Combine the state and action ###
            action_t = actions[:, t, :]
            obs_t = obs[:, t, :] if obs is not None else torch.zeros(B, self.embedding_dim, device=actions.device)
            state_t = posterior_states_list[-1][:, 0, :] if obs is not None else prior_states_list[-1][:, 0, :]
            state_t = state_t if dones is None else state_t * (1 - dones[:, t, :])
            hidden_t = hiddens_list[-1][:, 0, :]
            
            state_action = torch.cat([state_t, action_t], dim=-1)
            state_action = self.act_fn(self.project_state_action(state_action))

            ### Update the deterministic hidden state ###
            for i in range(len(self.rnn)):
                hidden_t = self.rnn[i](state_action, hidden_t)

            ### Determine the prior distribution ###
            hidden_action = torch.cat([hidden_t, action_t], dim=-1)
            hidden_action = self.act_fn(self.project_hidden_action(hidden_action))
            prior_params = self.prior(hidden_action)
            prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)

            ### Sample from the prior distribution ###
            prior_dist = torch.distributions.Normal(prior_mean, torch.exp(F.softplus(prior_logvar)))
            prior_state_t = prior_dist.rsample()

            ### Determine the posterior distribution ###
            # If observations are not available, we just use the prior
            if obs is None:
                posterior_mean = prior_mean
                posterior_logvar = prior_logvar
            else:
                hidden_obs = torch.cat([hidden_t, obs_t], dim=-1)
                hidden_obs = self.act_fn(self.project_hidden_obs(hidden_obs))
                posterior_params = self.posterior(hidden_obs)
                posterior_mean, posterior_logvar = torch.chunk(posterior_params, 2, dim=-1)

            ### Sample from the posterior distribution ###
            posterior_dist = torch.distributions.Normal(posterior_mean, torch.exp(F.softplus(posterior_logvar)))
            
            # Make sure to use rsample to enable the gradient flow
            # Otherwise you could also use code the reparameterization trick by hand
            posterior_state_t = posterior_dist.rsample()

            ### Store results in lists (instead of in-place modification) ###
            posterior_means_list.append(posterior_mean.unsqueeze(1))
            posterior_logvars_list.append(posterior_logvar.unsqueeze(1))
            prior_means_list.append(prior_mean.unsqueeze(1))
            prior_logvars_list.append(prior_logvar.unsqueeze(1))
            prior_states_list.append(prior_state_t.unsqueeze(1))
            posterior_states_list.append(posterior_state_t.unsqueeze(1))
            hiddens_list.append(hidden_t.unsqueeze(1))

        # Convert lists to tensors using torch.cat()
        hiddens = torch.cat(hiddens_list, dim=1)
        prior_states = torch.cat(prior_states_list, dim=1)
        posterior_states = torch.cat(posterior_states_list, dim=1)
        prior_means = torch.cat(prior_means_list, dim=1)
        prior_logvars = torch.cat(prior_logvars_list, dim=1)
        posterior_means = torch.cat(posterior_means_list, dim=1)
        posterior_logvars = torch.cat(posterior_logvars_list, dim=1)

        return hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars
    
class RSSM:
    def __init__(self,
                 encoder: EncoderCNN,
                 decoder: DecoderCNN,
                 reward_model: RewardModel,
                 dynamics_model: nn.Module,
                 hidden_dim: int,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int,
                 device: str = device):
        super(RSSM,self).__init__()
        self.dynamics = dynamics_model
        self.encoder = encoder
        self.decoder = decoder
        self.reward_model = reward_model

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        
        self.dynamics.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.reward_model.to(device)
        
    def generate_rollout(self, actions:torch.Tensor, hiddens:torch.Tensor=None, obs:torch.Tensor = None, dones:torch.Tensor = None):
        if hiddens is None:
            hiddens = torch.zeros(actions.size(0),self.hidden_dim).to(actions.device)
        if states is None:
            states = torch.zeros(actions.size(0),self.state_dim).to(actions.device)
        

        dynamics_result = self.dynamics(hiddens, states, actions, obs, dones)
        hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = dynamics_result
        
        return hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars
    
    
        