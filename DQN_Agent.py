from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym

class ReplayBuffer:

    def __init__(self, obs_dim: int, capacity=1000, batch_size = 32):
        self.s0_buf = np.zeros([capacity, obs_dim], dtype=np.float32)
        self.a0_buf = np.zeros([capacity], dtype=np.float32)
        self.reward_buf = np.zeros([capacity], dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.s1_buf = np.zeros([capacity, obs_dim], dtype=np.float32)
        self.capacity, self.batch_size = capacity, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        s0,
        a0,
        reward: float,
        done: bool,
        s1
    ):
        # Helper to flatten observations of various types
        def _prepare_obs(obs):
            # If observation is a dict, flatten by sorted key order
            if isinstance(obs, dict):
                parts = [np.asarray(obs[k], dtype=np.float32).reshape(-1)
                         for k in sorted(obs.keys())]
                return np.concatenate(parts)
            # If observation is list/tuple, flatten each element
            if isinstance(obs, (list, tuple)):
                parts = [np.asarray(x, dtype=np.float32).reshape(-1) for x in obs]
                return np.concatenate(parts)
            # Otherwise assume array-like
            return np.asarray(obs, dtype=np.float32).reshape(-1)

        # Prepare states
        s0_arr = _prepare_obs(s0)
        s1_arr = _prepare_obs(s1)

        # Prepare action as scalar
        try:
            a0_val = float(np.asarray(a0).reshape(()))
        except Exception:
            a0_val = float(np.asarray(a0).flatten()[0])

        # Store into buffers (ensure correct shape)
        self.s0_buf[self.ptr] = s0_arr
        self.a0_buf[self.ptr] = a0_val
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.s1_buf[self.ptr] = s1_arr

        # Advance pointer and update size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(s0_bat=self.s0_buf[idxs],
                    a0_bat=self.a0_buf[idxs],
                    reward_bat=self.reward_buf[idxs],
                    done_bat=self.done_buf[idxs],
                    s1_bat=self.s1_buf[idxs])

    def __len__(self) -> int:
        return self.size


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DQNAgent:
    """DQN Agent 与环境的交互

    Attribute:
        env (gym.Env): openAI Gym 环境
        memory (ReplayBuffer): 存储transitions的经验池
        batch_size (int): 一批次训练的样本数（从经验池一次采样的trans数量）
        epsilon (float): 探索因子
        epsilon_decay (float): 探索因子衰减率
        max_epsilon (float): epsilon最大值
        min_epsilon (float): epsilon最小值
        target_update_freq (int): 目标网络参数值复制频率
        gamma (float): 折扣因子
        dqn (Network): 主网络
        dqn_target (Network): 目标网络
        optimizer (torch.optim): 主网络优化器
        transition (list): 一个样本信息，包括 (state, action, reward, next_state, done)
    """

    def __init__(self, env: gym.Env,
                 capacity: int,
                 batch_size: int,
                 target_update_freq: int,
                 epsilon_decay: float,
                 max_epsilon: float = 1.0,
                 min_epsilon: float = 0.1,
                 gamma: float = 0.99):

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.buffer = ReplayBuffer(obs_dim, capacity, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update_freq = target_update_freq
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # 网络: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # 优化器
        self.optimizer = optim.Adam(self.dqn.parameters())

        self.transition = list()

        # 模型: train / test
        self.is_test = False

    def select_action(self, s0: np.ndarray) -> np.ndarray:
        # epsilon greedy policy
        if np.random.random() < self.epsilon:
            a0 = self.env.action_space.sample()
        else:
            a0 = self.dqn(torch.FloatTensor(s0).to(self.device)).argmax()
            a0 = a0.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [s0, a0]

        return a0

    def step(self, a0: np.ndarray):
        s1, reward, terminated, truncated, info = self.env.step(a0)
        done = terminated or truncated
        if not self.is_test:
            self.transition += [reward, done, s1]
            self.buffer.store(*self.transition)

        return s1, reward, done

    def update_network(self) -> torch.Tensor:
        samples = self.buffer.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, steps: int, plotting_interval: int = 500):
        self.is_test = False

        s0,_ = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for steps_idx in range(1, steps + 1):
            a0 = self.select_action(s0)
            s1, reward, done, truncated, _ = self.env.step(a0)
            done = done or truncated
            if not self.is_test:
                # 存 transition
                self.transition += [reward, done, s1]
                self.buffer.store(*self.transition)
            s0 = s1
            score += reward

            if done:
                s0,_ = self.env.reset()
                scores.append(score)
                score = 0

            # 如果经验池数据集达到采样数时，开始训练；同时开始减小epsilon
            if len(self.buffer) >= self.batch_size:
                loss = self.update_network()
                losses.append(loss)
                update_cnt += 1

                # 逐步减小epsilon
                self.epsilon = max(self.min_epsilon,
                                   self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
                epsilons.append(self.epsilon)

                # 当满足目标网络参数更新条件时，更新目标网络
                if update_cnt % self.target_update_freq == 0:
                    self._target_hard_update()

            # 显示训练过程
            if steps_idx % plotting_interval == 0:
                self._plot(steps_idx, scores, losses, epsilons)

        self.env.close()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """返回dqn loss"""
        device = self.device
        s0_bat = torch.FloatTensor(samples["s0_bat"]).to(device)
        a0_bat = torch.LongTensor(samples["a0_bat"].reshape(-1, 1)).to(device)
        reward_bat = torch.FloatTensor(samples["reward_bat"].reshape(-1, 1)).to(device)
        done_bat = torch.FloatTensor(samples["done_bat"].reshape(-1, 1)).to(device)
        s1_bat = torch.FloatTensor(samples["s1_bat"]).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        q0 = self.dqn(s0_bat).gather(1, a0_bat)
        q1 = self.dqn_target(s1_bat).max(dim=1, keepdim=True)[0].detach()  # 不计算目标网络的梯度
        mask = 1 - done_bat
        target = (reward_bat + self.gamma * q1 * mask).to(self.device)

        # 计算dqn损失
        loss = F.smooth_l1_loss(q0, target)

        return loss

    def _target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(self, steps_idx: int, scores, losses, epsilons):
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title(f'frame {steps_idx}. score: {np.mean(scores[-10:])}')
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.tight_layout()
        plt.show()
        clear_output(True)


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False   # = True 时，cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，
                                                  # 来达到优化运行效率的问题
        torch.backends.cudnn.deterministic = True

if '__name__' == '__main__':
    seed = 777
    np.random.seed(seed)
    seed_torch(seed)

    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    #超参数设置
    steps = 50000
    capacity = 1000
    batch_size = 32
    target_update_freq = 100
    epsilon_decay = 1/2000

    agent = DQNAgent(env, capacity, batch_size, target_update_freq, epsilon_decay)

    agent.train(steps)