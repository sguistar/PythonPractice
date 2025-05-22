import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# ================= Prioritized Replay Buffer ====================
class SumTree:
    def __init__(self, capacity):
        self.cap = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        self.size = 0

    def add(self, priority, data):
        idx = self.ptr + self.cap - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size+1, self.cap)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1)//2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while True:
            left = 2*idx + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = idx
                break
            else:
                idx = left if s <= self.tree[left] else right
                if idx == right:
                    s -= self.tree[left]
        data_idx = leaf - self.cap + 1
        return leaf, self.tree[leaf], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-5

    def store(self, transition):
        priority = (abs(transition[2]) + self.epsilon) ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size, beta=0.4):
        batch, idxs, priorities = [], [], []
        seg = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = seg * i, seg * (i+1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        sampling_prob = np.array(priorities) / self.tree.total
        weights = (self.tree.size * sampling_prob) ** (-beta)
        weights /= weights.max()
        batch = list(zip(*batch))
        return [np.vstack(x) for x in batch], idxs, weights

    def update_priorities(self, idxs, td_errors):
        for idx, td in zip(idxs, td_errors):
            p = (abs(td) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

# ================= Dueling Network ====================
class DuelingDQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = self.feature(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + (a - a.mean(dim=1, keepdim=True))

# ================= Agent ====================
class RainbowAgent:
    def __init__(self, env, capacity=10000, batch_size=64,
                 gamma=0.99, lr=1e-3, target_update=1000):
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.policy_net = DuelingDQN(obs_dim, n_actions).to(self.device)
        self.target_net = DuelingDQN(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.min_eps = 0.1
        self.eps_decay = 1e-4
        self.learn_step = 0
        self.target_update = target_update

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state)
        return q.argmax(1).item()

    def store_transition(self, s, a, r, s2, done):
        self.buffer.store((s, a, r, s2, done))

    def update(self, beta=0.4):
        samples, idxs, weights = self.buffer.sample(self.batch_size, beta)
        s, a, r, s2, done = [torch.FloatTensor(x).to(self.device) for x in samples]
        a = a.long()  # shape: (batch_size, 1)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        q_eval = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            # Double DQN target
            next_actions = self.policy_net(s2).argmax(1, keepdim=True)
            q_next = self.target_net(s2).gather(1, next_actions)
            q_target = r + self.gamma * q_next * (1 - done)
        td_error = q_target - q_eval
        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities
        self.buffer.update_priorities(idxs, td_error.detach().cpu().numpy().flatten())

        # epsilon decay
        self.epsilon = max(self.min_eps, self.epsilon - self.eps_decay)

        # target network update
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, num_frames=200000):
        state, _ = self.env.reset()
        episode_reward = 0
        for frame in range(1, num_frames+1):
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.store_transition(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if self.buffer.tree.size >= self.batch_size:
                self.update(beta=min(1.0, 0.4 + frame*(1.0-0.4)/num_frames))

            if done:
                state, _ = self.env.reset()
                print(f"Frame: {frame}, Episode Reward: {episode_reward}, Epsilon: {self.epsilon:.3f}")
                episode_reward = 0
        print('Completed Training')


# ================= Main ====================
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    agent = RainbowAgent(env)
    agent.train()
    env.close()
