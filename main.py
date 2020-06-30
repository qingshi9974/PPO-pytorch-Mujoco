import gym
import torch
import numpy as np
import argparse
from parameters import *
from PPO import Ppo
from collections import deque
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="Humanoid-v2",
                    help='name of Mujoco environement')
args = parser.parse_args()


env = gym.make(args.env_name)
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
#初始化随机种子
env.seed(500)
torch.manual_seed(500)
np.random.seed(500)
##状态的归一化
class Nomalize:
    def __init__(self, N_S):
        self.mean = np.zeros((N_S,))
        self.std = np.zeros((N_S, ))
        self.stdd = np.zeros((N_S, ))
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            #更新样本均值和方差
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.stdd = self.stdd + (x - old_mean) * (x - self.mean)
            #状态归一化
        if self.n > 1:
            self.std = np.sqrt(self.stdd / (self.n - 1))
        else:
            self.std = self.mean

        x = x - self.mean

        x = x / (self.std + 1e-8)

        x = np.clip(x, -5, +5)


        return x



ppo = Ppo(N_S,N_A)
nomalize = Nomalize(N_S)
episodes = 0
eva_episodes = 0
for iter in range(Iter):
    memory = deque()
    scores = []
    steps = 0
    while steps <2048: #Horizen
        episodes += 1
        s = nomalize(env.reset())
        score = 0
        for _ in range(MAX_STEP):
            steps += 1
            #选择行为
            a=ppo.actor_net.choose_action(torch.from_numpy(np.array(s).astype(np.float32)).unsqueeze(0))[0]
            s_ , r ,done,info = env.step(a)
            s_ = nomalize(s_)

            mask = (1-done)*1
            memory.append([s,a,r,mask])

            score += r
            s = s_
            if done:
                break
        with open('log_' + args.env_name  + '.txt', 'a') as outfile:
            outfile.write('\t' + str(episodes)  + '\t' + str(score) + '\n')
        scores.append(score)
    score_avg = np.mean(scores)
    print('{} episode score is {:.2f}'.format(episodes, score_avg))
    #每隔一定的timesteps 进行参数更新
    ppo.train(memory)





