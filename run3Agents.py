import os
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"

from ple.games.flappybird import FlappyBird
from ple import PLE
import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym
from parl.utils import logger
from datetime import datetime
import cv2

LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 200000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 32   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001 # 学习率
GAMMA = 0.99

class fc3Model(parl.Model):
    def __init__(self, act_dim):
        hid0_size = 64
        hid1_size = 32
        hid2_size = 16
        # 3层全连接网络
        self.fc0 = layers.fc(size=hid0_size, act='relu', name="fc0")
        self.fc1 = layers.fc(size=hid1_size, act='relu', name="fc1")
        self.fc2 = layers.fc(size=hid2_size, act='relu', name="fc2")
        self.fc3 = layers.fc(size=act_dim, act=None, name="fc3")

    def value(self, obs):
        # 定义网络
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
        h0 = self.fc0(obs)
        h1 = self.fc1(h0)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q

class fc2Model(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 32
        hid2_size = 16
        self.fc1 = layers.fc(size=hid1_size, act='relu', name="sfc1")
        self.fc2 = layers.fc(size=hid2_size, act='relu', name="sfc2")
        self.fc3 = layers.fc(size=act_dim, act=None, name="sfc3")

    def value(self, obs):
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q

class catModel(parl.Model):
    def __init__(self, act_dim):
        hid0_size = 64
        hid1_size = 32
        hid2_size = 16

        self.fc0 = layers.fc(size=hid0_size, act='relu', name="catfc0")
        self.fc1 = layers.fc(size=hid1_size, act='relu', name="catfc1")
        self.fc2 = layers.fc(size=hid2_size, act='relu', name="catfc2")
        self.fc3 = layers.fc(size=act_dim, act=None, name="catfc3")

    def value(self, last_obs, obs):
        oobs = fluid.layers.concat(input=[last_obs, obs], axis=-1, name='concat')
        
        h0 = self.fc0(oobs)
        h1 = self.fc1(h0)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q

class fcDQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ DQN algorithm
        
        Args:
            model (parl.Model): 定义Q函数的前向网络结构
            act_dim (int): action空间的维度，即有几个action
            gamma (float): reward的衰减因子
            lr (float): learning rate 学习率.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        """ 使用self.model的value网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_pred_value = self.target_model.value(next_obs)
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True  # 阻止梯度传递
        terminal = layers.cast(terminal, dtype='float32')
        target = reward + (1.0 - terminal) * self.gamma * best_v

        pred_value = self.model.value(obs)  # 获取Q预测值
        # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        #  ==> pred_action_value = [[3.9]]
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr)  # 使用Adam优化器
        optimizer.minimize(cost)
        return cost
    
    def sync_target(self):
        """ 把 self.model 的模型参数值同步到 self.target_model
        """
        self.model.sync_weights_to(self.target_model)

class catDQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ DQN algorithm
        
        Args:
            model (parl.Model): 定义Q函数的前向网络结构
            act_dim (int): action空间的维度，即有几个action
            gamma (float): reward的衰减因子
            lr (float): learning rate 学习率.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, last, obs):
        """ 使用self.model的value网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.model.value(last, obs)

    def learn(self, last_obs, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_pred_value = self.target_model.value(obs, next_obs)
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True  # 阻止梯度传递
        terminal = layers.cast(terminal, dtype='float32')
        target = reward + (1.0 - terminal) * self.gamma * best_v

        pred_value = self.model.value(last_obs, obs)  # 获取Q预测值
        # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        #  ==> pred_action_value = [[3.9]]
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr)  # 使用Adam优化器
        optimizer.minimize(cost)
        return cost
    
    def sync_target(self):
        """ 把 self.model 的模型参数值同步到 self.target_model
        """
        self.model.sync_weights_to(self.target_model)

class fc3Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(fc3Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim) 
            #act = 0 # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost

class fc2Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(fc2Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim) 
            #act = 0 # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost

class catAgent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(catAgent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            last_obs = layers.data(
                name='last_obs', shape=[self.obs_dim], dtype='float32') 
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(last_obs, obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            last_obs = layers.data(
                name='last_obs', shape=[self.obs_dim], dtype='float32')       
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(last_obs, obs, action, reward, next_obs, terminal)

    def sample(self, last_obs, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim) 
            #act = 0 # 探索：每个动作都有概率被选择
        else:
            act = self.predict(last_obs, obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, last_obs, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        last_obs = np.expand_dims(last_obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={
                'obs': obs.astype('float32'),
                'last_obs': last_obs.astype('float32')
            },
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, last_obs, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'last_obs': last_obs.astype('float32'),
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost

import random
import collections
import numpy as np

totale = 0

import sys
videoname = sys.argv[1]

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(agent1, agent2, agent3):
    input("开始比赛")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    
    frame_number = 0
    env = PLE(game, fps=30, display_screen=True)
    actionset = env.getActionSet()
    eval_reward = []
    
    for i in range(5):
        output_movie = cv2.VideoWriter(videoname+'_'+str(i)+'.mp4', fourcc, 20, (288, 512))
        env.init()
        env.reset_game()
        dstate = env.getGameState()
        # print(dstate)
        obs = list(dstate.values())
        
        last_obs = np.zeros_like(obs[0:8])
        episode_reward = 0
        while True:
            obs1 = obs[0:8]
            obs2 = obs[8:16]
            obs3 = obs[16:24]
            action1 = agent1.predict(obs1)
            action2 = agent2.predict(obs2)
            action3 = agent3.predict(last_obs,obs3)

            finalaction = 0
            if action1 == 0:
                finalaction += 1
            if action2 == 0:
                finalaction += 2
            if action3 == 0:
                finalaction += 4
            # print("action1: ", action1)
            # print("action2: ", action2)
            # print("action3: ", action3)
            # print("action: ", finalaction)
            # print(obs)
            # print(obs1)
            # print(obs2)
            # print(obs3)
            if finalaction == 0:
                finalaction = None
            score  = env.score()
            
            observation = env.getScreenRGB()
            observation = cv2.transpose(observation)
            font = cv2.FONT_HERSHEY_SIMPLEX
            observation = cv2.putText(observation, str(int(score)), (0, 25), font, 1.2, (255, 255, 255), 2)
            ss = observation.shape
            observation = cv2.resize(observation, (ss[1] * 2, ss[0] * 2))
            output_movie.write(observation)
            cv2.imshow("ss", observation)
            cv2.waitKey(30)  # 预测动作，只选最优动作

            reward = env.act(finalaction)
            last_obs = obs3
            dstate = env.getGameState()
            # print(dstate)
            obs = list(dstate.values())
            done = env.game_over()
            episode_reward += reward
            if done:
                break
            # input()
        eval_reward.append(episode_reward)
        cv2.destroyAllWindows()
        output_movie.release()
        input()
    return np.mean(eval_reward)


game = FlappyBird()
env = PLE(game, fps=30, display_screen=False)  # CartPole-v0: 预期最后一次评估总分 > 180（最大值是200）
action_dim = 2 # CartPole-v0: 2
obs_shape = len(env.getGameState()) // 3  # CartPole-v0: (4,)

print(env.getActionSet())

print(obs_shape)

# 根据parl框架构建agent
fc3model = fc3Model(act_dim=action_dim)
fc3algorithm = fcDQN(fc3model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
fc3agent = fc3Agent(fc3algorithm, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.1, e_greed_decrement=1e-6)

fc2model = fc2Model(act_dim=action_dim)
fc2algorithm = fcDQN(fc2model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
fc2agent = fc2Agent(fc2algorithm, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.1, e_greed_decrement=1e-6)

catmodel = catModel(act_dim=action_dim)
catalgorithm = catDQN(catmodel, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
catagent = catAgent(catalgorithm, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.1, e_greed_decrement=1e-6)

# 加载模型
save_path = './model3_8400_1256.2.ckpt'
fc3agent.restore(save_path)

save_path = './modelsm_16400_95.8.ckpt'
fc2agent.restore(save_path)

save_path = './modelconcat_30900_157.6.ckpt'
catagent.restore(save_path)

eval_reward = evaluate(fc3agent, fc2agent, catagent)  # render=True 查看显示效果
#logger.info('episode:{}    time:{}    e_greed:{}   test_reward:{}'.format(1, (end-start).seconds, agent.e_greed, eval_reward))