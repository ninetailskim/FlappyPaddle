import os
from ple.games.flappybird import FlappyBird
from ple import PLE
import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import time
from datetime import datetime
import cv2
import sys

LEARN_FREQ = 5
MEMORY_SIZE = 200000
MEMORY_WARMUP_SIZE = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
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

def main(width=200, height=400):
    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=True)
    action_dim = 2 
    print(env.getGameState())
    obs_shape = len(env.getGameState()) // 2

    fc3model = fc3Model(act_dim=action_dim)
    fc3algorithm = fcDQN(fc3model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent1 = fc3Agent(fc3algorithm, obs_dim=obs_shape, act_dim=action_dim, e_greed=0.1, e_greed_decrement=1e-6)

    save_path = './model3_8400_1256.2.ckpt'
    agent1.restore(save_path)

    while True:
        env.init()
        env.reset_game()
        obs = list(env.getGameState().values())       
        
        while True:
            obs1 = obs[0:8]
            action1 = agent1.predict(obs1)

            finalaction = 1 if action1 == 0 else 0

            print("Obs:",[int(obst) for obst in obs1]," Action:","跳" if finalaction == 1 else "什么都不做")
            env.act(finalaction)
            obs = list(env.getGameState().values())
            done = env.game_over()
            if done:
                break
            cvob = env.getScreenRGB()
            cvob = cv2.transpose(cvob)
            cvob = cv2.cvtColor(cvob, cv2.COLOR_BGR2RGB)
            cvob = cv2.resize(cvob, (width, height))
            cv2.imshow("FlappyPaddle", cvob)
            cv2.waitKey(50)

if __name__ == "__main__":
    width = 200
    height = 600
    if len(sys.argv) > 2:
        width = int(sys.argv[1])
        height = int(sys.argv[2])
    main(width, height)