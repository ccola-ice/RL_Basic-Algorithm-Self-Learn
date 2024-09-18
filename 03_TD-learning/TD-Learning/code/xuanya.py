import argparse
import datetime
import math
import os
import time
import turtle
from collections import defaultdict
import dill
import gym
# 这里需要改成自己的RL_Utils.py文件的路径
from RL_Utils import *


# 悬崖行走地图
class CliffWalkingWapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.t = None
        self.unit = 50
        self.max_x = 12
        self.max_y = 4

    def draw_x_line(self, y, x0, x1, color='gray'):
        assert x1 > x0
        self.t.color(color)
        self.t.setheading(0)
        self.t.up()
        self.t.goto(x0, y)
        self.t.down()
        self.t.forward(x1 - x0)

    def draw_y_line(self, x, y0, y1, color='gray'):
        assert y1 > y0
        self.t.color(color)
        self.t.setheading(90)
        self.t.up()
        self.t.goto(x, y0)
        self.t.down()
        self.t.forward(y1 - y0)

    def draw_box(self, x, y, fillcolor='', line_color='gray'):
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for i in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()

    def move_player(self, x, y):
        self.t.up()
        self.t.setheading(90)
        self.t.fillcolor('red')
        self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)

    def render(self):
        if self.t == None:
            self.t = turtle.Turtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.max_x + 100,
                          self.unit * self.max_y + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.max_x,
                                        self.unit * self.max_y)
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('gray')
            for _ in range(2):
                self.t.forward(self.max_x * self.unit)
                self.t.left(90)
                self.t.forward(self.max_y * self.unit)
                self.t.left(90)
            for i in range(1, self.max_y):
                self.draw_x_line(
                    y=i * self.unit, x0=0, x1=self.max_x * self.unit)
            for i in range(1, self.max_x):
                self.draw_y_line(
                    x=i * self.unit, y0=0, y1=self.max_y * self.unit)

            for i in range(1, self.max_x - 1):
                self.draw_box(i, 0, 'black')
            self.draw_box(self.max_x - 1, 0, 'yellow')
            self.t.shape('turtle')

        x_pos = self.s % self.max_x
        y_pos = self.max_y - 1 - int(self.s / self.max_x)
        self.move_player(x_pos, y_pos)


# Q_Learning智能体对象
class Q_Learning:
    def __init__(self, arg_dict):
        # 采样次数
        self.sample_count = 0
        # 动作数
        self.n_actions = arg_dict['n_actions']
        # 学习率
        self.lr = arg_dict['lr']
        # 未来奖励衰减系数
        self.gamma = arg_dict['gamma']
        # 当前的epsilon值
        self.epsilon = arg_dict['epsilon_start']
        # 初始的epsilon值
        self.epsilon_start = arg_dict['epsilon_start']
        # 最后的epsilon值
        self.epsilon_end = arg_dict['epsilon_end']
        # epsilon衰变参数
        self.epsilon_decay = arg_dict['epsilon_decay']
        # 使用嵌套字典表示Q（s，a），这里首先将所有Q（s、a）设置为0
        self.Q_table = defaultdict(lambda: np.zeros(self.n_actions))

    # 训练过程: 用e-greedy policy获取行动
    def sample_action(self, state):
        # 采样数更新
        self.sample_count += 1
        # 计算当前epsilon值
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.sample_count / self.epsilon_decay)
        # 根据均匀分布获取一个0-1的随机值，如果随机值大于当前epsilon，则按照最大Q值来选择动作，否则随机选择一个动作
        return np.argmax(self.Q_table[str(state)]) if np.random.uniform(0, 1) > self.epsilon else np.random.choice(
            self.n_actions)

    # 测试过程: 用最大Q值获取行动
    def predict_action(self, state):
        return np.argmax(self.Q_table[str(state)])

    # 更新Q表格
    def update(self, state, action, reward, next_state, done):
        # 计算Q估计
        Q_predict = self.Q_table[str(state)][action]
        # 计算Q现实
        if done:
            # 如果回合结束，则直接等于当前奖励
            Q_target = reward
        else:
            # 如果回合每结束，则按照
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)])
        # 根据Q估计和Q现实，差分地更新Q表格
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)

    # 保存模型
    def save_model(self, path):
        # 如果路径不存在，就自动创建
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(
            obj=self.Q_table,
            f=path + "checkpoint.pkl",
            pickle_module=dill
        )

    # 加载模型
    def load_model(self, path):
        self.Q_table = torch.load(f=path + 'checkpoint.pkl', pickle_module=dill)


# 训练函数
def train(arg_dict, env, agent):
    # 开始计时
    startTime = time.time()
    print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
    print("开始训练智能体......")
    # 记录每个epoch的奖励
    rewards = []
    # 记录每个epoch的智能体到达终点用的步数
    steps = []
    for epoch in range(arg_dict['train_eps']):
        # 每个epoch的总奖励
        ep_reward = 0
        # 每个epoch的步数记录器
        ep_step = 0
        # 重置环境，并获取初始状态
        state = env.reset()
        while True:
            # 画图
            if arg_dict['train_render']:
                env.render()
            # 根据e-贪心策略获取动作
            action = agent.sample_action(state)
            # 执行动作，获得下一个状态、奖励和是否结束当前回合的标志，并更新环境
            next_state, reward, done, _ = env.step(action)
            # 智能体更新，根据当前状态和动作、下一个状态和奖励，改进Q函数
            agent.update(state, action, reward, next_state, done)
            # 更新当前状态为下一时刻状态
            state = next_state
            # 累加记录奖励
            ep_reward += reward
            # 步数+1
            ep_step += 1
            # 如果当前回合结束，则跳出循环
            if done:
                break
        # 记录奖励、步数信息
        rewards.append(ep_reward)
        steps.append(ep_step)
        # 每隔10次迭代就输出一次
        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch: {epoch + 1}/{arg_dict["train_eps"]}, Reward: {ep_reward:.2f}, Steps:{ep_step}, Epislon: {agent.epsilon:.3f}')
    print("智能体训练结束 , 用时: " + str(time.time() - startTime) + " s")
    return {'epochs': range(len(rewards)), 'rewards': rewards, 'steps': steps}


# 测试函数
def test(arg_dict, env, agent):
    startTime = time.time()
    print("开始测试智能体......")
    print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
    # 记录每个epoch的奖励
    rewards = []
    # 记录每个epoch的智能体到达终点用的步数
    steps = []
    for epoch in range(arg_dict['test_eps']):
        # 每个epoch的总奖励
        ep_reward = 0
        # 每个epoch的步数记录器
        ep_step = 0
        # 重置环境，并获取初始状态
        state = env.reset()
        while True:
            # 画图
            if arg_dict['test_render']:
                env.render()
            # 根据最大Q值选择动作
            action = agent.predict_action(state)
            # 执行动作，获得下一个状态、奖励和是否结束当前回合的标志，并更新环境
            next_state, reward, done, _ = env.step(action)
            # 更新当前状态为下一时刻状态
            state = next_state
            # 累加记录奖励
            ep_reward += reward
            # 步数+1
            ep_step += 1
            # 如果当前回合结束，则跳出循环
            if done:
                break
        # 记录奖励、步数信息
        rewards.append(ep_reward)
        steps.append(ep_step)
        # 输出测试信息
        print(f"Epochs: {epoch + 1}/{arg_dict['test_eps']}, Steps:{ep_step}, Reward: {ep_reward:.2f}")
    print("测试结束 , 用时: " + str(time.time() - startTime) + " s")
    return {'episodes': range(len(rewards)), 'rewards': rewards, 'steps': steps}


# 创建环境和智能体
def create_env_agent(arg_dict):
    # 创建环境
    env = gym.make(arg_dict['env_name'])
    env = CliffWalkingWapper(env)
    # 设置随机种子
    all_seed(env, seed=arg_dict["seed"])
    # 获取状态数
    try:
        n_states = env.observation_space.n
    except AttributeError:
        n_states = env.observation_space.shape[0]
    # 获取动作数
    n_actions = env.action_space.n
    print(f"状态数: {n_states}, 动作数: {n_actions}")
    # 将状态数和动作数加入算法参数字典
    arg_dict.update({"n_states": n_states, "n_actions": n_actions})
    # 实例化智能体对象
    agent = Q_Learning(arg_dict)
    # 返回环境，智能体
    return env, agent


if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 获取当前路径
    curr_path = os.path.dirname(os.path.abspath(__file__))
    # 获取当前时间
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    # 相关参数设置
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='Q-learning', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='CliffWalking-v0', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=400, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.90, type=float, help="discounted factor")
    parser.add_argument('--epsilon_start', default=0.95, type=float, help="initial value of epsilon")
    parser.add_argument('--epsilon_end', default=0.01, type=float, help="final value of epsilon")
    parser.add_argument('--epsilon_decay', default=300, type=int, help="decay rate of epsilon")
    parser.add_argument('--lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('--device', default='cpu', type=str, help="cpu or cuda")
    parser.add_argument('--seed', default=520, type=int, help="seed")
    parser.add_argument('--show_fig', default=False, type=bool, help="if show figure or not")
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    parser.add_argument('--train_render', default=False, type=bool,
                        help="Whether to render the environment during training")
    parser.add_argument('--test_render', default=True, type=bool,
                        help="Whether to render the environment during testing")
    args = parser.parse_args()
    default_args = {'result_path': f"{curr_path}/outputs/{args.env_name}/{curr_time}/results/",
                    'model_path': f"{curr_path}/outputs/{args.env_name}/{curr_time}/models/",
                    }
    # 将参数转化为字典 type(dict)
    arg_dict = {**vars(args), **default_args}
    print("算法参数字典:", arg_dict)

    # 创建环境和智能体
    env, agent = create_env_agent(arg_dict)
    # 传入算法参数、环境、智能体，然后开始训练
    res_dic = train(arg_dict, env, agent)
    print("算法返回结果字典:", res_dic)
    # 保存相关信息
    agent.save_model(path=arg_dict['model_path'])
    save_args(arg_dict, path=arg_dict['result_path'])
    save_results(res_dic, tag='train', path=arg_dict['result_path'])
    plot_rewards(res_dic['rewards'], arg_dict, path=arg_dict['result_path'], tag="train")

    # =================================================================================================
    # 创建新环境和智能体用来测试
    print("=" * 300)
    env, agent = create_env_agent(arg_dict)
    # 加载已保存的智能体
    agent.load_model(path=arg_dict['model_path'])
    res_dic = test(arg_dict, env, agent)
    save_results(res_dic, tag='test', path=arg_dict['result_path'])
    plot_rewards(res_dic['rewards'], arg_dict, path=arg_dict['result_path'], tag="test")
