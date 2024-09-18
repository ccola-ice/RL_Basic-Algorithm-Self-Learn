import numpy as np
from utils import get_reward
import random
import time
import os
import matplotlib
from matplotlib import pyplot as plt
import torch
import json

"""默认不显示支持中文，现在设置格式使其可以显示中文"""
font = {"family":"MicroSoft YaHei",
       "weight":"bold",
       "size":"9"}
matplotlib.rc("font", **font)
# plt.rcParams['font.sans-serif'] = ['SimHei']

# random.seed(0)
# np.random.seed(4)
# torch.manual_seed(0)


class Solver(object):
    """基类"""
    def __init__(self, r: int, c: int):
        """
        :param r: 代表地图行数
        :param c: 代表地图列数
        """
        # 动作空间
        self.idx_to_action = {0: '↑', 1: '↓', 2: '←', 3: '→', 4: 'O'}
        # 地图行数、列数、动作个数
        self.r, self.c, self.action_nums = r, c, len(self.idx_to_action)
        self.state_value_matrix = np.random.randn(r, c)  # 状态价值矩阵初始化
        self.action_value_matrix = np.random.randn(r, c, len(self.idx_to_action))  # 动作价值矩阵初始化
        self.cur_best_policy = np.random.choice(len(self.idx_to_action), size=(r, c))  # 当前最优策略初始化

    # 打印最优策略
    def show_policy(self):
        for i in self.cur_best_policy.tolist():
            print(*[self.idx_to_action[idx] for idx in i], sep=' ')

    # 显示地图
    def _show_graph(self, graph):
        for i in graph:
            print(*i, sep=' ')

    # 清空控制台
    def _clear_console(self):
        if os.name == 'nt':  # for Windows
            _ = os.system('cls')
        else:  # for Linux and Mac
            _ = os.system('clear')

    # 动态显示
    def show_point_to_point(self, start_point, end_point, graph):
        assert (0 <= start_point[0] < self.r) and (0 <= start_point[1] < self.c), f'The start_point is {start_point}, is out of range.'
        assert (0 <= end_point[0] < self.r) and (0 <= end_point[1] < self.c), f'The end_point is {end_point}, is out of range.'

        row, col = start_point
        i = 0
        while True:
            graph[row][col] = self.idx_to_action[self.cur_best_policy[row][col]]
            self._clear_console()  # 清空控制台
            self._show_graph(graph)  # 显示地图
            time.sleep(0.5)
            row, col, _ = get_reward((row, col), self.cur_best_policy[row][col], graph)
            if (row, col) == end_point or i > self.r * self.c:
                break
            i += 1

    # epsilon贪婪法，当epsilon=0时，完全贪婪法
    def get_epsilon_greedy_action(self, state, epsilon=0.1):
        row, col = state
        # 找最优动作
        best_action = np.argmax(self.action_value_matrix[row][col]).item()
        # epsilon贪婪法，当epsilon != 0时，才有可能进入该if语句，否则直接返回最优动作
        if random.random() < epsilon * (self.action_nums - 1) / self.action_nums:
            actions = list(self.idx_to_action.keys())
            actions.remove(best_action)
            return random.choice(actions)
        return best_action

    def mplot(self, x, y, ax, fmt, title, x_label, y_label, legend):
        ax.plot(x, y, fmt)
        ax.set_xlim(x[0], x[-1] + 0.4)  # 设置X轴范围
        # ax.set_xticks(x)  # 用于设置X轴上要显示的刻度值
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(legend)
        ax.set_title(title)


class Sarsa(Solver):
    def __init__(self, r: int, c: int, ord):
        super(Sarsa, self).__init__(r, c)
        self.ord = ord  # 傅里叶级数阶数
        self.w = np.random.randn(1, (ord+1)**3) * 0.001  # 权重初始化

    def get_feature(self, state, action):
        """
        :param state: 状态
        :param ord: 特征函数最高阶次数/傅里叶q(对应书)
        :return: 代入state后的计算结果
        """
        feature_vector = []
        row, col = state
        # 归一化到[0,1]之间
        x_normalized = (col + 1) / self.c
        y_normalized = (row + 1) / self.r
        action_normalized = (action + 1) / self.action_nums
        for i in range(self.ord+1):
            for j in range(self.ord+1):
                for k in range(self.ord+1):
                    feature_vector.append(np.cos(np.pi * (i * x_normalized + j * action_normalized + k * y_normalized)))
        return np.array(feature_vector).reshape(-1, 1)

    def action_value_fun(self, x):
        self.w = np.array(self.w).reshape(1, -1)
        x = np.array(x).reshape(-1, 1)
        return (np.dot(self.w, x))[0][0]

    def get_epsilon_greedy_action(self, state, epsilon=0.1):
        action_value = []
        for i in range(len(self.idx_to_action)):
            x = self.get_feature(state, i)
            action_value.append(self.action_value_fun(x))
        best_action = np.argmax(np.array(action_value)).item()

        if random.random() < epsilon * (self.action_nums - 1) / self.action_nums:
            actions = list(self.idx_to_action.keys())
            actions.remove(best_action)
            return random.choice(actions)
        return best_action

    def _update(self, cur_state, cur_action, reward, next_state, next_action, alpha_k=1e-3, gama=0.9):
        cur_x = self.get_feature(cur_state, cur_action)
        next_x = self.get_feature(next_state, next_action)
        t = alpha_k * (reward + gama * self.action_value_fun(next_x) - self.action_value_fun(cur_x))
        self.w = self.w + t * cur_x.reshape(1, -1)

    def update(self, epoch, graph, start_state=None, alpha_k=1e-3, gama=0.9):
        # 指定起点
        cur_state = start_state
        cache, err = [], []
        for i in range(epoch):
            if (i+1)%100 == 0:
                print(f"第{i+1}次迭代")
            # 起点随机生成
            if start_state is None:
                cur_state = (random.randint(0, self.r - 1), random.randint(0, self.c - 1))  # 随机生成初始状态
            else:
                cur_state = start_state
            cur_action = self.get_epsilon_greedy_action(cur_state)
            j = 0
            while graph[cur_state[0]][cur_state[1]] != '●':
                *next_state, reward = get_reward(cur_state, cur_action, graph)
                next_action = self.get_epsilon_greedy_action(next_state)
                self._update(cur_state, cur_action, reward, next_state, next_action, alpha_k, gama)
                cur_state = next_state
                cur_action = next_action
                j += 1
                if j>100:
                    # break
                    pass
            cache.append(j)

        """保存权重"""
        with open('weight.json', 'w') as f:
            json.dump(self.w.tolist(), f)
        # _, axs = plt.subplots(1, 1, figsize=(4, 3), dpi=150)
        # self.mplot(list(range(len(cache))), cache, axs, 'b', 'Episode长度变化图', 'Episode index', 'Episode length', ['Sarsa'])
        # plt.savefig("./2.png", bbox_inches='tight')  # 这里有横轴截断问题
        # plt.tight_layout()
        # plt.show()

        for i in range(self.r):
            for j in range(self.c):
                for k in range(self.action_nums):
                    x = self.get_feature([i, j], k)
                    self.action_value_matrix[i][j][k] = self.action_value_fun(x)
        # 最后收敛的状态价值矩阵
        self.state_value_matrix = np.max(self.action_value_matrix, axis=2)  # 状态价值矩阵
        # 最优策略
        self.cur_best_policy = np.argmax(self.action_value_matrix, axis=2)  # 当前最优策略
        self.cur_best_policy[cur_state[0], cur_state[1]] = 4  # '●'动作为原地转圈，在训练过程我们没有赋值

    def load_model(self, path="./weight.json"):
        with open(path, 'r') as f:
            self.w = np.array(json.load(f)).reshape(1, -1)

        for i in range(self.r):
            for j in range(self.c):
                for k in range(self.action_nums):
                    x = self.get_feature([i, j], k)
                    self.action_value_matrix[i][j][k] = self.action_value_fun(x)
        # 最后收敛的状态价值矩阵
        self.state_value_matrix = np.max(self.action_value_matrix, axis=2)  # 状态价值矩阵
        # 最优策略
        self.cur_best_policy = np.argmax(self.action_value_matrix, axis=2)  # 当前最优策略
        self.cur_best_policy[3, 2] = 4  # '●'动作为原地转圈，在训练过程我们没有赋值

class DQN(Solver):
    def __init__(self, r: int, c: int, n_features: int, hidden_dims: int, learning_rate: float):
        super(DQN, self).__init__(r, c)
        """目标网络，延迟更新"""
        self.target_model = torch.nn.Sequential(torch.nn.Linear(n_features, hidden_dims), torch.nn.ReLU(),
                                                torch.nn.Linear(hidden_dims, hidden_dims), torch.nn.ReLU(),
                                                torch.nn.Linear(hidden_dims, self.action_nums))
        """评估网络, 频繁更新"""
        self.eval_model = torch.nn.Sequential(torch.nn.Linear(n_features, hidden_dims), torch.nn.ReLU(),
                                              torch.nn.Linear(hidden_dims, hidden_dims), torch.nn.ReLU(),
                                              torch.nn.Linear(hidden_dims, self.action_nums))
        self.loss_fn = torch.nn.MSELoss()  # 损失函数
        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=learning_rate)
        self.datas = []  # 记录数据

    def get_feature(self, state):
        """
        :param state: 状态
        :return: 代入state后归一化的结果
        """
        row, col = state
        # 归一化到[0,1]之间
        x_normalized = (col + 1) / self.c
        y_normalized = (row + 1) / self.r
        data = [x_normalized, y_normalized]
        return torch.tensor(data, dtype=torch.float32).reshape(-1, 2)

    def update_data(self, graph, start_state=None):
        # 起点随机生成
        if start_state is None:
            cur_state = (random.randint(0, self.r - 1), random.randint(0, self.c - 1))  # 随机生成初始状态
        else:
            cur_state = start_state  # 指定起点

        cur_action = self.get_epsilon_greedy_action(cur_state)
        j = 0
        while graph[cur_state[0]][cur_state[1]] != '●' or len(self.datas) < 100:
            *next_state, reward = get_reward(cur_state, cur_action, graph)
            next_action = self.get_epsilon_greedy_action(next_state)
            self.datas.append((cur_state, cur_action, reward, next_state))
            cur_state = next_state
            cur_action = next_action
            j += 1
            if j > 100:
                break
        self.datas=self.datas[-1000:]  # 只保留最近的1000个数据

    # epsilon等于1，采用离线学习
    def get_epsilon_greedy_action(self, state, epsilon=1):
        x = self.get_feature(state)  # 特征向量
        with torch.no_grad():
            out_put = self.eval_model(x.reshape(-1, 2))  # 输出所有的动作价值
        best_action = torch.argmax(out_put).item()  # 选取最优动作

        if random.random() < epsilon * (self.action_nums - 1) / self.action_nums:
            actions = list(self.idx_to_action.keys())
            actions.remove(best_action)
            return random.choice(actions)
        return best_action

    def sample_data(self, batch_size=32):
        data = random.sample(self.datas, batch_size)  # 随机采样
        cur_states = torch.stack([self.get_feature(i[0])[0] for i in data], dim=0) # 当前状态
        cur_actions = torch.tensor([i[1] for i in data])  # 当前动作
        rewards = torch.tensor([i[2] for i in data])  # 奖励
        next_states = torch.stack([self.get_feature(i[3])[0] for i in data], dim=0)  # 下一个状态
        return cur_states, cur_actions, rewards, next_states

    def update_model(self, epoch, graph, start_state=None, gama=0.9):
        loss_cache = []  # 记录损失
        for i in range(epoch):
            self.update_data(graph, start_state)  # 更新数据
            loss = 0
            for j in range(200):
                cur_states, cur_actions, rewards, next_states = self.sample_data()  # 采样数据
                value = self.eval_model(cur_states).gather(1, cur_actions.reshape(-1, 1))  # 当前状态价值
                with torch.no_grad():
                    target_value = self.target_model(next_states).max(dim=1)[0].reshape(-1,1)  # 下一个状态价值
                target_value = rewards.reshape(-1, 1) + gama * target_value  # 目标价值
                loss = self.loss_fn(value, target_value)  # 计算损失
                self.optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数
            loss_cache.append(loss.item())

            # i每增加5更新一次目标网络模型参数
            if (i+1) % 5 == 0:
                self.target_model.load_state_dict(self.eval_model.state_dict())  # 更新目标网络

            print(f"epoch:{i+1}, loss:{loss.item()}")

        # 打印策略
        self.show_policy(graph)
        # 保存模型参数
        torch.save(self.eval_model.state_dict(), "model.pkl")
        _, axs = plt.subplots(1, 1, figsize=(4, 3), dpi=150)
        self.mplot(list(range(len(loss_cache))), loss_cache, axs, 'b', 'loss变化图', 'epoch', 'loss',['loss曲线'])
        # plt.savefig("./2.png", bbox_inches='tight')  # 这里有横轴截断问题
        plt.tight_layout()
        plt.show()

    # 加载模型
    def load_model(self, path=None):
        assert path is not None, "path is None!!!"
        self.eval_model.load_state_dict(torch.load(path))

    def show_policy(self, graph=None):
        assert graph is not None, "graph is None!!!"
        # 打印策略
        with torch.no_grad():
            for i in range(self.r):
                for j in range(self.c):
                    if graph[i][j] != '●':
                        print(self.idx_to_action[self.get_epsilon_greedy_action((i, j), epsilon=0)], end=" ")
                    else:
                        print('●', end=" ")
                print()


if __name__ == "__main__":

    # graph = [['□', '□', '□', '□', '□', '□'],
    #          ['□', '×', '×', '□', '□', '×'],
    #          ['□', '□', '×', '□', '□', '×'],
    #          ['□', '×', '●', '×', '□', '×'],
    #          ['□', '×', '□', '□', '□', '×']]

    # graph = [['□', '□', '□'],
    #          ['□', '□', '×'],
    #          ['×', '□', '●']]

    graph = [['□', '□', '□', '□', '□'],
             ['□', '×', '×', '□', '□'],
             ['□', '□', '×', '□', '□'],
             ['□', '×', '●', '×', '□'],
             ['□', '×', '□', '□', '□']]
    r = len(graph)
    c = len(graph[0])

    start_state = (0, 0)
    end_point = (3, 2)  # 改地图的时候注意改终点坐标位置
    ord = 3  # 傅里叶阶数

    """Sarsa"""
    sarsa_iterator = Sarsa(r, c, 3)
    sarsa_iterator.update(1000, graph, start_state=start_state)  # 指定起点
    sarsa_iterator.update(75000, graph)  # 随机起点
    # sarsa_iterator.load_model("./SARSA_weight/weight_one.json")  # 指定起点为(0,0),加载训练好的参数
    # sarsa_iterator.load_model("./SARSA_weight/weight_all.json")  # 未指定起点，加载参数
    sarsa_iterator.show_policy()
    sarsa_iterator.show_point_to_point(start_state, end_point, graph)

    """DQN"""
    # dqn_iterator = DQN(r, c, 2, 64, 5e-3)
    # dqn_iterator.update_model(300, graph)
    # dqn_iterator.load_model("./model.pkl")  # 加载模型，自己没录视频训练的
    # # dqn_iterator.load_model("./DQN_model/model_v2.pkl")  # 录视频的时候训练的
    # dqn_iterator.show_policy(graph)
