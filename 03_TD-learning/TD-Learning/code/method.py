import numpy as np
from utils import get_reward
import random
import time
import os
import matplotlib
from matplotlib import pyplot as plt
from collections import deque

font = {"family":"MicroSoft YaHei",
       "weight":"bold",
       "size":"9"}
matplotlib.rc("font", **font)
# plt.rcParams['font.sans-serif'] = ['SimHei']

class Solver(object):
    """基类"""
    def __init__(self, r: int, c: int):
        """
        :param r: 代表地图行数
        :param c: 代表地图列数
        """
        # 动作空间
        self.idx_to_action = {0: '↑', 1: '↓', 2: '←', 3: '→', 4: 'O'}
        self.r, self.c, self.action_nums = r, c, len(self.idx_to_action) #地图行数、列数、动作个数
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

    def gradient_descent(self, x, y, w, alpha=0.01, epoch=100):
        for i in range(epoch):
            w = w - alpha * np.dot(x.T, np.dot(x, w) - y)
        return w

class Sarsa(Solver):
    def __init__(self, r: int, c: int):
        super(Sarsa, self).__init__(r, c)

    #更新动作价值矩阵，即Sarsa五要素：st,at,rt,st+1,at+1，以及alpha_k和gama
    def _update(self, cur_state, cur_action, reward, next_state, next_action, alpha_k=1e-1, gama=0.9):
        cur_row, cur_col = cur_state
        next_row, next_col = next_state
        cur_action_value  = self.action_value_matrix[cur_row, cur_col, cur_action]
        next_action_value = self.action_value_matrix[next_row, next_col, next_action]
        next_action_value = cur_action_value - alpha_k * (cur_action_value - reward - gama * next_action_value)
        self.action_value_matrix[cur_row, cur_col, cur_action] = next_action_value

    def update(self, epoch, graph, start_state=None, alpha_k=1e-1, gama=0.9):
        # 指定起点
        cur_state = start_state
        cache = []
        for i in range(epoch):
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
                j += 1 # j 表示每次迭代episode的长度

            cache.append(j) #存储每次迭代的episode的长度

        _, axs = plt.subplots(1, 1, figsize=(4, 3), dpi=150)
        self.mplot(list(range(len(cache))), cache, axs, 'b', 'Episode长度变化图', 'Episode index', 'Episode length', ['Sarsa'])
        # plt.savefig("./2.png", bbox_inches='tight')  # 这里有横轴截断问题
        plt.tight_layout()
        plt.show()

        # 最后收敛的状态价值矩阵
        self.state_value_matrix = np.max(self.action_value_matrix, axis=2)  # 状态价值矩阵
        # 最优策略
        self.cur_best_policy = np.argmax(self.action_value_matrix, axis=2)  # 当前最优策略
        self.cur_best_policy[cur_state[0], cur_state[1]] = 4  # '●'动作为原地转圈，在训练过程我们没有赋值

class Qlearning(Solver):
    def __init__(self, r, c):
        super(Qlearning, self).__init__(r, c)

    def _update(self, cur_state, cur_action, reward, next_state, next_action, alpha_k=0.1, gama=0.9):
        cur_row, cur_col = cur_state
        next_row, next_col = next_state
        cur_action_value = self.action_value_matrix[cur_row, cur_col, cur_action]
        next_action_value = np.max(self.action_value_matrix[next_row, next_col])  # 求最大的q值
        next_action_value = cur_action_value - alpha_k * (cur_action_value - reward - gama * next_action_value)
        self.action_value_matrix[cur_row, cur_col, cur_action] = next_action_value

    def update(self, epoch, graph, start_state=None, epsilon=1, alpha_k=0.1, gama=0.9):
        # 指定起点
        cur_state = start_state
        cache = []
        for i in range(epoch):
            # 起点随机生成
            if start_state is None:
                cur_state = (random.randint(0, self.r - 1), random.randint(0, self.c - 1))  # 随机生成初始状态
            else:
                cur_state = start_state
            cur_action = self.get_epsilon_greedy_action(cur_state, epsilon=epsilon)
            j = 0
            while graph[cur_state[0]][cur_state[1]] != '●':
                *next_state, reward = get_reward(cur_state, cur_action, graph)
                # epsilon为1的时候就是离线学习，否则就是在线学习
                next_action = self.get_epsilon_greedy_action(next_state, epsilon=epsilon)
                self._update(cur_state, cur_action, reward, next_state, next_action, alpha_k, gama)
                cur_state = next_state
                cur_action = next_action
                j += 1
            cache.append(j)

        _, axs = plt.subplots(1, 1, figsize=(4, 3), dpi=150)
        self.mplot(list(range(len(cache))), cache, axs, 'b', 'Episode长度变化图', 'Episode index', 'Episode length',
                   ['Qlearning'])
        # plt.savefig("./2.png", bbox_inches='tight')  # 这里有横轴截断问题
        plt.tight_layout()
        plt.show()

        # 最后收敛的状态价值矩阵
        self.state_value_matrix = np.max(self.action_value_matrix, axis=2)  # 状态价值矩阵
        # 最优策略
        self.cur_best_policy = np.argmax(self.action_value_matrix, axis=2)  # 当前最优策略
        self.cur_best_policy[cur_state[0], cur_state[1]] = 4  # '●'动作为原地转圈，在训练过程我们没有赋值


if __name__ == "__main__":
    graph = [['□', '□', '□', '□', '□', '□'],
             ['□', '×', '×', '□', '□', '×'],
             ['□', '□', '×', '□', '□', '×'],
             ['□', '×', '●', '×', '□', '×'],
             ['□', '×', '□', '□', '□', '×']]

    # graph = [['□', '□', '□'],
    #          ['□', '□', '×'],
    #          ['×', '□', '●']]

    # graph = [['□', '□', '□', '□', '□'],
    #          ['□', '×', '×', '□', '□'],
    #          ['□', '□', '×', '□', '□'],
    #          ['□', '×', '●', '×', '□'],
    #          ['□', '×', '□', '□', '□']]
    r = len(graph)
    c = len(graph[0])

    start_state = (0, 0)    #起点
    end_point = (3, 2)      #终点

    """Sarsa"""
    sarsa_iterator = Sarsa(r, c)
    sarsa_iterator.update(500, graph, start_state=start_state)  # 指定起点
    # sarsa_iterator.update(3000, graph)  # 随机起点
    sarsa_iterator.show_policy()
    sarsa_iterator.show_point_to_point(start_state, end_point, graph)

    """Qlearning"""
    # qlearn_iterator = Qlearning(r, c)#初始化
    # qlearn_iterator.update(800, graph, epsilon=0.1)
    # qlearn_iterator.show_policy()
