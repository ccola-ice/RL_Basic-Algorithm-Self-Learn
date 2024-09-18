def get_reward(location, action, graph):
    r, c = len(graph), len(graph[0])
    reward = -1  # 默认奖励为-1，是要求走最短路径
    row, col = location
    if action == 0:
        row = row - 1
    elif action == 1:
        row = row + 1
    elif action == 2:
        col = col - 1
    elif action == 3:
        col = col + 1

    if row < 0 or row > r - 1 or col < 0 or col > c - 1:
        reward = -10
    elif graph[row][col] == '×':
        reward = -10
    elif graph[row][col] == '●':
        reward = 1

    # 控制边界约束
    row = max(0, row)
    row = min(r - 1, row)
    col = max(0, col)
    col = min(c - 1, col)

    # 返回下一个状态以及奖励
    return row, col, reward