import numpy as np
import json
ord = 2

# aa = np.random.randn(1, (ord+1)**3) * 0.001  # 权重初始化
# print(aa)

# 保存权重
# with open('weight.json', 'w') as f:
#     json.dump(aa.tolist(), f)

# 读取权重
with open('weight.json', 'r') as f:
    bb = np.array(json.load(f))

print(bb)