import random

input_json = "/nfs/3x4090/data/workspace/dataset/DADA/sunny_ori.json"
output_json = "/nfs/3x4090/data/workspace/dataset/DADA/sunny.json"

# 读取原始数据
with open(input_json, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

# 随机抽取十分之一的索引
num_sample = max(1, len(lines) // 10)
selected_indices = set(random.sample(range(len(lines)), num_sample))

# 保持顺序写入
with open(output_json, 'w') as f:
    for i, line in enumerate(lines):
        if i in selected_indices:
            f.write(line + "\n")

print(f"原始数据 {len(lines)} 条，随机抽取 {num_sample} 条写入 {output_json}")
