import random

# ===== 配置 =====
input_json = "/nfs/3x4090/data/workspace/dataset/BDDA-ALL/BDDA-frame/test_sunny.json"
output_json = "/nfs/3x4090/data/workspace/dataset/BDDA-ALL/BDDA-frame/test_sunny.json"

ratio = 0.2  # 抽取比例（十分之一）

# ===== 读取原文件 =====
with open(input_json, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# ===== 获取所有前四位编号 =====
prefixes = sorted({line[1:5] for line in lines if len(line) > 6 and line[1:5].isdigit()})

# ===== 随机抽取十分之一编号 =====
sample_size = max(1, int(len(prefixes) * ratio))
selected_prefixes = set(random.sample(prefixes, sample_size))

print(f"📦 共 {len(prefixes)} 个编号，随机选中 {len(selected_prefixes)} 个编号：{sorted(selected_prefixes)[:10]} ...")

# ===== 只保留这些编号对应的行 =====
filtered_lines = [line for line in lines if any(line.startswith(f"\"{rid}/") for rid in selected_prefixes)]

# ===== 写回新文件 =====
with open(output_json, "w") as f:
    f.write("\n".join(filtered_lines))

print(f"✅ 已生成新 JSON：{output_json}")
print(f"原始 {len(lines)} 行 → 保留 {len(filtered_lines)} 行")
