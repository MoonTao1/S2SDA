import os
import shutil

# ======== 参数设置 ========
src_root = "/nfs/3x4090/data/workspace/dataset/BDDA-ALL/BDDA-salmm/gazemap_frames"
dst_root = "/nfs/3x4090/data/workspace/dataset/BDDA-ALL/BDDA-salmm/gazemap_frames/"
start_index = 4201  # 新编号起始

# ======== 要复制的文件夹编号（自己设定） ========
folder_ids = ["1013", "1167", "1215", "1390", "1461",
              "1551", "1588", "1653", "1732",]

# ======== 确保目标目录存在 ========
os.makedirs(dst_root, exist_ok=True)

# ======== 执行复制 ========
count = 0
for fid in folder_ids:
    src_path = os.path.join(src_root, fid)
    if not os.path.exists(src_path):
        print(f"⚠️ 未找到文件夹: {src_path}，跳过")
        continue

    new_name = f"{start_index + count:04d}"
    dst_path = os.path.join(dst_root, new_name)

    print(f"📂 复制 {fid} → {new_name}")
    shutil.copytree(src_path, dst_path)
    count += 1

print(f"\n✅ 完成！共复制 {count} 个文件夹到 {dst_root}")
