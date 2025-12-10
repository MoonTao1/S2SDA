from network.MINet import MINet_Res50  # 假设你有这个实现
from PIL import Image
import torchvision.transforms as T
import torch

# 1️⃣ 读取图片
img_path = "/data9102/workspace/mwt/dataset/night/trafficframe/05/000005.jpg"  # 替换成你的图片路径
img = Image.open(img_path).convert("RGB")  # 保证是三通道

# 2️⃣ 定义预处理
transform = T.Compose([
    T.Resize((320, 320)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])  # 官方训练归一化
])


# 3️⃣ 应用预处理
input_tensor = transform(img)  # [3,320,320]

# 4️⃣ 添加 batch 维度
input_tensor = input_tensor.unsqueeze(0)  # [1,3,320,320]

# 5️⃣ 放到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)

model = MINet_Res50()
model.load_state_dict(torch.load("/data9102/workspace/mwt/PODA/MINet_Res50.pth", map_location=device))
model.to(device)
model.eval()

# 推理
with torch.no_grad():
    pred_map = model(input_tensor)  # 输出 [B,1,H,W]
    binary_map = (pred_map[0, 0] > 0.5).float()  # 大于0.5为1，其余为0

# 可视化
import matplotlib
matplotlib.use('Agg')  # 或 'PDF', 'SVG' 等
import matplotlib.pyplot as plt

plt.imshow(binary_map.cpu(), cmap='gray')
plt.axis('off')
plt.show()
plt.savefig("example_binary.png", bbox_inches='tight', pad_inches=0)