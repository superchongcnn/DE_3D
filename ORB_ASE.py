import cv2
import numpy as np
import torch

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LocalGlobalDepthNet().to(device)
model.eval()  # 进入推理模式

def estimate_depth(image_path):
    # 读取 ORB-SLAM 关键帧图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))  # 调整大小
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # 归一化
    image = image.unsqueeze(0).to(device)

    # 计算深度图
    with torch.no_grad():
        depth_map = model(image)
    
    depth_map = depth_map.squeeze().cpu().numpy()
    return depth_map

# 计算深度图
depth_result = estimate_depth("orb_slam_keyframe.jpg")
cv2.imwrite("depth_result.png", (depth_result * 255).astype(np.uint8))
