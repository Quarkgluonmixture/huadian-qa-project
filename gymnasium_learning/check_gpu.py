import torch

if torch.cuda.is_available():
    print("✅ 恭喜！您的 GPU 已被 PyTorch 正确识别。")
    print(f"显卡名称: {torch.cuda.get_device_name(0)}")
    print("DQN 训练将自动使用 GPU 加速。")
else:
    print("❌ 未检测到可用的 GPU。")
    print("DQN 训练将使用 CPU，速度会比较慢。")
    print("\n要启用 GPU 加速，请确保您已：")
    print("1. 安装了 NVIDIA 显卡。")
    print("2. 安装了最新的 NVIDIA 驱动程序。")
    print("3. 安装了与 PyTorch 兼容的 CUDA Toolkit。")
