from generator.load_swin_unet_model import SwinUnet
import torch

# D:\Machine_Learning\VirtualEnvironments\Scripts\activate.bat

if __name__ == "__main__":
    y = torch.rand(2, 4, 3, 256, 256)

    model = SwinUnet(config=None)
    total_params = sum(
        param.numel() for param in model.parameters()
    )

    print(f"total_parameters = {total_params / 1e6:.3f} M")
    output = model(y)
