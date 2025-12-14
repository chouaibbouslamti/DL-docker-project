from torchvision import models, transforms
from PIL import Image
import torch

model = models.resnet18(pretrained=True)
model.eval()
print("ResNet18 chargé.")

# test rapide : créer un tenseur aléatoire
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = model(x)
print("Inference OK, sortie shape:", out.shape)
