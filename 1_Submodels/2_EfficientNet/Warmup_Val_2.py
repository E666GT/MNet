import json
from PIL import Image
import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-b7')
model.load_state_dict(torch.load("efficientnet-b7-dcc49843.pth"))

