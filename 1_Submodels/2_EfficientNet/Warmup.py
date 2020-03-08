# ##
# ## Deprecated
# import timm
# m = timm.create_model('tf_efficientnet_b7', pretrained=True)
# print(m.eval())
# #C:\Users\Evan/.cache\torch\checkpoints\tf_efficientnet_b7_ra-6c08e654.pth
# # python validate.py /cifar-100-python/test/ --model tf_efficientnet_b7 --pretrained

# Test on EfficientNet-PyTorch
# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_name('efficientnet-b7')
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b7')