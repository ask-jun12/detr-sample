import torch, torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import requests

# 学習済みモデルの取得
checkpoint = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
    map_location='cpu',
    check_hash=True
)

# 分類ヘッドの削除
del checkpoint['model']['class_embed.weight']
del checkpoint['model']['class_embed.bias']

# 保存
torch.save(checkpoint, 'detr-r50_no-class-head.pth')