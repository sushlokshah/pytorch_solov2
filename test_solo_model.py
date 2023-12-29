from modules.solov2 import SOLOV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.config import cfg

model = SOLOV2(
    cfg,
    pretrained="/home/awi-docker/video_summarization/pytorch_solov2/pretrained/solov2_448_r18_epoch_36.pth",
    mode="train",
)
print(model)

output = model.forward_dummy(torch.rand(1, 3, 512, 512))
print(len(output))
print(len(output[0]))
print(len(output[1]))
print(output[0][0].shape)
print(output[1][0].shape)
