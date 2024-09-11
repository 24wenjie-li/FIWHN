import torch
from torchsummaryX import summary
import model.MultiAdd as MultiAdd
model = MultiAdd.MODEL()

summary(model, torch.zeros((1, 3, 320, 180)))#HR:1280 x 720

# input LR x2, HR size is 720p
# summary(model, torch.zeros((1, 3, 640, 360)))

# input LR x3, HR size is 720p
# summary(model, torch.zeros((1, 3, 426, 240)))

# input LR x4, HR size is 720p
# summary(model, torch.zeros((1, 3, 320, 180)))