from torch import nn
from torchvision.transforms import ToTensor, Compose
import torch
from torchmetrics.regression import CosineSimilarity
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Fundamental(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 16),
                                     nn.ReLU(),
                                     nn.Flatten()
        )

    def encode(self, path_to_image: str = None, to_vector: bool = False, to_tensor: bool = True):
        transform = Compose([ToTensor()])
        x = Image.open(path_to_image)
        x = x.resize((512, 512))
        x = transform(x)
        x = x.reshape(-1, 512)
        x = x.to(device)
        output = self.encoder(x)
        if to_vector:
            output = output.cpu().detach().numpy()
        else:
            output = output

        return output

    def distance(self, im1_path: str = None, im2_path: str = None) -> float:
        vec1 = self.encode(im1_path)
        vec2 = self.encode(im2_path)
        distance = CosineSimilarity(reduction='mean')
        return distance(vec1, vec2).tolist()