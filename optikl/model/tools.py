from torch import nn
from torchvision.transforms import ToTensor, Compose, Resize
import torch
from torchmetrics.regression import CosineSimilarity
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Fundamental(nn.Module):

    def __init__(self):
        super(Fundamental, self).__init__()

        self.linear_encoder = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 16),
                                     nn.ReLU(),
                                     nn.Linear(16, 8),
                                     nn.ReLU(),
                                     nn.Flatten()
        )

        self.conv_encoder = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

    def encode(self, path_to_image: str = None, to_vector: bool = False, to_tensor: bool = True, type: str = 'linear'):
        if type not in ['linear', 'conv']:
            raise TypeError(f'There is no such a method like {type} \n'
                            f'Choose either "linear" or "conv"')

        transform = Compose([Resize((256, 256)), ToTensor()])
        x = Image.open(path_to_image)
        # x = x.resize((512, 512))
        x = transform(x)
        if type=='linear':
            x = x.reshape(-1, 512)
            x = x.to(device)
            output = self.linear_encoder(x)
        else:
            x = x.to(device)
            output = self.conv_encoder(x)

        if to_vector:
            output = output.cpu().detach().numpy()
        else:
            output = output

        return output

    def distance(self, im1_path: str = None, im2_path: str = None, type_of_encoding: str = 'linear') -> float:
        if type_of_encoding not in ['linear', 'conv']:
            raise TypeError(f'There is no such a method like {type_of_encoding} \n'
                            f'Choose either "linear" or "conv"')
        if type_of_encoding=='linear':
            vec1 = self.encode(im1_path, type=type_of_encoding)
            vec2 = self.encode(im2_path, type=type_of_encoding)
        else:
            vec1 = self.encode(im1_path, type=type_of_encoding)
            vec2 = self.encode(im2_path, type=type_of_encoding)
        distance = CosineSimilarity(reduction='mean')
        return distance(vec1, vec2).tolist()