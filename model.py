import os.path
from torchvision import transforms
from torchvision.models import resnext101_32x8d, resnext50_32x4d, mobilenet_v3_small
import torch
import torch.nn as nn

resize_size = 360
padded_size = 360


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin, cuda):
        super(ContrastiveLoss, self).__init__()
        self._margin = margin
        self._cuda = cuda

    def forward(self, y1, y2, cls):
        if self._cuda:
            cls = cls.cuda()
        euclidean_distance = torch.nn.functional.pairwise_distance(y1, y2)
        pos = cls * torch.pow(euclidean_distance, 2.)
        neg = (1. - cls) * torch.pow(torch.clamp(self._margin - euclidean_distance, min=0.), 2.)
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive


def compose_train_transforms(resize_size: int, padded_size: int):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize_size),
        transforms.CenterCrop(padded_size),
        transforms.RandomResizedCrop(
            padded_size,
            scale=(0.975, 1.025),
            ratio=(0.975, 1.025)
        ),
        transforms.RandomRotation(360),
        transforms.ColorJitter(
            brightness=(.70, 1.1),
            contrast=(.70, 1.2),
            saturation=(.95, 1.5),
            hue=(-0.05, 0.05)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])


def compose_valid_transforms(resize_size: int, padded_size: int):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize_size),
        transforms.CenterCrop(padded_size),
        transforms.RandomRotation(360),
    ])


class EmbeddingNet_internal(nn.Module):

    def __init__(self,
                 output_size: int = 1024,
                 embeddings_weights: str = None
                 ):
        super().__init__()

        pretrained_network = mobilenet_v3_small
        trainable_layers = None

        self.convnet = pretrained_network(pretrained=True)
        if trainable_layers is not None:
            for (name, _), child in zip(self.convnet.named_children(), self.convnet.children()):
                if name not in trainable_layers:
                    for param in child.parameters():
                        param.requires_grad = False

        self.convnet = nn.Sequential(*(list(self.convnet.children())[:-2]))
        self.aavgp3d = nn.AdaptiveAvgPool3d(output_size=(output_size, 1, 1))
        self.flatten = nn.Flatten()

        if embeddings_weights is not None:
            if not os.path.isfile(embeddings_weights):
                print(f"Couldn't load file {embeddings_weights}")
                exit(-1)
            self.load_state_dict(
                torch.load(embeddings_weights, weights_only=True)
            )

    def forward(self, x):
        output = self.convnet(x)
        output = self.aavgp3d(output)
        output = self.flatten(output)
        return output


class SiameseNet(nn.Module):
    def __init__(self,
                 embeddings_size: int,
                 embeddings_weights: None or str = None,
                 cuda: bool = False
                 ):
        super(SiameseNet, self).__init__()

        self._embedding_net = EmbeddingNet_internal(
            embeddings_size,
            embeddings_weights
        )
        self._cuda = cuda
        if self._cuda:
            self._embedding_net.cuda()

    def forward(self, x1, x2):

        if self._cuda:
            y1 = self._embedding_net(x1.cuda())
            y2 = self._embedding_net(x2.cuda())
        else:
            y1 = self._embedding_net(x1)
            y2 = self._embedding_net(x2)

        return y1, y2

    def embedding_net(self):
        return self._embedding_net


class EmbeddingNet(nn.Module):
    def __init__(
            self,
            embeddings_size: int,
            embeddings_weights: None or str = None,
            cuda=False
    ):
        super().__init__()

        self._embedding_net = EmbeddingNet_internal(embeddings_size, None)
        if embeddings_weights is not None and not os.path.isfile(embeddings_weights):
            print(f"Couldn't load file {embeddings_weights}")
            exit(-1)
        if embeddings_weights is not None:
            self._embedding_net.load_state_dict(
                torch.load(embeddings_weights, weights_only=True)
            )
        self._cuda = cuda

        if self._cuda:
            self._embedding_net.cuda()

    def forward(self, x):
        if self._cuda:
            y = self._embedding_net(x.cuda())
        else:
            y = self._embedding_net(x)
        return y


