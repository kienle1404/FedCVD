
from .dtn import DeepTransformerNetwork, dtn
from .resnet import ResNet, resnet1d34
from .unet import Unet, unet
from .vgg import vgg1d11, vgg2d11
from .unetr import unetr


def get_model(name: str, **kwargs):
    if name == "resnet1d34":
        return resnet1d34()
    elif name == "unet":
        return unet()
    elif name == "dtn":
        return dtn()
    elif name == "vgg1d11":
        return vgg1d11()
    elif name == "vgg2d11":
        return vgg2d11()
    elif name == "unetr":
        return unetr()
    else:
        raise ValueError(f"Model {name} not found")