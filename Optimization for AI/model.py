import torch.nn as nn
import torchvision


class EfficientNet(nn.Module):
    def __init__(self, in_ch=1, dim_output=101, pretrained=False):
        super(EfficientNet, self).__init__()
        
        self.model = torchvision.models.efficientnet.efficientnet_b4(pretrained=pretrained)
        self.model.features[0][0] = nn.Conv2d(in_ch, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1] = nn.Linear(1792, dim_output)

    def forward(self, img):
        return self.model(img)


class VGG13(nn.Module):
    def __init__(self, in_ch=1, dim_output=101, pretrained=False):
        super(VGG13, self).__init__()
        
        self.model = torchvision.models.vgg13(pretrained=pretrained)
        self.model.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.classifier[6] = nn.Linear(4096, dim_output)

    def forward(self, img):
        return self.model(img)


class VGG16(nn.Module):
    def __init__(self, in_ch=1, dim_output=101, pretrained=False):
        super(VGG16, self).__init__()
        
        self.model = torchvision.models.vgg16(pretrained=pretrained)
        self.model.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.classifier[6] = nn.Linear(4096, dim_output)

    def forward(self, img):
        return self.model(img)


class VGG19(nn.Module):
    def __init__(self, in_ch=1, dim_output=101, pretrained=False):
        super(VGG19, self).__init__()
        
        self.model = torchvision.models.vgg19(pretrained=pretrained)
        self.model.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.classifier[6] = nn.Linear(4096, dim_output)

    def forward(self, img):
        return self.model(img)


class ResNet18(nn.Module):
    def __init__(self, in_ch=1, dim_output=101, pretrained=False):
        super(ResNet18, self).__init__()
        
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        self.model.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, dim_output, bias=True)

    def forward(self, img):
        return self.model(img)


class ResNet34(nn.Module):
    def __init__(self, in_ch=1, dim_output=101, pretrained=False):
        super(ResNet34, self).__init__()
        
        self.model = torchvision.models.resnet34(pretrained=pretrained)
        self.model.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, dim_output, bias=True)

    def forward(self, img):
        return self.model(img)


class ResNet50(nn.Module):
    def __init__(self, in_ch=1, dim_output=101, pretrained=False):
        super(ResNet50, self).__init__()
        
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        self.model.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, dim_output, bias=True)

    def forward(self, img):
        return self.model(img)


class ResNet101(nn.Module):
    def __init__(self, in_ch=1, dim_output=101, pretrained=False):
        super(ResNet101, self).__init__()
        
        self.model = torchvision.models.resnet101(pretrained=pretrained)
        self.model.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(512, dim_output, bias=True)

    def forward(self, img):
        return self.model(img)


class MobileNet_V2(nn.Module):
    def __init__(self, in_ch=1, dim_output=101, pretrained=False):
        super(MobileNet_V2, self).__init__()
        
        self.model = torchvision.models.mobilenet_v2(pretrained=pretrained)
        self.model.features[0][0] = nn.Conv2d(in_ch, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1] = nn.Linear(1280, dim_output)

    def forward(self, img):
        return self.model(img)


class MobileNet_V3_small(nn.Module):
    def __init__(self, in_ch=1, dim_output=101, pretrained=False):
        super(MobileNet_V3_small, self).__init__()
        
        self.model = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        self.model.features[0][0] = nn.Conv2d(in_ch, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[3] = nn.Linear(1024, dim_output)

    def forward(self, img):
        return self.model(img)


class MobileNet_V3_large(nn.Module):
    def __init__(self, in_ch=1, dim_output=101, pretrained=False):
        super(MobileNet_V3_large, self).__init__()
        
        self.model = torchvision.models.mobilenet_v3_large(pretrained=pretrained)
        self.model.features[0][0] = nn.Conv2d(in_ch, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[3] = nn.Linear(1280, dim_output)

    def forward(self, img):
        return self.model(img)