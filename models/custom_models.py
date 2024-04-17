import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
from models.base_model import CustomModel


'''
    .. Enhance model customization by extending the layers in the model's output layer (head) through modifications in 
        the following section:
            model.fc = nn.Sequential(
                ...
            )
        Consider incorporating attention mechanisms, as they can be pivotal for enhancing model accuracy even further.

    .. As this models is parts of the ensemble model, the number of outputs is 1 (defined by num_classes).

    .. Next model are provided by default:
        [ResNet101, ResNet50, ResNet34, ResNet18, EfficientNet, VGG16, VGG19, InceptionV3]
'''
class CustomResNet101Model(CustomModel):
    def __init__(self, num_classes):
        resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        resnet101.fc = nn.Sequential(
            nn.Linear(2048, num_classes)
        )
        super(CustomResNet101Model, self).__init__(num_classes, resnet101)

class CustomResNet50Model(CustomModel):
    def __init__(self, num_classes):
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet50.fc = nn.Sequential(
            nn.Linear(2048, num_classes)
        )
        super(CustomResNet50Model, self).__init__(num_classes, resnet50)

class CustomResNet34Model(CustomModel):
    def __init__(self, num_classes):
        resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        resnet34.fc = nn.Sequential(
            nn.Linear(512, num_classes)
        )
        super(CustomResNet34Model, self).__init__(num_classes, resnet34)

class CustomResNet18Model(CustomModel):
    def __init__(self, num_classes):
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet18.fc = nn.Sequential(
            nn.Linear(512, num_classes)
        )
        super(CustomResNet18Model, self).__init__(num_classes, resnet18)

class CustomEfficientNetModel(CustomModel):
    def __init__(self, num_classes):
        efficientnet = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
        efficientnet._fc = nn.Sequential(
            nn.Linear(efficientnet._fc.in_features, num_classes)
        )

        super(CustomEfficientNetModel, self).__init__(num_classes, efficientnet)

class CustomVGG16Model(CustomModel):
    def __init__(self, num_classes):
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        vgg16.fc = nn.Sequential(
            nn.Linear(4096, num_classes)
        )
        super(CustomVGG16Model, self).__init__(num_classes, vgg16)

class CustomVGG19Model(CustomModel):
    def __init__(self, num_classes):
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        vgg19.fc = nn.Sequential(
            nn.Linear(4096, num_classes)
        )
        super(CustomVGG19Model, self).__init__(num_classes, vgg19)

class CustomInceptionV3Model(CustomModel):
    def __init__(self, num_classes):
        inception_v3 = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
        '''
            .. aux_logits = False: 
                This disabling auxiliary branch. By doing so, you are telling the model not to use the auxiliary 
                classifier. The auxiliary branch is an auxiliary classifier designed to help with training and combat 
                the vanishing gradient problem.
        '''
        inception_v3.aux_logits = False
        inception_v3.fc = nn.Sequential(
            nn.Linear(inception_v3.fc.in_features, num_classes)
        )
        super(CustomInceptionV3Model, self).__init__(num_classes, inception_v3)