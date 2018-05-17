import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


def modify_squeezenets(model):
    # /!\ Beware squeezenets do not have any last_linear module

    # Modify attributs
    model.dropout = model.classifier[0]
    model.last_conv = model.classifier[1]
    model.relu = model.classifier[2]
    model.avgpool = model.classifier[3]
    del model.classifier

    def logits(self, features):
        x = self.dropout(features)
        x = self.last_conv(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x

    def forward(self, input):
        x1 = self.features(input)
        x2 = x1
        x1 = self.logits(x1)
        print(x1, x2)
        return x1

    # Modify methods
    setattr(model.__class__, 'logits', logits)
    setattr(model.__class__, 'forward', forward)
    return model





def squeezenet1_1(num_classes=9):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    model = models.squeezenet1_1(pretrained=False, num_classes=num_classes)
    model = modify_squeezenets(model)

    return model