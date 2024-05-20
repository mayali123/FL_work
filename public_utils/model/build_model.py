import torch.nn as nn
from torchvision import models

def build_model(args):
    # choose different Neural network model for different args
    if args.model == 'Resnet18':
        model = ResNet18(args.n_classes, args.pretrained)
        model = model.to(args.device)
    else:
        raise ValueError(f'Name of model unknown {args.model}')

    return model



class ResNet18(nn.Module):
    def __init__(self, num_classes,pretrained):
        super(ResNet18, self).__init__()
        
        # 加载预训练的 ResNet 模型
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # 修改最后一个全连接层的输出大小
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # 为了和FedNed保持一致
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, is_need_feature=False,is_need_drop=False):
        # 获取模型提取的特征
        features = self.resnet.conv1(x)
        features = self.resnet.bn1(features)
        features = self.resnet.relu(features)
        features = self.resnet.maxpool(features)
        
        features = self.resnet.layer1(features)
        features = self.resnet.layer2(features)
        features = self.resnet.layer3(features)
        features = self.resnet.layer4(features)
        
        features = self.resnet.avgpool(features)
        # 为了和FedNed保持一致
        drop = features
        drop = self.dropout(drop)
        drop = drop.view(drop.size(0), -1)


        features = features.view(features.size(0), -1)

        
        
        # 输出特征和分类结果
        output = self.resnet.fc(features)            
        if is_need_feature:
            if is_need_drop:
                return features,drop,output
            else:
                return output,features
        else:
            return output