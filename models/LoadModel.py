import numpy as np
from torch import nn
import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import pretrainedmodels
import matplotlib.pyplot as plt
from config import pretrained_model

import pdb

class MainModel(nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.use_dcl = config.use_dcl
        self.num_classes = config.numcls
        self.backbone_arch = config.backbone
        self.use_Asoftmax = config.use_Asoftmax
        self.mask_guide = config.mask_guide
        # print(self.backbone_arch)

        if self.backbone_arch in dir(models):
            self.model = getattr(models, self.backbone_arch)()
            if self.backbone_arch in pretrained_model:
                self.model.load_state_dict(torch.load(pretrained_model[self.backbone_arch]))
        else:
            if self.backbone_arch in pretrained_model:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained=None)
            else:
                self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000)

        if self.backbone_arch == 'resnet50' or self.backbone_arch == 'se_resnet50':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'senet154':
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if self.backbone_arch == 'se_resnext101_32x4d':
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        if self.backbone_arch == 'se_resnet101':
            self.model = nn.Sequential(*list(self.model.children())[:-2])


        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)

        if self.use_dcl:
            if config.cls_2:
                self.classifier_swap = nn.Linear(2048, 2, bias=False)
            if config.cls_2xmul:
                self.classifier_swap = nn.Linear(2048, 2*self.num_classes, bias=False)
            self.Convmask = nn.Conv2d(2048, 1, 1, stride=1, padding=0, bias=True)
            self.avgpool2 = nn.AvgPool2d(2, stride=2)

        # if self.use_Asoftmax:
        #     self.Aclassifier = AngleLinear(2048, self.num_classes, bias=False)

        self.attention = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.att_type = "two"

    def getAttFeats(self,att_map,sec_map,features,type=None):
        # params: one for simple att*features
        # two for cat att*feat and features
        if type=='one':
            features=att_map*features
        elif type=='two':
            features=0.65*features+0.35*(att_map*features)
        elif type=='three':
            features=torch.cat((features,att_map*features),dim=1)
        elif type=='four':
            features = 0.2 * features + 0.7 * (att_map * features) + 0.1 * (sec_map * features)
        else:
            pass
        return features

    def forward(self, x, mask=None, sec_mask=None, last_cont=None):
        x = self.model(x)

        if self.mask_guide:
            fg_att = self.attention(torch.cat((torch.mean(x, dim=1).unsqueeze(1), \
                                               torch.max(x, dim=1)[0].unsqueeze(1)), dim=1))
            sec_att = self.attention(torch.cat((torch.mean(x, dim=1).unsqueeze(1), \
                                               torch.max(x, dim=1)[0].unsqueeze(1)), dim=1))
            # fg_att=torch.flatten(torch.sigmoid(fg_att),1)
            fg_att = torch.sigmoid(fg_att)
            sec_att = torch.sigmoid(sec_att)

            features = self.getAttFeats(fg_att, sec_att, x, type=self.att_type)
            if self.backbone_arch == 'resnet50':
                features = F.relu(features)

            if self.backbone_arch == 'vgg19':
                x = F.adaptive_avg_pool2d(features, (7, 7))
            else:
                x = F.adaptive_avg_pool2d(features, (1, 1))

            #
            x = self.avgpool(x)
            # print(x.shape)
            # x = torch.flatten(x, 1)
        else:
            x = self.avgpool(x)


        if self.use_dcl:
            mask = self.Convmask(x)
            mask = self.avgpool2(mask)
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)

        x = x.view(x.size(0), -1)
        out = []
        out.append(self.classifier(x))

        if self.mask_guide and mask is not None:
            h,w = fg_att.shape[2],fg_att.shape[3]
            mask=F.adaptive_avg_pool2d(mask, (h, w))

            fg_att = fg_att.view(fg_att.shape[0],-1)
            mask = mask.view(mask.shape[0],-1)

            mask += 1e-12
            max_elmts=torch.max(mask,dim=1)[0].unsqueeze(1)
            mask = mask/max_elmts
            out.append(fg_att)

            out.append(mask)


            if sec_mask is not None:
                h, w = sec_att.shape[2], sec_att.shape[3]
                sec_mask = F.adaptive_avg_pool2d(sec_mask, (h, w))

                sec_att = sec_att.view(sec_att.shape[0], -1)
                sec_mask = mask.view(sec_mask.shape[0], -1)

                sec_mask += 1e-12
                max_elmts = torch.max(sec_mask, dim=1)[0].unsqueeze(1)
                sec_mask = sec_mask / max_elmts
                out.append(sec_att)
                out.append(sec_mask)

        out.append(x)

        if self.use_dcl:
            out.append(self.classifier_swap(x))
            out.append(mask)

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))

        return out
