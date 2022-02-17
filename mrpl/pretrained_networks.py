from collections import namedtuple
import torch
from torchvision import models as tv
import torch.nn.functional as F

def gram_matrix(input):
    if len(input.size())==3:
        input= input.unsqueeze(-1)
    a, b, c, d = input.size()
    features = input.view(a* b, c*d)
    G = torch.mm(features, features.t())
    return G.div(a*b*c*d)

class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True,features=['linear'],resolution=['x1'],MRPL=False):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        self.MRPL = MRPL

        self.resolution = resolution
        self.features = features

        if self.MRPL :
            self.features = ['linear','quadratic']
            self.resolution = ['x1','x2']

        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):

        if ('x1' in self.resolution) or ('quadratic' in self.features) :
           
            h = self.slice1(X)
            h_relu1 = h
            h = self.slice2(h)
            h_relu2 = h
            h = self.slice3(h)
            h_relu3 = h
            h = self.slice4(h)
            h_relu4 = h
            h = self.slice5(h)
            h_relu5 = h

        if ('x2' in self.resolution) :

            X1 = F.interpolate(X, size=(128, 128))
            h = self.slice1(X1)
            h_relu1_2 = h
            h = self.slice2(h)
            h_relu2_2 = h
            h = self.slice3(h)
            h_relu3_2 = h
            h = self.slice4(h)
            h_relu4_2 = h
            h = self.slice5(h)
            h_relu5_2 = h

        if ('quadratic' in self.features):

            h_relu1_gram = gram_matrix(h_relu1).unsqueeze(0).unsqueeze(0)
            h_relu2_gram = gram_matrix(h_relu2).unsqueeze(0).unsqueeze(0)
            h_relu3_gram = gram_matrix(h_relu3).unsqueeze(0).unsqueeze(0)
            h_relu4_gram = gram_matrix(h_relu4).unsqueeze(0).unsqueeze(0)
            h_relu5_gram = gram_matrix(h_relu5).unsqueeze(0).unsqueeze(0)

        if self.MRPL :
            alexnet_outputs = namedtuple("AlexnetOutputs", ['X2relu2', 'X2relu3', 'X2relu4','relu4','relu5','X2relu5', 'gramrelu2', 'gramrelu3', 'gramrelu4', 'gramrelu5'])
            out = alexnet_outputs(h_relu2_2,h_relu3_2, h_relu4_2, h_relu5_2, h_relu4, h_relu5,h_relu2_gram,h_relu3_gram,h_relu4_gram,h_relu5_gram)

        elif ('x1' in self.resolution) and ('x2' in self.resolution) and ('linear' in self.features) and ('quadratic' in self.features):
            alexnet_outputs = namedtuple("AlexnetOutputs", ['X2relu1','X2relu2', 'X2relu3', 'X2relu4', 'X2relu5','relu1','relu2','relu3','relu4','relu5', 'gramrelu1','gramrelu2', 'gramrelu3', 'gramrelu4', 'gramrelu5'])
            out = alexnet_outputs(h_relu1_2,h_relu2_2,h_relu3_2, h_relu4_2,h_relu5_2,h_relu1,h_relu2,h_relu3,h_relu4,h_relu5,h_relu1_gram,h_relu2_gram,h_relu3_gram,h_relu4_gram,h_relu5_gram)

        elif ('x1' in self.resolution) and (not('x2' in self.resolution)) and (not('linear' in self.features)) and (not('quadratic' in self.features)):
            alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1','relu2','relu3','relu4','relu5'])
            out = alexnet_outputs(h_relu1,h_relu2,h_relu3,h_relu4,h_relu5)

        elif (not('x1' in self.resolution)) and ('x2' in self.resolution) and ('linear' in self.features) and (not('quadratic' in self.features)):
            alexnet_outputs = namedtuple("AlexnetOutputs", ['X2relu1','X2relu2', 'X2relu3', 'X2relu4', 'X2relu5'])
            out = alexnet_outputs(h_relu1_2,h_relu2_2,h_relu3_2, h_relu4_2,h_relu5_2)

        elif ('x1' in self.resolution) and ('x2' in self.resolution) and ('linear' in self.features) and (not('quadratic' in self.features)):
            alexnet_outputs = namedtuple("AlexnetOutputs", ['X2relu1','X2relu2', 'X2relu3', 'X2relu4', 'X2relu5','relu1','relu2','relu3','relu4','relu5'])
            out = alexnet_outputs(h_relu1_2,h_relu2_2,h_relu3_2, h_relu4_2,h_relu5_2,h_relu1,h_relu2,h_relu3,h_relu4,h_relu5)

        elif ('x1' in self.resolution) and (not('x2' in self.resolution)) and ('linear' in self.features) and (not('quadratic' in self.features)):
            alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1','relu2','relu3','relu4','relu5'])
            out = alexnet_outputs(h_relu1,h_relu2,h_relu3,h_relu4,h_relu5)

        elif (not('x1' in self.resolution)) and ('x2' in self.resolution) and ('linear' in self.features) and ('quadratic' in self.features):
            alexnet_outputs = namedtuple("AlexnetOutputs", ['X2relu1','X2relu2', 'X2relu3', 'X2relu4', 'X2relu5','relu1','relu2','relu3','relu4','relu5'])
            out = alexnet_outputs(h_relu1_2,h_relu2_2,h_relu3_2, h_relu4_2,h_relu5_2,h_relu1_gram,h_relu2_gram,h_relu3_gram,h_relu4_gram,h_relu5_gram)

        elif (not('x1' in self.resolution)) and (not('x2' in self.resolution)) and (not('linear' in self.features)) and ('quadratic' in self.features):
            alexnet_outputs = namedtuple("AlexnetOutputs", ['gramrelu1','gramrelu2', 'gramrelu3', 'gramrelu4', 'gramrelu5'])
            out = alexnet_outputs(h_relu1_gram,h_relu2_gram,h_relu3_gram,h_relu4_gram,h_relu5_gram)
        else :
            raise('Not implemented !')

        return out