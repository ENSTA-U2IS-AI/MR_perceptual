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

def gram_matrix2(input1,input2):
    assert input1.size() == input1.size()
    if len(input1.size())==3:
        input1 = input1.unsqueeze(-1)
        input2 = input2.unsqueeze(-1)
    a, b, c, d = input1.size()
    features1 = input1.view(a * b, c * d)
    features2 = input2.view(a* b, c*d)
    G = torch.mm(features1.t(), features1) + torch.mm(features2.t(), features2) -2*torch.mm(features1.t(), features2)
    #G = torch.mm(features1, features1.t()) + torch.mm(features2, features2.t()) - 2 * torch.mm(features1, features2.t())
    return torch.exp(G.div(a*b*c*d).unsqueeze(0).unsqueeze(0))


class squeezenet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True,gram=False):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        self.gram = gram
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2,5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
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
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple("SqueezeOutputs", ['relu1','relu2','relu3','relu4','relu5','relu6','relu7'])
        if self.gram:
            h_relu1 = gram_matrix(h_relu1).unsqueeze(0).unsqueeze(0)
            h_relu2 = gram_matrix(h_relu2).unsqueeze(0).unsqueeze(0)
            h_relu3 = gram_matrix(h_relu3).unsqueeze(0).unsqueeze(0)
            h_relu4 = gram_matrix(h_relu4).unsqueeze(0).unsqueeze(0)
            h_relu5 = gram_matrix(h_relu5).unsqueeze(0).unsqueeze(0)
            h_relu6 = gram_matrix(h_relu6).unsqueeze(0).unsqueeze(0)
            h_relu7 = gram_matrix(h_relu7).unsqueeze(0).unsqueeze(0)
        out = vgg_outputs(h_relu1,h_relu2,h_relu3,h_relu4,h_relu5,h_relu6,h_relu7)

        return out

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

        elif (not('x1' in resolution)) and ('x2' in self.resolution) and ('linear' in self.features) and (not('quadratic' in self.features)):
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

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True,gram=False):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        #vgg_pretrained_features.load_state_dict(torch.load('carl_vgg16.pt'))
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        self.gram = gram
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        if self.gram:
            h_relu1_2 = gram_matrix(h_relu1_2).unsqueeze(0).unsqueeze(0)
            h_relu2_2 = gram_matrix(h_relu2_2).unsqueeze(0).unsqueeze(0)
            h_relu3_3 = gram_matrix(h_relu3_3).unsqueeze(0).unsqueeze(0)
            h_relu4_3 = gram_matrix(h_relu4_3).unsqueeze(0).unsqueeze(0)
            h_relu5_3 = gram_matrix(h_relu5_3).unsqueeze(0).unsqueeze(0)
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out



class resnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, num=18,training='imagenet',gram=False):
        super(resnet, self).__init__()
        if(num==18):
            self.net = tv.resnet18(pretrained=pretrained)
        elif(num==34):
            self.net = tv.resnet34(pretrained=pretrained)
        elif(num==50):
            self.net = tv.resnet50(pretrained=pretrained)
            if training=='densecl':
                PATH='./ckpts/latest_net_densecl.pth'
                checkpoint = torch.load(PATH, map_location='cuda:0')
                print(len(list(checkpoint.keys())))
                for k in list(checkpoint.keys()):
                    '''if k.startswith('net.net.'):
                        # remove prefix
                            checkpoint[k[len('net.net.'):]] =checkpoint[k]
                            del checkpoint[k]'''
                    if k.startswith('net.'):
                        # remove prefix
                        #print('1111',k[len('net.'):])
                        checkpoint[k[len('net.'):]] =checkpoint[k]
                        del checkpoint[k]
                print(checkpoint.keys())
                print(len(list(checkpoint.keys())))
                print("Model's state_dict:")
                for param_tensor in self.net.state_dict():
                    print(param_tensor, "\t",param_tensor in list(checkpoint.keys()) )#self.net.state_dict()[param_tensor].size())
                self.net.load_state_dict(checkpoint,strict=False)

        elif(num==101):
            self.net = tv.resnet101(pretrained=pretrained)
        elif(num==152):
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5
        self.gram = gram

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_relu2 = h
        h = self.layer2(h)
        h_relu3 = h
        h = self.layer3(h)
        h_relu4 = h
        h = self.layer4(h)
        h_relu5 = h
        if self.gram:
            h_relu1 = gram_matrix(h_relu1).unsqueeze(0).unsqueeze(0)
            h_relu2 = gram_matrix(h_relu2).unsqueeze(0).unsqueeze(0)
            h_relu3 = gram_matrix(h_relu3).unsqueeze(0).unsqueeze(0)
            h_relu4 = gram_matrix(h_relu4).unsqueeze(0).unsqueeze(0)
            h_relu5 = gram_matrix(h_relu5).unsqueeze(0).unsqueeze(0)
        outputs = namedtuple("Outputs", ['relu1','conv2','conv3','conv4','conv5'])
        out = outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out