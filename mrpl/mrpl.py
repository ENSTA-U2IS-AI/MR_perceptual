from __future__ import absolute_import

import mrpl as mrpl


import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from . import pretrained_networks as pn
import torch.nn
import math
from PIL import Image
from skimage.morphology import erosion, dilation,binary_erosion, opening, closing, white_tophat, reconstruction, area_opening, area_closing
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM



def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)

# Learned perceptual metric

class MRPL(nn.Module):
    def __init__(self, pretrained=True, net='alex', version='0.1', lpips=False,mrpl_like=False,mrpl=False,spatial=False,
        pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True,RBF=False,ssim=False,randomRBF=False,loss_type=None,
        norm=None,feature=None,resolution=None,verbose=True):

        super(MRPL, self).__init__()
        if(verbose):
            print('Setting up [%s] perceptual loss: trunk [%s], v[%s], spatial [%s], loss  [%s], norm  [%s], feature  [%s], resolution  [%s],'%
                ('MRPL' if mrpl else 'baseline', net, version, 'on' if spatial else 'off','on' if loss_type else 'off','on' if norm else 'off','on' if feature else 'off','on' if resolution else 'off'))

        self.feature = feature
        self.resolution = resolution
        self.norm = norm
        self.loss_type = loss_type
        self.mrpl_like = mrpl_like
        self.mrpl = mrpl
        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.RBF=RBF
        self.randomRBF=randomRBF
        self.loss = torch.nn.BCELoss(reduce=False)
        self.entropy=entropy
        self.ssim=ssim
        self.lpips = lpips # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()
        self.training='imagenet'

        if mrpl :
            if self.loss_type == None :
                self.loss_type = 'CE'
            if self.norm == None :
                self.norm = 'sigmoid'
            self.mrpl_like = True

        num=0

        if(self.pnet_type=='alex'):
            net_type = pn.alexnet
            self.chns = [64,192,384,256,256]
            self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune,
            features=self.feature,resolution=self.resolution,MRPL=self.mrpl)            
        else: 
            raise('Network not implemented !')

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            self.lins = nn.ModuleList(self.lins)

        if(randomRBF):

            self.lin0_RBF = RFFKernel(self.chns[0], num_samples=512,length_scale=100.0)
            self.lin1_RBF = RFFKernel(self.chns[1], num_samples=512,length_scale=100.0)
            self.lin2_RBF = RFFKernel(self.chns[2], num_samples=512,length_scale=100.0)
            self.lin3_RBF = RFFKernel(self.chns[3], num_samples=512,length_scale=100.0)
            self.lin4_RBF = RFFKernel(self.chns[4], num_samples=512,length_scale=100.0)
            self.lins_RBF = [self.lin0_RBF,self.lin1_RBF,self.lin2_RBF,self.lin3_RBF,self.lin4_RBF]
            self.lins_RBF = nn.ModuleList(self.lins_RBF)

            if(pretrained):
                if(model_path is None):
                    import inspect
                    import os
                    model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth'%(version,net)))

                if(verbose):
                    print('Loading model from: %s'%model_path)
                self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

        if(eval_mode):
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1
            in1 = 2 * in1  - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)

        feats0, feats1, diffs = {}, {}, {}

        if self.mrpl_like :
            self.L = len(outs0)

        else :
            self.L = len(self.chns)

        for kk in range(self.L):
            if ((self.randomRBF) and (not self.entropy) and (not self.ssim)):
                o0 = self.lins_RBF[kk](outs0[kk])
                o1 = self.lins_RBF[kk](outs1[kk])
                feats0[kk], feats1[kk] = mrpl.normalize_tensor(o0), mrpl.normalize_tensor(o1)
                diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
                
            elif ((not self.randomRBF) and (self.loss_type == 'CE') and (not self.ssim)):
                feats0[kk], feats1[kk] = mrpl.normalize_tensor(outs0[kk],norm=self.norm), mrpl.normalize_tensor(outs1[kk],norm=self.norm)
                diffs[kk] = self.loss(feats0[kk],feats1[kk])
            elif ((not self.randomRBF) and (self.loss_type == 'MSE') and (not self.ssim)):
                feats0[kk], feats1[kk] = mrpl.normalize_tensor(outs0[kk],norm=self.norm), mrpl.normalize_tensor(outs1[kk],norm=self.norm)
                diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
            elif ((self.RBF) and (not self.entropy) and (not self.ssim)):
                feats0[kk], feats1[kk] = mrpl.normalize_tensor(outs0[kk]), mrpl.normalize_tensor(outs1[kk])
                diffs[kk] = pn.gram_matrix2(feats0[kk], feats1[kk])
            elif ((self.ssim) and (not self.RBF) and (not self.entropy)):
                feats0[kk], feats1[kk] = mrpl.normalize_tensor(outs0[kk]), mrpl.normalize_tensor(outs1[kk])
                diffs[kk] = ssim( feats0[kk], feats1[kk], data_range=1, size_average=False)
            else:
                feats0[kk], feats1[kk] = mrpl.normalize_tensor(outs0[kk]), mrpl.normalize_tensor(outs1[kk])
                diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if(self.lpips):
            if(self.spatial):
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [upsample(diffs[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                if (not self.RBF) and (not self.ssim):
                    res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]
                elif  (self.ssim):
                    res = [1-diffs[kk] for kk in range(self.L)]
                else:
                    res = [spatial_average(diffs[kk], keepdim=True) for kk in range(self.L)] #[spatial_average(diffs[kk], keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]

        if(retPerLayer):
            return (val, res)
        else:
            return val

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class RFFKernel(nn.Module):

    def __init__(self, num_imput, num_samples, length_scale):
        super().__init__()
        self.raw_lengthscale = length_scale
        self.num_samples=num_samples
        self.projlayer = nn.Conv2d(num_imput, num_samples, 1, stride=1, padding=0) #nn.Linear(num_imput, num_samples)
        std=1.0/self.raw_lengthscale
        self.projlayer.weight.data.normal_(0.0, std)
        self.projlayer.bias.data.uniform_(0.0, 2 * np.pi)
        #print('self.projlayer.weight.data',self.projlayer.weight.data,'std',std)


    def forward(self, x1):
        self.projlayer.requires_grad_(False)
        x1 = self.projlayer(x1)
        self.projlayer.requires_grad_(True)
        z = torch.cos(x1)
        D = self.num_samples
        z = z *math.sqrt((2/ D))


        return z
    def evaluatefeature(self, x1):
        self.projlayer.requires_grad_(False)
        x1 = self.projlayer(x1)
        self.projlayer.requires_grad_(True)
        z = torch.cos(x1)
        D = self.num_samples
        z = z *math.sqrt((2/ D))

        return z


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))

class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2.
        self.logit = self.net.forward(d0,d1)
        return self.loss(self.logit, per)

# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace

class L2(FakeNet):
    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            (N,C,X,Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0-in1)**2,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        elif(self.colorspace=='Lab'):
            value = mrpl.l2(mrpl.tensor2np(mrpl.tensor2tensorlab(in0.data,to_norm=False)),
                mrpl.tensor2np(mrpl.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
            ret_var = Variable( torch.Tensor((value,) ) )
            if(self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var

class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            value = mrpl.dssim(1.*mrpl.tensor2im(in0.data), 1.*mrpl.tensor2im(in1.data), range=255.).astype('float')
        elif(self.colorspace=='Lab'):
            value = mrpl.dssim(mrpl.tensor2np(mrpl.tensor2tensorlab(in0.data,to_norm=False)),
                mrpl.tensor2np(mrpl.tensor2tensorlab(in1.data,to_norm=False)), range=100.).astype('float')
        ret_var = Variable( torch.Tensor((value,) ) )
        if(self.use_gpu):
            ret_var = ret_var.cuda()
        return ret_var

# Definition of differents flat structuring element
se0 = square(2) # square
se1 = square(7) # square
se2 = square(15) # square
se3 = square(25) # square
se4 = square(35) # square

def opening_patterns(img, se0, se1, se2):
    Reco1 = []
    Reco2 = []
    Reco3 = []
    img=np.asarray(img).astype(np.uint8)
    for i in range(3):
        imFAS = erosion(img[:,:,i], se0)
        imFASbisreco1 = reconstruction(imFAS, img[:,:,i])
        residue_opening1 = (np.asarray(img[:,:,i]).astype(np.float16) - imFASbisreco1.astype(np.float16)).astype(np.uint8)
        Reco1.append(np.expand_dims(residue_opening1, axis=2))
        imFAS = erosion(img[:,:,i], se1)
        imFASbisreco2 = reconstruction(imFAS, img[:,:,i])
        residue_opening2 = (imFASbisreco1.astype(np.float16) - imFASbisreco2.astype(np.float16)).astype(np.uint8)
        Reco2.append(np.expand_dims(residue_opening2, axis=2))
        imFAS = erosion(img[:,:,i], se2)
        imFASbisreco3 = reconstruction(imFAS, img[:,:,i])
        residue_opening3 = (imFASbisreco2.astype(np.float16) - imFASbisreco3.astype(np.float16)).astype(np.uint8)
        Reco3.append(np.expand_dims(residue_opening3, axis=2))
    imreco1 = np.concatenate(Reco1, axis=2)
    imreco2 = np.concatenate(Reco2, axis=2)
    imreco3 = np.concatenate(Reco3, axis=2)

    return Image.fromarray(imreco1.astype(np.uint8)),Image.fromarray(imreco2.astype(np.uint8)), Image.fromarray(imreco3.astype(np.uint8))

def openingclosing_patterns(img, se0, se1, se2):
    Reco1 = []
    Reco2 = []
    Reco3 = []
    img=np.asarray(img).astype(np.uint8)
    for i in range(3):
        imFAS = erosion(img[:,:,i], se0)
        imFASbisreco1 = reconstruction(imFAS, img[:,:,i])
        imFAS_dual = erosion(255-imFASbisreco1, se0)
        imFASbisreco1_clos = 225-reconstruction(imFAS_dual, 255-imFASbisreco1)
        Reco1.append(np.expand_dims(imFASbisreco1_clos, axis=2))
        imFAS = erosion(img[:,:,i], se1)
        imFASbisreco2 = reconstruction(imFAS, img[:,:,i])
        imFAS_dual = erosion(255-imFASbisreco2, se1)
        imFASbisreco2_clos = 255-reconstruction(imFAS_dual, 255-imFASbisreco2)
        #residue_opening2 = (imFASbisreco1.astype(np.float16) - imFASbisreco2.astype(np.float16)).astype(np.uint8)
        Reco2.append(np.expand_dims(imFASbisreco2_clos, axis=2))
        imFAS = erosion(img[:,:,i], se2)
        imFASbisreco3 = reconstruction(imFAS, img[:,:,i])
        imFAS_dual = erosion(255-imFASbisreco3, se0)
        imFASbisreco3_clos = 225-reconstruction(imFAS_dual, 255-imFASbisreco3)
        #residue_opening3 = (imFASbisreco3.astype(np.float16) - imFASbisreco3.astype(np.float16)).astype(np.uint8)
        Reco3.append(np.expand_dims(imFASbisreco3_clos, axis=2))
    imreco1 = np.concatenate(Reco1, axis=2)
    imreco2 = np.concatenate(Reco2, axis=2)
    imreco3 = np.concatenate(Reco3, axis=2)

    return Image.fromarray(imreco1.astype(np.uint8)),Image.fromarray(imreco2.astype(np.uint8)), Image.fromarray(imreco3.astype(np.uint8))


def PS_opening_patterns(img, se0, se1, se2):
    imFAS = erosion(img, se0)
    imFASbisreco1 = reconstruction(np.asarray(imFAS).astype(np.uint8), np.asarray(img).astype(np.uint8))
    residue_opening1 = (np.asarray(img).astype(np.float16) - imFASbisreco1.astype(np.float16)).astype(np.uint8)
    imFAS = erosion(img, se1)
    imFASbisreco2 = reconstruction(np.asarray(imFAS).astype(np.uint8), np.asarray(img).astype(np.uint8))
    residue_opening2 = (imFASbisreco1.astype(np.float16) - imFASbisreco2.astype(np.float16)).astype(np.uint8)

    imFAS = erosion(img, se2)
    imFASbisreco3 = reconstruction(np.asarray(imFAS).astype(np.uint8), np.asarray(img).astype(np.uint8))
    residue_opening3 = (imFASbisreco2.astype(np.float16) - imFASbisreco3.astype(np.float16)).astype(np.uint8)


    return imFASbisreco3.astype(np.float16),residue_opening1.astype(np.float16),residue_opening2.astype(np.float16), residue_opening3.astype(np.float16)


class Pattern_Spectrum(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1
        #print(in0)
        #print(type(in0),in0.size())

        if(self.colorspace=='RGB'):
            img0=mrpl.tensor2im(in0.data)
            img1 = mrpl.tensor2im(in1.data)
        elif (self.colorspace == 'Lab'):
            print('error')
        img0low_res_ch0, img0R1_open_ch0, img0R2_open_ch0, img0R3_open_ch0 = PS_opening_patterns(img0[:, :, 0], se0, se1,
                                                                                          se2)
        img0low_res_ch1, img0R1_open_ch1, img0R2_open_ch1, img0R3_open_ch1 = PS_opening_patterns(img0[:, :, 1], se0, se1,
                                                                                          se2)
        img0low_res_ch2, img0R1_open_ch2, img0R2_open_ch2, img0R3_open_ch2 = PS_opening_patterns(img0[:, :, 2], se0, se1,
                                                                                          se2)

        img1low_res_ch0, img1R1_open_ch0, img1R2_open_ch0, img1R3_open_ch0 = PS_opening_patterns(img1[:, :, 0], se0, se1,
                                                                                          se2)
        img1low_res_ch1, img1R1_open_ch1, img1R2_open_ch1, img1R3_open_ch1 = PS_opening_patterns(img1[:, :, 1], se0, se1,
                                                                                          se2)
        img1low_res_ch2, img1R1_open_ch2, img1R2_open_ch2, img1R3_open_ch2 = PS_opening_patterns(img1[:, :, 2], se0, se1,
                                                                                                      se2)

        img0low_res_ch0_closi, img0R1_closi_ch0, img0R2_closi_ch0, img0R3_closi_ch0 = PS_opening_patterns(255-img0[:, :, 0], se0, se1,
                                                                                          se2)
        img0low_res_ch1_closi, img0R1_closi_ch1, img0R2_closi_ch1, img0R3_closi_ch1 = PS_opening_patterns(255-img0[:, :, 1], se0, se1,
                                                                                          se2)
        img0low_res_ch2_closi, img0R1_closi_ch2, img0R2_closi_ch2, img0R3_closi_ch2 = PS_opening_patterns(255-img0[:, :, 0], se0, se1,
                                                                                          se2)
        img1=mrpl.tensor2im(in1.data)
        img1low_res_ch0_closi, img1R1_closi_ch0, img1R2_closi_ch0, img1R3_closi_ch0 = PS_opening_patterns(255-img1[:, :, 0], se0, se1,
                                                                                          se2)
        img1low_res_ch1_closi, img1R1_closi_ch1, img1R2_closi_ch1, img1R3_closi_ch1 = PS_opening_patterns(255-img1[:, :, 1], se0, se1,
                                                                                          se2)
        img1low_res_ch2_closi, img1R1_closi_ch2, img1R2_closi_ch2, img1R3_closi_ch2 = PS_opening_patterns(255-img1[:, :, 0], se0, se1,
                                                                                                       se2)
        PS_img0= torch.tensor(np.concatenate((np.expand_dims(img0low_res_ch0, axis=0), np.expand_dims(img0R1_open_ch0, axis=0),
                        np.expand_dims(img0R2_open_ch0, axis=0), np.expand_dims(img0R3_open_ch0, axis=0),
                        np.expand_dims(img0low_res_ch1, axis=0), np.expand_dims(img0R1_open_ch1, axis=0),
                        np.expand_dims(img0R2_open_ch1, axis=0), np.expand_dims(img0R3_open_ch1, axis=0),
                        np.expand_dims(img0low_res_ch2, axis=0), np.expand_dims(img0R1_open_ch2, axis=0),
                        np.expand_dims(img0R2_open_ch2, axis=0), np.expand_dims(img0R3_open_ch2, axis=0),
                        np.expand_dims(img0low_res_ch0_closi, axis=0), np.expand_dims(img0R1_closi_ch0, axis=0),
                        np.expand_dims(img0R2_closi_ch0, axis=0), np.expand_dims(img0R3_closi_ch0, axis=0),
                        np.expand_dims(img0low_res_ch1_closi, axis=0), np.expand_dims(img0R1_closi_ch1, axis=0),
                        np.expand_dims(img0R2_closi_ch1, axis=0), np.expand_dims(img0R3_closi_ch1, axis=0),
                        np.expand_dims(img0low_res_ch2_closi, axis=0), np.expand_dims(img0R1_closi_ch2, axis=0),
                        np.expand_dims(img0R2_closi_ch2, axis=0), np.expand_dims(img0R3_closi_ch2, axis=0)), axis=0)).float()

        PS_img1 = torch.tensor(np.concatenate((np.expand_dims(img1low_res_ch0, axis=0), np.expand_dims(img1R1_open_ch0, axis=0),
                              np.expand_dims(img1R2_open_ch0, axis=0), np.expand_dims(img1R3_open_ch0, axis=0),
                              np.expand_dims(img1low_res_ch1, axis=0), np.expand_dims(img1R1_open_ch1, axis=0),
                              np.expand_dims(img1R2_open_ch1, axis=0), np.expand_dims(img1R3_open_ch1, axis=0),
                              np.expand_dims(img1low_res_ch2, axis=0), np.expand_dims(img1R1_open_ch2, axis=0),
                              np.expand_dims(img1R2_open_ch2, axis=0), np.expand_dims(img1R3_open_ch2, axis=0),
                              np.expand_dims(img1low_res_ch0_closi, axis=0), np.expand_dims(img1R1_closi_ch0, axis=0),
                              np.expand_dims(img1R2_closi_ch0, axis=0), np.expand_dims(img1R3_closi_ch0, axis=0),
                              np.expand_dims(img1low_res_ch1_closi, axis=0), np.expand_dims(img1R1_closi_ch1, axis=0),
                              np.expand_dims(img1R2_closi_ch1, axis=0), np.expand_dims(img1R3_closi_ch1, axis=0),
                              np.expand_dims(img1low_res_ch2_closi, axis=0), np.expand_dims(img1R1_closi_ch2, axis=0),
                              np.expand_dims(img1R2_closi_ch2, axis=0), np.expand_dims(img1R3_closi_ch2, axis=0)),
                             axis=0)).float()

        (N,C,X,Y) = in0.size()
        value = torch.mean(torch.mean(torch.mean((PS_img0 - PS_img1) ** 2, dim=0).view(1, X, Y), dim=1).view(1, 1, Y),
                           dim=2)
        return value

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network',net)
    print('Total number of parameters: %d' % num_params)
