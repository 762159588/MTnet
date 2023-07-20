import random
import numpy as np
import torch.nn as nn
from models.resnet import ResNet
from patchmaker import Patchmake,Hole_make
import torch.nn.functional as F
import torch
import math

import matplotlib.pyplot as plt
##############################################################################
def compute_accuracy(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.float).mean().item() * 100.

def euclidean_metric(a, b):#query，suport
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)

    return logits




class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss


class SupCCL(nn.Module):
    def __init__(self, args):
        super(SupCCL, self).__init__()
        self.args = args
        self.kl = KLDiv(T=args.kd_T)

    def forward(self, ebg, ebp, labels):

        # embedingg（N，C）,embedignp(p*n,c)
        # labelsg（N）labelsp(p*n)


        ebp = ebp.reshape(self.args.patch_num-self.args.mask_num,self.args.batch,-1)
        ebp_bar = ebp.mean(dim=0)


        soft_dil_loss = 0.

        for i,j in zip(ebg,ebp_bar):

            soft_dil_loss += self.kl(i, j.detach(),dim=0)
            soft_dil_loss += self.kl(j, i.detach(),dim=0)
####################################################################3



        soft_dcl_loss = 0.
        ebp = ebp.transpose(0,1) #(n,p-m,c)
        for i, j ,k in zip(ebg, ebp_bar,ebp):
            for l in k:
                soft_dcl_loss += self.kl((i-l)**2, (j-l)**2,dim=0)  # 计算kl散度

                soft_dcl_loss += self.kl((j-l)**2, (i-l)**2,dim=0)
        soft_dcl_loss = soft_dcl_loss/(self.args.patch_num-self.args.mask_num)
        return  soft_dil_loss,soft_dcl_loss


class Sup_CCL_Loss(nn.Module):
    def __init__(self, args):
        super(Sup_CCL_Loss, self).__init__()

        self.args = args
        self.glo = Embed(640, 128)
        self.pat = Embed(640, 128)
        ##########这里开始在做一个局部和全局的非线性映射


        self.contrast = SupCCL(args)

    def forward(self, ebg,ebp, labelsg):

        ebg = self.glo(ebg)
        ebp = self.pat(ebp)

        soft_dil_loss,  soft_dcl_loss = \
            self.contrast(ebg, ebp, labelsg)

        return soft_dil_loss,  soft_dcl_loss


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.proj_head = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out)
        )
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.proj_head(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class KLDiv(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(KLDiv, self).__init__()
        self.T = T

    def forward(self, y_s, y_t,dim=1):
        p_s = F.log_softmax(y_s / self.T, dim=dim)
        p_t = F.softmax(y_t / self.T, dim=dim)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss





##############################################################################
class ResNet_n(nn.Module):
    def __init__(self,  args):
        #(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, number_net=number_net)
        super(ResNet_n, self).__init__()
        self.args = args
        self.encoderg = ResNet(args)
        self.encoderp = ResNet(args)
        self.fc = nn.Linear(640, args.num_classes)
        self.fc2 = nn.Linear(640, args.num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_div = DistillKL(3)
        self.criterion_ccl = Sup_CCL_Loss(args)

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def random_mask(self,patch_num,mask_num):
        parch_index = range(0,patch_num)
        index = random.sample(parch_index,mask_num)
        mask = torch.ones(patch_num)
        for i in index:
            mask[i] = 0
        return mask

    def random_delete(self,tensor,patch_num,mask_num):
        #in:(p,n,c,w,h)
        parch_index = range(0,patch_num)
        index = random.sample(parch_index,mask_num)
        index.sort(reverse=True)
        for j in index:
            tensor = tensor[torch.arange(tensor.size(0)) != j]
        return tensor

    def forward(self, x,target,train = False):
        ##input(N,3,84,84)
        if train:
            ###patch_mode
            targetg = target.cuda()
            targetp = target.repeat(self.args.patch_num-self.args.mask_num).cuda()
            patch = Patchmake(self.args,x) #(N,P,C,W,H)
            N, P, C, W, H=patch.size()
            patch = patch.transpose(0,1) #(P,N,C,W,H)
            patch = self.random_delete(patch,self.args.patch_num,self.args.mask_num)

            patch = patch.reshape(-1,C,W,H)##(P*N,C,W,H)

            outputg = self.encoderg(x)    #n,c,w,h
            embedingg = self.avgpool(outputg).squeeze() #n,c
            embedingg = self.gaussian_normalize(embedingg,dim = 1)

            outputp = self.encoderp(patch)   #(p*n,c,w,h)
            embedingp = self.avgpool(outputp).squeeze()  #p*n,c
            embedingp = self.gaussian_normalize(embedingp, dim=1)

            logitg = self.fc(embedingg).cuda()
            logitp = self.fc2(embedingp).cuda()
            logitpmean = logitp.view(self.args.patch_num-self.args.mask_num,-1,logitp.shape[-1]).mean(dim=0)


            loss_cls = self.criterion_ce(logitp, targetp) + self.criterion_ce(logitg, targetg)  ##总的分类损失######################这里改成了pmean
            loss_cls =loss_cls.cuda()
            ensemble_logits = (logitpmean + logitg)/2
            loss_logit_kd = self.criterion_div(logitpmean, ensemble_logits)+self.criterion_div(logitg, ensemble_logits) ###两个支路的预测偏差
            loss_logit_kd = loss_logit_kd.cuda()
            loss_patch_bias = (logitp.view(self.args.patch_num-self.args.mask_num,-1,logitp.shape[-1])-logitpmean.unsqueeze(0))/math.sqrt(self.args.num_classes)   ##patch内部差异
            loss_patch_bias = loss_patch_bias.sum().cuda()/math.sqrt(self.args.patch_num-self.args.mask_num)
            l2,l4 = self.criterion_ccl(embedingg,embedingp,targetg)
            loss_ccl = l2+l4

            loss = loss_ccl+loss_logit_kd+loss_cls+loss_patch_bias
            # loss = loss_logit_kd + loss_cls + loss_patch_bias
            acc = compute_accuracy(logitg, targetg)+compute_accuracy(logitp, targetp)
            accg = compute_accuracy(logitg, targetg)
            accp = compute_accuracy(logitp, targetp)
            return loss,acc,accg,accp


            # ###mask_mode
            # targetg = target.cuda()
            # targetp = target.cuda()
            # masked = Hole_make(self.args, x)
            #
            # outputg = self.encoderg(x)  # n,c,w,h
            # embedingg = self.avgpool(outputg).squeeze()  # n,c
            # embedingg = self.gaussian_normalize(embedingg, dim=1)
            #
            # outputp = self.encoderp(masked)  # (p*n,c,w,h)
            # embedingp = self.avgpool(outputp).squeeze()  # p*n,c
            # embedingp = self.gaussian_normalize(embedingp, dim=1)
            #
            # logitg = self.fc(embedingg).cuda()
            # logitp = self.fc2(embedingp).cuda()
            #
            # loss_cls = self.criterion_ce(logitp, targetp) + self.criterion_ce(logitg, targetg)  ##总的分类损失
            # loss_cls = loss_cls.cuda()
            # ensemble_logits = (logitp + logitg) / 2
            # loss_logit_kd = self.criterion_div(logitp, ensemble_logits) + self.criterion_div(logitg,
            #                                                                                      ensemble_logits)  ###两个支路的预测偏差
            # loss_logit_kd = loss_logit_kd.cuda()
            #
            # l2, l4 = self.criterion_ccl(embedingg, embedingp, targetg)
            # loss_ccl = l2 + l4
            #
            # loss = lossccl + loss_logit_kd + loss_cls
            # acc = compute_accuracy(logitg, targetg) + compute_accuracy(logitp, targetp)
            # accg = compute_accuracy(logitg, targetg)
            # accp = compute_accuracy(logitp, targetp)
            # return loss, acc, accg, accp
        else:

            # #patch_mode
            outputg = self.encoderg(x)  #n,c,w,h
            patch = Patchmake(self.args,x)
            N, P, C, W, H=patch.size()
            patch = patch.transpose(0,1)
            patch = patch.reshape(-1,C,W,H)
            outputp = self.encoderp(patch) #(p*n,c,w,h)
            outputg = outputg.mean([-1,-2])
            outputp = outputp.mean([-1,-2])

            outputp = self.gaussian_normalize(outputp, dim=1)
            outputg = self.gaussian_normalize(outputg, dim=1)

            outputp = outputp.view(P,N,-1).mean(dim=0)
            emb = torch.cat([outputp,outputg],dim=1)
            supset,queset = emb[:self.args.way*self.args.shot],emb[self.args.way*self.args.shot:]
            proto = supset.contiguous().view(self.args.shot,self.args.way,-1).mean(dim = 0)
            queset = queset.contiguous().view(queset.shape[0],-1)
            logit_global =  euclidean_metric(queset, proto)
            logit_global = F.softmax(logit_global,dim = 1)

            #masked_mode
            # outputg = self.encoderg(x)  #n,c,w,h
            # masked = Hole_make(self.args, x)
            #
            #
            #
            # outputp = self.encoderp(masked ) #(p*n,c,w,h)
            # outputg = outputg.mean([-1,-2])
            # outputp = outputp.mean([-1,-2])
            #
            # outputp = self.gaussian_normalize(outputp, dim=1)
            # outputg = self.gaussian_normalize(outputg, dim=1)
            #
            # emb = torch.cat([outputp,outputg],dim=1)
            # supset,queset = emb[:self.args.way*self.args.shot],emb[self.args.way*self.args.shot:]
            # proto = supset.contiguous().view(self.args.shot,self.args.way,-1).mean(dim = 0)
            # queset = queset.contiguous().view(queset.shape[0],-1)
            # logit_global =  euclidean_metric(queset, proto)
            # logit_global = F.softmax(logit_global,dim = 1)

            return 1,logit_global,1,1



class Args():
    def __init__(self,ways,shots):
        self.way = ways
        self.shot = shots
        self.patch_num = 10
        self.num_classes = 64
        self.feat_dim=640
        self.kd_T = 3
        self.tau = 3
        self.batch = 5
    def forward(self):

        return 0



#
if __name__ == '__main__':

    args = Args(5,5)

    a = torch.rand(5,3,84,84).cuda()
    t = torch.randint(0,8,(5,1)).squeeze().cuda()
    v = ResNet_n(args).cuda()
    p = v(a,t,True)
    print(p)

    print('代码没问题真是太好了')
         #查看可训练参数



