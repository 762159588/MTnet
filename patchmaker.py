import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random


# class Patchmake(nn.Module):
#     def __init__(self,args):
#         super(Patchmake, self).__init__()
#         self.args = args
#
#
#     def forward(self,x):
#         #input X(N,C,W,H)
#         #OUTPUT P(N,P,c,w,h)
#         data = x
#         N,C,W,H = data.size()
#         W_patchsize = W // self.args.patch_num
#         H_patchsize = H // self.args.patch_num
#         W_patchsizehalf = W_patchsize // 2
#         H_patchsizehalf = H_patchsize // 2
#
#         img_patchlist = list()
#         for img in data:
#             patch_list = list()
#             for i in range(self.args.patch_num):
#                 patch = img.clone()
#                 randomcenterx = random.randint(W_patchsizehalf, W - W_patchsizehalf-1)
#                 randomcentery = random.randint(H_patchsizehalf, H - H_patchsizehalf-1)
#                 patchi = patch[:,(randomcenterx-W_patchsizehalf):(randomcenterx+W_patchsizehalf+1),(randomcentery-H_patchsizehalf):(randomcentery+H_patchsizehalf+1)]
#                 patch_list.append(patchi)
#             patch_list = torch.stack(patch_list,dim=0)
#             img_patchlist.append(patch_list)
#
#         img_patchlist = torch.stack(img_patchlist,dim=0)  #(N,P,c,w,h)
#         return img_patchlist

# def Patchmake(args,x,random=False):
#     if random:
#         data = x
#         N,C,W,H = data.size()
#         bsize = math.floor(math.sqrt(args.patch_num)) #bsize*bsize = patch_num 即每行/列的batch数目
#         W_patchsize = W // bsize  #w上每个batch尺寸
#         H_patchsize = H // bsize  #h上每个batch尺寸
#         W_patchsizehalf = W_patchsize // 2
#         H_patchsizehalf = H_patchsize // 2
#         img_patchlist = list()
#         for img in data:
#             patch_list = list()
#             for i in range(args.patch_num):
#                 patch = img.clone()
#                 randomcenterx = random.randint(W_patchsizehalf, W - W_patchsizehalf-1)
#                 randomcentery = random.randint(H_patchsizehalf, H - H_patchsizehalf-1)
#                 patchi = patch[:,(randomcenterx-W_patchsizehalf):(randomcenterx+W_patchsizehalf),(randomcentery-H_patchsizehalf):(randomcentery+H_patchsizehalf)]
#                 patch_list.append(patchi)
#             patch_list = torch.stack(patch_list,dim=0)
#             img_patchlist.append(patch_list)
#
#         img_patchlist = torch.stack(img_patchlist,dim=0)  #(N,P,c,w,h)
#         return img_patchlist
#     else:
#         data = x
#         N, C, W, H = data.size()
#         bsize = math.floor(math.sqrt(args.patch_num))
#         W_patchsize = W // bsize
#         H_patchsize = H // bsize
#         img_patchlist = list()
#         for img in data:
#             patch_list = list()
#             for i in range(0,bsize):
#                 for j in range(0,bsize):
#                     patch = img.clone()
#                     patchi = patch[:, (i*W_patchsize):((i+1)*W_patchsize),
#                              (j*H_patchsize):((j+1)*H_patchsize)]
#                     patch_list.append(patchi)
#             patch_list = torch.stack(patch_list,dim=0)
#             img_patchlist.append(patch_list)
#
#         img_patchlist = torch.stack(img_patchlist,dim=0)  #(N,P,c,w,h)
#         return img_patchlist


def Patchmake(args, x, ran_san=False):
###默认不随机采样，想随机采样的话ransan-true
    data = x
    N, C, W, H = data.size()
    bsize = math.floor(math.sqrt(args.patch_num))  # bsize*bsize = patch_num 即每行/列的batch数目
    W_patchsize = W // bsize  # w上每个batch尺寸
    H_patchsize = H // bsize  # h上每个batch尺寸
    W_patchsizehalf = W_patchsize // 2
    H_patchsizehalf = H_patchsize // 2
    img_patchlist = list()
    for img in data:
        patch_list = list()
        if ran_san:
            for i in range(args.patch_num):
                patch = img.clone()
                randomcenterx = random.randint(W_patchsizehalf, W - W_patchsizehalf - 1)
                randomcentery = random.randint(H_patchsizehalf, H - H_patchsizehalf - 1)
                patchi = patch[:, (randomcenterx - W_patchsizehalf):(randomcenterx + W_patchsizehalf+1),
                         (randomcentery - H_patchsizehalf):(randomcentery + H_patchsizehalf+1)]
                patch_list.append(patchi)
        else:
            for i in range(0, bsize):
                for j in range(0, bsize):
                    patch = img.clone()
                    patchi = patch[:, (i * W_patchsize):((i + 1) * W_patchsize),
                             (j * H_patchsize):((j + 1) * H_patchsize)]
                    patch_list.append(patchi)
        patch_list = torch.stack(patch_list, dim=0)
        img_patchlist.append(patch_list)

    img_patchlist = torch.stack(img_patchlist, dim=0)  # (N,P,c,w,h)
    return img_patchlist



def Hole_make(args, x):
##这个函数在输入的图片上挖洞，每个洞的尺寸，洞的数量预先定义好
    data = x
    N, C, W, H = data.size()#图片的尺寸
    rate = args.mask_rate





    mask_list = list()

    for _ in range(N):
        mask = torch.ones((C,W,H))
        mask_num = random.randint(1, 80)
        rate_per_patch = rate / mask_num
        bian_rate = math.sqrt(rate_per_patch)
        w_mid = W * bian_rate
        h_mid = H * bian_rate
        w_min = round(w_mid * 0.5)
        w_max = round(w_mid * 1.5)
        h_min = round(h_mid * 0.5)
        h_max = round(h_mid * 1.5)
        for i in range(mask_num):
            length_ww = random.randint(w_min, w_max)
            length_hh = random.randint(h_min, h_max)
            W_patchsizehalf = length_ww // 2
            H_patchsizehalf = length_hh // 2
            randomcenterx = random.randint(W_patchsizehalf, W - W_patchsizehalf - 1)
            randomcentery = random.randint(H_patchsizehalf, H - H_patchsizehalf - 1)
            mask[:, (randomcenterx - W_patchsizehalf):(randomcenterx + W_patchsizehalf + 1),
            (randomcentery - H_patchsizehalf):(randomcentery + H_patchsizehalf + 1)] = 0
        mask_list.append(mask)
    mask_list = torch.stack(mask_list, dim=0).cuda()
    new_data = mask_list*data.clone()
    return  new_data







class Args():
    def __init__(self):
        self.patch_num = 16
        self.mask_min = 2  # patch的最小尺寸
        self.mask_rate = 0.5 ##patch的最大尺寸
        self.mask_num = 3  ##patch的数量

    def forward(self):

        return 0



#
if __name__ == '__main__':

    args = Args()

    a = torch.rand(5,3,8,8).cuda()
    v = Hole_make(args,a)

    print(v.size())

    print('代码没问题真是太好了')




