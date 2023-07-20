import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import ResNet
from dataprocess import process
from feat import MultiHeadAttention
from tworesnet import ResNet_n


class SandGlassBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_c,
                                 out_features=in_c * 2,
                                 bias=False)

        # self.bn1 = nn.BatchNorm1d(in_c * 2)
        self.linear2 = nn.Linear(in_features=in_c * 2,
                                 out_features=2,
                                 bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.linear1(x)

        # output = self.bn1(output)
        output = F.normalize(output,2,dim=1)

        output = self.relu(output)
        output = self.linear2(output)
        output = torch.tanh(output)
        output = 1 + output

        return output




def euclidean_metric(a, b):#query，suport
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)

    return logits

def uncertainty_metric(a, b,beta):#query，suport
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    distance = ((a - b)**2).sum(dim=2)
    beta = beta.unsqueeze(0).repeat(n,1)
    score = torch.div(beta,5*distance)

    logits = -torch.mul(distance,torch.exp(-score))


    return logits


class MTNet(nn.Module):
    def __init__(self,args,mode = None):
        super(MTNet, self).__init__()

        self.args = args
        self.mode = mode

        self.encoderg = ResNet(args)
        self.encoderp = ResNet(args)
        self.pixel_att = MultiHeadAttention()
        self.alphas = SandGlassBlock(640)

        # self.pretrain = ResNet_n(args)



    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x


    def forward(self,x):   ####################微调
    # def forward(self,x,label,flag):   ####?###########预训练
        #input(N,3,84,84)
        if self.mode == 'pretraining':

            label = label

            loss,acc,accg,accp = self.pretrain(x,label,train=flag)
            return loss,acc,accg,accp

        if self.mode == 'encoder':


            # x4,lvl = self.encoder(x)

            # with torch.no_grad():
            embg = self.encoderg(x)

            # patches = Patchmake(self.args,x)
            # N, P, C, W, H=patches.size()
            # patches = patches.transpose(0,1)
            # patches = patches.reshape(-1,C,W,H)#p*n,c,w,h
            embp = self.encoderp(x)

            return embg,embp #(N,640,5,5),(P*N,640,ww,hh)
        if self.mode == 'ma':
            embg, embp = x
            #embg(n,640,5,5)
            #embp(p*n,640,1,1)

            ##global
            data_global = embg.mean([-1,-2])
            data_global = process(data_global)
            supset,queset = data_global[:self.args.way*self.args.shot],data_global[self.args.way*self.args.shot:]
            proto = supset.contiguous().view(self.args.shot,self.args.way,-1).mean(dim = 0)
            queset = queset.contiguous().view(queset.shape[0],-1)
            logit_global =  euclidean_metric(queset, proto)
            logit_global = F.softmax(logit_global,dim = 1)


            ##patch
            # datapatch = process(embp)
            datapatch = embp

            #n,c,w,h
            supsetp, quesetp = datapatch[:self.args.way * self.args.shot], datapatch[self.args.way * self.args.shot:]
            supsetp = supsetp.reshape(self.args.shot,self.args.way,supsetp.shape[1],supsetp.shape[2],supsetp.shape[3])#s,w,c,w,h
            supsetp = supsetp.mean(dim = 0)#w,c,w,h
            way,c,w,h = supsetp.size()
            q,_,_,_ = quesetp.size()
            #que:q,c,w,h
            supsetp =  self.pixel_att(supsetp,supsetp,supsetp).reshape(way,c,-1)
            quesetp = self.pixel_att(quesetp, quesetp,quesetp).reshape(q, c, -1)
            # quesetp = quesetp.reshape(quesetp.shape[0],quesetp.shape[1],-1)
            supsetp = self.gaussian_normalize(supsetp, dim=1)
            quesetp = self.gaussian_normalize(quesetp, dim=1)
            logit_patch = torch.einsum('wcp,qcs->qwps',supsetp,quesetp)
            logit_patch,_ = torch.topk(logit_patch,5,dim=-1,sorted = False)
            logit_patch,_ = torch.topk(logit_patch,5 , dim=-2, sorted=False)
            logit_patch = logit_patch.mean([-1,-2])

            logit_patch = F.softmax(logit_patch, dim=1)

            ##pixel
            # logit_pixel = self.pixel_att(x)

            # ##alpha
            protoa = proto.unsqueeze(1)
            protob = proto.unsqueeze(0)
            protoc = (protoa-protob)**2
            goal,_ = torch.max(protoc,dim=1)
            goal_sum = torch.sum(goal,dim = 0,keepdim=True)
            alpha = self.alphas(goal_sum)
            alpha = F.softmax(alpha,dim=1).squeeze()
            #
            #
            # ##logit
            logit = alpha[0]*logit_global+alpha[1]*logit_patch

            logit = F.softmax(logit,dim=1)



            return logit,logit_global,logit_patch

        if self.mode == 'fc':
            x = x.mean(dim=[-1, -2])
            x = self.fc2(x)

            logit = F.softmax(x,dim=1)




            return logit










###################################testcode#####################

class Args():
    def __init__(self,ways,shots):
        self.ways = ways
        self.shots = shots
        self.num_class = 64

    def forward(self):

        return 0



#
if __name__ == '__main__':

    args = Args(5,5)

    a = torch.rand(75,3,84,84)
    v = MTNet(args)
    p = v(a)
    print(p)

    print('代码没问题真是太好了')
         #查看可训练参数
