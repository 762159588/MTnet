import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        #(Nclass, demb, w*h)
        attn = torch.bmm(q.transpose(1, 2), k)#way,w*h,w*h
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v.transpose(1, 2))
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head=3, d_model=640, d_k=128, d_v=128, dropout=0.1):
        super().__init__()
        self.n_head = n_head   #1
        self.d_k = d_k  #640
        self.d_v = d_v  #640

        # self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        # self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        # self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.w_qs = nn.Conv2d(d_model, n_head * d_k, kernel_size=1, stride=1, bias=False)
        self.w_ks = nn.Conv2d(d_model, n_head * d_k, kernel_size=1, stride=1, bias=False)
        self.w_vs = nn.Conv2d(d_model, n_head * d_v, kernel_size=1, stride=1, bias=False)
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.kaiming_normal_(self.w_qs.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.w_ks.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.w_vs.weight, mode='fan_out', nonlinearity='leaky_relu')

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.BatchNorm2d(d_model)

        self.fc = nn.Conv2d(n_head * d_v, d_model, kernel_size=1, stride=1, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):#proto,proto,proto  (Nclass,demb,w,h)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        way, demb,w,h = q.size()
        _, demb,w,h = k.size()
        _, demb,w,h = v.size()

        residual = q#way,c,w,h
        q = self.w_qs(q)#(Nclass,demb,w,h)
        k = self.w_ks(k)#(Nclass,demb,w,h)
        v = self.w_vs(v)#(Nclass,demb,w,h)
        
        q = q.view(way,n_head * d_k,-1)
        k = k.view(way,n_head * d_k,-1)
        v = v.view(way, n_head * d_v, -1)
        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(way, w,h, n_head * d_v)
        output = output.permute(0, 3, 1, 2)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output
    
# class FEAT(FewShotModel):
#     def __init__(self, args):
#         super().__init__(args)
#         if args.backbone_class == 'ConvNet':
#             hdim = 64
#         elif args.backbone_class == 'Res12':
#             hdim = 640
#         elif args.backbone_class == 'Res18':
#             hdim = 512
#         elif args.backbone_class == 'WRN':
#             hdim = 640
#         else:
#             raise ValueError('')
#
#         self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
#
#     def _forward(self, instance_embs, support_idx, query_idx):
#         emb_dim = instance_embs.size(-1)
#
#         # organize support/query data
#         support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(  *(support_idx.shape + (-1,)))
#         query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))
#
#         # get mean of the support
#         proto = support.mean(dim=1) # Ntask x NK x d
#         num_batch = proto.shape[0]
#         num_proto = proto.shape[1]
#         num_query = np.prod(query_idx.shape[-2:])
#
#         # query: (num_batch, num_query, num_proto, num_emb)
#         # proto: (num_batch, num_proto, num_emb)
#         proto = self.slf_attn(proto, proto, proto)#
#         if self.args.use_euclidean:
#             query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nway, 1, d)
#             proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
#             proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
#
#             logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
#         else:
#             proto = F.normalize(proto, dim=-1) # normalize for cosine distance
#             query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
#
#             logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
#             logits = logits.view(-1, num_proto)
#
#         # for regularization
#         if self.training:
#             aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim),
#                                   query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
#             num_query = np.prod(aux_task.shape[1:3])
#             aux_task = aux_task.permute([0, 2, 1, 3])
#             aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
#             # apply the transformation over the Aug Task
#             aux_emb = self.slf_attn(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
#             # compute class mean
#             aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
#             aux_center = torch.mean(aux_emb, 2) # T x N x d task_center
#
#             if self.args.use_euclidean:
#                 aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
#                 aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
#                 aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
#
#                 logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
#             else:
#                 aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
#                 aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
#
#                 logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
#                 logits_reg = logits_reg.view(-1, num_proto)
#
#             return logits, logits_reg
#         else:
#             return logits
