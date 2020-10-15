from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
from core.model.tv2d_layer_2 import TV2DFunction
from entmax import sparsemax
from functools import partial
from torch import Tensor

from core.model.basis_functions import GaussianBasisFunctions
from core.model.continuous_sparsemax import ContinuousSparsemax
from core.model.continuous_softmax import ContinuousSoftmax
import math 


# --------------------------------------------------------------
# ---- Flatten the sequence (image in continuous attention) ----
# --------------------------------------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C, gen_func=torch.softmax):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.attention=__C.attention
        self.gen_func=gen_func

        if str(gen_func)=='tvmax':
            self.sparsemax = partial(sparsemax, k=512)
            self.tvmax = TV2DFunction.apply

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True)

        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,__C.FLAT_OUT_SIZE)

        if (self.attention=='cont-sparsemax'):
            self.transform = ContinuousSparsemax(psi=None) # use basis functions in 'psi' to define continuous sparsemax
        else:
            self.transform = ContinuousSoftmax(psi=None) # use basis functions in 'psi' to define continuous softmax
        
        device='cuda'

        # compute F and G offline for one length = 14*14 = 196
        self.Gs = [None]
        self.psi = [None]
        max_seq_len=14*14 # 196 grid features
        attn_num_basis=100 # 100 basis functions
        nb_waves=attn_num_basis
        self.psi.append([])
        self.add_gaussian_basis_functions(self.psi[1],nb_waves,device=device)


        # stack basis functions
        padding=True
        length=max_seq_len
        if padding:
            shift=1/float(2*math.sqrt(length))
            positions_x = torch.linspace(-0.5+shift, 1.5-shift, int(2*math.sqrt(length)))
            positions_x, positions_y=torch.meshgrid(positions_x,positions_x)
            positions_x=positions_x.flatten()
            positions_y=positions_y.flatten()
        else:
            shift = 1 / float(2*math.sqrt(length))
            positions_x = torch.linspace(shift, 1-shift, int(math.sqrt(length)))
            positions_x, positions_y=torch.meshgrid(positions_x,positions_x)
            positions_x=positions_x.flatten()
            positions_y=positions_y.flatten()

        positions=torch.zeros(len(positions_x),2,1).to(device)
        for position in range(1,len(positions_x)+1):
            positions[position-1]=torch.tensor([[positions_x[position-1]],[positions_y[position-1]]])

        F = torch.zeros(nb_waves, positions.size(0)).unsqueeze(2).unsqueeze(3).to(device) # torch.Size([N, 196, 1, 1])
        # print(positions.size()) # torch.Size([196, 2, 1])
        basis_functions = self.psi[1][0]
        # print(basis_functions.evaluate(positions[0]).size()) # torch.Size([N, 1, 1])

        for i in range(0,positions.size(0)):
            F[:,i]=basis_functions.evaluate(positions[i])[:]

        penalty = .01  # Ridge penalty
        I = torch.eye(nb_waves).to(device)
        F=F.squeeze(-2).squeeze(-1) # torch.Size([N, 196])
        G = F.t().matmul((F.matmul(F.t()) + penalty * I).inverse()) # torch.Size([196, N])
        if padding:
            G = G[length:-length, :]
            G=torch.cat([G[7:21,:],G[35:49,:],G[63:77,:],G[91:105,:],G[119:133,:],G[147:161,:],G[175:189,:],G[203:217,:],G[231:245,:],G[259:273,:],G[287:301,:],G[315:329,:],G[343:357,:],G[371:385,:]])
        
        self.Gs.append(G.to(device))

    def add_gaussian_basis_functions(self, psi, nb_basis, device):
        
        steps=int(math.sqrt(nb_basis))

        mu_x=torch.linspace(0,1,steps)
        mu_y=torch.linspace(0,1,steps)
        mux,muy=torch.meshgrid(mu_x,mu_y)
        mux=mux.flatten()
        muy=muy.flatten()

        mus=[]
        for mu in range(1,nb_basis+1):
            mus.append([[mux[mu-1]],[muy[mu-1]]])
        mus=torch.tensor(mus).to(device)

        sigmas=[]
        for sigma in range(1,nb_basis+1):
            sigmas.append([[0.001,0.],[0.,0.001]]) # it is possible to change this matrix
        sigmas=torch.tensor(sigmas).to(device) # in continuous softmax we have sigmas=torch.DoubleTensor(sigmas).to(device)

        assert mus.size(0) == nb_basis
        psi.append(GaussianBasisFunctions(mu=mus, sigma=sigmas))

    def value_function(self, values, mask=None):
        # Approximate B * F = values via multivariate regression.
        # Use a ridge penalty. The solution is B = values * G
        # x:(batch,L,D)
        G = self.Gs[1]
        B = torch.transpose(values,-1,-2) @ G
        return B

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2),-1e9)

        if str(self.gen_func)=='tvmax':
            att = att.squeeze(-1).view(-1,14,14)
            for i in range(att.size(0)):
                att[i] = self.tvmax(att[i])
            att = self.sparsemax(att.view(-1,14*14)).unsqueeze(-1)
        
        else:
            att = self.gen_func(att.squeeze(-1), dim=-1).unsqueeze(-1)

        # compute distribution parameters
        max_seq_len=196
        length=max_seq_len

        positions_x = torch.linspace(0., 1., int(math.sqrt(length)))
        positions_x, positions_y=torch.meshgrid(positions_x,positions_x)
        positions_x=positions_x.flatten()
        positions_y=positions_y.flatten()
        positions=torch.zeros(len(positions_x),2,1).to(x.device)
        for position in range(1,len(positions_x)+1):
            positions[position-1]=torch.tensor([[positions_x[position-1]],[positions_y[position-1]]])

        # positions: (196, 2, 1)
        # positions.unsqueeze(0): (1, 196, 2, 1)
        # att.unsqueeze(-1): (batch, 196, 1, 1)
        Mu= torch.sum(positions.unsqueeze(0) @ att.unsqueeze(-1), 1) # (batch, 2, 1)
        Sigma=torch.sum(((positions @ torch.transpose(positions,-1,-2)).unsqueeze(0) * att.unsqueeze(-1)),1) - (Mu @ torch.transpose(Mu,-1,-2)) # (batch, 2, 2)
        Sigma=Sigma + (torch.tensor([[1.,0.],[0.,1.]])*1e-6).to(x.device) # to avoid problems with small values


        if (self.attention=='cont-sparsemax'):
            Sigma=9.*math.pi*torch.sqrt(Sigma.det().unsqueeze(-1).unsqueeze(-1))*Sigma

        # get `mu` and `sigma` as the canonical parameters `theta`
        theta1 = ((1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2))) @ Mu).flatten(1)
        theta2 = (-1. / 2. * (1/2. * (Sigma.inverse() + torch.transpose(Sigma.inverse(),-1,-2)))).flatten(1)
        theta = torch.zeros(x.size(0), 6, device=x.device ) #torch.Size([batch, 6])
        theta[:,0:2]=theta1
        theta[:,2:6]=theta2

        # map to a probability density over basis functions
        self.transform.psi = self.psi[1]
        r = self.transform(theta)  # batch x nb_basis

        # compute B using a multivariate regression
        # batch x D x N
        B = self.value_function(x, mask=None)

        # (bs, nb_basis) -> (bs, 1, nb_basis)
        r = r.unsqueeze(1)  # batch x 1 x nb_basis

        # (bs, hdim, nb_basis) * (bs, nb_basis, 1) -> (bs, hdim, 1)
        # get the context vector
        # batch x values_size x 1
        context = torch.matmul(B, r.transpose(-1, -2))
        context = context.transpose(-1, -2)  # batch x 1 x values_size

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))

        x_atted = torch.cat(att_list, dim=1) # don't need this for continuous attention
        
        x_atted=context.squeeze(1) # for continuous softmax/sparsemax

        x_atted = self.linear_merge(x_atted) # linear_merge is used to compute Wx
        return x_atted





# ----------------------------------------------------------------
# ---- Flatten the sequence (question and discrete attention) ----
# ----------------------------------------------------------------
# this is also used to flatten the image features with discrete attention
class AttFlatText(nn.Module):
    def __init__(self, __C, gen_func=torch.softmax):
        super(AttFlatText, self).__init__()
        self.__C = __C

        self.gen_func=gen_func

        if str(gen_func)=='tvmax':
            self.sparsemax = partial(sparsemax, k=512)
            self.tvmax = TV2DFunction.apply

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True)

        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,__C.FLAT_OUT_SIZE)

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2),-1e9)

        if str(self.gen_func)=='tvmax':
            att = att.squeeze(-1).view(-1,14,14)
            for i in range(att.size(0)):
                att[i] = self.tvmax(att[i])
            att = self.sparsemax(att.view(-1,14*14)).unsqueeze(-1)
        
        else:
            att = self.gen_func(att.squeeze(-1), dim=-1).unsqueeze(-1)
        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size, gen_func=torch.softmax):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=token_size, embedding_dim=__C.WORD_EMBED_SIZE)

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.attention=__C.attention #added this 


        #if __C.USE_IMG_POS_EMBEDDINGS:
        #    self.img_pos_x_embeddings = nn.Embedding(num_embeddings=14, embedding_dim=int(__C.HIDDEN_SIZE/2))
        #    torch.nn.init.xavier_uniform_(self.img_pos_x_embeddings.weight)
        #    self.img_pos_y_embeddings = nn.Embedding(num_embeddings=14, embedding_dim=int(__C.HIDDEN_SIZE/2))
        #    torch.nn.init.xavier_uniform_(self.img_pos_y_embeddings.weight)
        #    self.use_img_pos_embeddings = __C.USE_IMG_POS_EMBEDDINGS

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True)

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE)

        self.gen_func=gen_func
        self.backbone = MCA_ED(__C, gen_func)

        if (self.attention=='discrete'):
            self.attflat_img = AttFlatText(__C, self.gen_func)
        else: # use continuous attention 
            self.attflat_img = AttFlat(__C, self.gen_func)

        self.attflat_lang = AttFlatText(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

        


    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        #if self.use_img_pos_embeddings:
        #    for i in range(img_feat.size(0)):
        #        pos = torch.LongTensor(np.mgrid[0:14,0:14]).cuda()
        #        img_feat[i]+=torch.cat([self.img_pos_x_embeddings(pos[0].view(-1)), self.img_pos_y_embeddings(pos[1].view(-1))],1)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask)

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask)

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask)

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature),dim=-1) == 0).unsqueeze(1).unsqueeze(2)
