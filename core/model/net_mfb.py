# --------------------------------------------------------
# OpenVQA
# Licensed under The MIT License [see LICENSE for details]
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

from core.model.mfb import CoAtt
import torch
import torch.nn as nn


# -------------------------------------------------------
# ---- Main MFB/MFH model with Co-Attention Learning ----
# -------------------------------------------------------


class Net_mfb(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size, gen_func=torch.softmax):
        super(Net_mfb, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(num_embeddings=token_size,embedding_dim=__C.WORD_EMBED_SIZE)

        self.img_feat_linear = nn.Linear(__C.IMG_FEAT_SIZE,__C.HIDDEN_SIZE)


        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.LSTM_OUT_SIZE,
            num_layers=1,
            batch_first=True)

        self.gen_func=gen_func

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.dropout_lstm = nn.Dropout(__C.DROPOUT_R)
        self.backbone = CoAtt(__C, gen_func)

        if __C.HIGH_ORDER:      # MFH
            self.proj = nn.Linear(2*__C.MFB_O, answer_size)
        else:                   # MFB
            self.proj = nn.Linear(__C.MFB_O, answer_size)

    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Pre-process Language Feature
        ques_feat = self.embedding(ques_ix)     # (N, T, WORD_EMBED_SIZE)
        ques_feat = self.dropout(ques_feat)
        ques_feat, _ = self.lstm(ques_feat)     # (N, T, LSTM_OUT_SIZE)
        ques_feat = self.dropout_lstm(ques_feat)

        z = self.backbone(img_feat, ques_feat)  # MFH:(N, 2*O) / MFB:(N, O)
        proj_feat = self.proj(z)                # (N, answer_size)

        return proj_feat

    # Masking
    def make_mask(self, feature):
        return (torch.sum(torch.abs(feature),dim=-1) == 0).unsqueeze(1).unsqueeze(2)
