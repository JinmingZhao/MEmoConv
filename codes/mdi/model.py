import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, itertools, random, copy, math
import copy
from transformers import BertModel, BertConfig, AutoConfig, AutoModel, AutoModelForSequenceClassification
import os
import pynvml

class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            #torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim) cand_dim == mem_dim?
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2) # batch, seq_len, mem_dim
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            # alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim
        return attn_pool, alpha


class LSTMModel(nn.Module):

    def __init__(self, D_m, D_e, D_h, n_classes=7, dropout=0.5):

        super(LSTMModel, self).__init__()

        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type='general2')
        self.linear = nn.Linear(2 * D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []

        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))

        return hidden


class AutoModelBert(nn.Module):
    def __init__(self, args, num_labels):
        super().__init__()
        config = AutoConfig.from_pretrained(args.bert_path, num_labels=num_labels)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            args.bert_path,
            from_tf=bool(".ckpt" in args.bert_path),
            config=config
        )

    def forward(self, content_ids, content_masks):

        logits = self.classifier(content_ids, attention_mask=content_masks, return_dict=False)[0]
        logits = logits# [B, cls]

        return logits


def get_local_attention_mask(length, batch, window, nheads=4):
    # local attention mask
    # mask: B*heads, n, n
    attention_matrix = torch.zeros((batch, length, length)).cuda()
    for i in range(0, min(window, length)):
        temp = torch.eye(length-i).cuda()
        if i == 0:
            attention_matrix[:, i:, :] += temp
            attention_matrix[:, :, i:] += temp
        else:
            attention_matrix[:, i:, :-i] += temp
            attention_matrix[:, :-i, i:] += temp
    attention_mask = (attention_matrix == 0)
    heads_attn_mask = []
    for b in range(len(attention_mask)):
        heads_attn_mask.append(attention_mask[b].unsqueeze(0).repeat(nheads, 1, 1))  # heads, n, n

    heads_attn_mask = torch.cat(heads_attn_mask, dim=0)  # B*heads, n, n
    return torch.cuda.BoolTensor(heads_attn_mask)


def get_seq_attention_mask(length, batch, reverse=False, nheads=4):
    # global attention mask
    print(length, batch)
    attention_matrix = torch.zeros((batch, length, length)).cuda()
    if not reverse:
        attention_matrix += torch.triu(torch.ones(length, length), diagonal=0).cuda()
    else:
        attention_matrix += torch.tril(torch.ones(length, length), diagonal=0).cuda()
    attention_mask = (attention_matrix == 0)
    heads_attn_mask = []
    for b in range(len(attention_mask)):
        heads_attn_mask.append(attention_mask[b].unsqueeze(0).repeat(nheads, 1, 1))  # heads, n, n

    heads_attn_mask = torch.cat(heads_attn_mask, dim=0)  # B*heads, n, n
    return torch.cuda.BoolTensor(heads_attn_mask)

def get_attention_mask(speaker_ids, mode="inter", nheads=4):
    #inter: speaker间，speaker相同为0，不同为1；intra：speaker内，speaker相同为1，不同为0
    #speaker_ids: [n, B]
    # print('mode {} speaker ids {}'.format(mode, speaker_ids))
    temp_speaker_ids = speaker_ids.transpose(0, 1)
    #result: [B, n, n]
    #L -> [n, n]: speaker_ids行向量复制L次，分别减去L列向量，如果结果为0说明speaker相同，否则不同
    matrix_a = torch.repeat_interleave(temp_speaker_ids.unsqueeze(dim=2), repeats=temp_speaker_ids.shape[1], dim=2)
    matrix_b = torch.repeat_interleave(temp_speaker_ids.unsqueeze(dim=1), repeats=temp_speaker_ids.shape[1], dim=1)
    matrix = matrix_a - matrix_b
    attention_mask = torch.zeros_like(matrix, dtype=torch.uint8)
    if mode == "inter":
        attention_mask = ((matrix + torch.eye(matrix.shape[-1], dtype=torch.uint8).cuda())==0) # True不允许attention，False允许
    elif mode == "intra":
        attention_mask = (matrix!=0) # B, n, n
    # print('---- results is OK on bs=1: {}'.format(attention_mask))

    heads_attn_mask = []
    for b in range(len(attention_mask)):
        heads_attn_mask.append(attention_mask[b].unsqueeze(0).repeat(nheads, 1, 1)) #heads, n, n
    heads_attn_mask = torch.cat(heads_attn_mask, dim=0) #B*heads, n, n
    return torch.cuda.BoolTensor(heads_attn_mask)


class speaker_TRM(nn.Module):
    def __init__(self, layers, in_dim, n_heads, ff_dim, dropout,
                 entire=False, attn_type=('global', 'intra', 'inter', 'local'),
                 same=False, window=3):
        super().__init__()
        self.layers = layers
        self.n_heads = n_heads
        self.entire = entire  # entire 只进行一次交互 否则每层一次交互
        self.attn_type = attn_type
        self.window = window

        encoder_layer = nn.TransformerEncoderLayer(in_dim, nhead=n_heads,
                                                   dim_feedforward=ff_dim, dropout=dropout)
        encoder_norm = nn.LayerNorm(in_dim)
        self.norm = copy.deepcopy(encoder_norm)
        if entire:
            encoder =  nn.TransformerEncoder(encoder_layer, layers, encoder_norm)
        else:
            encoder = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(layers)])
        if same:
            if 'global' in attn_type:
                self.global_encoder = encoder
            if 'intra' in attn_type:
                self.intra_encoder = encoder
            if 'inter' in attn_type:
                self.inter_encoder = encoder
            if 'local' in attn_type:
                self.local_encoder = encoder
        else:
            if 'global' in attn_type:
                self.global_encoder = copy.deepcopy(encoder)
            if 'intra' in attn_type:
                self.intra_encoder = copy.deepcopy(encoder)
            if 'inter' in attn_type:
                self.inter_encoder = copy.deepcopy(encoder)
            if 'local' in attn_type:
                self.local_encoder = copy.deepcopy(encoder)

    def forward(self, inputs, src_key_padding_mask, speaker_ids, mask=None): #[n, B, utr_dim]
        features = inputs
        if 'intra' in self.attn_type:
            intra_mask = get_attention_mask(speaker_ids, mode="intra", nheads=self.n_heads)
        if 'inter' in self.attn_type:
            inter_mask = get_attention_mask(speaker_ids, mode="inter", nheads=self.n_heads)
        if 'local' in self.attn_type:
            local_mask = get_local_attention_mask(speaker_ids.shape[0], speaker_ids.shape[1], self.window, nheads=self.n_heads)
        global_features, local_features, intra_features, inter_features = torch.zeros_like(features).cuda(), \
                                torch.zeros_like(features).cuda(), torch.zeros_like(features).cuda(), torch.zeros_like(features).cuda()
        if self.entire:
            if 'global' in self.attn_type:
                global_features = self.global_encoder(features, mask=mask, src_key_padding_mask=src_key_padding_mask)
            if 'intra' in self.attn_type:
                intra_features = self.intra_encoder(features, mask=intra_mask, src_key_padding_mask=src_key_padding_mask)
            if 'inter' in self.attn_type:
                inter_features = self.inter_encoder(features, mask=inter_mask, src_key_padding_mask=src_key_padding_mask)
            if 'local' in self.attn_type:
                local_features = self.local_encoder(features, mask=local_mask, src_key_padding_mask=src_key_padding_mask)
            features = global_features + intra_features + inter_features + local_features
        else:
            for l in range(self.layers):
                if 'global' in self.attn_type:
                    global_features = self.global_encoder[l](features, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
                if 'intra' in self.attn_type:
                    intra_features = self.intra_encoder[l](features, src_mask=intra_mask, src_key_padding_mask=src_key_padding_mask)
                if 'inter' in self.attn_type:
                    inter_features = self.inter_encoder[l](features, src_mask=inter_mask, src_key_padding_mask=src_key_padding_mask)
                if 'local' in self.attn_type:
                    local_features = self.local_encoder[l](features, src_mask=local_mask, src_key_padding_mask=src_key_padding_mask)
                features = global_features + intra_features + inter_features + local_features
        features = self.norm(features)
        return features

class MMGatedAttention(nn.Module):

    def __init__(self, dim, dropout=0.1):
        '''
        以a,v modal的信息作为背景信息融合到l模态中
        return seq, hidden
        '''
        super(MMGatedAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.transform_l = nn.Linear(dim, dim, bias=True)
        self.transform_v = nn.Linear(dim, dim, bias=True)
        self.transform_a = nn.Linear(dim, dim, bias=True)
        self.transform_av = nn.Linear(dim*3,1)
        self.transform_al = nn.Linear(dim*3,1)
        self.transform_vl = nn.Linear(dim*3,1)

    def forward(self, l, a=None, v=None): #[n, b, dim]
        hl = self.dropout(nn.Tanh()(self.transform_l(l)))
        if a is not None:
            ha = self.dropout(nn.Tanh()(self.transform_a(a)))
            z_al = torch.sigmoid(self.transform_al(torch.cat([a, l, a * l], dim=-1)))
            h_al = z_al * ha + (1 - z_al) * hl
        if v is not None:
            hv = self.dropout(nn.Tanh()(self.transform_v(v)))
            z_vl = torch.sigmoid(self.transform_vl(torch.cat([v, l, v * l], dim=-1)))
            h_vl = z_vl * hv + (1 - z_vl) * hl

        if a is not None and v is not None:
            z_av = torch.sigmoid(self.transform_av(torch.cat([a, v, a * v], dim=-1)))
            h_av = z_av * ha + (1 - z_av) * hv
            return h_al + h_vl + h_av
        elif a is None:
            return h_vl
        elif v is None:
            return h_al


class GatedAttention(nn.Module):

    def __init__(self, dim1, dim2, dropout=0.1):
        super(GatedAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.transform_x = nn.Linear(dim1, dim2, bias=True)
        self.transform_y = nn.Linear(dim1, dim2, bias=True)
        self.transform_xy = nn.Linear(dim1*3, 1)

    def forward(self, x, y): #[n, b, dim]
        hx = self.dropout(nn.Tanh()(self.transform_x(x)))
        hy = self.dropout(nn.Tanh()(self.transform_y(y)))
        zxy = torch.sigmoid(self.transform_xy(torch.cat([x, y, x * y], dim=-1)))
        hxy = zxy * hx + (1-zxy) * hy

        return hxy


class RelativePositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)
        self.d_model = d_model
        self.clamp_len = -1

    def forward(self, x):
        qlen = x.shape[0]
        bsz = x.shape[1]
        pos_emb = self.relative_positional_encoding(qlen, bsz)

        return x+pos_emb

    def relative_positional_encoding(self, klen, bsz=None):
        def positional_embedding(pos_seq, inv_freq, bsz=None):
            sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
            pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
            pos_emb = pos_emb[:, None, :]

            if bsz is not None:
                pos_emb = pos_emb.expand(-1, bsz, -1)

            return pos_emb

        freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
        inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))

        beg, end = klen-1, -1

        fwd_pos_seq = torch.arange(beg, end, -1.0)
        if self.clamp_len > 0:
            fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
        pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz)
        pos_emb = pos_emb.cuda()
        return pos_emb


class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :] * 0.1
        return self.dropout(x)


def get_trm_encoder(layers, in_dim, n_heads, ff_dim, dropout):
    encoder_layer = nn.TransformerEncoderLayer(in_dim, nhead=n_heads,
                                               dim_feedforward=ff_dim, dropout=dropout)
    encoder_norm = nn.LayerNorm(in_dim)
    encoder = nn.TransformerEncoder(encoder_layer, layers, encoder_norm)
    return encoder


class BertEncoder(nn.Module):
    def __init__(self, code_length, args):  # code_length为最终获得embedding的维度
        super().__init__()

        self.args = args
        self.code_length = code_length
        modelConfig = AutoConfig.from_pretrained(args.bert_path)
        if not args.use_utt_text_features:
            self.textExtractor = AutoModel.from_pretrained(args.bert_path, config=modelConfig)
            embedding_dim = args.bert_dim
        else:
            embedding_dim = args.text_dim

        fc_in_dim = embedding_dim
        if args.bert_feature_type == 'cat':
            fc_in_dim += embedding_dim

        if len(args.modals) > 1 and args.mm_type == 'ecat':
            fc_in_dim = 0
            if 'a' in args.modals:
                fc_in_dim += args.audio_dim
            if 'v' in args.modals:
                fc_in_dim += args.visual_dim
            if 'l' in args.modals:
                fc_in_dim += args.text_dim
        elif len(args.modals) > 1 and args.mm_type in ['eadd', 'egate']:
            if 'a' in args.modals:
                self.a_fc = nn.Linear(args.audio_dim, code_length)
            if 'v' in args.modals:
                self.v_fc = nn.Linear(args.visual_dim, code_length)
            if 'l' in args.modals:
                self.l_fc = nn.Linear(args.text_dim, code_length)
            if args.mm_type == 'egate':
                self.gate = MMGatedAttention(fc_in_dim, dropout=args.dropout)

        if not self.args.bert_wo_fc:
            self.fc = nn.Linear(fc_in_dim, code_length)
        self.act = nn.Tanh()

    def get_text_embeddings(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks,
                                    output_hidden_states=True)
        if self.args.bert_feature_type == 'cls':
            text_embeddings = output[0][:, 0, :]
        elif self.args.bert_feature_type == 'mpl':
            text_embeddings = torch.mean(output[0], dim=1)  # [B, emb]
        elif self.args.bert_feature_type == 'cat':
            text_embeddings = torch.cat([output[0][:, 0, :], torch.mean(output[0], dim=1)], dim=1)
        elif self.args.bert_feature_type == 'pool':
            text_embeddings = output[1]
        elif self.args.bert_feature_type == 'l4m':
            hidden_states = output[2]
            text_embeddings = torch.zeros_like(hidden_states[-1][:, 0, :]).cuda()
            for i in range(len(hidden_states) - 4, len(hidden_states)):
                text_embeddings += hidden_states[i][:, 0, :]  # [B, emb]
            text_embeddings /= 4

        return text_embeddings

    def forward(self, tokens, segments, input_masks, audf=None, visf=None, texf=None): #[B, Ln, emb]
        #tokens: input id
        #segments: zero for all sentences
        #input_masks: [1 1 1 ... 0 0 0]
        if not self.args.use_utt_text_features:
            features = []
            batch = tokens.shape[0]
            max_batch = self.args.max_bert_batch
            for start in range(0, batch, max_batch):
                end = min(start+max_batch, batch)
                features.append(self.get_text_embeddings(tokens[start:end], segments[start:end], input_masks[start:end]))
            features = torch.cat(features, dim=0)
        else:
            features = texf.reshape(-1, texf.shape[-1])

        if not self.args.bert_wo_fc:
            if len(self.args.modals) > 1 and self.args.mm_type == 'ecat':
                all_feat = []
                if 'a' in self.args.modals:
                    all_feat.append(audf.reshape(-1, audf.shape[-1]))
                if 'v' in self.args.modals:
                    all_feat.append(visf.reshape(-1, visf.shape[-1]))
                if 'l' in self.args.modals:
                    all_feat.append(features)
                features = torch.cat(all_feat, dim=-1)
            elif len(self.args.modals) > 1 and self.args.mm_type == 'eadd':
                result_features = torch.zeros((features.shape[0], self.code_length))
                if 'a' in self.args.modals:
                    result_features += self.a_fc(audf.reshape(-1, audf.shape[-1]))
                if 'v' in self.args.modals:
                    result_features += self.v_fc(visf.reshape(-1, visf.shape[-1]))
                if 'l' in self.args.modals:
                    result_features += self.l_fc(features)
                return self.act(result_features)
            elif len(self.args.modals) > 1 and self.args.mm_type == 'egate':
                a_feat, v_feat = None, None
                if 'a' in self.args.modals:
                    a_feat = self.a_fc(audf.reshape(-1, audf.shape[-1]))
                if 'v' in self.args.modals:
                    v_feat = self.v_fc(visf.reshape(-1, visf.shape[-1]))
                features = self.gate(features, a=a_feat, v=v_feat)
            features = self.fc(features)
            features = self.act(features)
        return features #[B, emb]


class new_ERC_HTRM(nn.Module):
    def __init__(self, args, n_classes, use_cls = False, emo_emb=None):
        super().__init__()
        def get_mlp(in_dim, hidden_dim, out_dim, n_layers):
            layers = []
            if n_layers != 0:
                layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
                for _ in range(n_layers - 1):
                    layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
                layers += [nn.Linear(hidden_dim, out_dim)]
            else:
                layers += [nn.Linear(in_dim, out_dim)]

            out_mlp = nn.Sequential(*layers)
            return out_mlp

        self.args = args
        self.use_cls = use_cls
        self.n_classes = n_classes
        if args.dataset_name == 'MELD':
            self.n_speakers = 9
        else:
            self.n_speakers = 2

        self.dropout = nn.Dropout(args.dropout)
        self.bert_encoder = BertEncoder(code_length=args.utr_dim, args=args)

        if args.pos_emb_type == 'sin':
            self.trm_pos_embedding = PositionalEncoding(d_model=args.utr_dim, dropout=args.dropout)
        elif args.pos_emb_type == 'learned':
            self.trm_pos_embedding = LearnedPositionEncoding(d_model=args.utr_dim, dropout=args.dropout)
        elif args.pos_emb_type == 'relative':
            self.trm_pos_embedding = RelativePositionEncoding(d_model=args.utr_dim, dropout=args.dropout)

        self.modals = args.modals
        trm_encoder = speaker_TRM(args.trm_layers, args.utr_dim, args.trm_heads, args.trm_ff_dim,
                                               args.dropout, entire=args.residual_spk_attn, attn_type=args.attn_type,
                                               same=args.same_encoder)
        if len(self.args.modals) > 1 and args.mm_type in ['lcat', 'add', 'gate']:
            if 'a' in self.modals:
                self.a_encoder = copy.deepcopy(trm_encoder)
                self.a_fc = nn.Linear(args.audio_dim, args.utr_dim)
            if 'v' in self.modals:
                self.v_encoder = copy.deepcopy(trm_encoder)
                self.v_fc = nn.Linear(args.visual_dim, args.utr_dim)
            if 'l' in self.modals:
                self.l_encoder = copy.deepcopy(trm_encoder)
                self.l_fc = nn.Linear(args.text_dim, args.utr_dim)
            if args.mm_type == 'gate':
                self.gate = MMGatedAttention(args.utr_dim, dropout=args.dropout)
        else:
            self.trm_encoder = copy.deepcopy(trm_encoder)

        if args.use_spk_emb:
            self.spk_embedding = nn.Embedding(self.n_speakers, args.utr_dim)

        if  len(self.modals) > 1 and args.mm_type == 'lcat':
            in_dim = len(self.modals) * args.utr_dim
        else:
            in_dim = args.utr_dim

        self.htrm_out_mlp = get_mlp(in_dim, args.hidden_dim, n_classes, args.mlp_layers)

    def forward(self, content_ids, content_masks, speaker_ids, segment_masks, audf=None, visf=None, texf=None):
        #content_ids: [n, B, L]
        #content_mask: [n, B, L]
        #speaker_ids: [n, B]
        #segment_masks: [n, B]
        #audf/visf/texf: [n, B, dim]
        #tgt: [n, B]
        final_output = {}

        segment_masks = segment_masks.transpose(0, 1) #[B, n]
        content_shape = content_ids.shape #(n, B, L)
        content_ids = content_ids.reshape(-1, content_shape[-1])
        content_masks = content_masks.reshape(-1, content_shape[-1]) #[n*B, L]
        seg_ids = torch.zeros_like(content_ids).cuda()

        if len(self.args.modals) > 1 and self.args.mm_type in ['lcat', 'add', 'gate']:
            features = []
            if 'l' in self.args.modals:
                l_out = self.bert_encoder(content_ids, segments=seg_ids, input_masks=content_masks, texf=texf)
                l_out = l_out.reshape(content_shape[0], content_shape[1], -1)
                l_out = self.trm_pos_embedding(self.dropout(l_out))
                l_out = self.l_encoder(l_out, src_key_padding_mask=segment_masks, speaker_ids=speaker_ids)
                features.append(l_out)
            if 'a' in self.args.modals:
                a_out = self.trm_pos_embedding(self.dropout(self.a_fc(audf)))
                a_out = self.a_encoder(a_out, src_key_padding_mask=segment_masks, speaker_ids=speaker_ids)
                features.append(a_out)
            if 'v' in self.args.modals:
                v_out = self.trm_pos_embedding(self.dropout(self.v_fc(visf)))
                v_out = self.v_encoder(v_out, src_key_padding_mask=segment_masks, speaker_ids=speaker_ids)
                features.append(v_out)
            if self.args.mm_type == 'lcat':
                output = torch.cat(features, dim=-1)
            elif self.args.mm_type == 'add':
                output = torch.zeros_like(features[0])
                for f in features:
                    output += f
        else:
            uttr_out = self.bert_encoder(content_ids, segments=seg_ids, input_masks=content_masks, audf=audf, visf=visf, texf=texf)
            uttr_out = uttr_out.reshape(content_shape[0], content_shape[1], -1) #[n, B, emb]
            uttr_out = self.trm_pos_embedding(self.dropout(uttr_out))
            output = self.trm_encoder(uttr_out, src_key_padding_mask=segment_masks, speaker_ids=speaker_ids) #[n, B, emb]

        if self.args.use_residual:
            output = torch.cat([output, uttr_out], dim=-1)

        output_logits = self.htrm_out_mlp(output) #[n, B, cls], 没有经过softmax
        output_logits = output_logits.transpose(0, 1).transpose(1, -1) #[cls, n, B]
        final_output['logits'] = output_logits

        return final_output


