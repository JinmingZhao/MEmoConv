import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertPreTrainedModel, BertForSequenceClassification
# from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import (
    RobertaModel, 
    RobertaForSequenceClassification, 
)
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaPreTrainedModel
)
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
# from models.networks.tools import init_weights

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, fc1_size, output_size, dropout_rate):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1_size = fc1_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.rnn1 = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 4, fc1_size)
        self.fc2 = nn.Linear(fc1_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm((hidden_size * 2, ))
        self.bn = nn.BatchNorm1d(hidden_size * 4)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)
        _, (final_h2, _) = rnn2(packed_normed_h1)
        return final_h1, final_h2

    def rnn_flow(self, x, lengths):
        batch_size = lengths.size(0)
        h1, h2 = self.extract_features(x, lengths, self.rnn1, self.rnn2, self.layer_norm)
        h = torch.cat((h1, h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return self.bn(h)

    def mask2length(self, mask):
        ''' mask [batch_size, seq_length, feat_size]
        '''
        _mask = torch.mean(mask, dim=-1).long() # [batch_size, seq_len]
        length = torch.sum(_mask, dim=-1)       # [batch_size,]
        return length 

    def forward(self, x, mask):
        lengths = self.mask2length(mask)
        h = self.rnn_flow(x, lengths)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.relu(h)
        o = self.fc2(h)
        return o, h

class SimpleClassifier(nn.Module):
    ''' Linear classifier, use embedding as input
        Linear approximation, should append with softmax
    '''
    def __init__(self, embd_size, output_dim, dropout):
        super(SimpleClassifier, self).__init__()
        self.dropout = dropout
        self.C = nn.Linear(embd_size, output_dim)
        self.dropout_op = nn.Dropout(dropout)
        
    def forward(self, x):
        if self.dropout > 0:
            x = self.dropout_op(x)
        return self.C(x)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class FcClassifier(nn.Module):
    def __init__(self, input_dim, layers, output_dim, dropout=0.3, use_bn=False):
        ''' Fully Connect classifier
            Parameters:
            --------------------------
            input_dim: input feature dim
            layers: [x1, x2, x3] will create 3 layers with x1, x2, x3 hidden nodes respectively.
            output_dim: output feature dim
            activation: activation function
            dropout: dropout rate
        '''
        super().__init__()
        self.all_layers = []
        for i in range(0, len(layers)):
            self.all_layers.append(nn.Linear(input_dim, layers[i]))
            self.all_layers.append(nn.ReLU())
            if use_bn:
                self.all_layers.append(nn.BatchNorm1d(layers[i]))
            if dropout > 0:
                self.all_layers.append(nn.Dropout(dropout))
            input_dim = layers[i]
        
        if len(layers) == 0:
            layers.append(input_dim)
            self.all_layers.append(Identity())
        
        self.fc_out = nn.Linear(layers[-1], output_dim)
        self.module = nn.Sequential(*self.all_layers)
    
    def forward(self, x):
        feat = self.module(x)
        out = self.fc_out(feat)
        return out, feat

class EF_model_AL(nn.Module):
    def __init__(self, fc_classifier, lstm_classifier, out_dim_a, out_dim_v, fusion_size, num_class, dropout):
        ''' Early fusion model classifier
            Parameters:
            --------------------------
            fc_classifier: acoustic classifier
            lstm_classifier: lexical classifier
            out_dim_a: fc_classifier output dim
            out_dim_v: lstm_classifier output dim
            fusion_size: output_size for fusion model
            num_class: class number
            dropout: dropout rate
        '''
        super(EF_model_AL, self).__init__()
        self.fc_classifier = fc_classifier
        self.lstm_classifier = lstm_classifier
        self.out_dim = out_dim_a + out_dim_v
        self.dropout = nn.Dropout(dropout)
        self.num_class = num_class
        self.fusion_size = fusion_size
        # self.out = nn.Sequential(
        #     nn.Linear(self.out_dim, self.fusion_size),
        #     nn.ReLU(),
        #     nn.Linear(self.fusion_size, self.num_class),
        # )
        self.out1 = nn.Linear(self.out_dim, self.fusion_size)
        self.relu = nn.ReLU()
        self.out2 = nn.Linear(self.fusion_size, self.num_class)

    def forward(self, A_feat, L_feat, L_mask):
        _, A_out = self.fc_classifier(A_feat)
        _, L_out = self.lstm_classifier(L_feat, L_mask)
        feat = torch.cat([A_out, L_out], dim=-1)
        feat = self.dropout(feat)
        feat = self.relu(self.out1(feat))
        out = self.out2(self.dropout(feat))
        return out, feat
    

class MaxPoolFc(nn.Module):
    def __init__(self, hidden_size, num_class=4):
        super(MaxPoolFc, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_class),
            nn.ReLU()
        )
    
    def forward(self, x):
        ''' x shape => [batch_size, seq_len, hidden_size]
        '''
        batch_size, seq_len, hidden_size = x.size()
        x = x.view(batch_size, hidden_size, seq_len)
        # print(x.size())
        out = torch.max_pool1d(x, kernel_size=seq_len)
        out = out.squeeze()
        out = self.fc(out)
        
        return out
    

class BertClassifier(BertPreTrainedModel):
    '''
    model = BertClassifier.from_pretrained('xxx_model')
    '''
    def __init__(self, config, num_classes, embd_method): # 
        super().__init__(config)
        self.num_labels = num_classes
        self.embd_method = embd_method
        if self.embd_method not in ['cls', 'mean', 'max']:
            raise NotImplementedError('Only [cls, mean, max] embd_method is supported, \
                but got', config.embd_method)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_layer = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask):
        # Feed the input to Bert model to obtain contextualized representations
        '''
        ['i']
        '''
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.last_hidden_state
        cls_token = outputs.pooler_output
        hidden_states = outputs.hidden_states
        # using different embed method
        if self.embd_method == 'cls':
            cls_reps = cls_token
        elif self.embd_method == 'mean':
            cls_reps = torch.mean(last_hidden, dim=1)
        elif self.embd_method == 'max':
            cls_reps = torch.max(last_hidden, dim=1)[0]
        
        cls_reps = self.dropout(cls_reps)
        logits = self.cls_layer(cls_reps)
        return logits, hidden_states

class RobertaCLS(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class RobertaClassifier(RobertaPreTrainedModel):
    def __init__(self, config, num_classes, embd_method='cls'):
        super().__init__(config)
        self.num_labels = num_classes
        self.embd_method = embd_method
        config.num_labels = self.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaCLS(config)
        self.init_weights()
        
    def forward(self, input_ids, attention_mask,):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.last_hidden_state
        # cls_token = outputs.pooler_output
        hidden_states = outputs.hidden_states
        # using different embed method
        if self.embd_method == 'cls':
            cls_reps = last_hidden[:, 0]
        elif self.embd_method == 'mean':
            cls_reps = torch.mean(last_hidden, dim=1)
        elif self.embd_method == 'max':
            cls_reps = torch.max(last_hidden, dim=1)[0]
        
        logits = self.classifier(cls_reps)
        return logits, hidden_states

class Wav2VecClassifier(Wav2Vec2PreTrainedModel):
    def __init__(self, config, num_classes, embd_method): # 
        super().__init__(config)
        self.num_labels = num_classes
        self.embd_method = embd_method
        if self.embd_method not in ['last', 'mean', 'max']:
            raise NotImplementedError('Only [last, mean, max] embd_method is supported, \
                but got', embd_method)

        self.model = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.cls_layer = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None):
        # Feed the input to Bert model to obtain contextualized representations
        outputs = self.model(
            input_ids,
            # attention_mask=attention_mask,
            # output_hidden_states=True,
        )
        last_hidden = outputs.last_hidden_state
        # using different embed method
        if self.embd_method == 'last':
            cls_reps = last_hidden[:, -1]
        elif self.embd_method == 'mean':
            cls_reps = torch.mean(last_hidden, dim=1)
        elif self.embd_method == 'max':
            cls_reps = torch.max(last_hidden, dim=1)[0]
        
        cls_reps = self.dropout(cls_reps)
        logits = self.cls_layer(cls_reps)
        return logits, last_hidden

def bert_classifier(num_classes, bert_name):
    model = BertForSequenceClassification.from_pretrained(
        bert_name, num_labels=num_classes,
        output_attentions=False, output_hidden_states=True
    )
    return model

def roberta_classifier(num_classes, bert_name):
    model = RobertaForSequenceClassification.from_pretrained(
        bert_name, num_labels=num_classes,
        output_attentions=False, output_hidden_states=True
    )
    return model

if __name__ == '__main__':
    # from transformers import AutoConfig
    # config = AutoConfig.from_pretrained('bert-base-uncased')
    # cls_net = BertClassifier(config, 4)

    # input = torch.ones([2, 24]).long()
    # attention_mask = torch.ones([2, 24]).long()
    # label = torch.ones(2).long()
    # cls_net = RobertaClassifier.from_pretrained('roberta-base', num_classes=4, embd_method='max')
    # cls_net = nn.DataParallel(cls_net, device_ids=[0])
    # cls_net.module.save_pretrained('./test/')
    # logits, hidden_states = cls_net(input, attention_mask)
    # print(logits.shape)
    # print(len(hidden_states))
    # print(hidden_states[0].shape)
    # print(hidden_states[1].shape)

    # model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    # output = model(input)
    # last_hidden, cls_token, hidden_states = output
    # print(last_hidden.shape)
    # print(cls_token.shape)
    # print(len(hidden_states))
    # print((hidden_states[0] == last_hidden).all())
    # print((hidden_states[-1] == last_hidden).all())
    # print((cls_token == last_hidden[:, 0, :]).all())
    # cls_net = RobertaClassifier.from_pretrained('data/pretrained_model', num_classes=6, embd_method='max')
    # print(cls_net)
    # cls_net = BertClassifier.from_pretrained('./checkpoints/finetune_5cls_bert_movie_max_combined_onlyValFalse_lr1e-05_run2/1/3_net_BC.pth', num_classes=5, embd_method='max')
    # print(cls_net)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/data7/MEmoBert/emobert/exp/mlm_pretrain/results/opensub/bert_base_uncased_500w_linear_lr1e4_warm4k_bs256_acc2_4gpu/checkpoint-48820')
    print(tokenizer)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print(tokenizer)
    # cls_net = BertClassifier.from_pretrained('bert-base', num_classes=5, embd_method='max')