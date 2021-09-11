dataset_name: 数据集的名称，数据地址在dataloader.py里，用的绝对地址
bert_dim, utr_dim：bert特征维度和提取后的句特征维度
lr, lr2: 如果使用BERTfinetune，lr是BERT的学习率，lr2是其他结构的学习率，使用特征的话lr没有用，只看lr2
finetune_layers: 如果使用BERT fintune，控制finetune层数
bert_frozen: 从文本抽取Bert相关的特征，但是不fintune
use_trm: 使用transformer结构 (还可以使用lstm结构)
trm_heads, trm_ff_dim, trm_layers: transformer的相关参数  # 单模态的实验 384/6/1024 多模态的实验  512/8/1536
use_spk_attn: 是否使用各种attention mask
residual_spk_attn： 表示所有attention mask每层一次相加，否则所有attention mask将在整个encoder最后相加
attn_type: 使用attention mask的种类，可选global intra inter local
same_encoder: 表示所有attentionmask使用同一个encoder，否则每个attention mask用一个encoder
use_utt_text_features: 文本模态使用提取好的特征，否则就finetune bert
mm_type: ecat lcat可选，分别表示前拼接和每个模态过一个encoder后拼接
bert_path: huggineface 库里的名字 

if dataset == 'MELD':
    path = '/data1/lyc/HTRM/data/MELD/MELD_features_raw.pkl'
elif dataset == 'IEMOCAP':
    path = '/data1/lyc/HTRM/data/IEMOCAP/IEMOCAP_features.pkl'
elif dataset == 'DailyDialog':
    path = '/data1/lyc/HTRM/data/DailyDialog/DailyDialog.pkl'
elif dataset == 'EmoryNLP':
    path = '/data1/lyc/HTRM/data/EmoryNLP/EmoryNLP.pkl'
elif dataset == 'M3ED':
    path = '/data1/lyc/HTRM_for_M3ED/data/M3ED.pkl'


## [Bug] 不支持 attention 机制不支持 batch-size大于1的情况