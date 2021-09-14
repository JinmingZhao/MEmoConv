/data9/memoconv/modality_fts/{text, speech, visual}

Text Modality:
    Bert-Base -- Done
    RoBerta-Base
    Glove

Speech Modality:
    ComparE_norm --Done
    wav2vec_zh -- Done
    wav2vec_zh_finetuned -- Done
    IS10_norm

Visual Modality:
    Denseface -- Done
    openface: 68 facial landmarks, 17 facial action units, head pose, head orientation, and eye gaze.  --OpenFace
    AffectNet: -- affectnet-env

## extracting affectnet features
export PYTHONPATH=/data9/memoconv/tools/AffectNet
采用的是mobilenet的模型，整体效果60%.
CUDA_VISIBLE_DEVICES=6 python inference.py

看了fendou_1 和 fendou_3 两个例子, DenseNet 一般不太明显的都会识别为Neutral. 
而 AffectNet 会倾向于识别为 Sad Afraid 等情感, 对于细微的表情很敏感。


## 调研一下如何抽取句子级别的文本特征
Sentence-Level Feature for Text
    1. sen_avg_bert 
    2. sen_cls_bert
    3. finetune_cls_roberta  

## 如何添加新的token.
需求，对话中的人名都替换为了 [PERSON], 可以在已经模型的基础上，利用一个 unused token 来表示 [PERSON]
1. 手动添加一个新的special token, 还要 修改 vocab, 修改token.json。 仿照其他的special token 利用 [unused88] 这个 position.
self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)
self.tokenizer.additional_special_tokens = '[unused88]'
print(self.tokenizer.special_tokens_map) --Done
2. 直接将文本中的 [PERSON] 修改为 [unused88] -- 方法不行，必须添加到 special token中才不会分词，否则会把 [unused88] 分成多个词。