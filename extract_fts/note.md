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