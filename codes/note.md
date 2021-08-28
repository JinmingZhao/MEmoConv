### 数据集集合的划分
movie independent split
56 movies = 38 train / 7 val / 10 trian
990 dialogs = 685 train / 126 val / 179 test
如果要确定集合划分的话，考虑到每个集合的情感分布、情感刺激、情感惰性等分布。
总体情感分布: utts: 24449 turns: 9082
final_emo2count rate {'Anger': 5234, 'Neutral': 10028, 'Sad': 3957, 'Happy': 2287, 'Surprise': 1051, 'Disgust': 1497, 'Fear': 395}
emotion shift 5396
emotion inertia 2696
emotion shift distribution Counter({'Neutral_Sad': 420, 'Sad_Neutral': 417, 'Neutral_Anger': 406, 'Neutral_Surprise': 381, 'Surprise_Neutral': 377, 'Anger_Neutral': 357, 'Neutral_Happy': 338, 'Sad_Anger': 280, 'Happy_Neutral': 276, 'Anger_Sad': 270, 'Neutral_Disgust': 205, 'Disgust_Neutral': 173, 'Anger_Disgust': 129, 'Disgust_Anger': 125, 'Surprise_Anger': 100, 'Sad_Surprise': 99, 'Surprise_Sad': 93, 'Surprise_Happy': 87, 'Happy_Surprise': 73, 'Anger_Surprise': 72, 'Disgust_Sad': 61, 'Sad_Happy': 53, 'Sad_Disgust': 52, 'Happy_Anger': 50, 'Fear_Anger': 50, 'Happy_Disgust': 49, 'Anger_Fear': 48, 'Fear_Neutral': 48, 'Happy_Sad': 47, 'Anger_Happy': 46, 'Neutral_Fear': 46, 'Disgust_Happy': 45, 'Surprise_Disgust': 31, 'Disgust_Surprise': 18, 'Surprise_Fear': 13, 'Sad_Fear': 12, 'Fear_Surprise': 9, 'Fear_Sad': 9, 'Happy_Fear': 9, 'Fear_Happy': 9, 'Fear_Disgust': 7, 'Disgust_Fear': 6})
emotion inertia distribution Counter({'Neutral_Neutral': 1552, 'Anger_Anger': 539, 'Happy_Happy': 289, 'Sad_Sad': 234, 'Disgust_Disgust': 44, 'Surprise_Surprise': 33, 'Fear_Fear': 5})
intra emotion shift 2879
intra emotion inertia 10891
intra emotion shift distribution Counter({'Sad_Neutral': 275, 'Neutral_Sad': 268, 'Neutral_Happy': 257, 'Neutral_Anger': 236, 'Anger_Neutral': 232, 'Happy_Neutral': 199, 'Anger_Sad': 164, 'Sad_Anger': 156, 'Neutral_Disgust': 118, 'Disgust_Neutral': 117, 'Surprise_Neutral': 101, 'Neutral_Surprise': 83, 'Disgust_Anger': 75, 'Anger_Disgust': 65, 'Happy_Sad': 47, 'Sad_Disgust': 45, 'Disgust_Sad': 37, 'Sad_Happy': 34, 'Surprise_Happy': 34, 'Surprise_Sad': 33, 'Happy_Anger': 31, 'Surprise_Anger': 28, 'Anger_Fear': 27, 'Fear_Neutral': 25, 'Happy_Disgust': 25, 'Neutral_Fear': 21, 'Happy_Surprise': 21, 'Disgust_Happy': 21, 'Sad_Surprise': 19, 'Fear_Anger': 18, 'Anger_Happy': 17, 'Anger_Surprise': 13, 'Surprise_Disgust': 9, 'Fear_Sad': 7, 'Sad_Fear': 6, 'Disgust_Surprise': 5, 'Happy_Fear': 3, 'Fear_Happy': 2, 'Disgust_Fear': 2, 'Fear_Disgust': 1, 'Fear_Surprise': 1, 'Surprise_Fear': 1})
intra emotion inertia distribution Counter({'Neutral_Neutral': 4871, 'Anger_Anger': 2523, 'Sad_Sad': 1830, 'Happy_Happy': 798, 'Disgust_Disgust': 564, 'Fear_Fear': 156, 'Surprise_Surprise': 149})
统计每个 movie 的 情感分布 , 情感刺激的分布， 情感惰性的分布 等
然后保证每个 set 内的符合整体的分布  --- Done
/data9/MEmoConv/memoconv/split_set/{train, val, test}_movie_names.txt
train utts 17427 val utts 2821  test utts 4201 
### 定义统一的情感类别标签 
/data9/MEmoConv/extract_fts/extract_label.py
{'Happy':0, 'Neutral':1, 'Sad':2, 'Disgust':3, 'Anger': 4, 'Fear': 5, 'Surprise':6}

### Baselin0:
Bert + text + Finetune
wav2vec-zh + speech + Finetune
七分类的结果都在F1=30%, WA=50%左右
### Baseline1: 多个encoder --Done
根据模型要求的数据格式，划分数据集并准备对应的数据
注意的是有一句的spk错误，所以遇到 B_jimaofeishangtian_13_6 的时候，如果key不存在，那么改为读取 A_jimaofeishangtian_13_6 的值。
设置最大的长度:
    text bert_base_chinese avg 9.405348023182418 mid 9 p80 12 p95 15
    speech wav2vec avg 71.39415848969989 mid 65 p80 93 p95 131
    visual denseface avg 35.959717679462905 mid 33 p80 47 p95 66
    max_text_tokens = 20 
    max_acoustic_tokens = 128 (wav2vec) 256(comparE)
    max_visual_tokens = 64

### Baseline2: MulT --DDL 0825 --

### 对话中的情感 
先把特征都平均得到句子级别的特征，然后跑一下，dialogRNN 试试

text:
    sent_cls_bert_base_chinese
    sent_cls_bert_base_chinese4chmed
    sent_cls_robert_wwm_base_chinese4chmed
speech:
    IS10_norm 
    sent_avg_wav2vec_zh
    wav2vec_zh4chmed --Pending, 需要进行模型转换
visual:
    sent_avg_denseface
    

features 4 dialogRNN/DialogueGCN:
    /data9/memoconv/modality_fts/dialogrnn/Asent_avg_wav2vec_zh_Vsent_avg_denseface_Lsent_cls_bert_base_chinese.pkl
    /data9/memoconv/modality_fts/dialogrnn/AIS10_norm_Vsent_avg_denseface_Lsent_cls_robert_wwm_base_chinese4chmed.pkl
result_dir
    /data9/MEmoConv/memoconv/results/dialogrnn
    /data9/MEmoConv/memoconv/results/dialoggcn


对话情感识别，需要首先获取句子级别的情感表示
https://github.com/declare-lab/conv-emotion
DialogRNN 数据格式要求:
/data1/Muse_hjw/gated/DialogueGCN/tmp/IEMOCAP_features_v_frame_wth_frm_len.pkl
videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid, vid2framelen = 
data = pkl.load(open('IEMOCAP_features_v_frame_wth_frm_len.pkl', 'rb'))
videoIDs
    video_names = data[0].keys()
    video_uttIds = data[0]['Ses05F_impro08']
videoSpeakers
    video_names = data[1].keys()
    video_spks = data[1]['Ses05F_impro08']  # 序列跟 video_uttIds 一一对应
    # 注意spk目前是AB的形式，而不是MF的形式
videoLabels
    video_names = data[2].keys()
    video_labels = data[2]['Ses05F_impro08'] # 序列跟 video_uttIds 一一对应
videoText
    video_names = data[3].keys()
    video_labels = data[3]['Ses05F_impro08']  # 每句话对应100维的向量
videoAudio
    video_names = data[4].keys()
    video_labels = data[4]['Ses05F_impro08']  # 每句话对应1582维的IS10特征
videoVisual
    video_names = data[5].keys()
    video_labels = data[5]['Ses05F_impro08']  # 每句话对应时间序列的特征 (50,342)
videoSentence
    video_names = data[6].keys()
    video_labels = data[6]['Ses05F_impro08']  #每句话的文本内容 跟 video_uttIds 一一对应
trainVid
    video_names = data[7] # 序列存储属于训练集合的 videoIDs
testVid
    video_names = data[8] # 序列存储属于测试集合的 videoIDs
vid2framelen
    video_names = data[9] # 序列存储属于训练集合的 videoIDs