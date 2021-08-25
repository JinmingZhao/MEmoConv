### 数据集集合的划分
movie independent split
56 movies = 38 train / 7 val / 10 trian
990 dialogs = 750 train / 100 val / 140 test
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
/data9/MEmoConv/memoconv/split_set

### 特征根据数据集进行划分，UtteranceId 以Target中的信息为标准 -- Going

### Baseline1: 多个encoder --DDL 0825

### Baseline2: MulT --DDL 0825


### 对话中的情感
对话情感识别，需要首先获取句子级别的情感表示
https://github.com/declare-lab/conv-emotion