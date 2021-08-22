'''
分析目前的人脸数据中的情感分布
/data9/datasets/AffectNetDataset
export PYTHONPATH=/data9/MEmoConv
val:
{'Neutral': 500, 'Anger': 500, 'Happiness': 500, 'Fear': 500, 'Disgust': 500, 'Surprise': 500, 'Sadness': 500, 'Contempt': 499}
train:
{'Happiness': 134415, 'Neutral': 74874, 'Surprise': 14090, 'Sadness': 25459, 'Anger': 24882, 'Disgust': 3803, 'Fear': 6378, 'Contempt': 3750}
Fer数据集的数据28K图片，类别也是8类 ['neu', 'hap', 'sur', 'sad', 'ang', 'dis', 'fea', 'con']
10倍的数据希望能表现的更好。
'''
import os
import glob
from tqdm import tqdm
import numpy as np
from collections import Counter
from preprocess.FileOps import read_file, read_json

root_dir = '/data9/datasets/AffectNetDataset'
emomap_filepath = os.path.join(root_dir, 'emomap.json')
emomap = read_json(emomap_filepath)
print(emomap)
emo2count = {}
for setname in ['train']:
    print(setname)
    annotation_dir = os.path.join(root_dir, '{}_set'.format(setname), 'annotations')
    frames = glob.glob(os.path.join(annotation_dir, '*_exp.npy'))
    for filepath in tqdm(frames):
        label = np.load(filepath)
        label = str(label)
        if emo2count.get(emomap[label]) is None:
            emo2count[emomap[label]] = 1
        else:
            emo2count[emomap[label]] += 1
print(emo2count)