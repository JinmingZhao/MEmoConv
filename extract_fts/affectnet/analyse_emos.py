'''
分析目前的人脸数据中的情感分布
/data9/datasets/AffectNetDataset
export PYTHONPATH=/data9/MEmoConv
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
for setname in ['train', 'val']:
    annotation_dir = os.path.join(root_dir, '{}_set'.format(setname), 'annotations')
    frames = glob.glob(os.path.join(annotation_dir, '*_exp.npy'))
    for filepath in tqdm(frames):
        label = np.load(filepath)
        label = str(label)
        if emo2count.get(label) is None:
            emo2count[emomap[label]] = 1
        else:
            emo2count[emomap[label]] += 1
print(emo2count)