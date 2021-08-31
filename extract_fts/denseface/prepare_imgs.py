'''
export PYTHONPATH=/data9/MEmoConv
将图片处理成DenseNet中用到的64*64的黑白图片
'''
from re import I
import cv2
import os
import glob
import h5py
from collections import Counter
from tqdm import tqdm
import numpy as np
from preprocess.FileOps import read_json

def transfer2fer_target(emo_cate):
    fer_idx_to_class = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']
    emo_index = fer_idx_to_class.index(emo_cate)
    return emo_index

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    return img

def gen_affectnet_dataset_data(out_data_dir):
    '''
    training data
    Parameters:
    -----------
    finetune_data_dir: where to store the output file, [trn/val]_img.npy and [trn/val]_target.npy
    '''
    root_dir = '/data9/datasets/AffectNetDataset'
    emomap_filepath = os.path.join(root_dir, 'emomap.json')
    emomap = read_json(emomap_filepath)
    for setname in ['val', 'train']:
        print('current {}'.format(setname))
        targets = []
        imgs = []
        img_dir = os.path.join(root_dir, '{}_set'.format(setname), 'images')
        annotation_dir = os.path.join(root_dir, '{}_set'.format(setname), 'annotations')
        frames = glob.glob(os.path.join(annotation_dir, '*_exp.npy'))
        for filepath in tqdm(frames):
            label = np.load(filepath)
            label = str(label)
            fer_label = transfer2fer_target(emomap[label])
            targets.append(fer_label)
            # img
            img_id = filepath.split('/')[-1].split('_')[0]
            img_path = os.path.join(img_dir, '{}.jpg'.format(img_id))
            assert os.path.exists(img_path) == True
            img = read_img(img_path)
            imgs.append(img)
        assert len(targets) == len(imgs)
        emo2count = Counter(targets)
        print(emo2count)
        imgs = np.array(imgs, np.int)
        targets = np.array(targets, np.float)
        np.save(os.path.join(out_data_dir, '{}_img.npy'.format(setname)), imgs)
        np.save(os.path.join(out_data_dir, '{}_target.npy'.format(setname)), targets)


def trans_npy2h5py(output_dir):
    for setname in ['val', 'trn','tst']:
        imgs = np.load(os.path.join(output_dir, '{}_img.npy'.format(setname)))
        targets = np.load(os.path.join(output_dir, '{}_target.npy'.format(setname)))

        h5img_filepath = os.path.join(output_dir, '{}_img.h5'.format(setname))
        h5target_filepath = os.path.join(output_dir, '{}_target.h5'.format(setname))
        utt_ids = range(len(targets))
        print('there are ')
        h5f = h5py.File(h5img_filepath, 'w')
        for utt_id in tqdm(utt_ids):
            feature = imgs[utt_id]
            h5f[str(utt_id)] = feature
        h5f.close()
        h5f = h5py.File(h5target_filepath, 'w')
        for utt_id in tqdm(utt_ids):
            feature = targets[utt_id]
            h5f[str(utt_id)] = feature
        h5f.close()

def compute_mean_std_by_channel(output_dir):
    images = np.load(os.path.join(output_dir, 'trn_img.npy'))
    mean_std_filepath = os.path.join(output_dir, 'trn_mean0_std1.npy')
    print(images.shape)
    means = []
    stds = []
    # for every channel in image(assume this is last dimension)
    means.append(np.mean(images))
    stds.append(np.std(images))
    means = np.array(means, np.float32)
    stds = np.array(stds, np.float32)
    np.save(mean_std_filepath, [means, stds])

def combine_with_fer(output_dir,fer_data_dir, combine_dir):
    # 将训练集合合并，验证集和验证集合合并，测试集合用Fer的测试集合
    for setname in ['val', 'trn']:
        imgs = np.load(os.path.join(output_dir, '{}_img.npy'.format(setname)))
        targets = np.load(os.path.join(output_dir, '{}_target.npy'.format(setname)))
        fer_imgs = np.load(os.path.join(fer_data_dir, '{}_img.npy'.format(setname)))
        fer_targets = np.load(os.path.join(fer_data_dir, '{}_target.npy'.format(setname)))

        fer_targets = np.argmax(fer_targets, axis=1)            
        total_targets = np.concatenate([targets, fer_targets])
        print('{} total targets {}'.format(setname, total_targets.shape))
        np.save(os.path.join(combine_dir, '{}_target.npy'.format(setname)), total_targets)

        total_imgs = np.concatenate([imgs, fer_imgs])
        print('{} total imgs {}'.format(setname, total_imgs.shape))
        np.save(os.path.join(combine_dir, '{}_img.npy'.format(setname)), total_imgs)

if __name__ == '__main__':
    output_dir = '/data9/datasets/AffectNetDataset/npy_data'
    fer_data_dir = '/data3/zjm/dataset/ferplus/npy_data'
    combine_data_dir = '/data9/datasets/AffectNetDataset/combine_with_fer/npy_data'

    if False:
        gen_affectnet_dataset_data(output_dir)

    if False:
        trans_npy2h5py(combine_data_dir)
    
    if False:
        compute_mean_std_by_channel(combine_data_dir)
    
    if False:
        combine_with_fer(output_dir, fer_data_dir, combine_data_dir)

    if True:
        # prepare imgs for v3d (112,112)
        pass