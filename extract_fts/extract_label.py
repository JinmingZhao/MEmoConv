'''
读取数据并且整理对应的Target, 已经将所有 NotSure 的情感进行平滑
'''

import os
import collections
import numpy as np
from preprocess.FileOps import read_csv, read_pkl, write_pkl, read_file


def get_uttId2label(meta_filepath):
    '''
    UttId = {spk}_{dialogID}_{uttID}
    '''
    uttId2mullabel = collections.OrderedDict()
    uttId2label = collections.OrderedDict()
    instances = read_csv(meta_filepath, delimiter=';', skip_rows=1)
    for i in range(len(instances)):
        instance = instances[i]
        UtteranceId = instance[0]
        if UtteranceId is None:
            continue
        spk = instance[4]
        new_uttId = '{}_{}'.format(spk, UtteranceId)
        mul_emos, final_emo = instance[8].split(','), instance[9]     
        uttId2mullabel[new_uttId] = mul_emos
        uttId2label[new_uttId] = final_emo
    return uttId2mullabel, uttId2label

if __name__ == '__main__':
    # export PYTHONPATH=/data9/MEmoConv
    # CUDA_VISIBLE_DEVICES=7 python extract_label.py  
    label_filepath = '/data9/memoconv/modality_fts/target/all_label.pkl'
    multi_label_filepath = '/data9/memoconv/modality_fts/target/all_multi_label.pkl'
    movies_names = read_file('../preprocess/movie_list.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]

    if True:
        movie2uttId2label = collections.OrderedDict()
        movie2uttId2mul_label = collections.OrderedDict()
        all_smooth_count = 0
        for movie_name in movies_names:
            print(f'Current movie {movie_name}')
            output_label_filepath = '/data9/memoconv/modality_fts/target/movies/{}_label.pkl'.format(movie_name)
            output_mul_label_filepath = '/data9/memoconv/modality_fts/target/movies/{}_label.pkl'.format(movie_name)
            meta_filepath = '/data9/memoconv/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
            smoothed_meta_filepath = '/data9/memoconv/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
            uttId2mullabel, uttId2label = get_uttId2label(meta_filepath)
            write_pkl(output_label_filepath, uttId2label)
            write_pkl(output_mul_label_filepath, uttId2mullabel)
            movie2uttId2label.update(uttId2label)
            movie2uttId2mul_label.update(uttId2mullabel)
        print('total {} {} utts'.format(len(movie2uttId2label), len(movie2uttId2mul_label)))
        write_pkl(label_filepath, movie2uttId2label)
        write_pkl(multi_label_filepath, movie2uttId2mul_label)
    # transfer to label Id