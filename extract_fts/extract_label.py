'''
读取数据并且整理对应的Target, 将 Other 和 NotSure 的情感进行平滑
'''
import os
import collections
import numpy as np
from preprocess.FileOps import read_csv, read_pkl, write_pkl, read_file

def get_uttId2features(meta_filepath):
    '''
    UttId = {spk}_{dialogID}_{uttID}
    '''
    uttId2ft = collections.OrderedDict()
    instances = read_csv(meta_filepath, delimiter=';', skip_rows=1)
    for instance in instances:
        UtteranceId = instance[0]
        if UtteranceId is None:
            continue
        dialog_id = '_'.join(UtteranceId.split('_')[:2])
        spk,  = instance[4]
        new_uttId = '{}_{}'.format(spk, UtteranceId)


if __name__ == '__main__':
    # export PYTHONPATH=/data9/MEmoConv
    # CUDA_VISIBLE_DEVICES=7 python extract_speech_ft.py  
    feat_type = 'wav2vec'
    all_output_ft_filepath = '/data9/memoconv/modality_fts/speech/all_label_{}.pkl'.format(feat_type)
    movies_names = read_file('../preprocess/movie_list.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]
    movie2uttID2ft = collections.OrderedDict()

    # extract all faces, only in the utterance
    for movie_name in movies_names:
        print(f'Current movie {movie_name}')
        output_ft_filepath = '/data9/memoconv/modality_fts/speech/movies/{}_label_{}.pkl'.format(movie_name, feat_type)
        meta_filepath = '/data9/memoconv/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
        movie_audio_dir = '/data9/memoconv/memoconv_convs_talknetoutput/{}'.format(movie_name)
        uttId2ft = get_uttId2features(meta_filepath, movie_audio_dir)
        write_pkl(output_ft_filepath, uttId2ft)
        movie2uttID2ft.update(uttId2ft)
    write_pkl(all_output_ft_filepath, movie2uttID2ft)
