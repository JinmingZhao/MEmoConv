'''
采用原始的数据格式, 保留对应的语音文件的路径
export PYTHONPATH=/data9/MEmoConv
'''
from builtins import zip
from datasets.utils import file_utils
import numpy as np
import os
from preprocess.FileOps import read_file, write_csv, read_pkl, write_file

def get_set_wavpath_data(set_movie_names_filepath, output_int2name_filepath, modality_ft_dir):
    set_movie_names = read_file(set_movie_names_filepath)
    set_movie_names = [name.strip() for name in set_movie_names]

    all_audio_filepaths = []

    int2name = np.load(output_int2name_filepath)
    int2name_dict = {name:1 for name in int2name}
    for movie_name in set_movie_names:
        movie_audio_info_filepath = os.path.join(modality_ft_dir, '{}_speechpath_info.pkl'.format(movie_name))
        assert os.path.exists(movie_audio_info_filepath) == True
        uttId2ft = read_pkl(movie_audio_info_filepath)
        # spk错误case的修正
        if uttId2ft.get('A_jimaofeishangtian_13_6') is not None:
            print('Modify the spk error case A_jimaofeishangtian_13_6')
            uttId2ft['B_jimaofeishangtian_13_6'] = uttId2ft['A_jimaofeishangtian_13_6'].replace('A_', 'B_')
        for uttId in uttId2ft:
            if int2name_dict.get(uttId) is not None:
                filepath = uttId2ft[uttId]
                assert os.path.exists(filepath) == True
                all_audio_filepaths.append(uttId2ft[uttId] + '\n')
    return all_audio_filepaths

if __name__ == '__main__':
    output_dir = '/data9/memoconv/modality_fts/utt_baseline'
    split_info_dir = '/data9/MEmoConv/memoconv/split_set'
    target_dir = '/data9/memoconv/modality_fts/target/movies'
    audio_dir = '/data9/memoconv/memoconv_convs_talknetoutput'
    modality_ft_dir = '/data9/memoconv/modality_fts/speech/movies'
    for setname in ['train', 'val', 'test']:
        print('current setname {}'.format(setname))
        output_int2name_filepath = os.path.join(output_dir, setname, 'int2name.npy')
        set_movie_names_filepath = os.path.join(split_info_dir, '{}_movie_names.txt'.format(setname))
        set_audio_paths = get_set_wavpath_data(set_movie_names_filepath, output_int2name_filepath, modality_ft_dir)
        write_file(os.path.join(output_dir, setname, 'audio_filepaths.txt'), set_audio_paths)