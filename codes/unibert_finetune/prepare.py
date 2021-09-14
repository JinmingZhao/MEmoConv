'''
采用原始的数据格式
export PYTHONPATH=/data9/MEmoConv
'''
from builtins import zip
import numpy as np
import os
from preprocess.FileOps import read_file, write_csv, read_pkl, read_csv

def get_set_bert_data(set_movie_names_filepath, output_int2name_filepath, output_label_filepath, modality_ft_dir):
    set_movie_names = read_file(set_movie_names_filepath)
    set_movie_names = [name.strip() for name in set_movie_names]

    all_sents = []
    all_sents.append(['label', 'sentence1'])

    int2name2label = {}
    int2name = np.load(output_int2name_filepath)
    int2label = np.load(output_label_filepath)
    for name, label in zip(int2name, int2label):
        int2name2label[name] = label
    for movie_name in set_movie_names:
        movie_modality_ft_filepath = os.path.join(modality_ft_dir, '{}_text_info.pkl'.format(movie_name))
        assert os.path.exists(movie_modality_ft_filepath) == True
        uttId2ft = read_pkl(movie_modality_ft_filepath)
        # spk错误case的修正
        if uttId2ft.get('A_jimaofeishangtian_13_6') is not None:
            print('Modify the spk error case A_jimaofeishangtian_13_6')
            uttId2ft['B_jimaofeishangtian_13_6'] = uttId2ft['A_jimaofeishangtian_13_6']
        for uttId in uttId2ft:
            if int2name2label.get(uttId) is not None:
                 all_sents.append([int2name2label[uttId], uttId2ft[uttId]])
    return all_sents


def get_set_bert_data_from_csv(set_movie_names_filepath, output_int2name_filepath, output_label_filepath):
    set_movie_names = read_file(set_movie_names_filepath)
    set_movie_names = [name.strip() for name in set_movie_names]

    all_sents = []
    all_sents.append(['label', 'sentence1'])

    int2name2label = {}
    int2name = np.load(output_int2name_filepath)
    int2label = np.load(output_label_filepath)
    for name, label in zip(int2name, int2label):
        int2name2label[name] = label
    for movie_name in set_movie_names:
        print('Current movie_name {}'.format(movie_name))
        meta_filepath = '/data9/memoconv/memoconv_final_labels_csv_personname/meta_{}.csv'.format(movie_name)
        instances = read_csv(meta_filepath, delimiter=';', skip_rows=1)
        for instance in instances:
            UtteranceId = instance[0]
            if UtteranceId is None:
                continue
            text, spk = instance[3], instance[4]
            new_uttId = '{}_{}'.format(spk, UtteranceId)
            if int2name2label.get(new_uttId) is not None:
                all_sents.append([int2name2label[new_uttId], text])
    print('total utts {}'.format(len(all_sents)))
    return all_sents


if __name__ == '__main__':
    output_dir = '/data9/memoconv/modality_fts/utt_baseline'
    split_info_dir = '/data9/MEmoConv/memoconv/split_set'
    target_dir = '/data9/memoconv/modality_fts/target/movies'
    # Step2: 根据 int2name 获取对应的不同模态的特征, 注意统计长度，方便设计模型
    modality_ft_dir = os.path.join('/data9/memoconv/modality_fts/text', 'movies')
    for setname in ['train', 'val', 'test']:
        print('current setname {}'.format(setname))
        output_int2name_filepath = os.path.join(output_dir, setname, 'int2name.npy')
        output_int2label_filepath = os.path.join(output_dir, setname, 'label.npy')
        set_movie_names_filepath = os.path.join(split_info_dir, '{}_movie_names.txt'.format(setname))
        # set_texts = get_set_bert_data(set_movie_names_filepath, output_int2name_filepath, output_int2label_filepath, modality_ft_dir)
        # write_csv(os.path.join(output_dir, setname, 'bert_data.csv'), set_texts, delimiter=',')
        set_texts = get_set_bert_data_from_csv(set_movie_names_filepath, output_int2name_filepath, output_int2label_filepath)
        write_csv(os.path.join(output_dir, setname, 'bert_data_persontoken.csv'), set_texts, delimiter=',')