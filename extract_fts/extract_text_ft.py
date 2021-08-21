'''
TalkNet Conda
抽取中文文本特征，采用Bert-Base
bert-base-chinese
'''
import torch
import os, collections
import numpy as np
from typing import List
from transformers import BertModel, AlbertModel, BertTokenizer, TransfoXLTokenizer, AlbertTokenizer
from preprocess.FileOps import read_csv, read_pkl, write_pkl, read_file

class BertExtractor(object):
    def __init__(self, device=0, model_name='bert_base'):
        """
        :param model_name: bert size: base(768), small(512), medium(512), tiny(128), mini(256)
        Choice in [bert_base, bert_medium, bert_mini, bert_tiny, albert_base]
        """
        self.model_name = model_name
        self.pretrained_path = self.get_pretrained_path(model_name)

        self.device = torch.device('cuda:{}'.format(device))
        self.get_tokenizer(self.model_name.split('_')[0])
        self.max_length = 512
        if self.model_name.split('_')[0] == 'bert':
            self.model = BertModel.from_pretrained(self.pretrained_path).to(self.device)
        else:
            self.model = AlbertModel.from_pretrained('albert-base-v2').to(self.device)
        
        self.model.eval()
    
    def get_pretrained_path(self, model_name):
        path_config = {
            'bert_base': '/data7/hjw/Bert_pretrn/bert_base_uncased_L-12_H-768_A-12',
            'bert_medium': '/data7/hjw/Bert_pretrn/bert_medium_uncased_L-8_H-512_A-8',
            'bert_small': '/data7/hjw/Bert_pretrn/bert_small_uncased_L-4_H-512_A-8',
            'bert_mini': '/data7/hjw/Bert_pretrn/bert_mini_uncased_L-4_H-256_A-4',
            'bert_tiny': '/data7/hjw/Bert_pretrn/bert_tiny_uncased_L-2_H-128_A-2',
            'albert_base': '/data7/lrc/MuSe2020/hhh/pretrain_model/albert_base',
            'albert_large': '/data7/lrc/MuSe2020/hhh/pretrain_model/albert_large',
            'bert_base_chinese': '/data2/zjm/tools/LMs/bert_base_chinese',
            # finetune on MuSe arousal label
            'bert_base_arousal': '/data7/lrc/MuSe2020/MuSe2020_features/code/finetune_bert/output/arousal',
            # finetune on MuSe valence label
            'bert_base_valence': '/data7/lrc/MuSe2020/MuSe2020_features/code/finetune_bert/output/valence',
        }
        if path_config.get(model_name) is None:
            print()
        return path_config[model_name]

    def get_tokenizer(self, model_name):
        if model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)
        elif model_name == 'transformer-xl':
            self.tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
            self.tokenizer.pad_token = '[PAD]'
        elif model_name == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2') # 'albert-base-v2'
        else:
            raise ValueError('model name not supported')
    
    def extract_sentence(self, sentence: str):
        # sentence without cls and sep and the tokenizer will auto add these
        ids = self.tokenizer.encode(sentence)
        ids = torch.tensor(ids).unsqueeze(0)
        feats = self.model(ids.to(self.device))[0]
        # save two special tokens: fisrt index vector is cls and last index is sep
        return feats.detach().cpu().squeeze(0).numpy()
    
    def __call__(self, sentence: str):
        return self.extract_sentence(sentence)


def get_uttId2features(extractor, meta_filepath):
    '''
    UttId = {spk}_{dialogID}_{uttID}
    '''
    uttId2text = collections.OrderedDict()
    uttId2ft = collections.OrderedDict()
    instances = read_csv(meta_filepath, delimiter=';', skip_rows=1)
    total_rows = 0
    for instance in instances:
        UtteranceId = instance[0]
        if UtteranceId is None:
            continue
        total_rows += 1
        text, spk = instance[3], instance[4]
        new_uttId = '{}_{}'.format(spk, UtteranceId)
        ft = extractor(text)
        uttId2text[new_uttId] = text
        uttId2ft[new_uttId] = ft
    return uttId2text, uttId2ft


if __name__ == '__main__':
    # export PYTHONPATH=/data9/MEmoConv
    # CUDA_VISIBLE_DEVICES=7 python extract_text_ft.py
    if False:
        # demo of extract text features
        sentence = '我们开始健身吧'
        extract_bert = BertExtractor(device=0, model_name='bert_base_chinese')
        feature = extract_bert(sentence)
        print(feature.shape) # (9, 768) fisrt is cls and last is sep
    
    all_output_ft_filepath = '/data9/memoconv/modality_fts/text/movies/all_text_ft_bert_base_chinese.pkl'
    all_text_info_filepath = '/data9/memoconv/modality_fts/text/movies/all_text_info.pkl'
    movies_names = read_file('../preprocess/movie_list.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]
    movie2uttID2ft = collections.OrderedDict()
    movie2uttID2text = collections.OrderedDict()

    extract_bert = BertExtractor(device=0, model_name='bert_base_chinese')
    # extract all faces, only in the utterance
    for movie_name in movies_names:
        print(f'Current movie {movie_name}')
        output_ft_filepath = '/data9/memoconv/modality_fts/text/movies/{}_text_ft_bert_base_chinese.pkl'.format(movie_name)
        text_info_filepath = '/data9/memoconv/modality_fts/text/movies/{}_text_info.pkl'.format(movie_name)
        meta_filepath = '/data9/memoconv/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
        talkout_dialog_dir = '/data9/memoconv/memoconv_convs_talknetoutput/{}'.format(movie_name)
        uttId2text, uttId2ft = get_uttId2features(extract_bert, meta_filepath)
        write_pkl(output_ft_filepath, uttId2ft)
        write_pkl(text_info_filepath, uttId2text)
        movie2uttID2ft.update(uttId2ft)
        movie2uttID2text.update(uttId2text)
    write_pkl(all_output_ft_filepath, movie2uttID2ft)
    write_pkl(all_text_info_filepath, movie2uttID2text)