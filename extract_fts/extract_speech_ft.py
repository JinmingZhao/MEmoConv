import os
import collections
import pandas as pd
import numpy as np
import subprocess
import soundfile as sf
import scipy.signal as spsig
import librosa
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Processor, Wav2Vec2Model
from preprocess.FileOps import read_csv, read_pkl, write_pkl, read_file

class ComParEExtractor():
    ''' 抽取comparE特征, 输入音频路径, 输出npy数组, 每帧130d
    '''
    def __init__(self, opensmile_tool_dir=None, downsample=-1, tmp_dir='/data7/emobert/comparE_feature/raw_fts', no_tmp=False):
        ''' Extract ComparE feature
            tmp_dir: where to save opensmile csv file
            no_tmp: if true, delete tmp file
            downsample. if =-1, then use the raw comparE fts, else use the resampeld fts.
        '''
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if opensmile_tool_dir is None:
            opensmile_tool_dir = '/data2/zjm/tools/opensmile-3.0-linux-x64'
        self.opensmile_tool_dir = opensmile_tool_dir
        self.tmp_dir = tmp_dir
        self.downsample = downsample
        self.no_tmp = no_tmp
    
    def __call__(self, full_wav_path):
        # such as: /data7/emobert/data_nomask_new/audio_clips/No0079.The.Kings.Speech/188.wav
        movie_name = full_wav_path.split('/')[-2]
        basename = movie_name + '_' + os.path.basename(full_wav_path).split('.')[0]
        save_path = os.path.join(self.tmp_dir, basename+".csv")
        cmd = 'SMILExtract -C {}/config/compare16/ComParE_2016.conf \
            -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1 \
            -I {} -lldcsvoutput {} -instname xx -O ? -noconsoleoutput 1'
        # os.system(cmd.format(self.opensmile_tool_dir, wav, save_path))
        p = subprocess.Popen([cmd.format(self.opensmile_tool_dir, full_wav_path, save_path)], stderr=subprocess.PIPE, shell=True)
        err = p.stderr.read()
        if err:
            raise RuntimeError(err)
        
        df = pd.read_csv(save_path, delimiter=';')
        wav_ft_data = np.array(df.iloc[:, 2:])
        if self.downsample > 0:
            if len(wav_ft_data) > self.downsample:
                wav_ft_data = spsig.resample_poly(wav_ft_data, up=1, down=self.downsample, axis=0)
                if self.no_tmp:
                    os.remove(save_path) 
            else:
                raise ValueError('Error in {wav}, signal length must be longer than downsample parameter')
        return wav_ft_data


class IS10Extractor(object):
    ''' 抽取IS10特征, 输入音频路径, 输出npy数组, 每帧1568d
    '''
    def __init__(self, opensmile_tool_dir=None, downsample=-1, tmp_dir='/data7/emobert/comparE_feature/raw_fts', no_tmp=False):
        ''' Extract ComparE feature
            tmp_dir: where to save opensmile csv file
            no_tmp: if true, delete tmp file
            downsample. if =-1, then use the raw comparE fts, else use the resampeld fts.
        '''
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if opensmile_tool_dir is None:
            opensmile_tool_dir = '/data2/zjm/tools/opensmile-3.0-linux-x64'
        self.opensmile_tool_dir = opensmile_tool_dir
        self.tmp_dir = tmp_dir
        self.downsample = downsample
        self.no_tmp = no_tmp

    def read_feats(self, one_feats_file):
        if one_feats_file is None:
            feats_vector = np.zeros((1582))
        else:
            lines = read_file(one_feats_file)
            feat_line = lines[-1]
            splits = feat_line.strip().split(',')
            feats_vector = [eval(v) for v in splits[1:-1]]
            if len(feats_vector) != 1582:
                print('Too short {}'.format(one_feats_file))
                feats_vector = np.zeros((1582))
        return feats_vector

    def __call__(self, full_wav_path):
        # such as: /data7/emobert/data_nomask_new/audio_clips/No0079.The.Kings.Speech/188.wav
        movie_name = full_wav_path.split('/')[-2]
        basename = movie_name + '_' + os.path.basename(full_wav_path).split('.')[0]
        save_path = os.path.join(self.tmp_dir, basename+".csv")
        cmd = 'SMILExtract -l 0 -C {}/config/is09-13/IS10_paraling_compat.conf -I {} -O {} -noconsoleoutput 1'
        p = subprocess.Popen([cmd.format(self.opensmile_tool_dir, full_wav_path, save_path)], stderr=subprocess.PIPE, shell=True)
        err = p.stderr.read()
        if err:
            raise RuntimeError(err)
            print('Extract error {}'.format(save_path))
            save_path = None
        wav_ft_data = self.read_feats(save_path)
        return np.array(wav_ft_data)

class Wav2VecExtractor(object):
    ''' 
    # work on env=transformers 4.4 
    抽取comparE特征, 输入音频路径, 输出npy数组, 每帧768d
    '''
    def __init__(self, downsample=4, gpu=0, model_name='facebook/wav2vec2-base', use_asr_based_model=False):
        self.downsample = downsample
        self.device = torch.device('cuda:{}'.format(gpu))
        self.model_name = model_name
        print('[INFO] use {} model'.format(model_name))
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        
    @staticmethod
    def read_audio(wav_path):
        speech, sr = sf.read(wav_path)
        if sr != 16000:
            speech = librosa.resample(speech, sr, 16000)
            sr = 16000
        if sr * 10 < len(speech):
            print(f'{wav_path} long than 10 seconds and clip {speech.shape}')
            speech = speech[:int(sr * 10)]
        return speech, sr

    def __call__(self, wav_fileapth):
        input_values, sr = Wav2VecExtractor.read_audio(wav_fileapth)
        input_values = self.processor(input_values, return_tensors="pt", sampling_rate=sr).input_values.to(self.device)
        with torch.no_grad():
            try:
                ft = self.model(input_values).last_hidden_state
            except:
                print('{} audio is too short {}'.format(wav_fileapth, len(input_values)))
                ft = torch.zeros([])

        if self.downsample > 0:
            ft = torch.cat([
                torch.mean(ft[:, i:i+self.downsample], dim=1) for i in range(0, ft.shape[1], self.downsample)
            ], dim=0)
        return ft.cpu().numpy()

def get_uttId2features(extractor, meta_filepath, movie_audio_dir):
    '''
    UttId = {spk}_{dialogID}_{uttID}
    '''
    uttId2speechpath = collections.OrderedDict()
    uttId2ft = collections.OrderedDict()
    instances = read_csv(meta_filepath, delimiter=';', skip_rows=1)
    for instance in instances:
        UtteranceId = instance[0]
        if UtteranceId is None:
            continue
        dialog_id = '_'.join(UtteranceId.split('_')[:2])
        spk = instance[4]
        new_uttId = '{}_{}'.format(spk, UtteranceId)
        audio_filepath = os.path.join(movie_audio_dir, dialog_id, 'pyaudios', new_uttId + '.wav')
        assert os.path.exists(audio_filepath) == True
        ft = extractor(audio_filepath)
        uttId2ft[new_uttId] = ft # (bs, len, dim) / (dim) 
        uttId2speechpath[new_uttId] = audio_filepath
    return uttId2speechpath, uttId2ft

def get_sentence_level_ft(sent_type, output_ft_filepath, feat_dim=1024):
    new_utt2ft = collections.OrderedDict()
    utt2feat = read_pkl(output_ft_filepath)
    for uttId in utt2feat.keys():
        ft = utt2feat[uttId]
        if ft.size == 1:
            print('Utt {} is None Speech'.format(uttId))
            new_ft = np.zeros(feat_dim)
        else:
            ft = ft[0] # bs position
            if sent_type == 'sent_avg':
                new_ft = np.mean(ft, axis=0)
            elif sent_type == 'sent_cls':
                # 最后一个时刻用于分类
                new_ft = ft[-1]
            else:
                print('Error sent type {}'.format(sent_type))
        new_utt2ft[uttId] = new_ft
    assert len(utt2feat) == len(new_utt2ft)
    return new_utt2ft

if __name__ == '__main__':
    # export PYTHONPATH=/data9/MEmoConv
    # CUDA_VISIBLE_DEVICES=7 python extract_speech_ft.py  
    feat_type = 'wav2vec_zh'
    all_output_ft_filepath = '/data9/memoconv/modality_fts/speech/all_speech_ft_{}.pkl'.format(feat_type)
    all_text_info_filepath = '/data9/memoconv/modality_fts/speech/all_speech_path_info.pkl'
    movies_names = read_file('../preprocess/movie_list.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]
    movie2uttID2ft = collections.OrderedDict()
    movie2uttID2speechpath = collections.OrderedDict()

    if False:
        if feat_type == 'comparE':
            print('Using ComparE extactor')
            extractor = ComParEExtractor(tmp_dir='/data9/memoconv/modality_fts/speech/comparE_raw_fts')
        elif feat_type == 'wav2vec':
            print('Using wav2vec extactor')
            extractor = Wav2VecExtractor(downsample=-1, gpu=0, model_name='facebook/wav2vec2-base')
        elif feat_type == 'wav2vec_zh':
            print('Using new wav2vec zh extactor')
            extractor = Wav2VecExtractor(downsample=-1, gpu=0, model_name='jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
        elif feat_type == 'IS10':
            print('Using IS10 extactor')
            extractor = IS10Extractor(tmp_dir='/data9/memoconv/modality_fts/speech/IS10_raw_fts')
        else:
            print(f'Error feat type {feat_type}')
        # extract all faces, only in the utterance
        for movie_name in movies_names:
            print(f'Current movie {movie_name}')
            output_ft_filepath = '/data9/memoconv/modality_fts/speech/movies/{}_speech_ft_{}.pkl'.format(movie_name, feat_type)
            text_info_filepath = '/data9/memoconv/modality_fts/speech/movies/{}_speechpath_info.pkl'.format(movie_name)
            meta_filepath = '/data9/memoconv/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
            movie_audio_dir = '/data9/memoconv/memoconv_convs_talknetoutput/{}'.format(movie_name)
            uttId2speechpath, uttId2ft = get_uttId2features(extractor, meta_filepath, movie_audio_dir)
            write_pkl(output_ft_filepath, uttId2ft)
            if not os.path.exists(text_info_filepath):
                write_pkl(text_info_filepath, uttId2speechpath)
            movie2uttID2ft.update(uttId2ft)
            movie2uttID2speechpath.update(uttId2speechpath)
        write_pkl(all_output_ft_filepath, movie2uttID2ft)
        if not os.path.exists(all_text_info_filepath):
            write_pkl(all_text_info_filepath, movie2uttID2speechpath)
    
    if False:
        # get sentence-level features by cls and average pool
        sent_type = 'sent_cls' # sent_avg, sent_cls
        feat_dim = 1024 # wav2vec_zh 1024
        for movie_name in movies_names:
            print(f'Current movie {movie_name}')
            output_ft_filepath = '/data9/memoconv/modality_fts/speech/movies/{}_speech_ft_{}.pkl'.format(movie_name, feat_type)
            output_sent_ft_filepath = '/data9/memoconv/modality_fts/speech/movies/{}_speech_ft_{}_{}.pkl'.format(movie_name, sent_type, feat_type)
            uttId2ft = get_sentence_level_ft(sent_type, output_ft_filepath, feat_dim)
            write_pkl(output_sent_ft_filepath, uttId2ft)