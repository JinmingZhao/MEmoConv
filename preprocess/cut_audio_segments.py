'''
将语音按照句子进行切割，适当的前后扩1帧，也就是50ms.
dialogId_uttID.wav  比如 fendou_1_1.wav
'''
import os
import soundfile as sf
import numpy as np
import subprocess
import scipy.signal as spsig
from FileOps import read_file
from extract_spk_faces import get_spk2timestamps

def read_wav(wav_path):
    wav_data, sr = sf.read(wav_path, dtype='int16')
    return wav_data, sr

def write_wav(wav_path, wav_data, sr):
    sf.write(wav_path, wav_data, sr)
    assert os.path.exists(wav_path) == True

def transtimestamp2second(timestamp):
    # 不需要考虑fps的问题，经过验证，目前的都是合适的
    start_time, end_time = timestamp[0], timestamp[1]
    # print(start_time, end_time)
    h, m, s, fs = start_time.split(':')
    start_second = int(m)* 60  + int(s) + int(fs) * 40 * 0.001
    h, m, s, fs = end_time.split(':')
    end_second = int(m)* 60 + int(s) + int(fs) * 40 * 0.001
    return start_second, end_second

def get_audio_utts(dialog_audios_dir, cur_dialog_wav_filepath, spk2timestamps, spk2uttIds):
    wav_data, sr = read_wav(cur_dialog_wav_filepath)
    assert sr == 16000
    for spk in spk2timestamps.keys():
        timestamps = spk2timestamps[spk]
        uttIds = spk2uttIds[spk]
        assert len(timestamps) == len(uttIds)
        for timestamp, uttId in zip(timestamps, uttIds):
            start_second, end_second = transtimestamp2second(timestamp)
            # 前后扩充50ms
            # print(uttId, timestamp, start_second, end_second)
            current_wave_data = wav_data[int((start_second+0.05)*sr): int((end_second+0.05)*sr)]
            filepath = os.path.join(dialog_audios_dir, '{}_{}.wav'.format(spk, uttId))
            write_wav(filepath, current_wave_data, sr=sr)


if __name__ == '__main__':
    movies_names = read_file('movie_list.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]
    # extract all faces, only in the utterance
    for movie_name in movies_names:
        print(f'Current movie {movie_name}')
        meta_filepath = '/data9/memoconv/memoconv_final_labels_csv/meta_{}.csv'.format(movie_name)
        talkout_dialog_dir = '/data9/memoconv/memoconv_convs_talknetoutput/{}'.format(movie_name)
        dialog2spk2timestamps, dialog2spk2uttIds = get_spk2timestamps(meta_filepath)
        for dialog_id in list(dialog2spk2timestamps.keys()):
            print('current {}'.format(dialog_id))
            cur_dialog_dir = os.path.join(talkout_dialog_dir, dialog_id)
            dialog_audios_dir = os.path.join(cur_dialog_dir, 'pyaudios')
            if not os.path.exists(dialog_audios_dir):
                os.mkdir(dialog_audios_dir)
            cur_dialog_wav_filepath = os.path.join(cur_dialog_dir, 'pyavi', 'audio.wav')
            spk2timestamps, spk2uttIds = dialog2spk2timestamps[dialog_id], dialog2spk2uttIds[dialog_id]
            get_audio_utts(dialog_audios_dir, cur_dialog_wav_filepath, spk2timestamps, spk2uttIds)