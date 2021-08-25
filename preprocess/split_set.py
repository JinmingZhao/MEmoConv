import collections
import os
from preprocess.FileOps import read_file, read_pkl, read_csv, write_json, read_json

def get_movie_meta_info(movie_name, meta_dir, movie_meta_info_filepath):
    '''
    {turn_num:
    utt_num:
    emo_distribution：
    inter-emoshift:
    inter-emoinertia
    intra-emoshift:
    intra-emoinertia
    }
    '''
    emotion_shift, emotion_inertia = [], []
    intra_emotion_inertia, intra_emotion_shift = [], []
    meta_fileapth = os.path.join(meta_dir, 'meta_{}.csv'.format(movie_name))
    all_instances = read_csv(meta_fileapth, delimiter=';', skip_rows=1)
    # 统计情感分布
    final_emo2count = {}
    # 统计情感和spk，方便后续进行情感变化的分析
    dialog2emos = {}
    dialog2spks = {}
    num_turns = 0
    num_utts = 0
    for instance in all_instances:
        UtteranceId = instance[0]
        if UtteranceId is None:
            continue
        num_utts += 1
        spk, final_emo = instance[4], instance[9]
        if final_emo2count.get(final_emo) is None:
            final_emo2count[final_emo] = 1
        else:
            final_emo2count[final_emo] += 1
        dialog_id = '_'.join(UtteranceId.split('_')[:2])
        if dialog2emos.get(dialog_id) is None:
            dialog2emos[dialog_id] = [final_emo]
        else:    
            dialog2emos[dialog_id] += [final_emo]
        if dialog2spks.get(dialog_id) is None:
            dialog2spks[dialog_id] = [spk]
        else:
            dialog2spks[dialog_id] += [spk]

    # 统计情感turn之间的情感变化组合
    for dialog_id in dialog2emos.keys():
        dialog_emos = dialog2emos[dialog_id]
        dialog_spks = dialog2spks[dialog_id]
        assert len(dialog_emos) == len(dialog_spks)
        pre_spk = dialog_spks[0]
        pre_emo = dialog_emos[0]
        num_turns += 1
        # print(f'\t dialog id {dialog_id}')
        for spk, emo in zip(dialog_spks[1:], dialog_emos[1:]):
            if spk != pre_spk:
                num_turns += 1
                # spk 不同表示turn发生改变
                if emo == pre_emo:
                    emotion_inertia.append(pre_emo+'_'+emo)
                else:
                    emotion_shift.append(pre_emo+'_'+emo)
                    # print(f'emotion_shift {pre_emo} -> {emo}')
                pre_spk = spk
                pre_emo = emo     
    # 统计情感turn内的情感变化组合
    for dialog_id in dialog2emos.keys():
        dialog_emos = dialog2emos[dialog_id]
        dialog_spks = dialog2spks[dialog_id]
        pre_spk = dialog_spks[0]
        turn_emos = [dialog_emos[0]]
        # print(f'\t dialog id {dialog_id}')
        # print(dialog_spks)
        for spk, emo in zip(dialog_spks[1:], dialog_emos[1:]):
            if spk == pre_spk:
                turn_emos.append(emo)
            else:
                # analyse 
                if len(turn_emos) > 1:
                    # print('turn emos {}'.format(turn_emos))
                    pre_emo = turn_emos[0]
                    for emo in turn_emos[1:]:
                        if emo == pre_emo:
                            intra_emotion_inertia.append(pre_emo+'_'+emo)
                            # print(f'intra emotion_inertia {pre_emo} -> {emo}')
                        else:
                            intra_emotion_shift.append(pre_emo+'_'+emo)
                            # print(f'intra emotion_shift {pre_emo} -> {emo}')
                            pre_emo = emo
                # new turn 
                turn_emos = []
                turn_emos.append(emo)
                pre_spk = spk
    meta_info = collections.OrderedDict()
    meta_info['num_turns'] = num_turns
    meta_info['num_utts'] = num_utts
    meta_info['emo_distribution'] = final_emo2count
    meta_info['inter_emo_shift'] = emotion_shift
    meta_info['inter_emo_inertia'] = emotion_inertia
    meta_info['intra_emo_shift'] = intra_emotion_shift
    meta_info['intra_emo_inertia'] = intra_emotion_inertia
    write_json(movie_meta_info_filepath, meta_info)

if __name__ == '__main__':
    movies_names = read_file('movie_list.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]

    meta_dir = '/data9/memoconv/memoconv_final_labels_csv'
    target_dir = '/data9/memoconv/split_set/movie_meta_info'
    for movie_name in movies_names[0:1]:
        movie_meta_info_filepath = os.path.join(target_dir, '{}_.json'.format(movie_name))
        get_movie_meta_info(movie_name, meta_dir, movie_meta_info_filepath)