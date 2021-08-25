import collections
from collections import Counter
import os
import sys
import random
from preprocess.FileOps import read_file, read_pkl, read_csv, write_json, read_json

def get_movie_meta_info(movie_name, meta_dir, movie_meta_info_filepath):
    '''
    {turn_num:
    utt_num:
    kappa:
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
    meta_info['FleissKappa'] = all_instances[0][30]
    meta_info['emo_distribution'] = final_emo2count
    meta_info['inter_emo_shift'] = emotion_shift
    meta_info['inter_emo_inertia'] = emotion_inertia
    meta_info['intra_emo_shift'] = intra_emotion_shift
    meta_info['intra_emo_inertia'] = intra_emotion_inertia
    write_json(movie_meta_info_filepath, meta_info)

def get_sub_set_statistic(set_movies, target_dir):
    emo2count = {}
    num_turns = 0
    num_utts = 0
    inter_emoshift = []
    inter_emoinertia = []
    intra_emoshift = []
    intra_emoinertia = []
    for movie_name in set_movies:
        meta_filepath = os.path.join(target_dir, '{}_statistic.json'.format(movie_name))
        meta_info = read_json(meta_filepath)
        mov_emo2count = meta_info['emo_distribution']
        for emo in mov_emo2count.keys():
            if emo2count.get(emo) is None:
                emo2count[emo] = mov_emo2count[emo]
            else:
                emo2count[emo] += mov_emo2count[emo]
        inter_emoshift.extend(meta_info['inter_emo_shift'])
        inter_emoinertia.extend(meta_info['inter_emo_inertia'])
        intra_emoshift.extend(meta_info['intra_emo_shift'])
        intra_emoinertia.extend(meta_info['intra_emo_inertia'])
        num_turns += meta_info['num_turns']
        num_utts += meta_info['num_utts']
    inter_emoshift = Counter(inter_emoshift)
    inter_emoinertia = Counter(inter_emoinertia)
    intra_emoshift = Counter(intra_emoshift)
    intra_emoinertia = Counter(intra_emoinertia)
    return emo2count, num_turns, num_utts, inter_emoshift, inter_emoinertia, intra_emoshift, intra_emoinertia

def get_rate_info(total_emo2count, emo2count):
    rate_emo_count = collections.OrderedDict()
    for emo in total_emo2count.keys():
        if emo2count.get(emo) is None:
            rate = 0
        else:
            rate = emo2count[emo] / total_emo2count[emo]
        rate_emo_count[emo] = round(rate, 4)
    return rate_emo_count

def get_emo_change_rate(inter_emoshift, inter_emoinertia, intra_emoshift, intra_emoinertia):
    total_emo_shift = {'Neutral_Sad': 420, 'Sad_Neutral': 417, 'Neutral_Anger': 406, 'Neutral_Surprise': 381, 'Surprise_Neutral': 377, 'Anger_Neutral': 357, 'Neutral_Happy': 338, 'Sad_Anger': 280, 'Happy_Neutral': 276, 'Anger_Sad': 270, 'Neutral_Disgust': 205, 'Disgust_Neutral': 173, 'Anger_Disgust': 129, 'Disgust_Anger': 125, 'Surprise_Anger': 100}
    total_emo_inertia = {'Neutral_Neutral': 1552, 'Anger_Anger': 539, 'Happy_Happy': 289, 'Sad_Sad': 234, 'Disgust_Disgust': 44, 'Surprise_Surprise': 33, 'Fear_Fear': 5}
    total_intra_emo_shift = {'Sad_Neutral': 275, 'Neutral_Sad': 268, 'Neutral_Happy': 257, 'Neutral_Anger': 236, 'Anger_Neutral': 232, 'Happy_Neutral': 199, 'Anger_Sad': 164, 'Sad_Anger': 156, 'Neutral_Disgust': 118, 'Disgust_Neutral': 117, 'Surprise_Neutral': 101}
    total_intra_emo_inertia = {'Neutral_Neutral': 4871, 'Anger_Anger': 2523, 'Sad_Sad': 1830, 'Happy_Happy': 798, 'Disgust_Disgust': 564, 'Fear_Fear': 156, 'Surprise_Surprise': 149}
    inter_emoshift_rate = get_rate_info(total_emo_shift, inter_emoshift)
    inter_emoinertia_rate = get_rate_info(total_emo_inertia, inter_emoinertia)
    intra_emoshift_rate = get_rate_info(total_intra_emo_shift, intra_emoshift)
    intra_emoinertia_rate = get_rate_info(total_intra_emo_inertia, intra_emoinertia)
    return inter_emoshift_rate, inter_emoinertia_rate, intra_emoshift_rate, intra_emoinertia_rate

def get_split_info(movies_names, target_dir, train_num=38, val_num=7, test_num=11):
    total_emo2count = {'Anger': 5234, 'Neutral': 10028, 'Sad': 3957, 'Happy': 2287, 'Surprise': 1051, 'Disgust': 1497, 'Fear': 395}
    train_samples = random.sample(movies_names, train_num)
    val_test_samples = []
    for movie in movies_names:
        if movie not in train_samples:
            val_test_samples.append(movie)
    assert len(val_test_samples) + len(train_samples) == len(movies_names)
    val_samples = random.sample(val_test_samples, val_num)
    test_samples = []
    for movie in val_test_samples:
        if movie not in val_samples:
            test_samples.append(movie)
    assert len(test_samples) == test_num
    assert len(test_samples) + len(val_samples) == len(val_test_samples)
    # 统计每一个set的情感分布，多次运行找合适的
    print(train_samples)
    print(val_samples)
    print(test_samples)
    emo2count,  num_turns, num_utts, inter_emoshift, inter_emoinertia, intra_emoshift, intra_emoinertia = get_sub_set_statistic(train_samples, target_dir)
    rate_emo_count = get_rate_info(total_emo2count, emo2count)
    print('******* Train set {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(num_turns/9082, num_utts/24449, sum(inter_emoshift.values())/5396, sum(inter_emoinertia.values())/2696, 
                                                            sum(intra_emoshift.values())/2879, sum(intra_emoinertia.values())/10891))
    print(rate_emo_count)
    inter_emoshift_rate, inter_emoinertia_rate, intra_emoshift_rate, intra_emoinertia_rate = get_emo_change_rate(inter_emoshift, inter_emoinertia, intra_emoshift, intra_emoinertia)
    print(inter_emoshift_rate)
    print(inter_emoinertia_rate)
    print(intra_emoshift_rate)
    print(intra_emoinertia_rate)
    emo2count,  num_turns, num_utts, inter_emoshift, inter_emoinertia, intra_emoshift, intra_emoinertia = get_sub_set_statistic(val_samples, target_dir)
    rate_emo_count = get_rate_info(total_emo2count, emo2count)
    print('*******  Val set {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(num_turns/9082, num_utts/24449, sum(inter_emoshift.values())/5396, sum(inter_emoinertia.values())/2696, 
                                                            sum(intra_emoshift.values())/2879, sum(intra_emoinertia.values())/10891))
    print(rate_emo_count)    
    inter_emoshift_rate, inter_emoinertia_rate, intra_emoshift_rate, intra_emoinertia_rate = get_emo_change_rate(inter_emoshift, inter_emoinertia, intra_emoshift, intra_emoinertia)
    print(inter_emoshift_rate)
    print(inter_emoinertia_rate)
    print(intra_emoshift_rate)
    print(intra_emoinertia_rate)
    emo2count, num_turns, num_utts, inter_emoshift, inter_emoinertia, intra_emoshift, intra_emoinertia = get_sub_set_statistic(test_samples, target_dir)
    rate_emo_count = get_rate_info(total_emo2count, emo2count)
    print('*******  Test set {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(num_turns/9082, num_utts/24449, sum(inter_emoshift.values())/5396, sum(inter_emoinertia.values())/2696, 
                                                            sum(intra_emoshift.values())/2879, sum(intra_emoinertia.values())/10891))
    print(rate_emo_count)
    inter_emoshift_rate, inter_emoinertia_rate, intra_emoshift_rate, intra_emoinertia_rate = get_emo_change_rate(inter_emoshift, inter_emoinertia, intra_emoshift, intra_emoinertia)
    print(inter_emoshift_rate)
    print(inter_emoinertia_rate)
    print(intra_emoshift_rate)
    print(intra_emoinertia_rate)

if __name__ == '__main__':
    movies_names = read_file('movie_list.txt')
    movies_names = [movie_name.strip() for movie_name in movies_names]

    meta_dir = '/data9/memoconv/memoconv_final_labels_csv'
    target_dir = '/data9/memoconv/split_set/movie_meta_info'

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    if False:
        for movie_name in movies_names:
            movie_meta_info_filepath = os.path.join(target_dir, '{}_statistic.json'.format(movie_name))
            get_movie_meta_info(movie_name, meta_dir, movie_meta_info_filepath)
    
    if False:
        run_num = sys.argv[1]
        print('---------------- {} -------------'.format(run_num))
        get_split_info(movies_names, target_dir, train_num=38, val_num=7, test_num=11)