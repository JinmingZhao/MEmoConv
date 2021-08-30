'''
统计所有在论文中需要汇报的数据
'''

import os
import numpy as np
import openpyxl
import collections
from collections import Counter
from FileOps import read_xls, read_file
from anno_check import compute_fleiss_kappa, compute_fleiss_kappa_noother

def compute_global_fleiss_kappa(movies_names):
    all_p1_emos, all_p2_emos, all_p3_emos = [], [], []
    for movie_name in movies_names:
        meta_fileapth = '/Users/jinming/Desktop/works/memoconv_final_labels/meta_{}.xlsx'.format(movie_name)
        instances = read_xls(meta_fileapth, sheetname='sheet1', skip_rows=1)
        for i in range(len(instances)):
            anno1_emo_str, anno2_emo_str, anno3_emo_str = instances[i][5].value, instances[i][6].value, instances[i][7].value
            anno1_emos, anno2_emos, anno3_emos = anno1_emo_str.split(','), anno2_emo_str.split(','), anno3_emo_str.split(',')
            all_p1_emos.append(anno1_emos[0])
            all_p2_emos.append(anno2_emos[0])
            all_p3_emos.append(anno3_emos[0])
    fleiss_kappa = compute_fleiss_kappa(all_p1_emos, all_p2_emos, all_p3_emos)
    fleiss_kappa_noother = compute_fleiss_kappa_noother(all_p1_emos, all_p2_emos, all_p3_emos)
    print('global fleiss_kappa (include other) {}'.format(fleiss_kappa))
    print('global fleiss_kappa (No other) {}'.format(fleiss_kappa_noother))
    
def compute_corpus_duration(movies_names):
    '''
    具体的每个句子的时间长度, 顺便统计一下文本的长度
    '''
    total_dialog_duration = []
    total_duration = []
    total_text_len = []
    for movie_name in movies_names:
        meta_fileapth = '/Users/jinming/Desktop/works/memoconv_final_labels/meta_{}.xlsx'.format(movie_name)
        all_instances = read_xls(meta_fileapth, sheetname='sheet1', skip_rows=1)
        dialog2time_stamp = collections.OrderedDict()
        for instance in all_instances:
            UtteranceId = instance[0].value
            if UtteranceId is None:
                continue
            start_time, end_time, text = instance[1:4]
            h, m, s, fs = start_time.value.split(':')
            start_second = int(m)* 60  + int(s) + float(int(fs)/25)
            h, m, s, fs = end_time.value.split(':')
            end_second = int(m)* 60  + int(s) + float(int(fs)/25)
            dur = end_second - start_second
            total_duration.append(dur)
            total_text_len.append(len(text.value))

            dialog_id = '_'.join(UtteranceId.split('_')[:2])
            if dialog2time_stamp.get(dialog_id) is None:
                dialog2time_stamp[dialog_id] = [[start_second, end_second]]
            else:
                dialog2time_stamp[dialog_id] += [[start_second, end_second]]
        for dialog_id in dialog2time_stamp:
            dur = dialog2time_stamp[dialog_id][-1][1] - dialog2time_stamp[dialog_id][0][0]
            total_dialog_duration.append(dur)
        
    print('total duration {} hour'.format(sum(total_duration)/3600, ))
    print('average utt duration {} second'.format(sum(total_duration)/len(total_duration)))
    print('average utt text {} charaters'.format(sum(total_text_len)/len(total_text_len)))
    print('total dialog duration {} hour'.format(sum(total_dialog_duration)/3600))
    print('average dialog duration {} second'.format(sum(total_dialog_duration)/len(total_dialog_duration)))

def compute_spk_age_gender(movies_names, dialog_spk_info_filepath):
    gender_map = {'Female':'Female', 'female':'Female', 'Femal':'Female', 'Male': 'Male', 'male':'Male'}
    age_map = {'Child':'Child', 'child':'Child', 'Young':'Young', 'young':'Young', 'Yong':'Young', 'yound': 'Young', 'Yound':'Young',
                        'Mid': 'Mid', 'mid': 'Mid', 'Old':'Old', 'old':'Old'}
    age2count = {'Child': 0, 'Young':0, 'Mid':0, 'Old': 0}
    gender2count = {'Female': 0, 'Male':0}
    all_actors = 0
    for movie_name in movies_names:
        print('Current movie {}'.format(movie_name))
        actors = 0
        all_instances = read_xls(dialog_spk_info_filepath, sheetname=movie_name, skip_rows=0)
        actor_names = all_instances[0][1:]
        age_names = all_instances[1][1:]
        gender_names = all_instances[2][1:]
        for actor, age, gender in zip(actor_names, age_names, gender_names):
            actor, age, gender = actor.value, age.value, gender.value
            if actor == age == gender == None:
                continue
            if age_map.get(age) is None:
                print('Error Age {}'.format(age))
            else:
                age = age_map[age]
            if gender_map.get(gender) is None:
                print('Error Gender {}'.format(gender))
            else:
                gender = gender_map[gender]
            actors += 1
            age2count[age] += 1    
            gender2count[gender] += 1
        all_actors += actors
    assert all_actors == sum(age2count.values()) == sum(gender2count.values())
    print(f'all actors {all_actors}')
    print(f'Gender distribution {gender2count}')
    print(f'Age distribution {age2count}')

def compute_emotion_distribution(movies_names):
    '''
    turn 间的情感变化 inter-shift, inter-inertia 和 turn 内的情感变化 intra-shift, intra-inertia
    '''
    multi_emos2count = {}
    final_emo2count = {}
    emotion_shift, emotion_inertia = [], []
    intra_emotion_inertia, intra_emotion_shift = [], []
    for movie_name in movies_names:
        print('Current movie {}'.format(movie_name))
        meta_fileapth = '/Users/jinming/Desktop/works/memoconv_final_labels/meta_{}.xlsx'.format(movie_name)
        all_instances = read_xls(meta_fileapth, sheetname='sheet1', skip_rows=1)
        dialog2emos = {}
        dialog2spks = {}
        for instance in all_instances:
            UtteranceId = instance[0].value
            if UtteranceId is None:
                continue
            mul_emos, final_emo = instance[8].value, instance[9].value
            # 统计多情感的分布
            if ',' in mul_emos:
                if multi_emos2count.get(mul_emos) is None:
                    multi_emos2count[mul_emos] = 1
                else:
                    multi_emos2count[mul_emos] += 1
            if final_emo2count.get(final_emo) is None:
                final_emo2count[final_emo] = 1
            else:
                final_emo2count[final_emo] += 1
            # 统计对话 Turn 之间情感变化
            dialog_id = '_'.join(UtteranceId.split('_')[:2])
            spk = instance[4].value
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
            # print(f'\t dialog id {dialog_id}')
            for spk, emo in zip(dialog_spks[1:], dialog_emos[1:]):
                if spk != pre_spk:
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
    multi_emo_rate = sum(multi_emos2count.values()) / sum(final_emo2count.values())
    print(f'multi-emo distribution {multi_emos2count}')
    print(f'multi-emo rate {multi_emo_rate}')
    print(f'final_emo2count rate {final_emo2count}')
    emotion_shift_count = len(emotion_shift)
    emotion_inertia_count = len(emotion_inertia)
    print(f'emotion shift {emotion_shift_count}')  
    print(f'emotion inertia {emotion_inertia_count}')
    emo_shift2count = Counter(emotion_shift)
    print(f'emotion shift distribution {emo_shift2count}')
    emo_inertia2count = Counter(emotion_inertia)
    print(f'emotion inertia distribution {emo_inertia2count}')
    # intra-turn emotion shift and inertia
    intra_emotion_shift_count = len(intra_emotion_shift)
    intra_emotion_inertia_count = len(intra_emotion_inertia)
    print(f'intra emotion shift {intra_emotion_shift_count}')  
    print(f'intra emotion inertia {intra_emotion_inertia_count}')
    intra_emo_shift2count = Counter(intra_emotion_shift)
    print(f'intra  emotion shift distribution {intra_emo_shift2count}')
    intra_emo_inertia2count = Counter(intra_emotion_inertia)
    print(f'intra  emotion inertia distribution {intra_emo_inertia2count}')

def get_basic_info(movies_names, statistic_filepath):
    all_instances = read_xls(statistic_filepath, sheetname='sheet1', skip_rows=1)
    count_movies = 0
    count_dialogs = 0
    count_turns = 0
    count_utts = 0
    for instance in all_instances:
        movies_name, num_dialog, num_turn, num_utt = [ins.value for ins in instance[:4]]
        if movies_name in movies_names:
            count_movies += 1
            count_dialogs += int(num_dialog)
            count_turns += int(num_turn)
            count_utts += int(num_utt)
    avg_turns_dialog = round(count_turns/count_dialogs, 2)
    avg_utts_turn = round(count_utts/count_turns, 2)
    avg_utts_dialog = round(count_utts/count_dialogs, 2)
    print('movies {} dialogs {} turns {} utts {} avg_turns {}, avg_utts_turn {} avg_utts_dialog {}'.format(
            count_movies, count_dialogs, count_turns, count_utts, avg_turns_dialog, avg_utts_turn, avg_utts_dialog))

def write_meta_json_info():
    '''
    {'dialogId':
        'spkA': xxx,
        'spkB': xxx, 
        'starttime_raw_episode': xxx
        'endtime_raw_episode': xxx
        {
            uttId:{
                'speaker': A
                'starttime': xxx,
                'endtime': xxx,
                'duration': xxx,
                'text': xxx,
                'final_mul_emos': [],
                'final_main_emo': [],
                'annotator1': [],
                'annotator2': [],
                'annotator3': [],
                'annotator4': []
            }
        }
    }
    '''
    pass

if __name__ == '__main__':

    set_name = 'total'
    movies_names = read_file('movie_list_{}.txt'.format(set_name))
    movies_names = [movie_name.strip() for movie_name in movies_names]
    statistic_filepath = '/Users/jinming/Desktop/works/memoconv_final_labels/statistic.xlsx'

    if True:
        get_basic_info(movies_names, statistic_filepath)

    if True:
        compute_global_fleiss_kappa(movies_names)
    
    if True:
        compute_corpus_duration(movies_names)
    
    if True:
        dialog_spk_info_filepath = '/Users/jinming/Desktop/works/memoconv_final_labels/dialogSpkAnnoUpdate.xlsx'
        compute_spk_age_gender(movies_names, dialog_spk_info_filepath)

    if True:
        compute_emotion_distribution(movies_names)
    
    
    