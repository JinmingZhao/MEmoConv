import os
import math
import collections
from FileOps import read_xls, write_xls, read_file

'''
step1: 验证三个文件的行数是否一致
step2: 验证三个文件的对话的个数以及每个对话的句子数目，以及句子的ID是否唯一
step3: 验证说话人个数是否是只有AB, 验证第一句话是否为A，统计对话的轮数
step4: 然后验证情感的标注信息，所有的句子数目，完全相同的情感句子数目，第一情感相同的句子数据，情感有交集的句子数目
step5: 文本数据的清洗，将所有的英文标点符合替换为中文标点符号, 以anno3为标准。
'''

def check_dialog(anno_instances):
    dialogs = collections.OrderedDict()
    dialogs2spks = collections.OrderedDict()
    for instance_id in range(len(anno_instances)):
        instance = anno_instances[instance_id]
        UtteranceId = instance[0].value
        if UtteranceId is None:
            continue
        dialog_id = '_'.join(UtteranceId.split('_')[:2])
        spkId = instance[4].value
        if spkId not in ['A', 'B']:
            print('\t Error dialog {} has more than 2 speakers'.format(dialog_id))      
            exit(0)
        if dialogs.get(dialog_id) is None:
            dialogs[dialog_id] = [UtteranceId]
            if spkId != 'A':
                print('\t Error dialog {} not start at speaker A'.format(dialog_id))   
                exit(0)
            dialogs2spks[dialog_id] = [spkId]
        else:
            dialogs[dialog_id] += [UtteranceId]
            dialogs2spks[dialog_id] += [spkId]
    num_dialogs = len(dialogs)
    # print('\t dialog nums {}'.format(num_dialogs))
    num_utts = 0
    for dialog_id in dialogs.keys():
        utts = dialogs[dialog_id]
        if len(set(utts)) != len(utts):
            print('In dialog {} the uttId is not unique {} {}'.format(dialog_id, len(set(utts)),  len(utts)))
            exit(0)
        num_utts += len(utts)
    # print('\t total utts nums {}'.format(num_utts))
    total_num_turns = 0
    dialogid2num_turns = collections.OrderedDict()
    for dialog_id in dialogs2spks.keys():
        spks = dialogs2spks[dialog_id]
        num_turns = 0
        cur_spk = 'A'
        for i in range(len(spks)):
            if spks[i] != cur_spk:
                num_turns +=1 
                cur_spk = spks[i]
        dialogid2num_turns[dialog_id] = num_turns
        total_num_turns += num_turns
    # print('\t total turns nums {}'.format(num_turns))
    return num_dialogs, total_num_turns, num_utts, dialogid2num_turns, dialogs

def analyse_dialog_emotions(anno_instances):
    dialogs2emos = collections.OrderedDict()
    pre_emos = None
    total_emo_anno = 0
    for instance_id in range(len(anno_instances)):
        instance = anno_instances[instance_id]
        UtteranceId = instance[0].value
        if UtteranceId is None:
            continue
        emos_str = instance[5].value
        dialog_id = '_'.join(UtteranceId.split('_')[:2])
        if dialogs2emos.get(dialog_id) is None:
            # 每个对话的第一句话的情感标注不可以为空
            if emos_str is None:
                print('\t \t Error emo anno is Empty in {}'.format(dialog_id))
                exit(0)
            emos = emos_str.split(',')
            dialogs2emos[dialog_id] = [emos]
            total_emo_anno += len(emos)
            # 更新 pre_emos
            pre_emos = emos
        else:
            if emos_str is None:
                emos = pre_emos
            else:
                # 随时更新 pre_emos
                emos = emos_str.split(',')
                pre_emos = emos
            dialogs2emos[dialog_id] += [emos]
            total_emo_anno += len(emos)
    return total_emo_anno, dialogs2emos

def analyse_emotion_three_consistency(anno1_dialogs2emos, anno2_dialogs2emos, anno3_dialogs2emos):
    '''
    三个人:
        完全相同的情感句子数目: num_total_same
        第一情感相同的句子数据: num_main_same
        情感有交集的句子数目: num_part_same
    '''
    num_total_same = 0
    num_main_same = 0
    num_part_same = 0
    for dialog_id in anno1_dialogs2emos.keys():
        anno1_dailog_emos = anno1_dialogs2emos[dialog_id]
        anno2_dailog_emos = anno2_dialogs2emos[dialog_id]
        anno3_dailog_emos = anno3_dialogs2emos[dialog_id]
        for utt_index in range(len(anno1_dailog_emos)):
            if anno1_dailog_emos[utt_index] == anno2_dailog_emos[utt_index] == anno3_dailog_emos[utt_index]:
                num_total_same += 1
            elif anno1_dailog_emos[utt_index][0] == anno2_dailog_emos[utt_index][0] == anno3_dailog_emos[utt_index][0]:
                num_main_same += 1
            else:
                part_same_emos = set(anno1_dailog_emos[utt_index]).intersection(set(anno2_dailog_emos[utt_index]),set(anno3_dailog_emos[utt_index]))
                if len(part_same_emos) > 0:
                    num_part_same += 1
    num_part_same = num_total_same + num_main_same + num_part_same
    num_main_same = num_total_same + num_main_same
    return num_total_same, num_main_same, num_part_same

def analyse_emotion_two_consistency(anno1_dialogs2emos, anno2_dialogs2emos, anno3_dialogs2emos):
    '''
    一般来说我们情感标注，只要两人的情感一致就可以进行决策。
    两个人的情感满足相似即可:
        完全相同的情感句子数目: num_total_same
        第一情感相同的句子数据: num_main_same
        情感有交集的句子数目: num_part_same
    '''
    num_total_same = 0
    num_main_same = 0
    num_part_same = 0
    for dialog_id in anno1_dialogs2emos.keys():
        anno1_dailog_emos = anno1_dialogs2emos[dialog_id]
        anno2_dailog_emos = anno2_dialogs2emos[dialog_id]
        anno3_dailog_emos = anno3_dialogs2emos[dialog_id]
        for utt_index in range(len(anno1_dailog_emos)):
            # 统计二者完全相同的
            if anno1_dailog_emos[utt_index] == anno2_dailog_emos[utt_index]:
                num_total_same += 1
            elif anno1_dailog_emos[utt_index] == anno3_dailog_emos[utt_index]:
                num_total_same += 1
            elif anno2_dailog_emos[utt_index] == anno3_dailog_emos[utt_index]:
                num_total_same += 1
            else:
                pass
            # 统计二者主情感相同
            if anno1_dailog_emos[utt_index][0] == anno2_dailog_emos[utt_index][0]:
                num_main_same += 1
            elif anno1_dailog_emos[utt_index][0] == anno3_dailog_emos[utt_index][0]:
                num_main_same += 1
            elif anno2_dailog_emos[utt_index][0] == anno3_dailog_emos[utt_index][0]:
                num_main_same += 1
            else:
                pass
            # 统计二者部分情感相同
            inter_anno12 = set(anno1_dailog_emos[utt_index]).intersection(set(anno2_dailog_emos[utt_index]))
            inter_anno13 = set(anno1_dailog_emos[utt_index]).intersection(set(anno3_dailog_emos[utt_index]))
            inter_anno23 = set(anno2_dailog_emos[utt_index]).intersection(set(anno3_dailog_emos[utt_index]))
            if len(inter_anno12) > 0 or len(inter_anno13) > 0 or  len(inter_anno23) > 0:
                num_part_same += 1
    return num_total_same, num_main_same, num_part_same


def write_xls_oneline():
    '''
    电视剧名字作为ID，统计每部电视剧的统计信息，如果存在那么更新，如果不存在那么追加
    '''
    pass


if __name__ == '__main__':
    movies_names = read_file('movie_list.txt')
    movie_name = movies_names[4].strip()
    print('Current movie {}'.format(movie_name))
    anno1_path = '/Users/jinming/Desktop/works/memoconv_labels/{}_anno1_done.xlsx'.format(movie_name)
    anno2_path = '/Users/jinming/Desktop/works/memoconv_labels/{}_anno2_done.xlsx'.format(movie_name)
    anno3_path = '/Users/jinming/Desktop/works/memoconv_labels/{}_anno3_done.xlsx'.format(movie_name)

    # 整理统计的结果
    result_filepath = '/Users/jinming/Desktop/works/memoconv_labels/aaaa_total.xlsx'

    # step1
    anno1_instances = read_xls(anno1_path, sheetname=movie_name, skip_rows=1)
    anno2_instances = read_xls(anno2_path, sheetname='工作表1', skip_rows=1)
    anno3_instances = read_xls(anno3_path, sheetname='工作表1', skip_rows=1)
    print('\t anno1 {} anno2 {} anno3 {}'.format(len(anno1_instances), len(anno2_instances), len(anno3_instances)))
    # step2 & 3
    print('\t checking dailog 1')
    anno1_num_dialogs, anno1_num_turns, anno1_num_utts, anno1_dialogid2num_turns, anno1_dialogs = check_dialog(anno1_instances)
    print('\t checking dailog 2')
    anno2_num_dialogs, anno2_num_turns, anno2_num_utts, anno2_dialogid2num_turns, anno2_dialogs = check_dialog(anno2_instances)
    print('\t checking dailog 3')
    anno3_num_dialogs, anno3_num_turns, anno3_num_utts, anno3_dialogid2num_turns, anno3_dialogs = check_dialog(anno3_instances)
    assert anno1_num_dialogs == len(anno1_dialogid2num_turns) == len(anno1_dialogs)
    print('\t dialog_num: anno1 {} anno2 {} anno3 {}'.format(anno1_num_dialogs, anno1_num_dialogs, anno1_num_dialogs))
    assert anno1_num_dialogs == anno2_num_dialogs == anno3_num_dialogs
    print('\t dialog_turns_num: anno1 {} anno2 {} anno3 {}'.format(anno1_num_turns, anno2_num_turns, anno3_num_turns))
    flag = (anno1_num_turns == anno2_num_turns == anno3_num_turns)
    if not flag:
        for dialog_id in anno1_dialogid2num_turns.keys():
            sub_flag = (anno1_dialogid2num_turns[dialog_id] == anno2_dialogid2num_turns[dialog_id] == anno3_dialogid2num_turns[dialog_id])
            if not sub_flag:
                print('\t \t Error in dailog {}: anno1 {} anno2 {} anno3 {}'.format(dialog_id, anno1_dialogid2num_turns[dialog_id], 
                                anno2_dialogid2num_turns[dialog_id], anno3_dialogid2num_turns[dialog_id]))
    print('\t dialog_utts_num: anno1 {} anno2 {} anno3 {}'.format(anno1_num_utts, anno2_num_utts, anno3_num_utts))
    assert anno1_num_utts == anno2_num_utts == anno3_num_utts
    # step4
    print('\t analyse emotion of dailog 1')
    anno1_total_emo_anno, anno1_dialogs2emos = analyse_dialog_emotions(anno1_instances)
    print('\t analyse emotion of dailog 2')
    anno2_total_emo_anno, anno2_dialogs2emos = analyse_dialog_emotions(anno2_instances)
    print('\t analyse emotion of dailog 3')
    anno3_total_emo_anno, anno3_dialogs2emos = analyse_dialog_emotions(anno3_instances)
    print('\t total emo annos: anno1 {} anno2 {} anno3 {}'.format(anno1_total_emo_anno, anno2_total_emo_anno, anno3_total_emo_anno))
    num_total_same, num_main_same, num_part_same = analyse_emotion_three_consistency(anno1_dialogs2emos, anno2_dialogs2emos, anno3_dialogs2emos)
    print('\t three people emo consistency: total_same {} main_same {} part_same {}'.format(num_total_same, num_main_same, num_part_same))
    num_total_same, num_main_same, num_part_same = analyse_emotion_two_consistency(anno1_dialogs2emos, anno2_dialogs2emos, anno3_dialogs2emos)
    print('\t two people emo consistency: total_same {} main_same {} part_same {}'.format(num_total_same, num_main_same, num_part_same))