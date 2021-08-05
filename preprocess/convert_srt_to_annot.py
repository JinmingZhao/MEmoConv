import os
import math
from FileOps import read_file, write_xls

'''
conver the srt to the, every 4 lines
1
00:00:00,000 --> 00:00:00,750
为什么去法国

2
00:00:01,500 --> 00:00:02,450
为学习

3
----
'''

def trans_ms_frames(str_time, is_start):
    # 如果是开始，那么向小的取整数，如果是结束，那么向大的取整数. 保证内容都包含在时间周期内。
    other, ms = str_time.split(',')
    if is_start:
        frames = math.floor(int(ms)/40)
    else:
        frames = math.ceil(int(ms)/40)
    new_str_time = '{}:{:02d}'.format(other, frames)
    return new_str_time

def read_srt(filepath):
    # 将时间信息中的毫秒转化为 frames, 跟ShotCut的时间格式一样，方便定位。
    all_instances = []
    lines = read_file(filepath)
    dialogId = filepath.split('/')[-1].replace('.srt', '')
    for i in range(0, len(lines), 4):
        uttIdx = lines[i].strip()
        if len(uttIdx) == 0:
            continue
        timestamp = lines[i+1]
        start_time, end_time = timestamp.split(' --> ')
        # modify ms to frames
        start_time = trans_ms_frames(start_time, is_start=True)
        end_time = trans_ms_frames(end_time, is_start=False)
        textInfo = lines[i+2]
        uttId = dialogId + '_' + str(uttIdx)
        all_instances.append([uttId, start_time, end_time, textInfo])
    print(len(all_instances), (len(lines)//4))
    assert len(all_instances) == (len(lines)//4)
    return all_instances

def save_annot_format(movie_name, movie_instancess, annot_format_output_path):
    '''
    表头: 
    UtteranceId	StartTime EndTime Text Speaker(First Speaker as A else B)	EmoAnnotator1	EmoAnnotator2	EmoAnnotator3																			
    '''
    movie_instancess.insert(0, ['UtteranceId', 'StartTime', 'EndTime', 'Text', 'Speaker(First Speaker as A else B)', 'EmoAnnotator1', 'Note'])
    write_xls(annot_format_output_path, movie_name, movie_instancess)

if __name__ == '__main__':
    root_dir = '/Users/jinming/Desktop/works/memoconv_annot'
    movie_name = 'womendexinshidai'
    annot_format_output_path = os.path.join(root_dir, movie_name + '_anno1' + '.xlsx')
    annot_dir = os.path.join(root_dir, movie_name)
    dialognames = os.listdir(annot_dir)
    count_valid_dialogs = 0
    movie_instancess = []
    for dialog_idx in range(1, 25):
        dialogname = movie_name + '_' + str(dialog_idx) + '.srt'
        filepath = os.path.join(annot_dir, dialogname)
        if not os.path.exists(filepath):
            print('{} not exist'.format(filepath))
            continue
        count_valid_dialogs += 1
        dialog_instances = read_srt(filepath)
        print('Done {} {} utterances'.format(dialogname, len(dialog_instances)))
        movie_instancess.extend(dialog_instances)
    print('\t there are total {} dialogs {} utterances'.format(count_valid_dialogs, len(movie_instancess)))
    save_annot_format(movie_name, movie_instancess, annot_format_output_path)