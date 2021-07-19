import os
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

def read_srt(filepath):
    all_instances = []
    lines = read_file(filepath)
    dialogId = filepath.split('/')[-1].replace('.srt', '')
    for i in range(0, len(lines), 4):
        uttIdx = lines[i].strip()
        if len(uttIdx) == 0:
            continue
        timestamp = lines[i+1]
        start_time, end_time = timestamp.split(' --> ')
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
    movie_name = 'xiaohuanxi'
    annot_format_output_path = os.path.join(root_dir, movie_name + '_anno1' + '.xlsx')
    movie_dir = os.path.join(root_dir, movie_name)
    dialognames = os.listdir(movie_dir)
    movie_instancess = []
    for dialog_idx in range(1, len(dialognames)+1):
        dialogname = movie_name + '_' + str(dialog_idx) + '.srt'
        filepath = os.path.join(movie_dir, dialogname)
        dialog_instances = read_srt(filepath)
        print('Done {} {} utterances'.format(dialogname, len(dialog_instances)))
        movie_instancess.extend(dialog_instances)
    print('\t there are total {} dialogs {} utterances'.format(len(dialognames), len(movie_instancess)))
    save_annot_format(movie_name, movie_instancess, annot_format_output_path)