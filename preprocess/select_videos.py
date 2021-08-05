import os
from FileOps import read_xls, write_xls



filepath = '/Users/jinming/Desktop/works/memoconv_convs/AnnoExpert/Part4.xlsx'
save_dir = '/Users/jinming/Desktop/works/memoconv_convs/AnnoExpert/Part4'
video_dir = '/Users/jinming/Desktop/works/memoconv_convs'

all_instances = read_xls(filepath, '工作表1', skip_rows=1)
print('all instances {}'.format(len(all_instances)))
all_video_paths = []
for instance in all_instances:
    UtteranceId = instance[0]
    if UtteranceId.value is None:
        continue
    dialog_id = UtteranceId.value.split('_')[0] + '_' + UtteranceId.value.split('_')[1]
    video_path = os.path.join(video_dir, UtteranceId.value.split('_')[0], dialog_id + '.mp4')
    if video_path in all_video_paths:
        continue
    if os.path.exists(video_path):
        all_video_paths.append(video_path)
    else:
        print('No video {}'.format(video_path))
print(all_video_paths)
for video_path in all_video_paths:
    os.system('cp {} {}/'.format(video_path, save_dir))