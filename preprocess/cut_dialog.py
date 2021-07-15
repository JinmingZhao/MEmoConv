import os
import openpyxl

'''
根据表格里面的片段信息和剧集信息，进行切割，并且编号，每个dialog保存为一个文件。
切割的命令: 
时间格式: hh:mm:ss.xxx
start: (float(hour) * 3600 + float(minite) * 60 + float(second))
end: ((float(hour) * 3600 + float(minite) * 60 + float(second)) - start) -> hh:mm:ss -> 00:00:54
ffmpeg -ss 126 -t 00:00:54 -i '/Users/jinming/Desktop/works/memoconv_rawmovies/fendou02.mp4' -c:v libx264 -c:a aac -strict experimental -b:a 180k '/Users/jinming/Desktop/works/memoconv_convs/fendou02_01.mp4' -y
'''

def read_xls(filepath, sheetname, skip_rows=0):
    '''
    :param filepath:
    :param sheetname: sheetname
    :return: list types of all instances
    '''
    workbook = openpyxl.load_workbook(filepath)
    # workbook.sheetnames
    booksheet = workbook[sheetname]
    rows = booksheet.rows
    all_rows = [r for r in rows]
    return all_rows[skip_rows:]

class VideoCutterOneClip():
    ''' 按句子的timestamp切分视频, 一次切一个视频片段出来
        save_root: 切出来的视频放在哪里, 
        padding: 每句话两端的padding时间
        return: sub_video_dir
    '''
    def __init__(self, save_root, **kwargs):
        super().__init__(**kwargs)
        self.save_root = save_root

    def calc_time(self, time):
        str_time = str(time)
        hour, minute, second, ms = str_time.split(':')
        return float(minute) * 60 + float(second) + float(ms) * 0.01
    
    def strptime(self, seconds):
        hour, minite, second = 0, 0, 0
        minite = int(seconds % 3600 // 60)
        second = seconds - hour*3600 - minite*60
        return f"{hour}:{minite}:{second:.2f}"
    
    def __call__(self, video_path, movie_name, index, start, end):
        '''
        start: mm:ss:ms
        '''
        # _cmd = 'ffmpeg -ss {} -t {} -i {} -c:v libx264 -c:a aac -strict experimental -b:a 180k {} -y'
        _cmd = 'ffmpeg -ss {} -t {} -i {} -c:v libx264 -c:a aac -strict experimental -b:a 180k {} -y >/dev/null 2>&1 '
        save_path = os.path.join(self.save_root, f"{movie_name}_{int(index)}.mp4")
        if not os.path.exists(save_path):
            print('doing cut video')
            duration = self.calc_time(end) - self.calc_time(start)
            duration = self.strptime(duration)
            start = self.calc_time(start)
            os.system(_cmd.format(start, duration, video_path, save_path))
        return save_path

if __name__ == '__main__':    
    # modify this
    movie_name = 'xinlianaishidai'

    raw_movies_dir = '/Users/jinming/Desktop/works/memoconv_rawmovies'
    conv_movies_dir = '/Users/jinming/Desktop/works/memoconv_convs'
    segment_info_path = os.path.join(raw_movies_dir, '多模态对话数据集对话选取.xlsx')
    cutter =  VideoCutterOneClip(save_root=conv_movies_dir)
    all_instances = read_xls(segment_info_path, movie_name, skip_rows=1)
    for instance in all_instances:
        Index, Episode,	startTime, endTime = instance[:4]
        index = Index.value
        episode = Episode.value
        # 空行处理
        if index is None or episode is None:
            break
        episode = int(episode)
        start_time = startTime.value
        end_time = endTime.value
        video_path = os.path.join(raw_movies_dir, movie_name + '{:02d}'.format(episode) + '.mp4')
        save_path = cutter(video_path, movie_name, index, start_time, end_time)
        print(save_path)