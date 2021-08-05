import os
import openpyxl

'''
Step1: cut video dialogs
(opt)format transfer: ffmpeg -i xiaohuanxi01_back.mp4 -vcodec h264 -acodec aac xiaohuanxi01.mp4
cutting videos
transfer to time fromat: hh:mm:ss.xxx
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
    ''' 
    save_root:
    return: sub_video_dir
    '''
    def __init__(self, save_root, **kwargs):
        super().__init__(**kwargs)
        self.save_root = save_root

    def calc_time(self, time):
        # ShotCut: hh:mm:ss.frames
        str_time = str(time)
        hour, minute, second, frames = str_time.split(':')
        return float(minute) * 60 + float(second) + (int(frames) + 1) * 40 * 0.001
    
    def strptime(self, seconds):
        hour, minite, second = 0, 0, 0
        minite = int(seconds % 3600 // 60)
        second = seconds - hour*3600 - minite*60
        return "{}:{}:{:.2f}".format(hour, minite, second)
    
    def __call__(self, video_path, movie_name, index, start, end):
        '''
        start: mm:ss:ms
        '''
        _cmd = 'ffmpeg -ss {} -t {} -i {}  -c:v libx264 -c:a aac -strict experimental -b:a 180k {} -y'
        # _cmd = 'ffmpeg -ss {} -t {}  -i {} -c:v libx264 -c:a aac -strict experimental -b:a 180k {} -y >/dev/null 2>&1 '
        save_path = os.path.join(self.save_root, "{}_{}.mp4".format(movie_name, int(index)))
        if not os.path.exists(save_path):
            print('doing cut video')
            duration = self.calc_time(end) - self.calc_time(start)
            print(duration)
            duration = self.strptime(duration)
            start = self.calc_time(start)
            os.system(_cmd.format(start, duration, video_path, save_path))
        return save_path

if __name__ == '__main__':
    # modify this
    # movie_names = ['shaonianpai', 'bianchengnidenayitian', 'xueselangman', \
    #     'womendexinshidai', 'jinhun', 'xiaozhangfu', 'waixiangren', 'jiayounishizuibangde']
    movie_names = ['zhengyangmenxia']
    for movie_name in movie_names:
        print('Current {}'.format(movie_name))
        # raw_movies_dir = '/data9/memoconv/memoconv_rawmovies'
        # conv_movies_dir = '/data9/memoconv/memoconv_convs/{}'.format(movie_name)
        raw_movies_dir = '/Users/jinming/Desktop/works/memoconv_rawmovies'
        conv_movies_dir = '/Users/jinming/Desktop/works/memoconv_convs/{}'.format(movie_name)
        if not os.path.exists(conv_movies_dir):
            os.mkdir(conv_movies_dir)
        segment_info_path = os.path.join(raw_movies_dir, 'dialog_selection_round2.xlsx')
        cutter =  VideoCutterOneClip(save_root=conv_movies_dir)
        all_instances = read_xls(segment_info_path, movie_name, skip_rows=1)
        for instance in all_instances:
            Index, Episode,	startTime, endTime = instance[:4]
            index = Index.value
            episode = Episode.value
            if index is None or episode is None:
                break
            episode = int(episode)
            start_time = startTime.value
            end_time = endTime.value
            if os.path.exists(os.path.join(raw_movies_dir, movie_name + '{:02d}'.format(episode) + '.mp4')):
                video_path = os.path.join(raw_movies_dir, movie_name + '{:02d}'.format(episode) + '.mp4')
            if os.path.exists(os.path.join(raw_movies_dir, movie_name + '{:02d}'.format(episode) + '.rmvb')):
                video_path = os.path.join(raw_movies_dir, movie_name + '{:02d}'.format(episode) + '.rmvb')
            if os.path.exists(os.path.join(raw_movies_dir, movie_name + '{:02d}'.format(episode) + '.mkv')):
                video_path = os.path.join(raw_movies_dir, movie_name + '{:02d}'.format(episode) + '.mkv')
            print(video_path)
            save_path = cutter(video_path, movie_name, index, start_time, end_time)
            print(save_path)