'''
统计所有在论文中需要汇报的数据
'''

import os
import numpy as np
from FileOps import read_xls, read_file

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
    pass
    