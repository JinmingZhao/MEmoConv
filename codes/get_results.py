import numpy as np
import sys
import os
import datetime
import collections
from glob import glob

def get_latest_result(path):
    logs = os.listdir(path)
    logs = list(filter(lambda x: True if x.startswith('none_') else False, logs))
    logs = sorted(logs,  key=lambda x: datetime.datetime.strptime(x, 'none_%Y-%m-%d-%H.%M.%S.log'))
    log = logs[-1]
    val_log, test_log = analysis_log(os.path.join(path, log))
    return val_log, test_log

def analysis_log(log):
    f = open(log)
    lines = f.readlines()
    f.close()
    res_lines = list(filter(lambda x: True if 'result WA' in x else False, lines))
    assert len(res_lines) == 2
    val_log = {}
    test_log = {}
    for line in res_lines:
        if 'Val' in line:
            WA = float(line[line.find('WA:')+3: line.find('UAR')].strip())
            UAR = float(line[line.find('UAR')+3: line.find('F1')].strip())
            F1 = float(line[line.find('F1')+2:].strip())
            val_log['WA'] = WA
            val_log['UAR'] = UAR
            val_log['F1'] = F1
        elif 'Tst' in line:
            WA = float(line[line.find('WA:')+3: line.find('UAR')].strip())
            UAR = float(line[line.find('UAR')+3: line.find('F1')].strip())
            F1 = float(line[line.find('F1')+2:].strip())
            test_log['WA'] = WA
            test_log['UAR'] = UAR
            test_log['F1'] = F1
        else:
            raise ValueError('Can not find correct pattern in {}'.format(line))
    return val_log, test_log

if __name__ == '__main__':
    output_name = 'Dlggcn_AVL_BaseLSTME180WP10WF10dp0.4_lr0.0003__AIS10_norm-Vsent_avg_denseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_inputproj'
    result_dir = '/data9/memoconv/results'
    model_name = 'dialoggcn'
    log_dirs = glob(os.path.join(result_dir, model_name, output_name + '_run*', 'log'))
    print(log_dirs)
    print(output_name)

    all_val_results = []
    all_tst_results = []
    for log_dir in log_dirs:
        val_log, test_log = get_latest_result(log_dir)
        val_results = [val_log['WA'], val_log['UAR'], val_log['F1']]
        test_results = [test_log['WA'], test_log['UAR'], test_log['F1']]
        all_val_results.append(val_results)
        all_tst_results.append(test_results)
    print('Validation')
    avg_all_val_results = np.mean(all_val_results, axis=0)
    print('\t'.join( ['{:.4f}'.format(v) for v in avg_all_val_results]))
    print('\t'.join( ['{:.4f}'.format(v) for v in all_val_results[0]]))
    print('\t'.join( ['{:.4f}'.format(v) for v in all_val_results[1]]))
    print('Testing')
    avg_all_tst_results = np.mean(all_tst_results, axis=0)
    print('\t'.join( ['{:.4f}'.format(v) for v in avg_all_tst_results]))
    print('\t'.join( ['{:.4f}'.format(v) for v in all_tst_results[0]]))
    print('\t'.join( ['{:.4f}'.format(v) for v in all_tst_results[1]]))