import numpy as np
import sys
import os
import datetime
import collections
from glob import glob

def get_latest_result(path, type_eval):
    logs = os.listdir(path)
    logs = list(filter(lambda x: True if x.startswith('none_') else False, logs))
    logs = sorted(logs,  key=lambda x: datetime.datetime.strptime(x, 'none_%Y-%m-%d-%H.%M.%S.log'))
    log = logs[-1]
    if type_eval == 'WF1':
        val_log, test_log = analysis_wf1_log(os.path.join(path, log), type_eval)
    else:
        val_log, test_log = analysis_f1_log(os.path.join(path, log), type_eval)
    return val_log, test_log

def analysis_wf1_log(log, type_eval='WF1'):
    f = open(log)
    lines = f.readlines()
    f.close()
    res_lines = list(filter(lambda x: True if type_eval+'-result' in x else False, lines))
    assert len(res_lines) == 2
    val_log = {}
    test_log = {}
    for line in res_lines:
        if 'Val' in line:
            WA = float(line[line.find('WA:')+3: line.find('UAR')].strip())
            UAR = float(line[line.find('UAR')+3: line.find(' F1')].strip())
            F1 = float(line[line.find(' F1')+3: line.find('WF1 ')].strip())
            WF1 = float(line[line.find('WF1 ')+4:].strip())
            val_log['WA'] = WA
            val_log['UAR'] = UAR
            val_log['F1'] = F1
            val_log['WF1'] = WF1
        elif 'Tst' in line:
            WA = float(line[line.find('WA:')+3: line.find('UAR')].strip())
            UAR = float(line[line.find('UAR')+3: line.find(' F1')].strip())
            F1 = float(line[line.find(' F1')+3: line.find('WF1 ')].strip())
            WF1 = float(line[line.find('WF1 ')+4:].strip())
            test_log['WA'] = WA
            test_log['UAR'] = UAR
            test_log['F1'] = F1
            test_log['WF1'] = F1
        else:
            raise ValueError('Can not find correct pattern in {}'.format(line))
    return val_log, test_log

def analysis_f1_log(log, type_eval='F1'):
    f = open(log)
    lines = f.readlines()
    f.close()
    res_lines = list(filter(lambda x: True if ' F1-result' in x else False, lines))
    assert len(res_lines) == 2
    val_log = {}
    test_log = {}
    for line in res_lines:
        if 'Val' in line:
            WA = float(line[line.find('WA:')+3: line.find('UAR')].strip())
            UAR = float(line[line.find('UAR')+3: line.find(' F1 ')].strip())
            F1 = float(line[line.find(' F1 ')+4: line.find('WF1')].strip())
            WF1 = float(line[line.find('WF1')+4:].strip())
            val_log['WA'] = WA
            val_log['UAR'] = UAR
            val_log['F1'] = F1
            val_log['WF1'] = WF1
        elif 'Tst' in line:
            WA = float(line[line.find('WA:')+3: line.find('UAR')].strip())
            UAR = float(line[line.find('UAR')+3: line.find(' F1 ')].strip())
            F1 = float(line[line.find(' F1 ')+4: line.find('WF1')].strip())
            WF1 = float(line[line.find('WF1')+4:].strip())
            test_log['WA'] = WA
            test_log['UAR'] = UAR
            test_log['F1'] = F1
            test_log['WF1'] = F1
        else:
            raise ValueError('Can not find correct pattern in {}'.format(line))
    return val_log, test_log

if __name__ == '__main__':
    output_name = 'V_lr0.0002_dp0.5_bnFalse_Asent_avg_wav2vec_zh256_Vsent_avg_affectdenseface256_Lsent_avg_robert_base_wwm_chinese256_F256,128'
    result_dir = '/data9/memoconv/results'
    model_name = 'utt_baseline/early_fusion_multi'
    log_dirs = glob(os.path.join(result_dir, model_name, output_name + '_run*', 'log'))
    print(log_dirs)
    print(output_name)

    for type_eval in ['WF1', 'F1']:
        print('--------- {} ---------------'.format(type_eval))
        all_val_results = []
        all_tst_results = []
        for log_dir in log_dirs:
            val_log, test_log = get_latest_result(log_dir, type_eval)
            val_results = [val_log['WA'], val_log['UAR'], val_log['F1'], val_log['WF1']]
            test_results = [test_log['WA'], test_log['UAR'], test_log['F1'], test_log['WF1']]
            all_val_results.append(val_results)
            all_tst_results.append(test_results)
        print('Validation')
        avg_all_val_results = np.mean(all_val_results, axis=0)
        print('\t'.join( ['{:.4f}'.format(v) for v in avg_all_val_results]))
        print('\t'.join( ['{:.4f}'.format(v) for v in all_val_results[0]]))
        print('\t'.join( ['{:.4f}'.format(v) for v in all_val_results[1]]))
        print('\t'.join( ['{:.4f}'.format(v) for v in all_val_results[2]]))
        print('Testing')
        avg_all_tst_results = np.mean(all_tst_results, axis=0)
        print('\t'.join( ['{:.4f}'.format(v) for v in avg_all_tst_results]))
        print('\t'.join( ['{:.4f}'.format(v) for v in all_tst_results[0]]))
        print('\t'.join( ['{:.4f}'.format(v) for v in all_tst_results[1]]))
        print('\t'.join( ['{:.4f}'.format(v) for v in all_tst_results[2]]))