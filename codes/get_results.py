import numpy as np
import sys
import os
import datetime
import collections
from glob import glob

def write_file(filepath, lines):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

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
            UAR = float(line[line.find('UAR')+3: line.find(' F1 ')].strip())
            F1 = float(line[line.find(' F1 ')+4: line.find(' WF1 ')].strip())
            WF1 = float(line[line.find(' WF1 ')+5:].strip())
            val_log['WA'] = WA
            val_log['UAR'] = UAR
            val_log['F1'] = F1
            val_log['WF1'] = WF1
        elif 'Tst' in line:
            WA = float(line[line.find('WA:')+3: line.find('UAR')].strip())
            UAR = float(line[line.find('UAR')+3: line.find(' F1 ')].strip())
            F1 = float(line[line.find(' F1 ')+4: line.find(' WF1 ')].strip())
            WF1 = float(line[line.find(' WF1 ')+5:].strip())
            test_log['WA'] = WA
            test_log['UAR'] = UAR
            test_log['F1'] = F1
            test_log['WF1'] = WF1
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
            F1 = float(line[line.find(' F1 ')+4: line.find(' WF1 ')].strip())
            WF1 = float(line[line.find(' WF1 ')+5:].strip())
            val_log['WA'] = WA
            val_log['UAR'] = UAR
            val_log['F1'] = F1
            val_log['WF1'] = WF1
        elif 'Tst' in line:
            WA = float(line[line.find('WA:')+3: line.find('UAR')].strip())
            UAR = float(line[line.find('UAR')+3: line.find(' F1 ')].strip())
            F1 = float(line[line.find(' F1 ')+4: line.find(' WF1 ')].strip())
            WF1 = float(line[line.find(' WF1 ')+5:].strip())
            test_log['WA'] = WA
            test_log['UAR'] = UAR
            test_log['F1'] = F1
            test_log['WF1'] = WF1
        else:
            raise ValueError('Can not find correct pattern in {}'.format(line))
    return val_log, test_log

def remove_bad_results(all_val_results, all_tst_results, type_eval):
    if type_eval == 'WF1':
        type_eval_index = 3
    else:
        type_eval_index = 2
    eval_metrics = [v[type_eval_index] for v in all_val_results]
    min_index = np.argmin(eval_metrics)
    all_val_results.pop(min_index)
    all_tst_results.pop(min_index)
    return all_val_results, all_tst_results


if __name__ == '__main__':
    result_dir = '/data9/memoconv/results'
    model_name = 'dialoggcn'
    ft_types = {
        'speech': 'sent_wav2vec_zh2chmed2e5last',
        'visual': 'sent_avg_affectdenseface',
        'text': 'sent_cls_robert_wwm_base_chinese4chmed',
    }
    output_names = [
        'Dlggcn_A_BaseLSTME180WP10WF10dp0.4_lr0.0003__Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_inputproj',
        'Dlggcn_V_BaseLSTME180WP10WF10dp0.4_lr0.0003__Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_inputproj',
        'Dlggcn_L_BaseLSTME180WP10WF10dp0.4_lr0.0003__Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_inputproj',
        'Dlggcn_LA_BaseLSTME180WP10WF10dp0.4_lr0.0003__Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_inputproj',
        'Dlggcn_LV_BaseLSTME180WP10WF10dp0.4_lr0.0003__Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_inputproj',
        'Dlggcn_AV_BaseLSTME180WP10WF10dp0.4_lr0.0003__Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_inputproj',
        'Dlggcn_LAV_BaseLSTME180WP10WF10dp0.4_lr0.0003__Asent_wav2vec_zh2chmed2e5last-Vsent_avg_affectdenseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_inputproj'
    ]
    postfix = 'lr0.0003_dp0.4_dialoggcn'
    result_path = os.path.join(result_dir, 'statistic', '_'.join(ft_types.values()) + '_' + postfix)
    all_lines = []
    for output_name in output_names:
        all_lines.append(output_name + '\n')
        log_dirs = glob(os.path.join(result_dir, model_name, output_name + '_run*', 'log'))
        for type_eval in ['WF1', 'F1']:
            all_lines.append('--------- {} ---------------'.format(type_eval) + '\n' )
            all_val_results = []
            all_tst_results = []
            for log_dir in log_dirs:
                val_log, test_log = get_latest_result(log_dir, type_eval)
                val_results = [val_log['WA'], val_log['UAR'], val_log['F1'], val_log['WF1']]
                test_results = [test_log['WA'], test_log['UAR'], test_log['F1'], test_log['WF1']]
                all_val_results.append(val_results)
                all_tst_results.append(test_results)
            # remove one bad result and average 
            all_val_results, all_tst_results = remove_bad_results(all_val_results, all_tst_results, type_eval)
            all_lines.append('Validation' + '\n')
            avg_all_val_results = np.mean(all_val_results, axis=0)
            all_lines.append('\t'.join( ['{:.4f}'.format(v) for v in avg_all_val_results]) + '\n')
            all_lines.append('\t'.join( ['{:.4f}'.format(v) for v in all_val_results[0]]) + '\n')
            all_lines.append('\t'.join( ['{:.4f}'.format(v) for v in all_val_results[1]]) + '\n')
            all_lines.append('Testing' + '\n')
            avg_all_tst_results = np.mean(all_tst_results, axis=0)
            all_lines.append('\t'.join( ['{:.4f}'.format(v) for v in avg_all_tst_results]) + '\n')
            all_lines.append('\t'.join( ['{:.4f}'.format(v) for v in all_tst_results[0]]) + '\n')
            all_lines.append('\t'.join( ['{:.4f}'.format(v) for v in all_tst_results[1]]) + '\n')
    write_file(result_path, all_lines)