调参结果:
1. project层会有帮助，但是该映射层不能加激活和 Dropout
2. Dropout 为 0.1 的时候效果最好
3. G和P的RNN模型维度一般设置为512/256，如果是融合的一般512，单模态的V和L是256，A是512


### IS10_norm
# set  Dlgrnn_G256P256E128H128A128_dp0.1_lr0.0005_A_AIS10_norm-Vsent_avg_denseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_run1
Save model at 36 epoch
Loading best model found on val set: epoch-26
[Val] result WA: 0.4300 UAR 0.3000 F1 0.2900
[Tst] result WA: 0.4000 UAR 0.2600 F1 0.2600
# set  Dlgrnn_G256P256E128H128A128_dp0.1_lr0.0005_A_AIS10_norm-Vsent_avg_denseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_run2
2021-09-01 02:32:48,447 - 2021-09-01-02.21.06 - INFO - Loading best model found on val set: epoch-18
2021-09-01 02:32:49,715 - 2021-09-01-02.21.06 - INFO - [Val] result WA: 0.3800 UAR 0.2700 F1 0.2600
2021-09-01 02:32:52,082 - 2021-09-01-02.21.06 - INFO - [Tst] result WA: 0.3300 UAR 0.2500 F1 0.2300

# set Dlgrnn_G150P150E100H100A100_dp0.1_lr0.0005_A_AIS10_norm-Vsent_avg_denseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_run1
Loading best model found on val set: epoch-21
[Val] result WA: 0.4300 UAR 0.2800 F1 0.2700
[Tst] result WA: 0.4200 UAR 0.2500 F1 0.2500
# set Dlgrnn_G150P150E100H100A100_dp0.1_lr0.0005_A_AIS10_norm-Vsent_avg_denseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_run2
2021-09-01 03:08:20,169 - 2021-09-01-02.54.14 - INFO - Loading best model found on val set: epoch-24
2021-09-01 03:08:21,493 - 2021-09-01-02.54.14 - INFO - [Val] result WA: 0.4100 UAR 0.2800 F1 0.2700
2021-09-01 03:08:23,924 - 2021-09-01-02.54.14 - INFO - [Tst] result WA: 0.3800 UAR 0.2700 F1 0.2500

# set  Dlgrnn_G512P512E128H128A128_dp0.1_lr0.0005_A_AIS10_norm-Vsent_avg_denseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_run1
2021-09-01 02:16:57,541 - 2021-09-01-01.58.27 - INFO - Loading best model found on val set: epoch-26
2021-09-01 02:16:59,020 - 2021-09-01-01.58.27 - INFO - [Val] result WA: 0.4200 UAR 0.2900 F1 0.2700
2021-09-01 02:17:01,802 - 2021-09-01-01.58.27 - INFO - [Tst] result WA: 0.3800 UAR 0.2700 F1 0.2500
# set  Dlgrnn_G512P512E128H128A128_dp0.1_lr0.0005_A_AIS10_norm-Vsent_avg_denseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_run2
2021-09-01 02:28:45,726 - 2021-09-01-02.20.30 - INFO - Loading best model found on val set: epoch-7
2021-09-01 02:28:47,198 - 2021-09-01-02.20.30 - INFO - [Val] result WA: 0.3500 UAR 0.2700 F1 0.2400
2021-09-01 02:28:50,564 - 2021-09-01-02.20.30 - INFO - [Tst] result WA: 0.3600 UAR 0.2500 F1 0.2300

# set Dlgrnn_G256P256E128H128A128_dp0.1_lr0.0005_A_AIS10_norm-Vsent_avg_denseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_inputproj_run1
2021-09-01 02:31:38,442 - 2021-09-01-02.12.52 - INFO - Loading best model found on val set: epoch-30
2021-09-01 02:31:39,796 - 2021-09-01-02.12.52 - INFO - [Val] result WA: 0.3900 UAR 0.3100 F1 0.2900
2021-09-01 02:31:43,060 - 2021-09-01-02.12.52 - INFO - [Tst] result WA: 0.3600 UAR 0.2600 F1 0.2500
# set Dlgrnn_G256P256E128H128A128_dp0.1_lr0.0005_A_AIS10_norm-Vsent_avg_denseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_inputproj_run2
Loading best model found on val set: epoch-53
[Val] result WA: 0.4500 UAR 0.3000 F1 0.3000
[Tst] result WA: 0.4300 UAR 0.2600 F1 0.2600

# set Dlgrnn_G512P512E128H128A128_dp0.1_lr0.0005_A_AIS10_norm-Vsent_avg_denseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_inputproj_run1
2021-09-01 02:26:05,366 - 2021-09-01-02.14.01 - INFO - ============ Evaluation Epoch 22 ============
2021-09-01 02:26:05,367 - 2021-09-01-02.14.01 - INFO - Cur learning rate 0.0005
2021-09-01 02:26:06,961 - 2021-09-01-02.14.01 - INFO - [Validation] Loss: 2.31,	 F1: 28.00,	 WA: 40.00,	 UA: 31.00,
2021-09-01 02:26:09,932 - 2021-09-01-02.14.01 - INFO - [Testing] Loss: 2.37,	 F1: 26.00,	 WA: 38.00,	 UA: 27.00,
# set Dlgrnn_G512P512E128H128A128_dp0.1_lr0.0005_A_AIS10_norm-Vsent_avg_denseface-Lsent_cls_robert_wwm_base_chinese4chmed_class_weight_inputproj_run2
2021-09-01 03:03:26,613 - 2021-09-01-02.44.49 - INFO - ============ Evaluation Epoch 37 ============
2021-09-01 03:03:26,614 - 2021-09-01-02.44.49 - INFO - Cur learning rate 0.00044366197183098594
2021-09-01 03:03:28,071 - 2021-09-01-02.44.49 - INFO - [Validation] Loss: 2.60,	 F1: 31.00,	 WA: 44.00,	 UA: 32.00,
2021-09-01 03:03:30,783 - 2021-09-01-02.44.49 - INFO - [Testing] Loss: 3.00,	 F1: 27.00,	 WA: 43.00,	 UA: 28.00,