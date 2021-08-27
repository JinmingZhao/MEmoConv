## A-ComparE
2021-08-26 19:40:00,414 - 2021-08-26-17.52.17 - INFO - [Val] result WA: 0.3832 UAR 0.2392 F1 0.2395
2021-08-26 19:40:03,899 - 2021-08-26-17.52.17 - INFO - [Tst] result WA: 0.3790 UAR 0.2090 F1 0.2112

## A-Wav2vec-zh > ComparE
2021-08-26 19:42:42,351 - 2021-08-26-17.54.29 - INFO - [Val] result WA: 0.3988 UAR 0.2446 F1 0.2476
2021-08-26 19:42:44,659 - 2021-08-26-17.54.29 - INFO - [Tst] result WA: 0.4006 UAR 0.2336 F1 0.2363

# A-Wav2vec-zh（512）要比256维度的要好
2021-08-27 03:12:19,937 - 2021-08-27-03.06.50 - INFO - Loading best model found on val set: epoch-9
2021-08-27 03:12:21,399 - 2021-08-27-03.06.50 - INFO - [Val] result WA: 0.4424 UAR 0.2605 F1 0.2647
2021-08-27 03:12:23,569 - 2021-08-27-03.06.50 - INFO - [Tst] result WA: 0.4275 UAR 0.2432 F1 0.2469

## L-Bert-base-Chinese
Loading best model found on val set: epoch-11
[Val] result WA: 0.4495 UAR 0.2821 F1 0.2801
[Tst] result WA: 0.4661 UAR 0.2555 F1 0.2650

## V-DenseFace
2021-08-25 19:26:37,008 - 2021-08-25-19.17.54 - INFO - Loading best model found on val set: epoch-17
2021-08-25 19:26:38,583 - 2021-08-25-19.17.54 - INFO - [Val] result WA: 0.4573 UAR 0.2671 F1 0.2640
2021-08-25 19:26:40,717 - 2021-08-25-19.17.54 - INFO - [Tst] result WA: 0.4268 UAR 0.2503 F1 0.2368

## V-DenseFace + L-Bert-base-Chinese（目前结果是最好的，加入语音信号没啥用）
2021-08-26 19:39:13,690 - 2021-08-26-17.53.04 - INFO - Loading best model found on val set: epoch-13
2021-08-26 19:39:25,922 - 2021-08-26-17.53.04 - INFO - [Val] result WA: 0.5101 UAR 0.3762 F1 0.3723
2021-08-26 19:39:43,091 - 2021-08-26-17.53.04 - INFO - [Tst] result WA: 0.4892 UAR 0.3459 F1 0.3452


## A-wav2vec-zh + L-Bert-base-Chinese
# AL_lr0.001_dp0.5_bnFalse_Awav2vec_zh256_Vdenseface256_Lbert_base_chinese256_F512,256_run1_self
2021-08-27 01:55:43,238 - 2021-08-27-01.46.43 - INFO - Loading best model found on val set: epoch-8
2021-08-27 01:55:46,156 - 2021-08-27-01.46.43 - INFO - [Val] result WA: 0.4952 UAR 0.3101 F1 0.3115
2021-08-27 01:55:50,469 - 2021-08-27-01.46.43 - INFO - [Tst] result WA: 0.4742 UAR 0.2773 F1 0.2734
# AL_lr0.001_dp0.5_bnFalse_Awav2vec_zh512_Vdenseface256_Lbert_base_chinese256_F512,256_run1_self
2021-08-27 02:39:53,703 - 2021-08-27-02.28.51 - INFO - Loading best model found on val set: epoch-8
2021-08-27 02:39:55,755 - 2021-08-27-02.28.51 - INFO - [Val] result WA: 0.4665 UAR 0.3288 F1 0.3220
2021-08-27 02:39:58,758 - 2021-08-27-02.28.51 - INFO - [Tst] result WA: 0.4563 UAR 0.3011 F1 0.3024


 ## A-ComparE + V-Denseface
 2021-08-26 19:30:55,396 - 2021-08-26-17.53.12 - INFO - Loading best model found on val set: epoch-15
2021-08-26 19:31:26,475 - 2021-08-26-17.53.12 - INFO - [Val] result WA: 0.4747 UAR 0.3090 F1 0.3056
2021-08-26 19:32:04,459 - 2021-08-26-17.53.12 - INFO - [Tst] result WA: 0.4385 UAR 0.2717 F1 0.2623

 ## A-wav2vec-zh + V-Denseface
2021-08-27 01:54:31,011 - 2021-08-27-01.47.10 - INFO - Loading best model found on val set: epoch-11
2021-08-27 01:54:32,971 - 2021-08-27-01.47.10 - INFO - [Val] result WA: 0.4467 UAR 0.3059 F1 0.3226
2021-08-27 01:54:36,122 - 2021-08-27-01.47.10 - INFO - [Tst] result WA: 0.4142 UAR 0.2666 F1 0.2727
# AV_lr0.001_dp0.5_bnFalse_Awav2vec_zh512_Vdenseface256_Lbert_base_chinese256_F512,256_run1_self
2021-08-27 02:39:40,424 - 2021-08-27-02.29.27 - INFO - Loading best model found on val set: epoch-11
2021-08-27 02:39:43,841 - 2021-08-27-02.29.27 - INFO - [Val] result WA: 0.4481 UAR 0.3081 F1 0.3137
2021-08-27 02:39:48,918 - 2021-08-27-02.29.27 - INFO - [Tst] result WA: 0.4297 UAR 0.2825 F1 0.2818

 ## A-ComparE + V-Denseface + L-Bert-base-Chinese
 2021-08-26 19:42:45,293 - 2021-08-26-17.52.46 - INFO - Loading best model found on val set: epoch-12
2021-08-26 19:42:47,149 - 2021-08-26-17.52.46 - INFO - [Val] result WA: 0.5310 UAR 0.3756 F1 0.3842
2021-08-26 19:42:49,738 - 2021-08-26-17.52.46 - INFO - [Tst] result WA: 0.4942 UAR 0.3255 F1 0.3297

## A-wav2vec-zh + V-Denseface + L-Bert-base-Chinese
2021-08-27 01:54:47,458 - 2021-08-27-01.47.43 - INFO - Loading best model found on val set: epoch-7
2021-08-27 01:54:49,489 - 2021-08-27-01.47.43 - INFO - [Val] result WA: 0.5463 UAR 0.3616 F1 0.3792
2021-08-27 01:54:52,590 - 2021-08-27-01.47.43 - INFO - [Tst] result WA: 0.5165 UAR 0.3148 F1 0.3203
# AVL_lr0.001_dp0.5_bnFalse_Awav2vec_zh512_Vdenseface256_Lbert_base_chinese256_F512,256_run1_self
2021-08-27 02:40:12,251 - 2021-08-27-02.29.53 - INFO - Loading best model found on val set: epoch-8
2021-08-27 02:40:14,791 - 2021-08-27-02.29.53 - INFO - [Val] result WA: 0.5232 UAR 0.3786 F1 0.3711
2021-08-27 02:40:18,855 - 2021-08-27-02.29.53 - INFO - [Tst] result WA: 0.4620 UAR 0.3427 F1 0.3279
