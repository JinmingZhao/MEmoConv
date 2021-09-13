# MEmoConv
在TalkNet_ASD的环境下
如果是3090的机器上，那么需要安装更新版本的cuda下的torch, 但是cap机器访问不了外网，所以需要手动下载
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0
https://download.pytorch.org/whl/cu111/torch-1.9.0+cu111-cp37-cp37m-linux_x86_64.whl
https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp37-cp37m-linux_x86_64.whl
https://download.pytorch.org/whl/torchaudio-0.9.0-cp37-cp37m-linux_x86_64.whl


## 下载数据, 可以从YouTube上下载 或者 从百度云下载
python37 环境, youtube-dl 工具
如何下载数据，高质量数据且包含更多信息的数据.
存储目录:
/Users/jinming/Desktop/works/memoconv_rawmovies

youtube-dl --ignore-errors -c -f bestvideo+bestaudio --merge-output-format mp4 --output '/Users/jinming/Desktop/works/memoconv_rawmovies/%(title)s' --playlist-items 16 'https://www.youtube.com/watch?v=AAT5NepFkaQ&list=PLwqZU7cJTZQ_uVVudaXaxIZ6O4xaUtARb'

目标是1000个对话，一部电视剧挑选20个对话(20/3=7集左右)，共需要50部电视剧

从百度云下载。 之前的用的搜索的地址，失效了。

## Leo emobert docker 更新ffmpeg -- 之前配置的 2.8.7 版本不行
还是很麻烦的
https://blog.csdn.net/yuxielea/article/details/103146362
配置 x264 和 环境

## ffmpeq 抽取语音信号
cd /Users/jinming/Desktop/works/memoconv_convs/fendou
ffmpeg -i fendou_2.mp4 -vn -f wav -acodec pcm_s16le -ac 1 -ar 16000 fendou_2.wav

## 采用讯飞的语音转写功能，使用双人对话场景，可以生成说话人，可以生成每个词对应的时间戳，完美契合我们的场景。
目前的讯飞听见的字幕生成、讯飞听见都不行。
等待开发者API的返回结果看看咋样。 --垃圾


##  找一个合适的demo
满足哪些要求呢？ 
语音视觉清晰，情感丰富，情感变化比较多 不要求有多情感标注。
最好是有turn内情感变化和turn之间的情感变化。 另外主题要符号通俗易懂且政治正确。

anjia1
xinlianaishidai_6
yipuerzhu_15
doutinghao_13
fumuaiqing_1
fumuaiqing_20