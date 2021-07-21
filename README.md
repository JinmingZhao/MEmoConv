# MEmoConv

## 下载数据, 可以从YouTube上下载，因为有字幕文件 --Going
python37 环境, youtube-dl 工具

如何下载数据，高质量数据且包含更多信息的数据.
存储目录:
/Users/jinming/Desktop/works/memoconv_rawmovies

youtube-dl --ignore-errors -c -f bestvideo+bestaudio --merge-output-format mp4 --output '/Users/jinming/Desktop/works/memoconv_rawmovies/%(title)s' --playlist-items 16 'https://www.youtube.com/watch?v=AAT5NepFkaQ&list=PLwqZU7cJTZQ_uVVudaXaxIZ6O4xaUtARb'

目标是1000个对话，一部电视剧挑选20个对话(20/3=7集左右)，共需要50部电视剧。

1. 下载电视剧 奋斗 看了几集，有很多质量比较高的对话
https://www.youtube.com/watch?v=AAT5NepFkaQ&list=PLwqZU7cJTZQ_uVVudaXaxIZ6O4xaUtARb

2. 都挺好 --
https://www.youtube.com/watch?v=YtzqsA-a8MM&list=PLQqbdnAgoRmYhfPJgYB9YQxDsNQ-ErQBd


## 更新ffmpeg -- 之前配置的 2.8.7 版本不行
还是很麻烦的
https://blog.csdn.net/yuxielea/article/details/103146362
source deactivate vlbert
