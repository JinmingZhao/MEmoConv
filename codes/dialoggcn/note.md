## 配置环境 --失败
创建 dialoggcn env 下面配置
conda install -y -c pytorch cudatoolkit=10.1 "pytorch=1.4.0=py3.6_cuda10.1.243_cudnn7.6.3_0"   torchvision=0.5.0=py36_cu101
pip install torch_geometric==1.6.3
pip install torch_sparse (很慢)
pip install torch_scatter
在.bashrc中添加环境变量
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
codes/dialoggcn/model.py#795 新版本没有 edge-norm，去掉了

Cap:
pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
https://download.pytorch.org/whl/cu110/torch-1.7.0%2Bcu110-cp37-cp37m-linux_x86_64.whl
https://download.pytorch.org/whl/cu110/torchvision-0.8.0-cp37-cp37m-linux_x86_64.whl
https://download.pytorch.org/whl/torchaudio-0.7.0-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse  torch_scatter -f   https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
配置就版本的太难了，都是最新版本很好配置，但是没法使用。

采用胡景文的docker.