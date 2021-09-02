## 配置环境
pip install torch_geometric==1.6.3
pip install torch_sparse (很慢)
pip install torch_scatter
在.bashrc中添加环境变量
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

codes/dialoggcn/model.py#795 新版本没有 edge-norm，去掉了