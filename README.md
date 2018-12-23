# NER-demo

### CPU 
```
pip install git+https://www.github.com/keras-team/keras-contrib.git
pip install -r requirements.txt
```

### GPU
To do   
```
环境配置
https://zhuanlan.zhihu.com/p/37924625
https://medium.com/@WhoYoung99/2018%E6%9C%80%E6%96%B0win10%E5%AE%89%E8%A3%9Dtensorflow-gpu-keras-8b3f8652509a
https://www.cnblogs.com/luruiyuan/p/6660142.html
https://www.tensorflow.org/install/gpu

tensorflow-gpu版本号要与cuda对应。
https://blog.csdn.net/wangkun1340378/article/details/82379525?utm_source=blogxgwz2

cat /usr/local/cuda/version.txt
CUDA Version 9.1.85
pip install tensorflow-gpu==1.3.0

然而这个版本适配的keras 与 keras-contrib（最新版本不兼容），得安装兼容版本的 keras-contrib才能用CRF layer。
```
