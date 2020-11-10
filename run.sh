export PYTHONNOUSERSITE=True
conda activate NER-demo-bert
CUDA_VISIBLE_DEVICES=4 python src/main.py


jupyter-lab
jupyter-lab --port 8001
jupyter-lab --ip 0.0.0.0 --port 6014 --no-browser --allow-root


# 转发
from your local machine, run
ssh -N -f -L localhost:6001:localhost:6001 your_ip -p port


# tensorboard
tensorboard --logdir records/simple/run1/log --port 6006


from your local machine, run
ssh -N -f -L localhost:16006:localhost:6006 your_ip -p port


http://localhost:16006/


lsof -ti:6006 | xargs kill -9




