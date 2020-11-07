# source ~/anaconda3/bin/activate
~/anaconda3/bin/conda create -n NER-demo-bert python=3.7 # encoding is more easier than in python3.6 :)
conda activate NER-demo-bert
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda info --env
pip install tqdm
pip install transformers
pip install seqeval
pip install sklearn-crfsuite==0.3.6


conda install -c conda-forge jupyterlab


sklearn_crfsuite




# conda install tensorboard
# pip install sklearn
# conda install pandas
# conda install keras
# pip install nltk==3.5
# pip install psutil

# 
# pip install jupyterlab

# jupyter-lab
# jupyter-lab --port 8001
# jupyter-lab --ip 0.0.0.0 --port 6004 --no-browser --allow-root

# https://github.com/kiteco/jupyterlab-kite
# pip install jupyter-kite

# conda install -c conda-forge nodejs
# conda install nodejs

# pip install jupyter-lsp


conda list > conda_env.txt



