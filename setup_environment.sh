# source ~/anaconda3/bin/activate
~/anaconda3/bin/conda create -n NER-demo-bert python=3.7 # encoding is more easier than in python3.6 :)
export PYTHONNOUSERSITE=True
conda activate NER-demo-bert
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda info --env
pip install tqdm
pip install transformers==3.3.1
pip install seqeval
pip install sklearn-crfsuite==0.3.6
conda install tensorboard -y
pip install pytorch-crf



conda install -c conda-forge jupyterlab
# https://github.com/kiteco/jupyterlab-kite
# pip install jupyter-kite
# conda install -c conda-forge nodejs
# conda install nodejs
# pip install jupyter-lsp

conda list > conda_env.txt