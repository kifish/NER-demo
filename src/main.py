from config.config import Config
import torch
import numpy as np
import random

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # for debug

def main():
    # fix seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = Config()
    trainer = config.trainer(config)
    trainer.run(config.mode)

if __name__ == '__main__':
    main()
