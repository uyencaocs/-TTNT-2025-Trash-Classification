import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    """
    Thiết lập global seed để đảm bảo tính Reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    
    # Một số setting cho CUDNN để kết quả deterministic nhất có thể
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
