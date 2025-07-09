import torch
import random
import numpy as np

def setup_seed(seed):
    """
    再現性のための乱数シード設定
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
