import os
import sys
import math
import random
import time
import logging
import argparse
import torch
import numpy as np

import logging

def setup_logger():
    logger = logging.getLogger("neural_lark")
    logger.setLevel(logging.DEBUG)
    # logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()


def setup_logger_file(logger, log_dir, run_name=None):
    """
    Send info to console, and detailed debug information in logfile.
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # 关键：文件名保持短，避免 Windows 路径过长
    logfile_path = os.path.join(log_dir, f"run_{timestr}.txt")
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(logfile_path, encoding="utf-8")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Logging to {}".format(logfile_path))




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

