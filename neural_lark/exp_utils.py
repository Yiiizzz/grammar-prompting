# neural_lark/exp_utils.py
import os
import re
from typing import Dict, Tuple

def sanitize(name: str) -> str:
    """Make a string safe for use in file/folder names (esp. on Windows)."""
    name = re.sub(r'[<>:"/\\|?*]+', '-', str(name))
    name = name.replace(' ', '_').strip(' .')
    return name or "unknown"

def build_induction_cfg_folder(cfg: Dict) -> str:
    static = "static-on" if cfg.get("use_static_grammar_induction", False) else "static-off"
    closure = "closure-on" if cfg.get("use_closure_for_induction", False) else "closure-off"
    spec = "spec-on" if cfg.get("use_nt_specialization_for_induction", False) else "spec-off"
    return f"indcfg_{static}_{closure}_{spec}"

def build_common_folders(cfg: Dict):
    """Return the list of path components from engine to seed."""
    engine_folder = sanitize(cfg.get("engine", "unknown_engine"))
    dataset_folder = sanitize(cfg.get("dataset", "unknown_dataset"))

    num_shot = cfg.get("num_shot", None)
    num_shot_folder = f"shot_{num_shot}" if num_shot is not None else "shot_none"

    mode_folder = f"mode_{sanitize(cfg.get('prompt_mode', 'std'))}"

    indcfg_folder = build_induction_cfg_folder(cfg)

    seed_folder = f"s{cfg.get('seed', 0)}"

    return [engine_folder, dataset_folder, num_shot_folder, mode_folder, indcfg_folder, seed_folder]

def build_log_dir(cfg: Dict, root: str = "log") -> Tuple[str, str]:
    """
    返回：
      - log_dir: 日志目录路径
      - run_name: 一个可读的实验名（用来给 wandb 和 log 文件命名）
    """
    folders = build_common_folders(cfg)
    log_dir = os.path.join(root, *folders)
    run_name = "-".join(folders)
    return log_dir, run_name

def build_cache_dir(cfg: Dict, root: str) -> str:
    """
    返回缓存目录路径（不含文件名）。
    """
    folders = build_common_folders(cfg)
    cache_dir = os.path.join(root, *folders)
    return cache_dir
