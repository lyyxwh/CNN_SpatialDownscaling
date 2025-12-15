import os
import logging
from datetime import datetime
logger = logging.getLogger(__name__)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def daily_checkpoint_path(out_dir, prefix, date_obj):
    """按日期保存权重名： prefix_YYYY-MM-DD.pt """
    ensure_dir(out_dir)
    name = f"{prefix}_{date_obj.strftime('%Y-%m-%d')}.pt"
    return os.path.join(out_dir, name)

def timestamped_path(out_dir, prefix):
    ensure_dir(out_dir)
    name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    return os.path.join(out_dir, name)
