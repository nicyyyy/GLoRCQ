import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import torch


def copy_small_files(src_folder, dst_folder, max_size_mb=10, info = False):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for root, dirs, files in os.walk(src_folder):
        for file_name in files:
            if file_name!= 'model.safetensors.index.json':
                src_file_path = os.path.join(root, file_name)

                file_size = os.path.getsize(src_file_path)
                file_size_mb = file_size / (4* 1024 * 1024)

                if file_size_mb < max_size_mb:

                    relative_path = os.path.relpath(src_file_path, src_folder)
                    dst_file_path = os.path.join(dst_folder, relative_path)

                    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

                    shutil.copy2(src_file_path, dst_file_path)
                    if info:
                        print(f"copy: {src_file_path} -> {dst_file_path}")