import os
from os.path import join as opj
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def backup_file(src, dst):
    if os.path.exists(opj(src, 'nobackup')):
        return
    if len([k for k in os.listdir(src) if k.endswith('.py') or k.endswith('.sh')]) == 0:
        return
    if not os.path.isdir(dst):
        os.makedirs(dst)
    all_files = os.listdir(src)
    for fname in all_files:
        fname_full = opj(src, fname)
        fname_dst = opj(dst, fname)
        if os.path.isdir(fname_full):
            backup_file(fname_full, fname_dst)
        elif fname.endswith('.py') or fname.endswith('.sh'):
            shutil.copy(fname_full, fname_dst)