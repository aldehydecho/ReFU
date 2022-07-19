import os
from os.path import join as opj
import shutil
from datetime import datetime
import numpy as np
# from plyfile import PlyData
import struct
import global_var

def write_obj(verts, faces, path, color_idx=None):
    faces = faces + 1
    with open(path, 'w') as f:
        for vidx, v in enumerate(verts):
            if color_idx is not None and color_idx[vidx]:
                f.write("v {:.5f} {:.5f} {:.5f} 1 0 0\n".format(v[0], v[1], v[2]))
            else:
                f.write("v {:.5f} {:.5f} {:.5f}\n".format(v[0], v[1], v[2]))
        for fa in faces:
            f.write("f {:d} {:d} {:d}\n".format(fa[0], fa[1], fa[2]))

def read_obj(path):
    with open(path, 'r') as obj:
        datas = obj.read()

    lines = datas.splitlines()
    
    vertices = []
    faces = []
    
    for line in lines:
        elem = line.split()
        if elem:
            if elem[0] == 'v':
                vertices.append([float(elem[1]), float(elem[2]), float(elem[3])])
            elif elem[0] == 'f':
                face = []
                for i in range(1, len(elem)):
                    face.append(int(elem[i].split('/')[0]))
                faces.append(face)
            else:
                pass
            
    vertices = np.array(vertices)
    faces = np.array(faces)-1
            
    return vertices, faces