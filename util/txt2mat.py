import numpy as np
import os
import scipy.io as scio
path = '../data/mydataset/1'

dict_ = {
    'cylindrical_ch1': [],
    'cylindrical_ch2': [],
    'hook_ch1': [],
    'hook_ch2': [],
    'lateral_ch1': [],
    'lateral_ch2': [],
    'normal_ch1': [],
    'normal_ch2': [],
    'palmar_ch1': [],
    'palmar_ch2': [],
    'spherical_ch1': [],
    'spherical_ch2': [],
    'tip_ch1': [],
    'tip_ch2': []
}
for name in os.listdir(path):
    file_path = os.path.join(path,name)
    f= open(file_path,'r',encoding='utf-8')
    tmp = name.split('.')[0]
    key = tmp.split('_')[0][:-1]+f'_ch{tmp.split("_")[1]}'
    print(name)
    value = f.read().replace('\n\n','\n').split('\n')[:3000]
    if len(value) < 3000:
        print(name)
        continue
    dict_[key] += [value]

result = {}
for key,value in dict_.items():
    value = np.array(value, dtype=np.int32)
    value = np.array(list(map(lambda a:a/np.max(a),value)))
    result[key] =value
scio.savemat('../data/mydataset/1/The_first_group.mat', result)



