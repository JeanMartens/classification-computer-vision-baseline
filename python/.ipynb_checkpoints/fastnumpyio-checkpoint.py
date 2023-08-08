import numpy as np
import os

class fastnumpyio:
    def load(file):
        file=open(file,"rb")
        header = file.read(128)
        descr = str(header[19:25], 'utf-8').replace("'","").replace(" ","")
        shape = tuple(int(num) for num in str(header[60:120], 'utf-8').replace(', }', '').replace('(', '').replace(')', '').split(','))
        datasize = np.lib.format.descr_to_dtype(descr).itemsize
        for dimension in shape:
            datasize *= dimension
        return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))