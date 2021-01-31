import numpy as np
import copy as copy
import math

import scipy.io as sio
from scipy import signal
from scipy.signal import hilbert
import tensorly as tl
from tensorly.decomposition import tucker
import matplotlib.pyplot as plt

__all__ = ['CompressWaveInfor', 'CompressWavefield', 'CompressWaveList']

class CompressWaveInfor(object):
    def __init__(self, rank, addrank, tensor_shape, relerr=1e-2):
        self.rank = rank
        self.addrank = addrank
        self.tensor_shape = tensor_shape
        self.relerr = relerr

class CompressWavefield(object):
    def __init__(self, compress_wave_infor):
        self.relerr = compress_wave_infor.relerr
        self.rank = compress_wave_infor.rank
        self.rankfloat = self.rank.astype(float)
        self.addrank = compress_wave_infor.addrank
        self.tensor_shape = compress_wave_infor.tensor_shape
        self.compress_error = 0
        self.wave_tensor = np.zeros(self.tensor_shape)
        self.n_record_slices = 0
        self.compress_waves_tensor = None
        self.is_compress = False
        
    def append(self, wave):
        self.wave_tensor[self.n_record_slices,:,:] = wave.reshape([self.tensor_shape[1], self.tensor_shape[2]])
        self.n_record_slices += 1
        if self.n_record_slices == self.tensor_shape[0]:
            tucker_tensor = tucker(self.wave_tensor, rank=self.rank)
            tensor_reconstruct=tl.tucker_to_tensor(tucker_tensor)
            dtensor = self.wave_tensor-tensor_reconstruct
            wave_norm = np.linalg.norm(self.wave_tensor.flatten())
            err_k = np.linalg.norm(dtensor.flatten()) / wave_norm
            while err_k > self.relerr:
                self.rankfloat += self.addrank
                self.rank = self.rankfloat.astype(int)
                tucker_tensor = tucker(self.wave_tensor, rank=self.rank)
                tensor_reconstruct=tl.tucker_to_tensor(tucker_tensor)
                dtensor = self.wave_tensor-tensor_reconstruct
                wave_norm = np.linalg.norm(self.wave_tensor.flatten())
                err_k = np.linalg.norm(dtensor.flatten()) / wave_norm

            if math.isnan(err_k):
                self.compress_waves_tensor = tucker_tensor
                self.compress_error = err_k
            else:     
                self.compress_waves_tensor = tucker_tensor
                self.is_compress = True
                self.wave_tensor = None
                self.compress_error = err_k
            
    def reconstruct_tensor(self):
        if self.is_compress is True:
            return tl.tucker_to_tensor(self.compress_waves_tensor)
        else:
            return self.wave_tensor
            
class CompressWaveList(list):
    def __init__(self, compress_wave_infor):
#         self.slice_size = slice_size
        self.compress_wave_infor = compress_wave_infor
        self.tensor_list = []
        self.n_waves = 0
        self.n_sub_waves = 0
        self.current_tensor = None
        self.current_tensor_id = None
        self.output_shape = [np.prod(compress_wave_infor.tensor_shape[1:3]), 1]
    
    def append(self, wave):
        if self.n_waves == 0:
            CompWave = CompressWavefield(self.compress_wave_infor)
            CompWave.append(wave)
            self.tensor_list.append(CompWave)
        elif self.n_sub_waves == 0:
            CompWave = CompressWavefield(self.compress_wave_infor)
            CompWave.append(wave)
            self.tensor_list.append(CompWave)
        else:
            self.tensor_list[-1].append(wave)
            
        self.n_waves += 1
        self.n_sub_waves += 1
        
        if self.n_sub_waves == self.compress_wave_infor.tensor_shape[0]:
            # print(self.tensor_list[-1].rank)
            self.update_compress_wave_infor()
            self.n_sub_waves = 0
            
        
    
    def update_compress_wave_infor(self):
        self.compress_wave_infor.rank = self.tensor_list[-1].rank
        self.compress_wave_infor.rankfloat = self.tensor_list[-1].rankfloat

        
    def __getitem__(self, indices):
        output = []
        if isinstance(indices, list):
            for index in indices:
                output.append(self.get_one_index(index))
        else:
            output = self.get_one_index(indices)
        return output
    
    def get_one_index(self, index):
        n_slices = self.compress_wave_infor.tensor_shape[0]
        id_sub_tensor, id_sub_wave = self.compute_subindex(index, n_slices)
        
        if id_sub_tensor == self.current_tensor_id:
            return self.current_tensor[id_sub_wave,:,:].reshape(self.output_shape)
        else:
            self.current_tensor_id = id_sub_tensor
            self.current_tensor = self.tensor_list[id_sub_tensor].reconstruct_tensor()
            return self.current_tensor[id_sub_wave,:,:].reshape(self.output_shape)
        
        
    def compute_subindex(self, index, n_slices):
        id_main = index // n_slices
        id_sub = index % n_slices
        
        return id_main, id_sub
    
    def __len__(self):
         return self.n_waves        