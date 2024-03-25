#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# This file is part of the nn_3D-anomaly-detection distribution (https://github.com/mpiForGate/mpiForGate).
# Copyright (c) 2022-2023 imec-Vision Lab, University of Antwerp.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.


from mpi4py import MPI
import numpy as np
from enum import Enum
from threading import Lock, Thread
import logging, os, sys, time, macfile, argparse
from utils import *
#import SimpleITK as sitk
from imageio import stack_images

sys.excepthook = global_except_hook
    
class collector:
    
    def __init__(self, logFolder):
        self.log = open(os.path.join(logFolder,'collector.log'), "w") 
        self.write_log('init')
        self.images = {}
        self.file_exts = None
        
    def write_log(self, message):
        self.log.write(getTimeString()+': ' +message+'\n')
        self.log.flush()
        
                
    def process_READ(self, queue_n, macfile):
        output_filepaths = getOutputImageFiles(macfile)
        images = []
        for output_filepath in output_filepaths:
            reader = stack_images(output_filepath)
            images.append(reader.get_stack())
            os.remove(output_filepath)
        stored_image = self.images.get(str(queue_n), None)
        if stored_image is None:
            self.images[str(queue_n)] = images
        else:
            for i,image in enumerate(images):
                self.images[str(queue_n)][i] = image + stored_image[i]
                
    def process_multiREAD(self, queue_list, curr_macfile_list):
        #self.write_log('process_multiREAD')
        files_to_read={}
        for queue_n in queue_list:
            files_to_read[str(queue_n)]=[]
            n_outputs = len(getOutputImageFiles(curr_macfile_list[0]))
            for j in range(n_outputs):
                files_to_read[str(queue_n)].append([])

        for i, queue_n in enumerate(queue_list):
            output_filepaths = getOutputImageFiles(curr_macfile_list[i])
            for j, output_filepath in enumerate(output_filepaths):
                files_to_read[str(queue_n)][j].append(output_filepath)

        for i, (queue_n, files) in enumerate(files_to_read.items()):
            images = []
            for j in range(n_outputs):
                self.write_log('multi reading {}'.format(files[j]))
                reader = stack_images(files[j])
                images.append(reader.get_stack())

                for file_to_del in files[j]:
                    os.remove(file_to_del)
                
                #nda = itk.GetArrayFromImage(images[-1])
                #nda = np.sum(images[-1],axis=0)
                #images[-1] = itk.GetImageFromArray(nda)
                images[-1] = np.sum(images[-1],axis=0)
            
            stored_image = self.images.get(queue_n, None)
            if stored_image is None:
                self.images[queue_n] = images
            else:
                for i,image in enumerate(images):
                    self.images[queue_n][i] = image + stored_image[i]


    def process_WRITE(self, queue_n, macfile):
        #self.write_log('process_WRITE')
        output_filepaths = getOutputImageFiles(macfile)
        for i,  output_filepath in enumerate(output_filepaths):
            ext = output_filepath[output_filepath.rfind('.'):]
            output_cumfilepath = output_filepath[:output_filepath.rfind('_')] + ext
            self.write_log('writing {}'.format(output_cumfilepath))
            #self.writer.SetFileName(output_cumfilepath)
            #self.writer.Execute(self.images.get(str(queue_n))[i])
            writer = stack_images(output_cumfilepath, mode='w')
            writer.write_image(self.images.get(str(queue_n))[i])
        del self.images[str(queue_n)]
        
        output_root_filepath = getOutputRootFile(macfile)
        if output_root_filepath is not None:
            ext = '.root'
            basename = output_root_filepath[:output_root_filepath.rfind('_')]
            output_cumfilepath = basename + ext
            # Launch system command to merge root files (hadd)
            os.system('hadd -f {} {}'.format(output_cumfilepath, basename+'_*'))
            os.system('rm {}_*'.format(basename))
            
        # if not self.keep_macfile:
        #     # All output files have been written, delete the macfiles related to that projection
        #     common_path = macfile[:macfile.rfind('_')]+'_*'
        #     os.system('rm {}'.format(common_path))

'''
def main(logFolder, keep_macfile):
    cl = collector(logFolder, keep_macfile)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_folder', required=True)
    parser.add_argument('--keep_macfile', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    
    main(args.log_folder, args.keep_macfile)
'''