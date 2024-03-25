#!/usr/bin/env python3
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



import os, sys, pathlib
import itk
import itk.support.types as itkt
import numpy as np
from utils import *

__all__ = [ 'stack_images']
sys.excepthook = global_except_hook

# This function will get the last number in a filename and return it as an integer. The filename may have a format like this: "image_0001.tif" or "image_1.tif" or "image_1-1.tif"
def get_number_from_filename(filename):
    number = os.path.splitext(filename)[0].split('_')[-1].split('-')[-1]
    return int(number)

itkEnum2itkType = {
    itk.CommonEnums.IOComponent_FLOAT: itk.F,
    itk.CommonEnums.IOComponent_LONG: itk.SL,
    itk.CommonEnums.IOComponent_ULONG: itk.UL,
    itk.CommonEnums.IOComponent_SHORT: itk.SS,
    itk.CommonEnums.IOComponent_USHORT: itk.US,
    itk.CommonEnums.IOComponent_CHAR: itk.SC,
    itk.CommonEnums.IOComponent_UCHAR: itk.UC
}

npType2itkType = {
    'float32': itk.F,
    'float64': itk.D,
}

class stack_images():
    def __init__(self, path, mode='r'):
        self.mode = mode
        self.path = path
        
        if self.mode in ['r','rw']:
            self.set_imageio()
            self.n_images = self.read_n_images()
            self.nx, self.ny = self.get_image_shape()
        else:
            assert isinstance(self.path, str) and '*' not in self.path, "The path must be a string without '*' when the stack is in write mode."
            
    def read(self, files):
        pixelType       = self.imageIO.GetPixelType()
        componentType   = self.imageIO.GetComponentType()
        
        out_dims = 2 if self.n_images==1 else 3
        imageType = itk.Image[itkEnum2itkType[componentType], out_dims]
        reader = itk.ImageFileReader[imageType].New() if self.n_images==1 else itk.ImageSeriesReader[imageType].New()
        reader.SetImageIO(self.imageIO)
        reader.SetFileName(files[0]) if self.n_images==1 else reader.SetFileNames(files)
        reader.Update()
        image = reader.GetOutput()
        return itk.array_from_image(image)
        
    def write(self, image):
        componentType = str(image.dtype)
        imageType = itk.Image[npType2itkType[componentType], 2]
        writer = itk.ImageFileWriter[imageType].New() 
        writer.SetFileName(self.path)
        writer.SetInput(itk.image_from_array(image))
        writer.SetImageIO(itk.ImageIOFactory.CreateImageIO(self.path, itk.CommonEnums.IOFileMode_WriteMode))
        writer.Update()
    
    def set_imageio(self):
        mode = itk.CommonEnums.IOFileMode_ReadMode if self.mode in ['r','rw'] else itk.CommonEnums.IOFileMode_WriteMode
        input_file = self.get_files()[0]
        imageIO = itk.ImageIOFactory.CreateImageIO(input_file, mode)
        # try
        try:
            imageIO.ReadImageInformation()
        except:
            raise Exception("An error. The file {} cannot be read.".format(input_file))
        self.imageIO = imageIO
        
    def get_image(self, i=0):
        if self.mode in ['r','rw']:
            img = self.read(self.get_files()[i])
            return img
        else:
            raise Exception("The image cannot be read because the stack has not been declared in read mode.")
        
    def get_stack(self):
        if self.mode in ['r','rw']:
            img = self.read(self.get_files())
            img = np.expand_dims(img, 0) if self.n_images==1 else img
            return img
        else:
            raise Exception("The image cannot be read because the stack has not been declared in read mode.")
        
    def write_image(self, img):
        if self.mode in ['w','rw']:
            self.write(img)
        else:
            raise Exception("The image cannot be saved because the stack has not been declared in write mode.")
    
    def make_filepath(self,i):
        formattable_path = self.path.replace('*','{}') if  '*' in self.path else self.path
        return formattable_path.format(i)
            
    def get_files(self):
        if self.mode in ['r','rw']:
            files = [self.path] if isinstance(self.path, str) else self.path
            for i, file in enumerate(files):
                if '*' in file:
                    files[i] = sorted(list(pathlib.Path(file).parent.glob(pathlib.Path(file).name)), key=get_number_from_filename)
            return files
        else:
            raise Exception("The image cannot be read because the stack has not been declared in read mode.")

    def read_n_images(self):
        if self.mode in ['r','rw']:
            return len(list(self.get_files()))
        else:
            return None
        
    def get_image_shape(self):
        if self.mode in ['r','rw']:
            return [self.imageIO.GetDimensions(i) for i in range(self.imageIO.GetNumberOfDimensions())]
        else:
            return None, None
    
    
def testing():
    read_image_path = '/user/antwerpen/206/vsc20649/scratch/008/out/0/prova.tiff'
    read_image_paths = [read_image_path, '/user/antwerpen/206/vsc20649/scratch/008/out/0/prova.tiff']
    write_image_path = '/user/antwerpen/206/vsc20649/data/Source/mpiForGate/tests/tmp_out'
    
    stack1 = stack_images(read_image_path)
    image = stack1.get_image()
    image = stack1.get_stack()
    
    stack2 = stack_images(read_image_paths)
    images = stack2.get_stack()
    
    w_stack1 = stack_images(write_image_path+'/image.tiff', mode='w')
    image = np.sum(image, axis=0)
    w_stack1.write_image(image)
    
    w_stack2 = stack_images(write_image_path+'/stack.tiff', mode='w')
    images = np.sum(images, axis=0)
    w_stack2.write_image(images)
    
    
if __name__ == '__main__':
    testing()