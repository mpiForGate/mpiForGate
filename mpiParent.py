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
import sys, os, shutil, time, stat, pathlib
from pathlib import Path
import socket, time, argparse
from utils import *
import split_job
import numpy as np
from collectorManager import collectorManager, collectState
from split_job import get_processCT_info_from_macfile, get_processed_macfile 


queue_size = 10 # Tells the worker how many projections should have in memory while performing the reading tasks

sys.excepthook = global_except_hook

# remove directory with read-only files
def rm_dir_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IRWXU)
    func(path)


def simulateGate(mac_file):
    output_files = getOutputImageFiles(mac_file)
    import SimpleITK as sitk
    
    subSim, proj = getSimulationParametersFromPath(mac_file)
    for i, output_file in enumerate(output_files):
        if os.path.isabs(output_file):
            output_file_path = output_file
        else:
            jobFolder = os.path.dirname(os.path.abspath(mac_file))
            output_file_path = os.path.dirname(os.path.join(jobFolder, output_file))
            
        nda = np.ones((64,64),dtype=np.float32) * (1+subSim +100*proj +i*10000)
        img = sitk.GetImageFromArray(nda)
        sitk.WriteImage(img, output_file_path)
    time.sleep(0.2)
    return 0
    
# The following function creates output folders IF:
# 1. Image output files are specified in the mac file
# 2. The output files are specified in the mac file 
def createOutputFolders(mac_file):
    # Create the output folders for images if they do not exist
    output_files = getOutputImageFiles(mac_file)
    for output_file in output_files:
        if os.path.isabs(output_file):
            parent_folder = os.path.dirname(output_file)
        else:
            jobFolder = str(Path(mac_file).parents[2])
            parent_folder = os.path.dirname(os.path.join(jobFolder, output_file))
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder, exist_ok=True)
            
    # Create the output root folder if it does not exist
    output_root_filepath = getOutputRootFile(mac_file)
    if output_root_filepath is not None:
        if os.path.isabs(output_root_filepath):
            parent_folder = os.path.dirname(output_root_filepath)
        else:
            jobFolder = os.path.dirname(os.path.abspath(mac_file))
            parent_folder = os.path.dirname(os.path.join(jobFolder, output_root_filepath))
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder, exist_ok=True)
            
# The function will recursively delete all files and folders in the given path. 
# The function will call itself for each subfolder.
def remove_files_and_subfolders(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

def deleteOutputFolders(mac_file):
    output_files = getOutputImageFiles(mac_file)
    for output_file in output_files:
        if os.path.isabs(output_file):
            parent_folder = os.path.dirname(output_file)
        else:
            jobFolder = os.path.dirname(os.path.abspath(mac_file))
            parent_folder = os.path.dirname(os.path.join(jobFolder, output_file))
        #shutil.rmtree(parent_folder, onerror=rm_dir_readonly) if os.path.exists(parent_folder) else None
        remove_files_and_subfolders(parent_folder)


'''
    Temporary macfiles are held in a ".tmp" folder in the job folder.
'''
def main(macfile_path, is_test=False, keep_macfile=False, keep_logs=False):
    nProjs, nSubSims = get_processCT_info_from_macfile(macfile_path)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    jobFolder = os.path.dirname(os.path.abspath(macfile_path))
    jobName = os.path.splitext(os.path.abspath(macfile_path))[0].split('/')[-1]
    logFolder = os.path.join(jobFolder, ".logs", jobName)
    pathlib.Path(logFolder).mkdir(parents=True, exist_ok=True)
    if (rank == 0) and (not os.path.exists(logFolder)):
        pathlib.Path(logFolder).mkdir(parents=True, exist_ok=True)
    
    # At this point, the log folder should exist and be empty
    comm.Barrier()
    start_time = time.time()

    logfile = open(os.path.join(logFolder,'master-' +str(rank) +'.log'), 'w') 
    logfile.write(getTimeString() +": Starting MPI job rank "+str(rank) +" of "+str(size)+". On: "+socket.gethostname()+"\n")
    logfile.flush()

    if rank == 0:
        cstate = collectState(logFolder, nSubSims, nProjs)    

    tmpFolder = os.path.join(jobFolder, ".tmp", jobName)
    if rank == 0:
        # delete temporary folder with macfiles
        shutil.rmtree(tmpFolder, onerror=rm_dir_readonly) if os.path.exists(tmpFolder) else None
        pathlib.Path(tmpFolder).mkdir(parents=True, exist_ok=True)
        # delete output folder with outputs (!!!)
        deleteOutputFolders(macfile_path)
    
    # At this point, the temporary folder should exist and be empty
    comm.Barrier() 
            
    macfiles_assignment = [[None for i in range(nProjs)] for j in range(nSubSims)]
    for p in range(nProjs):
        for s in range(nSubSims):
            # It creates the paths to the mac/batch files which will be created and executed later on
            file_to_execute = os.path.join(tmpFolder, jobName +'_' +str(p) +'_' +str(s) +'.mac')
            macfiles_assignment[s][p] = file_to_execute
        
    # Every rank has its "files_to_execute"
    files_to_execute = []
    cnt=0
    for p in range(nProjs):       
        for s in range(nSubSims):
            assigned_to_rank = cnt%(size-1)+1
            if assigned_to_rank == rank:
                files_to_execute.append(macfiles_assignment[s][p])
            if rank == 0:
                cstate.assign(macfiles_assignment[s][p], assigned_to_rank,s,p)
            cnt+=1

    if rank == 1:
        assert len(files_to_execute)!=0, 'Something went wrong..'

    if rank == 0: # The first rank do not execute any external code, just manage the collector
        logfile.write(getTimeString() +': launching collector manager..' +"\n")
        logfile.flush()
        cm = collectorManager(logFolder, cstate, queue_size, keep_macfile)
        logfile.write(getTimeString() +': joining..' +"\n")
        logfile.flush()
        cm.join()
    else:
        for file_to_execute in files_to_execute:
            subSim, proj = getSimulationParametersFromPath(file_to_execute)
            batch_log_file = os.path.join(logFolder,'job-'+str(proj)+'-'+str(subSim)+'.log')
            split_job.get_processed_macfile(macfile_path, file_to_execute, proj, subSim)
            createOutputFolders(file_to_execute)
            logfile.write(getTimeString() +': rank ' +str(rank) +' has started file '+file_to_execute+'\n')
            logfile.flush()  
            rc = os.system('Gate '+file_to_execute + ' > '+batch_log_file) if not is_test else simulateGate(file_to_execute)
            if rc != 0:
                raise Exception('Process '+str(rank)+' returned a non-zero value. Its arguments were: '+file_to_execute+' and the log file was '+batch_log_file)
            logfile.write(getTimeString() +': rank ' +str(rank) +' has finished file '+file_to_execute+'\n')
            logfile.flush()  
            data = np.array([signals.DONE.value, rank, subSim, proj], dtype=np.int32)
            comm.Send([data,MPI.INT], dest=0)
            os.remove(batch_log_file) if not keep_logs else None
  
    print("Rank "+str(rank)+" of " +str(size) +": Exiting without errors")

    comm.Barrier()
    logfile.close()
    if rank == 0:
        print('Total time in hh:mm:ss: ', time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
    if rank == 0 and not keep_logs:
        shutil.rmtree(logFolder)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--macfile', dest='macfile_path', required=True)
    parser.add_argument('--test', dest='is_test', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--keep_macfiles', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--keep_logs', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    
    main(args.macfile_path, args.is_test, args.keep_macfiles, args.keep_logs)