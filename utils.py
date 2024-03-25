
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

from datetime import datetime
from enum import Enum
import mpi4py, sys
import macfile

class signals(Enum):
    READ  = 0 
    WRITE = 1 
    CLOSE = 2
    DONE  = 3

class states(Enum):
    SLEEPING       = 0
    READY          = 1
    READING        = 2
    WRITING        = 3
    DONE           = 4
    
def getTimeString():
    return datetime.now().strftime("%H:%M:%S")

def getSimulationParametersFromPath(batchfile):
    parts  = batchfile.replace('.', '_') .split('_')
    subSim = int(parts[-2])
    proj   = int(parts[-3])
    return subSim, proj

def getSimulationParametersFromMacfile(batchfile):
    parts  = batchfile.replace('.', '_') .split('_')
    subSim = int(parts[-2])
    parts  = batchfile.split('/')
    proj   = int(parts[-3])
    return subSim, proj

def getSignal(msg):
    split_msg = msg.split()
    return int(split_msg[0])

def getQueue(msg):
    split_msg = msg.split()
    return int(split_msg[1])

def getMacfile(msg):
    split_msg = msg.split()
    return split_msg[2]

def getOutputImageFiles(mac_file):
    mr = macfile.macfile(mac_file)
    output = []
    outfile = mr.get('/gate/output/ProcessCT/setFileName')
    if outfile is not None:
        outfile = outfile[0]
        output.append(outfile)
    scatter_outfile = mr.get('/gate/output/ProcessCT/setScatterFileName')
    if scatter_outfile is not None:
        scatter_outfile = scatter_outfile[0]
        output.append(scatter_outfile)
    return output

def getOutputRootFile(mac_file):
    mr = macfile.macfile(mac_file)
    output = None
    root_outfile = mr.get('/gate/output/root/setFileName')
    if root_outfile is not None:
        output = root_outfile[0]
    return output


def global_except_hook(exctype, value, traceback):
    import sys
    try:
        import mpi4py.MPI
        sys.stderr.write("\n*****************************************************\n")
        sys.stderr.write("Uncaught exception was detected on rank {}. \n".format(
            mpi4py.MPI.COMM_WORLD.Get_rank()))
        from traceback import print_exception
        print_exception(exctype, value, traceback)
        sys.stderr.write("*****************************************************\n\n\n")
        sys.stderr.write("\n")
        sys.stderr.write("Calling MPI_Abort() to shut down MPI processes...\n")
        sys.stderr.flush()
    finally:
        try:
            import mpi4py.MPI
            mpi4py.MPI.COMM_WORLD.Abort(1)
        except Exception as e:
            sys.stderr.write("*****************************************************\n")
            sys.stderr.write("Sorry, we failed to stop MPI, this process will hang.\n")
            sys.stderr.write("*****************************************************\n")
            sys.stderr.flush()
            raise e