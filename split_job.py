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


from macfile import macfile as mf
import copy, numpy as np, os, pathlib, sys
from scipy.spatial.transform import Rotation as R
from enum import Enum
from utils import *


sys.excepthook = global_except_hook

def insert_number_before_point(stringa, number):
    file_start = stringa.rfind('/')
    file_start = 0 if file_start == -1 else file_start
    idx = stringa[file_start:].rfind('.')
    if idx == -1:
        idx = len(stringa)
    out = stringa[:idx+file_start] +'_' + str(number) + stringa[idx+file_start:]
    return out

def insert_number_as_parent(stringa, number):
    idx = stringa.rfind('/')
    if idx == -1:
        idx = 0
    if idx != 0:
        out = stringa[:idx] +'/' + str(number) + stringa[idx:]
    else:
        out = str(number) + '/' + stringa
    return out

def convert_letter_to_rot_axis(letter):
    if letter == 'x':
        out = np.array([1.,0.,0.])
    elif letter == 'y':
        out = np.array([0.,1.,0.])
    elif letter == 'z':
        out = np.array([0.,0.,1.])
    else:
        raise ValueError('Wrong letter for rotation axis (must be x, y or z)')
    return out


# Define enum for the different types of jobs (CT scan, energy swipe, etc.)
class ScanType(Enum):
    Radiograph = 1
    CT = 2
    EnergySwipe = 3
    ConveyBelt = 4

class proj_par_manager:
    def __init__(self, mac_dict):
        super().__init__()
        self._scan_type = ScanType.Radiograph
        self.load_useful_values(mac_dict)
        
    @property
    def scan_type(self):
        return self._scan_type

    @scan_type.setter
    def scan_type(self, value):
        if isinstance(value, ScanType):
            self._scan_type = value
        else:
            raise TypeError('Scan type must be a ScanType enum')

    def detect_scan_type(self, mac_dict):
        if mac_dict.get('/mpiForGate/simulateRotation') is not None:
            self.scan_type = ScanType.CT
        elif mac_dict.get('/mpiForGate/energySwipe') is not None:
            self.scan_type = ScanType.EnergySwipe
    
    def load_useful_values_ct_scan(self, mac_dict):
        # This is a list of 4 elements: start_angle, end_angle, n_proj, rot_axis (optional, 'y' by default)
        simulate_rot = mac_dict.pop('/mpiForGate/simulateRotation', None)
        start_angle, end_angle, self.n_projs = float(simulate_rot[0]), float(simulate_rot[1]), int(simulate_rot[2])
        self.rot_axis_lett = str(simulate_rot[3]) if len(simulate_rot) == 4 else 'z'
        self.rot_axis = convert_letter_to_rot_axis(self.rot_axis_lett)
        COR_axis = mac_dict.pop('/mpiForGate/CORaxis', None)
        self.COR_axis = np.array( COR_axis[0:3],dtype=np.float32) if COR_axis is not None else np.array([0.,0.,0.],dtype=np.float32)
        
        self.projs  = np.linspace(start_angle, end_angle, self.n_projs, endpoint=True) ## careful ! this and 2 following lines are related to a X-ray source rotation
        
        source_name = mac_dict.get('/gate/source/addSource')[0]
        self.src_pos = np.array(mac_dict.get('/gate/source/' +source_name +'/gps/pos/centre')[0:3],dtype=np.float32) #units will not change
        
        sensitive_det = mac_dict.find_cmd('attachCrystalSD') # should account for the perfect one as well
        if sensitive_det == []:
            sensitive_det = mac_dict.find_cmd('attachPerfectCrystalSD')
        module_name = sensitive_det[0].split('/')[2]
        cmds, _ = mac_dict.find_value(module_name)
        for cmd in cmds:
            if '/daughters/name' in cmd:
                scanner_name = cmd.split("/")[2]
        self.det_pos = np.array(mac_dict.get('/gate/' +scanner_name +'/placement/setTranslation')[0:3],dtype=np.float32) #units will not change
        init_rot_axis = mac_dict.get('/gate/' +scanner_name +'/placement/setRotationAxis', None)
        if init_rot_axis is not None:
            self.init_rot_axis  = np.array(init_rot_axis[0:3],dtype=np.float32)
            rot_angle = mac_dict.get('/gate/' +scanner_name +'/placement/setRotationAngle')
            rot_angle = float(rot_angle[0]) if len(rot_angle)==1 else (float(rot_angle[0])*180/np.pi if rot_angle[1]!='deg' else float(rot_angle[0]))
            self.init_rot_angle = np.array(rot_angle,dtype=np.float32) # I want it in degrees
        else:
            self.init_rot_axis  = np.array([0.,0.,1.],dtype=np.float32)
            self.init_rot_angle = np.array(0.,dtype=np.float32)
    
    def load_useful_values_energy_swipe(self, mac_dict):
        energies_string = mac_dict.pop('/mpiForGate/energySwipe', None) # this is a list of 4 elements: start_energy, end_energy, energy metric, step (default 1)
        start_energy = float(energies_string[0])
        end_energy   = float(energies_string[1])
        self.energy_metric = energies_string[2]
        if len(energies_string) == 4:
            step = float(energies_string[3])
        else:
            step = 1.
        self.energies = np.arange(start_energy, end_energy+step, step) # hack to include the last energy
        self.n_energies = self.n_projs = self.energies.size
    
    def load_useful_values(self, mac_dict):
        self.detect_scan_type(mac_dict)
        if self.scan_type == ScanType.CT:
            self.load_useful_values_ct_scan(mac_dict)
        elif self.scan_type == ScanType.EnergySwipe:
            self.load_useful_values_energy_swipe(mac_dict)
        #elif self.is_convey_belt:
        #   load_useful_values_convey_belt(mac_dict)
        else:
            self.n_projs = 1
        assert self.n_projs > 0, 'Number of projections must be > 0'
        
        if self.scan_type.value > ScanType.Radiograph.value:
            self.outfile = mac_dict.get('/gate/output/ProcessCT/setFileName')
            if self.outfile is not None:
                self.outfile = self.outfile[0]
                self.outfile = self.outfile if os.path.isabs(self.outfile) else os.path.join(os.path.dirname(mac_dict.macfile_path), self.outfile)
            self.scatter_outfile = mac_dict.get('/gate/output/ProcessCT/setScatterFileName')
            if self.scatter_outfile is not None:
                self.scatter_outfile = self.scatter_outfile[0]
                self.scatter_outfile = self.scatter_outfile if os.path.isabs(self.scatter_outfile) else os.path.join(os.path.dirname(mac_dict.macfile_path), self.scatter_outfile)
            self.rootoutfile = mac_dict.get('/gate/output/root/setFileName')
            if self.rootoutfile is not None:
                self.rootoutfile = self.rootoutfile[0]
                self.rootoutfile = self.rootoutfile if os.path.isabs(self.rootoutfile) else os.path.join(os.path.dirname(mac_dict.macfile_path), self.rootoutfile)
        else:
            self.outfile, self.scatter_outfile, self.rootoutfile = None, None, None

    def rotate(self, angle_n):
        r = R.from_euler(self.rot_axis_lett, self.projs[angle_n], degrees=True) ### unit
        rot_matrix = r.as_matrix()
        # Source position and orientation
        src_pos = np.matmul(rot_matrix,self.src_pos-self.COR_axis)+self.COR_axis
        if self.rot_axis_lett == 'z':
            posrot1 = rot_matrix[:,1]
            posrot2 = rot_matrix[:,2]
        else:
            raise NotImplementedError
        # Detector position and orientation
        det_pos = np.matmul(rot_matrix,self.det_pos-self.COR_axis)+self.COR_axis
        # Now, we have to concatenate the actual rotation found above with the initial rotation. The initial rotation is given by the user in the mac file.
        initial_rot_matrix = R.from_rotvec(self.init_rot_angle*self.init_rot_axis, degrees=True).as_matrix()
        total_rot_matrix = np.matmul(rot_matrix, initial_rot_matrix)
        rot_axis  = R.from_matrix(total_rot_matrix).as_rotvec(degrees=True)
        if np.linalg.norm(rot_axis) == 0.: # no rotation
            rot_axis  = self.rot_axis
            rot_angle = 0.
        else: # rotation has to be decomposed from a single axis notation to (angle, axis) notation
            rot_angle = np.linalg.norm(rot_axis)
            rot_axis /= rot_angle
        
        return src_pos, rot_axis, rot_angle, det_pos, posrot1, posrot2
    
    def build_output_paths(self, path, proj_n):
        outfile         = insert_number_as_parent(self.outfile,         proj_n) if (self.outfile         is not None) else None
        scatter_outfile = insert_number_as_parent(self.scatter_outfile, proj_n) if (self.scatter_outfile is not None) else None
        rootoutfile     = insert_number_as_parent(self.rootoutfile,     proj_n) if (self.rootoutfile     is not None) else None
        return outfile, scatter_outfile, rootoutfile
    
    def get_total_n_parameters(self):
        return self.n_projs
    
    def get_task_per_param(self, proj_n):
        commands, values = [], []
        if self.scan_type == ScanType.CT:
            src_pos, rot_axis, rot_angle, det_pos, posrot1, posrot2 = self.rotate(proj_n)
            commands.append('/gate/ProcessCTscanner/placement/setTranslation')
            values.append(det_pos.tolist())
            commands.append('/gate/ProcessCTscanner/placement/setRotationAxis')
            values.append(rot_axis.tolist())
            commands.append('/gate/ProcessCTscanner/placement/setRotationAngle') 
            values.append([rot_angle, 'deg'])
            commands.append('/gate/source/xraygun/gps/pos/centre')
            values.append(src_pos.tolist())
            commands.append('/gate/source/xraygun/gps/pos/rot1')
            values.append(posrot1.tolist())
            commands.append('/gate/source/xraygun/gps/pos/rot2')
            values.append(posrot2.tolist())
        elif self.scan_type == ScanType.EnergySwipe:
            commands.append('/gate/source/mybeam/gps/ene/mono')
            values.append([self.energies[proj_n], self.energy_metric])
        outfile, scatter_outfile, rootoutfile = self.build_output_paths(self.outfile, proj_n)
        if outfile is not None:
            commands.append('/gate/output/ProcessCT/setFileName')
            values.append([outfile])
        if scatter_outfile is not None:
            commands.append('/gate/output/ProcessCT/setScatterFileName')
            values.append([scatter_outfile])  
        if rootoutfile is not None:
            commands.append('/gate/output/root/setFileName')
            values.append([rootoutfile])      
        return commands, values

class cpu_par_manager:
    def __init__(self, mac_dict):
        super().__init__()
        self.load_useful_values(mac_dict)

    def load_useful_values(self, mac_dict):        
        self.outfile = mac_dict.get('/gate/output/ProcessCT/setFileName')
        self.outfile = self.outfile[0] if self.outfile is not None else None
        self.scatter_outfile = mac_dict.get('/gate/output/ProcessCT/setScatterFileName')
        self.scatter_outfile = self.scatter_outfile[0] if self.scatter_outfile is not None else None
        self.rootoutfile = mac_dict.get('/gate/output/root/setFileName')
        self.rootoutfile = self.rootoutfile[0] if self.rootoutfile is not None else None
        self.n_processes = mac_dict.pop('/mpiForGate/nProcesses', None)
        self.n_processes = 1 if self.n_processes is None else int(self.n_processes[0])
          
    def build_output_paths(self, cpu_n):
        outfile         = insert_number_before_point(self.outfile,         cpu_n) if (self.outfile         is not None) else None
        scatter_outfile = insert_number_before_point(self.scatter_outfile, cpu_n) if (self.scatter_outfile is not None) else None
        rootoutfile     = insert_number_before_point(self.rootoutfile,     cpu_n) if (self.rootoutfile     is not None) else None
        return outfile, scatter_outfile, rootoutfile
    
    def get_total_n_parameters(self):
        return self.n_processes
    
    def get_task_per_param(self, cpu_n):
        outfile, scatter_outfile, rootoutfile = self.build_output_paths(cpu_n)
        commands, values = [], []
        if outfile is not None:
            commands.append('/gate/output/ProcessCT/setFileName')
            values.append([outfile])
        if scatter_outfile is not None:
            commands.append('/gate/output/ProcessCT/setScatterFileName')
            values.append([scatter_outfile])
        if rootoutfile is not None:
            commands.append('/gate/output/root/setFileName')
            values.append([rootoutfile])
        return commands, values

class seed_par_manager:
    def __init__(self, mac_dict):
        pass
    
    def get_task_per_param(self, seed):
        commands, values = [], []
        commands.append('/gate/random/setEngineSeed')
        values.append([seed])
        return commands, values
    

def get_processed_macfile(macfile_path, new_macfile_path, proj_n, cpu_n):
    orig_macfile = mf(macfile_path)
    curr_macfile = copy.deepcopy(orig_macfile) # working on copy because the the managers have destructive reading on some macfile instructions

    ct   = proj_par_manager(curr_macfile)
    new_ct_cmds, new_ct_vals   = ct.get_task_per_param(proj_n)
    curr_macfile.update(new_ct_cmds, new_ct_vals)
    
    cpu  = cpu_par_manager(curr_macfile)
    new_cpu_cmds, new_cpu_vals  = cpu.get_task_per_param(cpu_n)
    curr_macfile.update(new_cpu_cmds, new_cpu_vals)
    
    seed = seed_par_manager(curr_macfile)
    new_seed_cmds, new_seed_vals = seed.get_task_per_param(cpu.get_total_n_parameters()*proj_n+cpu_n) #assign one particolar seed per simulation
    curr_macfile.update(new_seed_cmds, new_seed_vals)
    
    curr_macfile.write(new_macfile_path)
    
def get_processCT_info_from_macfile(macfile_path):
    orig_macfile = mf(macfile_path)
    # simulate_rot = orig_macfile.get('/mpiForGate/simulateRotation', None)
    # n_angles = 1 if simulate_rot is None else simulate_rot[2]
    # n_processes = orig_macfile.get('/mpiForGate/nProcesses', None)
    # n_processes = 1 if n_processes is None else n_processes[0]
    
    ct   = proj_par_manager(orig_macfile)
    n_projs   = ct.get_total_n_parameters()
    
    cpu  = cpu_par_manager(orig_macfile)
    n_processes  = cpu.get_total_n_parameters()
    
    return int(n_projs), int(n_processes)


if __name__ == '__main__':
    angle_n = 3 
    cpu_n   = 2 
    macfile_path     = 'tests/simple_macfile/processCT.mac'
    new_macfile_path = 'tests/simple_macfile/processCT_mod.mac'
    
    # Get absolute path
    macfile_path = pathlib.Path(macfile_path).absolute()
    new_macfile_path = pathlib.Path(new_macfile_path).absolute()
    
    get_processCT_info_from_macfile(macfile_path)
    get_processed_macfile(macfile_path, new_macfile_path, angle_n, cpu_n)
