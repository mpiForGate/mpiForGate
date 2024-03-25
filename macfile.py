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


import os


class macfile:
    def __init__(self, macfile=None):
        self.is_macfile_loaded = False
        if macfile is not None:
            self.macfile_path = os.path.abspath(macfile)
            self.load(macfile)

    def load(self, macfile):
        self.macfile_path = os.path.abspath(macfile)
        with open(macfile, 'r') as f:
            macfile_lines = f.readlines()
        
        lines_to_delete = []
        for i, line in enumerate(macfile_lines):
            line = line.strip()
            line = line.split('#', 1)[0]
            if line=='':
                lines_to_delete.append(i)
            macfile_lines[i] = line

        for i in reversed(lines_to_delete):
            macfile_lines.pop(i)

        self.commands = []
        self.values = []
        for line in macfile_lines:
            line_segm = line.split()
            self.commands.append(line_segm[0])
            self.values.append(line_segm[1:])
            
        self.is_macfile_loaded = True

    def write(self, new_path):
        assert self.is_macfile_loaded, 'No macfile has been read'
        to_be_last_command='/gate/application/start'
        with open(new_path, 'w') as f:
            for cmd, value in zip(self.commands, self.values):
                if cmd != to_be_last_command:
                    #print([line]+self.macfile_dict[line]+['\n'])
                    value_to_write = []
                    for value_elem in value:
                        value_elem = value_elem if not isinstance(value_elem, float) else '{0:,.5f}'.format(value_elem)
                        value_elem = value_elem if not isinstance(value_elem, int) else str(value_elem)
                        value_to_write.append(value_elem)
                    try:
                        f.write(' '.join([cmd,] + value_to_write + ['\n',]))
                    except Exception as e:
                        raise ValueError('Error in writing cmd {} with value {}, becase of {}'.format(cmd, value, e))
            f.write(' '.join([to_be_last_command] + ['\n',])) # last command

    def get(self, cmd, default=None):
        try:
            return self.values[self.commands.index(cmd)]
        except:
            return default
        
    # finds the all commands that contains cmd_pattern
    def find_cmd(self, cmd_pattern):
        list_of_cmds = []
        for cmd in self.commands:
            if cmd_pattern in cmd:
                list_of_cmds.append(cmd)
        return list_of_cmds
    
    def find_value(self, value_pattern):
        tuple_cmd_value = []
        for cmd, value in zip(self.commands, self.values):
            if value_pattern in value:
                tuple_cmd_value.append((cmd, value))
        return tuple_cmd_value
        
    def remove(self, cmd):
        idx = self.commands.index(cmd)
        self.commands.pop(idx)
        self.values.pop(idx)
        
    def pop(self, cmd, default=None):
        try:
            idx = self.commands.index(cmd)
            self.commands.pop(idx)
            return self.values.pop(idx)
        except:
            return default

    def update(self, curr_cmd, curr_value):
        # curr_cmd can be a list of commands. In this case, curr_value must be a list of lists
        # curr_cmd   = curr_cmd   if isinstance(curr_cmd, list)      else [curr_cmd]
        # if len(curr_cmd) > 0:
        #     curr_value = curr_value if isinstance(curr_value[0], list) else [curr_value]
        # else:
        #     curr_value = curr_value if isinstance(curr_value, list) else [curr_value]
        
            
        if not isinstance(curr_cmd, list):
            curr_cmd = [curr_cmd]
            if not isinstance(curr_value, list):
                curr_value = [[curr_value]]
            else:
                if isinstance(curr_value[0], list):
                    assert len(curr_value[0])==1, 'curr_value must be a list of values, not a list of lists'
                else:
                    curr_value = [curr_value]
        else:
            assert len(curr_cmd)==len(curr_value), 'curr_cmd and curr_value must have the same length'
        
        # separate the commands that appear more than once
        unique_cmds = []
        values_of_unique_cmds = []
        duplicate_cmds = []
        values_of_duplicate_cmds = []
        for cmd_to_check, value in zip(curr_cmd, curr_value):
            indices =[i for i, cmd in enumerate(self.commands) if cmd == cmd_to_check]
            if len(indices)==1:
                unique_cmds.append(cmd_to_check)
                values_of_unique_cmds.append(value)
            else:
                duplicate_cmds.append(cmd_to_check)
                values_of_duplicate_cmds.append(value)
        
        for cmd_to_modify, value_to_modify in zip(unique_cmds, values_of_unique_cmds):
            new_value = value_to_modify if isinstance(value_to_modify, list) else [value_to_modify]
            try: # cmd_to_modify is already known
                indices = [i for i, x in enumerate(self.commands) if x == cmd_to_modify]
                #idx = self.commands.index(cmd_to_modify)
                
                if len(indices)>1:# If there is more than one, remove all the duplicates after the first from self.commands and self.values
                    for index in sorted(indices, reverse=True):
                        del self.commands[index]
                        del self.values[index]
                    self.commands.append(cmd_to_modify)
                    self.values.append(new_value)
                else:
                    for i, new_value_element in enumerate(new_value):
                        self.values[indices[0]][i] = new_value_element # assumes that the new values have less or same number of elements
            except ValueError: # the cmd is new
                self.commands.append(cmd_to_modify)
                self.values.append(new_value)
        
        for cmd_to_check in duplicate_cmds:
            indices = [i for i, x in enumerate(self.commands) if x == cmd_to_check]
            for index in sorted(indices, reverse=True):
                del self.commands[index]
                del self.values[index]
        for cmd_to_add, value_to_add in zip(duplicate_cmds, values_of_duplicate_cmds):
            self.commands.append(cmd_to_add)
            self.values.append(value_to_add)
            
        
            
            
if __name__ == '__main__':
    mr = macfile()
    mr.load('tests/simple_macfile/processCT.mac')
    mr.update('/gate/source/mybeam/gps/pos/centre', [-1, -1, -1]) # update a cmd
    mr.update('/gate/source/mybeam/gps/pos/centre2', [0, 1, 2]) # insert a cmd
    mr.write('tests/simple_macfile/processCT_mod.mac')