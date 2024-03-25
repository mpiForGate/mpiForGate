
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
from threading import Lock, Thread
import logging, os, sys, time
from utils import *
from collector import collector
            

sys.excepthook = global_except_hook
            
class collectState:
    
    def __init__(self, logFolder, nSubSims, nProjs):
        self.nProjs   = nProjs
        self.nSubSims = nSubSims
        self.state       = np.zeros((nSubSims,nProjs),dtype=np.int32)
        self.assigned_to = -np.ones((nSubSims,nProjs),dtype=np.int32)
        self.macfile     = [[None]*nProjs for i in range(nSubSims)]
        self.mutex = Lock()
        self.log = open(os.path.join(logFolder,"state.log"), "w") 
        self.write_log("init")

    def write_log(self, message):
        self.log.write(getTimeString()+': ' +message+'\n')
        self.log.flush()

    def assign(self, macfile,rank, subSim, proj):
        proj   = proj
        subSim = subSim 
        self.assigned_to[subSim,proj] = rank 
        self.macfile[subSim][proj]     = macfile 

    def changeState(self,subSim,proj, new_state):
        proj   = proj
        subSim = subSim
        self.mutex.acquire()
        self.log.write(getTimeString()+": ("+str(subSim)+","+str(proj)+") "+states(self.state[subSim,proj]).name+" to "+new_state.name+'\n')
        self.log.flush()
        if new_state.value - self.state[subSim,proj] == 1:
            self.state[subSim,proj] = new_state.value
        elif (new_state == states.DONE) and (self.state[subSim,proj] == states.READING.value):
            self.state[subSim,proj] = new_state.value
        elif (new_state == states.DONE) and (self.state[subSim,proj] == states.READY.value):
            self.state[subSim,proj] = new_state.value
        else:
            self.mutex.release()
            raise Exception("collectState has been requested to perform an invalid changeState(): proj "+str(subSim)+" subSim "+str(proj)+" change "+states(self.state[subSim,proj]).name+" to "+new_state.name)
        self.mutex.release()

    def get_READY_processes(self):
        self.mutex.acquire()
        return_value = np.argwhere(self.state==states.READY.value)
        self.mutex.release()
        return return_value

    def shouldWrite(self,subSim,proj):
        proj   = proj
        subSim = subSim
        self.mutex.acquire()
        return_value=False
        done_jobs = (self.state[:,proj]==states.DONE.value) 
        if np.sum(done_jobs) == len(done_jobs)-1:
            i, = np.where(done_jobs == False)
            i = i[0]
            return_value = True if (i==subSim and self.state[i,proj]>=states.READING.value) else False

        if return_value: # debug
            self.write_log('Could write! ('+str(subSim)+','+str(proj)+')')
        self.mutex.release()
        return return_value

    def thereAreStillSleepingProcesses(self):
        self.mutex.acquire()
        return_value = np.any(self.state==states.SLEEPING.value)
        self.mutex.release()
        return return_value

    def thereIsStillWorkForManager(self):
        self.mutex.acquire()
        return_value = np.any(self.state!=states.DONE.value)
        self.mutex.release()
        return return_value

    def isWorkerBusy(self):
        self.mutex.acquire()
        return_value = np.any( (self.state==states.READING.value) | (self.state==states.WRITING.value))
        self.mutex.release()
        return return_value

class collectorManager:

    def __init__(self, logFolder, cstate, queue_size, keep_macfile=False):
        self.sleep_time_done_nothing = int(10)
        self.sleep_time_done_something = int(5)
        self.comm = MPI.COMM_WORLD
        self.keep_macfile = keep_macfile
        self.collector = collector(logFolder)
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.queue_size = queue_size
        self.cs = cstate
        if self.rank != 0:
            raise Exception(getTimeString()+": It should work on rank 0 instead of rank "+str(self.rank))
        self.shall_listen_comm = True
        self.log = open(os.path.join(logFolder, "manager.log"), "w") 
        self.log.write("init\n")
        self.mutex = Lock()
        self.queue_n = [-1 for i in range(self.queue_size)]
        self.threadList = []
        self.threadList.append(Thread(target=self.comm_listener, daemon=True))
        self.has_intercomm_ended = False
        self.threadList.append(Thread(target=self.intercomm_operator, daemon=True))
        for thread in self.threadList:
            thread.start()

    def comm_listener(self):
        while self.cs.thereAreStillSleepingProcesses():
            self.write_log("probing for message")
            data = np.zeros((1,4), dtype=np.int32)
            self.comm.Recv([data,MPI.INT], source=MPI.ANY_SOURCE)
            data = data.tolist()[0]
            signal, rank, subSim, proj  = [int(item) for item in data[0:4]]
            self.write_log("received "+signals(signal).name +" from rank "+str(rank))
            
            if signal == signals.DONE.value:
                self.cs.changeState(subSim, proj, states.READY)
            else:
                raise Exception(getTimeString()+": Received unknown signal from slave "+str(rank))
        self.write_log("comm_listener terminated")

    def intercomm_operator(self):
        while self.cs.thereIsStillWorkForManager():
            list_ready_sims = self.cs.get_READY_processes()
            done_something = False
            for subSim, proj in list_ready_sims:
                if -1 in self.queue_n:
                    if proj not in self.queue_n:
                        self.queue_n[self.queue_n.index(-1)] = proj
                
            to_be_processed_projs = [value for value in list_ready_sims if value[1] in self.queue_n]
            if len(to_be_processed_projs) > 0:
                self.write_log("processing jobs {}: next in queue".format(to_be_processed_projs))
                self.multi_process(to_be_processed_projs)
                done_something = True

            time.sleep(self.sleep_time_done_nothing) if not done_something else time.sleep(self.sleep_time_done_something)
        self.write_log("intercomm_operator terminated")
        self.has_intercomm_ended = True
        
    def write_log(self, message):
        self.log.write(getTimeString()+': ' +message+'\n')
        self.log.flush()
                
    def process(self, subSim, proj):
        self.write_log("processing job ("+str(subSim)+","+str(proj)+"): sending READ")
        curr_macfile = self.cs.macfile[subSim][proj]
        self.cs.changeState(subSim, proj, states.READING)
        self.collector.process_READ(self.queue_n.index(proj), curr_macfile)
        self.write_log("job ("+str(subSim)+","+str(proj)+"): received DONE reading")

        if self.cs.shouldWrite(subSim, proj): 
            #self.write_log("processing job ("+str(subSim)+","+str(proj)+"): sending WRITE")
            self.cs.changeState(subSim, proj, states.WRITING)
            self.collector.process_WRITE(self.queue_n.index(proj), curr_macfile)
            #self.write_log(": job ("+str(subSim)+","+str(proj)+"): received DONE writing")
            self.queue_n[self.queue_n.index(proj)] = -1
        self.cs.changeState(subSim, proj, states.DONE)
        if not self.keep_macfile:
            os.remove(curr_macfile)

    def multi_process(self, to_be_processed_projs):
        jobs = np.array(to_be_processed_projs)
        subsims = jobs[:,0]
        projs   = jobs[:,1]
        curr_macfile_list = []
        queue_list = []
        self.write_log("processing job(s) {}: sending READ".format(to_be_processed_projs))
        for subSim,proj in zip(subsims, projs):
            self.cs.changeState(subSim, proj, states.READING)
            curr_macfile_list.append(self.cs.macfile[subSim][proj])
            queue_list.append(self.queue_n.index(proj))
        self.collector.process_multiREAD(queue_list, curr_macfile_list)
        self.write_log("processing job(s) {}: received DONE reading".format(to_be_processed_projs))

        for i,(subSim,proj) in enumerate(zip(subsims, projs)):
            if self.cs.shouldWrite(subSim, proj): 
                self.write_log("processing job ("+str(subSim)+","+str(proj)+"): sending WRITE")
                self.cs.changeState(subSim, proj, states.WRITING)
                self.collector.process_WRITE(queue_list[i], curr_macfile_list[i])
                self.write_log(": job ("+str(subSim)+","+str(proj)+"): received DONE writing")
                self.queue_n[self.queue_n.index(proj)] = -1
            self.cs.changeState(subSim, proj, states.DONE)
            if not self.keep_macfile:
                os.remove(curr_macfile_list[i])
     
    def join(self):
        for thread in self.threadList:
            thread.join()
        self.log.close()
