# MPI wrapper for X-ray GATE simulations
The software is a wrapper for GATE simulations. It allows to run GATE simulations in parallel using MPI. The software is written in Python and uses the mpi4py library.
It doesn't need any modification to the actual macro file, rather few additions. The software will split the macro file into multiple files and run them in parallel. If the number of required jobs exceedes the number of MPI ranks, then the jobs will be queued. The software will also automatically merge the output files into a single file, per projection.

In order to have a better understanding of the software, please read the following sections:
- [Installation](#installation)
- [Usage](#usage)
- [Behind the scenes](#behind-the-scenes)
- [Notes](#notes)


# Installation
The python software requires the following libraries:
- mpi4py
- GATE. If you would like ROOT outputs to be merged together , then you also need to build GATE with its cluster-tools (more specifically, the 'hadd' tool).

# Usage
The software is meant to be started as a ordinary MPI job, for example through `srun` command. Each process will take care of executing the appropriate macro file.
```bash
srun mpiParent.py --macfile /path/to/macfiles.mac --keep_macfiles 
```
The mpiParent.py will take as input all the options that follow its name. The options are:
- `--macfile`: the path to the macro file that will be executed
- `--keep_macfiles`: if present, the macro files will not be deleted after the execution
- `--keep_logfiles`: if present, the log files will not be deleted after the execution

Where the number of MPI ranks (i.e. number of concurrent processes) is determined by the settings of the srun command. 

To simulate a rotation of the source and detector, an additional command in the macro file is required. The command is:
```bash
/mpiForGate/simulateRotation 0 360 10
/mpiForGate/CORaxis 0.00000 -0.05868 0.00000
```
which will instruct the software to rotate the source and detector around the CORaxis (center of rotation axis) by 360 degrees, with 10 steps. In other words, the software will simulate 10 projections, starting from 0 degrees and ending at 360 degrees and being equally spaced. For each projection, the software will simulate the numer of photons specified in the macro file. 

To expedite the simulation of one projection, you may want to execute the simulation of one projection on multiple MPI ranks. This can be done by specifying the number of processes that will run the simulation of one projection. This can be done by adding the following command in the macro file, which will instruct the software to run the simulation of one projection for 54 times:
```bash
/mpiForGate/nProcesses 54
```
Note that each process will simulate the same number of photons, so the total number of photons will be `nProcesses * <number of photons>`. This command can be useful not only for speeding the simulation of one projection, but also for circumventing a GATE/G4 (silent) limitation regarding the maximum number of photons that can be simulated in one run (somewhere between 10^9 and 10^10).

# License
The software is released under the GNU General Public License v3.0 - See the LICENSE file for more details.

# Behind the scenes
The macro files that will be executed by each MPI rank will be stored in a temporary hidden directory (.tmp) at the same directory level of the macfiles. Same goes for the log files (.logs).

## Notes
- The software does not support (intentionally) cumulative calls, i.e. the output of the simulation will be overwritten at each run.
