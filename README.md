Simulations of single and multi-joint assistive devices to reduce the metabolic 
cost of walking.

SimTK project page: https://simtk.org/projects/coupled-exo-sim

Bianco NA, Franks PW, Hicks JL, Delp SL. Coupled exoskeleton assistance 
simplifies control and maintains metabolic benefits: a simulation study. 
PLoS One. 2021, in review.   

preprint: https://doi.org/10.1101/2021.04.16.440073

Software requirements
---------------------
- Matlab (2018b)
- GPOPS-II (version 2.3)
- OpenSim 3.3
- Python 2.7 (an Anaconda environment can be loaded from conda_enviroment.yml)
  - doit
  - yaml
  - matplotlib
  - pandas
  - scipy
  - h5py
  - seaborn

The simulation pipeline may work with later versions of Matlab and GPOPS-II, but
these are the versions I used to generate my results. The pipeline will *not*
work with OpenSim 4.0 (or later) and Python 3.

Config file setup
-----------------
Before running any simulations, you will need create a file called "config.yml"
and place it in the top directory of the repository. This file will contain
various values and paths needed to run the simulation pipeline. The following
four entries can be copied into the file verbatim (these could have been 
hard-coded somewhere else in the pipeline, but I found it useful to have them
in the config file to keep track of what I was simulating at any given time): 

subjects: [01, 02, 04, 18, 19]
norm_hip_max_torque: 1.0
norm_knee_max_torque: 1.0
norm_ankle_max_torque: 2.0

The remaining entries are paths that will be specific to your system:

doit_path: Path to the repository root 
           (i.e., C:\Users\Nick\Repos\coupled-exo-sim)
optctrlmuscle_path: Path to the optctrlmuscle submodule 
                    (i.e., <doit_path>\optctrlmuscle)
motion_capture_data_path: Path to raw mocap data (download from SimTK project)
results_path: Path to the simulation results directory
validate_path: Path to the directory containing validation results
analysis_path: Path to the directory containing figure drafts
figures_path: Path to the directory containing final paper figures
opensim_home: Path to the home directory of your OpenSim 3.3 install

'doit' workflow
---------------
All steps in the simulation pipeline for this study are handled using the Python
package 'doit', which is a task automation tool. 'Doit' keeps track of the tasks
that have been run and which previous tasks need to be run in order to run tasks
later in the pipeline. 

The top-level 'doit' file is called 'dodo.py'; all tasks for the study eminate 
from this file. Subject-specific tasks are defined in 'subjectXX.py' files. These
files instantiate "tasks", which are individual blocks of simulation code that 
'doit' will execute. The core OpenSim pipeline tasks are contained in the 
'osimpipeline' submodule, and study-specific tasks are contained in the 
'tasks.py'. 

Tasks are executed by calling 'doit <task_name>' in PowerShell (or other
terminal environment). Call 'doit list' to see a list of available tasks that 
can be run. The file 'run_tasks.ps1' is a PowerShell script containing the tasks
needed to be run to reproduce the study. The tasks are listed in order, and if 
you run a task before running the necessary preceeding tasks, those tasks will
be run automatically. You can force a task to run without the preceeding tasks
by calling 'doit -s <task_name>'; this can be helpful if the 'doit' cache gets
corrupted and you know that all preceeding tasks have been run already.





