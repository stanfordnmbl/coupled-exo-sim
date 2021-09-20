# Allow using the osimpipeline git submodule.
import sys
sys.path.insert(1, 'osimpipeline')

import os

import yaml
with open('config.yaml') as f:
    config = yaml.load(f)
if not 'opensim_home' in config:
    raise Exception("You must define the field `opensim_home` in config.yaml "
            "to point to the root of your OpenSim 3.3 installation.")
sys.path.insert(1, os.path.join(config['opensim_home'], 'sdk', 'python'))
sys.path.insert(1, 'perimysium')

DOIT_CONFIG = {
        'verbosity': 2,
        'default_tasks': None,
        }

# Settings for plots.
import matplotlib
# matplotlib.use('TkAgg')
if matplotlib.__version__[0] == '1':
    raise Exception("Must have matplotlib version 2 to avoid "
            "incorrect bar plots.")
import matplotlib.pyplot as plt
plt.rc('font', family='Helvetica, Arial, sans-serif', size=8)
plt.rc('errorbar', capsize=1.5)
plt.rc('lines', markeredgewidth=1)
plt.rc('legend', fontsize=8)

import osimpipeline as osp

# This line is necessary for registering the tasks with python-doit.
from osimpipeline.vital_tasks import *
from osimpipeline.mrs_tasks import *

# Custom tasks for this project.
from tasks import *

# Custom helper functions for this project
from helpers import *

study = osp.Study('exotopology',
        'Rajagopal2015_18musc_muscle_names_probed_iliacus.osim',
        'Rajagopal2015_reserve_actuators.xml')

# Model markers to compute errors for
marker_suffix = ['ASI','PSI','TH1','TH2','TH3','TB1','TB2',
                 'CAL','TOE','MT5','ACR','LEL','MEL','UA1','UA2','UA3',
                 'FAsuperior','FAradius','FAradius']
error_markers = ['*' + marker for marker in marker_suffix] 
error_markers.append('CLAV')
error_markers.append('C7')
study.error_markers = error_markers

# List of modified muscle redundancy problems used in the study
# study.momentArms = 'BiLLEE'
study.momentArms = 'fixed_direction'

# 'Default' (activations squared) or 'Met' (metabolic cost). 
#  Default is assumed if omitted.
study.costFunction = 'Met' 
study.mod_names = get_exotopology_flags(study)[0]

study.dof_names = ['hip_flexion_r','knee_angle_r','ankle_angle_r']
study.muscle_names = ['bifemsh_r', 'med_gas_r', 'glut_max2_r', 'psoas_r',
                      'rect_fem_r','semimem_r','soleus_r','tib_ant_r',
                      'vas_int_r']
study.nice_muscle_names_dict = {'glut_max2_r': 'glut. max.',
                                'psoas_r': 'iliopsoas',
                                'semimem_r': 'semimemb.',
                                'rect_fem_r': 'rect. fem.',
                                'bifemsh_r': 'bi. fem. s.h.',
                                'vas_int_r': 'vas. int.',
                                'med_gas_r': 'gastroc.',
                                'soleus_r': 'soleus',
                                'tib_ant_r': 'tib. ant.',
                                }
# muscle-tendon parameter calibration settings
# study.calibrate_muscle_names = ['med_gas_r','glut_max2_r','rect_fem_r',
#                                 'semimem_r','soleus_r','tib_ant_r','vas_int_r',
#                                 'psoas_r']
study.param_dict = dict()
param_muscle_list = ['med_gas_r','glut_max2_r','rect_fem_r',
                     'semimem_r','soleus_r','tib_ant_r','vas_int_r']
study.param_dict['optimal_fiber_length'] = param_muscle_list
study.param_dict['tendon_slack_length'] = param_muscle_list
study.param_dict['muscle_strain'] = param_muscle_list

study.cost_dict = dict()
cost_muscle_list = ['med_gas_r','glut_max2_r','rect_fem_r',
                    'semimem_r','soleus_r','tib_ant_r','vas_int_r']
study.cost_dict['emg'] = cost_muscle_list                               

# Choose cycles to isolate for aggregate and plot tasks.
# Set to None to plot all cycles.
study.cycles_to_plot = ['cycle02','cycle03']

study.calibrate_cycles = ['cycle01'] # cycles used to calibrate subject models
study.test_cycles = ['cycle02','cycle03'] # cycles used to analyses
study.test_cycles_num = [2, 3]

# Which activation dyanmics model to use: "explicit" or "implicit"
study.actdyn = 'explicit'

# Results
# ----------
# Add tasks for each subject
subjects = config['subjects']
for subj in subjects:
    if subj == 01:
        import subject01
        subject01.add_to_study(study)

    if subj == 02:
        import subject02
        subject02.add_to_study(study)

    if subj == 04:
        import subject04
        subject04.add_to_study(study)

    if subj == 18:
        import subject18
        subject18.add_to_study(study)

    if subj == 19:
        import subject19
        subject19.add_to_study(study)

# Copy data files for all study subjects
study.add_task(TaskCopyMotionCaptureData, 
    walk100=(2, '_newCOP3'),
    walk125=(2, '_newCOP3'),
    walk150=(2, '_newCOP3'),
    walk175=(2, '_newCOP3'),
    run200=(2, '_newCOP3'),
    run300=(2, '_newCOP3'),
    run400=(2, '_newCOP3'),
    run500=(2, '_newCOP3'),
    )

study.add_task(TaskRemoveEMGFileHeaders)
study.add_task(TaskCopyGenericModelFilesToResults)

# Analysis
# --------
subjects = ['subject01', 'subject02', 'subject04', 'subject18', 'subject19']
conditions = ['walk2']
# Experiment results
study.add_task(TaskAggregateMuscleDataExperiment, cond_names=conditions)  
study.add_task(TaskPlotMuscleData, study.tasks[-1])
study.add_task(TaskAggregateMomentsExperiment, cond_names=conditions)
study.add_task(TaskPlotMoments, study.tasks[-1])
study.add_task(TaskAggregateReservesExperiment, cond_names=conditions)
study.add_task(TaskComputeReserves, study.tasks[-1])

# Validation
# ----------
study.add_task(TaskValidateMarkerErrors, cond_names=conditions)
study.add_task(TaskValidateMuscleActivity, cond_names=conditions)
study.add_task(TaskValidateKinematics, cond_name=conditions[0])
study.add_task(TaskValidateKinetics, cond_name=conditions[0])
task_names = ['experiment', 'mrsmod_deviceHe', 'mrsmod_deviceKe', 
               'mrsmod_deviceHf', 'mrsmod_deviceKf', 'mrsmod_deviceAp',
               'mrsmod_deviceHeKe_coupled', 'mrsmod_deviceHeKe_independent', 
               'mrsmod_deviceHfKf_coupled', 'mrsmod_deviceHfKf_independent', 
               'mrsmod_deviceKfAp_coupled', 'mrsmod_deviceKfAp_independent', 
               'mrsmod_deviceHfAp_coupled', 'mrsmod_deviceHfAp_independent', 
               'mrsmod_deviceHfKfAp_coupled', 'mrsmod_deviceHfKfAp_independent', 
               ]
study.add_task(TaskValidateReserves, task_names, cond_name=conditions[0])
study.add_task(TaskValidateSensitivity)

## Device comparisons
master_device_list = list()
Hf_color = 'darkred'
Kf_color = 'gold'
Ap_color = 'darkblue'
KfAp_color = 'darkgreen'
HfKf_color = 'darkorange'
HfAp_color = 'purple'
HfKfAp_color = 'saddlebrown'

Ke_color = 'darkcyan'
He_color = 'violet'
HeKe_color = 'slateblue'

######################################################
################### MRSMOD RESULTS ###################
######################################################

# Hip extension + knee extension
device_list = ['mrsmod_deviceHe', 
               'mrsmod_deviceKe', 
               'mrsmod_deviceHeKe_coupled',
               'mrsmod_deviceHeKe_independent']

master_device_list  = list(set().union(master_device_list, device_list))
label_list = [
              'hip ext.', 
              'knee ext.',
              'hip ext. + knee ext.', 
              'hip ext. + knee ext. (ind)']
color_list = [He_color, Ke_color, HeKe_color, HeKe_color]


plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
plot_lists['linestyle_list'] = ['-','-','-','--']
folder = 'mrsmod_HeKe'
study.add_task(TaskAggregateDevicePower, device_list, suffix=folder, 
    conditions=conditions)
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder,
    conditions=conditions)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects,
      max_metabolic_reduction=40, cond_names=conditions)

# Hip flexion + knee flexion + ankle plantarflexion
device_list = ['mrsmod_deviceHf', 
               'mrsmod_deviceKf', 
               'mrsmod_deviceAp',
               'mrsmod_deviceHfKfAp_coupled',
               'mrsmod_deviceHfKfAp_independent']
master_device_list  = list(set().union(master_device_list, device_list))
label_list = ['hip flex.', 
              'knee flex.',
              'ankle pl.',
              'hip flex. + knee flex. + ankle pl.',
              'hip flex. + knee flex. + ankle pl. (ind)']
color_list = [Hf_color, Kf_color, Ap_color, HfKfAp_color, HfKfAp_color] 

plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
plot_lists['linestyle_list'] = ['-','-','-','-','--']
folder = 'mrsmod_HfKfAp'
study.add_task(TaskAggregateDevicePower, device_list, suffix=folder,
    conditions=conditions)
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder,
    conditions=conditions)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects,
    max_metabolic_reduction=50, cond_names=conditions)

# Hip flexion + knee flexion
device_list = ['mrsmod_deviceHf', 
               'mrsmod_deviceKf', 
               'mrsmod_deviceHfKf_coupled',
               'mrsmod_deviceHfKf_independent']
master_device_list  = list(set().union(master_device_list, device_list))
label_list = ['hip flex.', 
              'knee flex.',
              'hip flex. + knee flex.',
              'hip flex. + knee flex. (ind)']
color_list = [Hf_color, Kf_color, HfKf_color, HfKf_color] 

plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
plot_lists['linestyle_list'] = ['-','-','-','--']
folder = 'mrsmod_HfKf'
study.add_task(TaskAggregateDevicePower, device_list, suffix=folder,
    conditions=conditions)
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder,
    conditions=conditions)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects,
    max_metabolic_reduction=40, cond_names=conditions)

# Hip flexion + ankle plantarflexion
device_list = ['mrsmod_deviceHf', 
               'mrsmod_deviceAp',
               'mrsmod_deviceHfAp_coupled',
               'mrsmod_deviceHfAp_independent']
master_device_list  = list(set().union(master_device_list, device_list))
label_list = ['hip flex.', 
              'ankle pl.',
              'hip flex. + ankle pl.',
              'hip flex. + ankle pl. (ind)']
color_list = [Hf_color, Ap_color, HfAp_color, HfAp_color] 

plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
plot_lists['linestyle_list'] = ['-','-','-','--']
folder = 'mrsmod_HfAp'
study.add_task(TaskAggregateDevicePower, device_list, suffix=folder,
    conditions=conditions)
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder,
    conditions=conditions)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects,
    max_metabolic_reduction=40, cond_names=conditions)

# Knee flexion + ankle plantarflexion
device_list = ['mrsmod_deviceKf', 
               'mrsmod_deviceAp',
               'mrsmod_deviceKfAp_coupled',
               'mrsmod_deviceKfAp_independent']
master_device_list  = list(set().union(master_device_list, device_list))
label_list = ['knee flex.',
              'ankle pl.',
              'knee flex. + ankle pl.',
              'knee flex. + ankle pl. (ind)']
color_list = [Kf_color, Ap_color, KfAp_color, KfAp_color] 

plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
plot_lists['linestyle_list'] = ['-','-','-','--']
folder = 'mrsmod_KfAp'
study.add_task(TaskAggregateDevicePower, device_list, suffix=folder,
    conditions=conditions)
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder,
    conditions=conditions)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects,
    max_metabolic_reduction=40, cond_names=conditions)


# device_list = ['mrsmod_deviceHKAp_independent_torque_limited', 
#                'mrsmod_deviceHKAp_independent']
# master_device_list  = list(set().union(master_device_list, device_list))
# label_list = ['limited\ntorque', 'full\ntorque']
# color_list = ['orange', 'blue'] 

# plot_lists = dict()
# plot_lists['device_list'] = device_list
# plot_lists['label_list'] = label_list
# plot_lists['color_list'] = color_list
# plot_lists['linestyle_list'] = ['-','-']
# folder = 'mrsmod_HKAp'
# study.add_task(TaskAggregateDevicePower, device_list, suffix=folder,
#     conditions=conditions)
# study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder,
#     conditions=conditions)
# study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects,
#     max_metabolic_reduction=60, cond_names=conditions, 
#     fig_width=2, fig_height=3.5)

# Wholebody metabolic cost reductions
# -----------------------------------
plot_lists = dict()
device_list = ['mrsmod_deviceHe', 'mrsmod_deviceKe', 
               'mrsmod_deviceHf', 'mrsmod_deviceKf', 'mrsmod_deviceAp',
               'mrsmod_deviceHeKe_coupled', 'mrsmod_deviceHeKe_independent', 
               'mrsmod_deviceHfKf_coupled', 'mrsmod_deviceHfKf_independent', 
               'mrsmod_deviceKfAp_coupled', 'mrsmod_deviceKfAp_independent', 
               'mrsmod_deviceHfAp_coupled', 'mrsmod_deviceHfAp_independent', 
               'mrsmod_deviceHfKfAp_coupled', 'mrsmod_deviceHfKfAp_independent', 
               ]
plot_lists['index_list'] = [0, 0.5, 1, 1.5, 2, 2.7, 2.7, 3.6, 3.6, 4.5, 4.5, 5.4, 5.4, 6.3, 6.3]
plot_lists['device_list'] = device_list
plot_lists['label_list'] = ['hip\next.', 'knee\next.', 
                            'hip\nflex.', 'knee\nflex.', 'ankle\npl.',
                            'hip ext.\nknee ext.',
                            'hip flex.\nknee flex.', 'knee flex.\nankle pl.',
                            'hip flex.\nankle pl.',
                            'hip, knee flex.\nankle pl.',
                            ]
plot_lists['xticks'] = [0, 0.5, 1, 1.5, 2, 2.7, 3.6, 4.5, 5.4, 6.3]
plot_lists['color_list'] = [He_color, Ke_color, Hf_color, Kf_color, 
                            Ap_color, HeKe_color, HfKf_color, KfAp_color, 
                            HfAp_color, HfKfAp_color]
master_device_list = list(set().union(master_device_list, device_list))
folder = 'metabolic_reductions'
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder,
    conditions=conditions)
study.add_task(TaskPlotMetabolicReductions, plot_lists, folder, subjects=subjects,
    max_metabolic_reduction=50, fig_height=5, fig_width=7,
    cond_names=conditions, multijoint_only=False)

# Muscle activations
# ------------------
plot_lists['index_list'] = [0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
plot_lists['label_list'] = ['hip\next.', 'knee\next.', 
                            'hip\nflex.', 'knee\nflex.', 'ankle\npl.',
                            'hip ext.\nknee ext.',
                            'hip flex.\nknee flex.', 'knee flex.\nankle pl.',
                            'hip flex.\nankle pl.',
                            'hip flex.\nknee flex.\nankle pl.',
                            ]
study.add_task(TaskPlotMuscleActivity, plot_lists, 'walk2')

# Device moments and powers
# -------------------------
study.add_task(TaskAggregateDeviceMoments, device_list, suffix=folder,
    conditions=conditions)
study.add_task(TaskCreateDeviceMomentTable, study.tasks[-1], suffix=folder)
study.add_task(TaskAggregateDevicePower, device_list, suffix=folder, 
    conditions=conditions)
study.add_task(TaskCreateDevicePowerTable, study.tasks[-1], suffix=folder)

# Muscle comparisons
# ------------------
study.add_task(TaskPlotMuscleMultijointComparison, 'walk2', 'mrsmod_deviceKfAp',
    'psoas_r', muscle_data=['norm_fiber_powers'],
    min_norm_power=-1, max_norm_power=2)
study.add_task(TaskPlotMuscleMultijointComparison, 'walk2', 'mrsmod_deviceKfAp',
    'rect_fem_r', muscle_data=['norm_fiber_powers'],
    min_norm_power=-3, max_norm_power=0.5)
study.add_task(TaskPlotMuscleMultijointComparison, 'walk2', 'mrsmod_deviceKfAp',
    'soleus_r', muscle_data=['activations', 'norm_fiber_powers'],
    min_norm_power=-0.5, max_norm_power=2.5)

# Wholebody metabolic cost reductions
# -----------------------------------
# plot_lists = dict()
# device_list = ['mrsmod_deviceHeKe_coupled', 'mrsmod_deviceHeKe_independent', 
#                'mrsmod_deviceHfKf_coupled', 'mrsmod_deviceHfKf_independent', 
#                'mrsmod_deviceKfAp_coupled', 'mrsmod_deviceKfAp_independent', 
#                'mrsmod_deviceHfAp_coupled', 'mrsmod_deviceHfAp_independent', 
#                'mrsmod_deviceHfKfAp_coupled', 'mrsmod_deviceHfKfAp_independent', 
#                ]

# offset = 0.2
# index_list = [0, 0, 0.9, 0.9, 1.8, 1.8, 2.7, 2.7, 3.6, 3.6]
# index_list = [idx + offset for idx in index_list]
# plot_lists['index_list'] = index_list
# plot_lists['device_list'] = device_list
# plot_lists['label_list'] = ['hip ext.\nknee ext.',
#                             'hip flex.\nknee flex.', 'knee flex.\nankle pl.',
#                             'hip flex.\nankle pl.',
#                             'hip, knee flex.\nankle pl.',
#                             ]

# xticks = [0, 0.9, 1.8, 2.7, 3.6]
# xticks = [tick + offset for tick in xticks]
# plot_lists['xticks'] = xticks
# plot_lists['color_list'] = [HeKe_color, HfKf_color, KfAp_color, 
#                             HfAp_color, HfKfAp_color]
# master_device_list = list(set().union(master_device_list, device_list))
# folder = 'metabolic_reductions_multijoint_only'
# study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder,
#     conditions=conditions)
# study.add_task(TaskPlotMetabolicReductions, plot_lists, folder, 
#     subjects=subjects,
#     max_metabolic_reduction=50, fig_height=5, fig_width=7,
#     cond_names=conditions, multijoint_only=True)

# plot_lists = dict()
# device_list = ['mrsmod_deviceHe', 'mrsmod_deviceKe', 'mrsmod_deviceHeKe', 
#                'mrsmod_deviceHeKe_independent']
# plot_lists['index_list'] = [0, 1, 2, 2]
# plot_lists['device_list'] = device_list
# plot_lists['label_list'] = ['hip ext.', 'knee ext.', 'hip ext.\nknee ext.']
# plot_lists['color_list'] = [He_color, Ke_color, HeKe_color]
# master_device_list = list(set().union(master_device_list, device_list))
# folder = 'metabolic_reductions_HeKe'
# study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder)
# study.add_task(TaskPlotMetabolicReductions, plot_lists, folder, subjects=subjects,
#     max_metabolic_reduction=20, fig_height=5, fig_width=3*width_per_col)

# Device showcase plots
# ---------------------
include_activations = False 

plot_lists = dict()
device_list = ['mrsmod_deviceHeKe_coupled', 'mrsmod_deviceHeKe_independent']
plot_lists['device_list'] = device_list
plot_lists['label_list'] = ['coupled', 'independent']
plot_lists['color_list'] = ['orange', 'blue']
master_device_list = list(set().union(master_device_list, device_list))
suffix = 'showcase_%s' % device_list[-1]
study.add_task(TaskAggregateDevicePower, device_list, suffix=suffix,
    conditions=['walk2'])
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=suffix,
    conditions=['walk2'])
study.add_task(TaskPlotDeviceShowcase, suffix, plot_lists, 'walk2', subjects,
    ['glut_max2_r', 'rect_fem_r', 'vas_int_r', 'soleus_r'], 'figure2',
    met_ylim=[-8, 2], met_yticks=[-8, -6, -4, -2, 0, 2],
    moment_ylim=[-1, 1], moment_yticks=[-1, 0, 1])

plot_lists = dict()
device_list = ['mrsmod_deviceHfKf_coupled', 'mrsmod_deviceHfKf_independent']
plot_lists['device_list'] = device_list
plot_lists['label_list'] = ['coupled', 'independent']
plot_lists['color_list'] = ['orange', 'blue']
master_device_list = list(set().union(master_device_list, device_list))
suffix = 'showcase_%s' % device_list[-1]
study.add_task(TaskAggregateDevicePower, device_list, suffix=suffix,
    conditions=['walk2'])
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=suffix,
    conditions=['walk2'])
study.add_task(TaskPlotDeviceShowcase, suffix, plot_lists, 'walk2', subjects,
    ['psoas_r', 'semimem_r', 'vas_int_r', 'med_gas_r'], 'figure3',
    met_ylim=[-25, 5], met_yticks=[-25, -20, -15, -10, -5, 0, 5],
    moment_ylim=[-1, 1], moment_yticks=[-1, 0, 1])

plot_lists = dict()
device_list = ['mrsmod_deviceKfAp_coupled', 'mrsmod_deviceKfAp_independent']
plot_lists['device_list'] = device_list
plot_lists['label_list'] = ['coupled', 'independent']
plot_lists['color_list'] = ['orange', 'blue']
master_device_list = list(set().union(master_device_list, device_list))
suffix = 'showcase_%s' % device_list[-1]
study.add_task(TaskAggregateDevicePower, device_list, suffix=suffix,
    conditions=['walk2'])
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=suffix,
    conditions=['walk2'])
study.add_task(TaskPlotDeviceShowcase, suffix, plot_lists, 'walk2', subjects,
    ['vas_int_r', 'semimem_r', 'med_gas_r', 'soleus_r'], 'figure4',
    met_ylim=[-20, 10], met_yticks=[-20, -15, -10, -5, 0, 5, 10])

plot_lists = dict()
device_list = ['mrsmod_deviceHfAp_coupled', 'mrsmod_deviceHfAp_independent']
plot_lists['device_list'] = device_list
plot_lists['label_list'] = ['coupled', 'independent']
plot_lists['color_list'] = ['orange', 'blue']
master_device_list = list(set().union(master_device_list, device_list))
suffix = 'showcase_%s' % device_list[-1]
study.add_task(TaskAggregateDevicePower, device_list, suffix=suffix,
    conditions=['walk2'])
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=suffix,
    conditions=['walk2'])
study.add_task(TaskPlotDeviceShowcase, suffix, plot_lists, 'walk2', subjects,
    ['psoas_r', 'semimem_r', 'med_gas_r', 'soleus_r'], 'figure5',
    met_ylim=[-25, 5], met_yticks=[-25, -20, -15, -10, -5, 0, 5])

plot_lists = dict()
device_list = ['mrsmod_deviceHfKfAp_coupled', 'mrsmod_deviceHfKfAp_independent']
plot_lists['device_list'] = device_list
plot_lists['label_list'] = ['coupled', 'independent']
plot_lists['color_list'] = ['orange', 'blue']
master_device_list = list(set().union(master_device_list, device_list))
suffix = 'showcase_%s' % device_list[-1]
study.add_task(TaskAggregateDevicePower, device_list, suffix=suffix,
    conditions=['walk2'])
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=suffix,
    conditions=['walk2'])
study.add_task(TaskPlotDeviceShowcase, suffix, plot_lists, 'walk2', subjects,
    ['psoas_r', 'bifemsh_r', 'med_gas_r', 'soleus_r'], 'figure6',
    met_ylim=[-25, 5], met_yticks=[-25, -20, -15, -10, -5, 0, 5],
    legend_loc=(0.10, 0.7))

# BiLLEE simulations
# plot_lists = dict()
# device_list = ['mrsmod_deviceHKAp_independent_torque_limited',
#                'mrsmod_deviceHKAp_independent']
# plot_lists['device_list'] = device_list
# plot_lists['label_list'] = ['limited torque', 'full torque']
# plot_lists['color_list'] = ['orange', 'blue']
# master_device_list = list(set().union(master_device_list, device_list))
# suffix = 'showcase_%s' % device_list[-1]
# study.add_task(TaskAggregateDevicePower, device_list, suffix=suffix,
#     conditions=['walk2'])
# study.add_task(TaskAggregateMetabolicRate, device_list, suffix=suffix,
#     conditions=['walk2'])
# study.add_task(TaskPlotDeviceShowcase, suffix, plot_lists, 'walk2', subjects,
#     ['psoas_r', 'semimem_r', 'med_gas_r', 'soleus_r'],
#     include_activations=include_activations, legend_loc=(0.2, 0.6),
#     met_ylim=[-20, 0], met_yticks=[-20, -15, -10, -5, 0])

for mod in master_device_list:
    study.add_task(TaskAggregateMomentsMod, mod, cond_names=conditions)
    study.add_task(TaskPlotMoments, study.tasks[-1])
    study.add_task(TaskAggregateMuscleDataMod, mod, cond_names=conditions)
    study.add_task(TaskPlotMuscleData, study.tasks[-1])
    study.add_task(TaskAggregateReservesMod, mod, cond_names=conditions)
    study.add_task(TaskComputeReserves, study.tasks[-1])