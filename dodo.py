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
        'Rajagopal2015_18musc_muscle_names_probed.osim',
        'Rajagopal2015_reserve_actuators.xml')

# Model markers to compute errors for
marker_suffix = ['ASI','PSI','TH1','TH2','TH3','TB1','TB2','TB3',
                 'CAL','TOE','MT5','ACR','LEL','MEL','UA1','UA2','UA3',
                 'FAsuperior','FAradius','FAradius']
error_markers = ['*' + marker for marker in marker_suffix] 
error_markers.append('CLAV')
error_markers.append('C7')
study.error_markers = error_markers

# List of modified muscle redundancy problems used in the study
study.momentArms = 'fixed_direction'
study.whichDevices = 'active_only'
# A number in the range [0 1] or just empty brackets '[]'
study.fixMomentArms = '[]'
# 'Default' (activations squared) or 'Met'. Default is assumed if omitted.
study.costFunction = 'Met' 
study.mod_names = get_exotopology_flags(momentArms=study.momentArms, 
    whichDevices=study.whichDevices, fixMomentArms=study.fixMomentArms)[0]
study.muscle_names = ['bifemsh_r', 'med_gas_r', 'glut_max2_r', 'psoas_r',
                      'rect_fem_r','semimem_r','soleus_r','tib_ant_r',
                      'vas_int_r']
# muscle-tendon parameter calibration settings
study.calibrate_muscle_names = ['med_gas_r','glut_max2_r','rect_fem_r',
                                'semimem_r','soleus_r','tib_ant_r','vas_int_r',
                                'psoas_r']
study.param_dict = dict()
param_muscle_list = ['bifemsh_r','med_gas_r','glut_max2_r','rect_fem_r',
                     'semimem_r','soleus_r','tib_ant_r','vas_int_r','psoas_r']
study.param_dict['optimal_fiber_length'] = param_muscle_list
study.param_dict['tendon_slack_length'] = param_muscle_list
study.param_dict['muscle_strain'] = param_muscle_list

study.cost_dict = dict()
cost_muscle_list = ['med_gas_r','glut_max2_r','rect_fem_r','semimem_r',
                    'soleus_r','tib_ant_r','vas_int_r']
study.cost_dict['emg'] = cost_muscle_list                               

# Choose cycles to isolate for aggregate and plot tasks.
# Set to None to plot all cycles.
study.cycles_to_plot = ['cycle03']

# Add tasks for each subject
subjects = config['subjects']
for subj in subjects:
    if subj == 01:
        import subject01
        subject01.add_to_study(study)

    if subj == 02:
        import subject02
        subject02.add_to_study(study)

    # if subj == 03:
    #     import subject03
    #     subject03.add_to_study(study)

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
 
study.add_task(TaskCopyEMGData)

study.add_task(TaskCopyGenericModelFilesToResults)

# Validation
# ----------
study.add_task(TaskValidateAgainstEMG)

# Analysis
# --------
HfAp_agg_tasks = list()

# Global results
study.add_task(TaskAggregateMuscleActivityExperiment)
study.add_task(TaskPlotMuscleActivity, study.tasks[-1])
study.add_task(TaskAggregateMomentsExperiment)
study.add_task(TaskPlotMoments, study.tasks[-1])
for mod in study.mod_names:
    study.add_task(TaskAggregateMomentsMod, mod)
    # if mod == 'actHfAp':
    #     HfAp_agg_tasks.insert(1, study.tasks[-1])

    study.add_task(TaskPlotMoments, study.tasks[-1], mod=mod)
# study.add_task(TaskAggregateMuscleActivityMod)
# study.add_task(TaskPlotMuscleActivity, study.tasks[-1])
# study.add_task(TaskAggregateTorqueParameters)
# study.add_task(TaskPlotTorqueParameters)

# Active devices only, fixed direction, free value moment arms
if ((study.momentArms == 'fixed_direction') and 
    (study.whichDevices == 'active_only') and
    (study.fixMomentArms == '[]')):
    actXxXxXx = get_exotopology_flags(momentArms='fixed_direction', 
        whichDevices='active_only')[0]
    study.add_task(TaskAggregatePeakPower, mods=actXxXxXx, suffix='actXxXxXx')
    study.add_task(TaskAggregateMetabolicRate, mods=actXxXxXx, 
        suffix='actXxXxXx')
    study.add_task(TaskPlotDeviceMetabolicRankings, mods=actXxXxXx, 
        suffix='actXxXxXx')
    study.add_task(TaskPlotMetabolicReductionVsPeakPower, mods=actXxXxXx, 
        suffix='actXxXxXx')
    actXxXxXx_multControls = get_mult_controls_mod_names(study)
    study.add_task(TaskAggregatePeakPower, mods=actXxXxXx_multControls, 
        suffix='actXxXxXx_multControls')
    study.add_task(TaskAggregateMetabolicRate, mods=actXxXxXx_multControls, 
        suffix='actXxXxXx_multControls')
    study.add_task(TaskPlotDeviceMetabolicRankings, mods=actXxXxXx_multControls, 
        suffix='actXxXxXx_multControls')
    study.add_task(TaskPlotMetabolicReductionVsPeakPower, 
        mods=actXxXxXx_multControls, 
        suffix='actXxXxXx_multControls')
    # study.add_task(TaskAggregateMuscleActivityMod, mods=actXxXxXx_multControls, 
    #     suffix='actXxXxXx_multControls')
    # study.add_task(TaskPlotMuscleActivity, study.tasks[-1], 
    #     suffix='actXxXxXx_multControls')
    # study.add_task(TaskAggregateTorqueParameters, mods=actXxXxXx_multControls, 
    #     suffix='actXxXxXx_multControls')
    # study.add_task(TaskPlotTorqueParameters, mods=actXxXxXx_multControls, 
    #     suffix='actXxXxXx_multControls')
    for mod in actXxXxXx_multControls:
        study.add_task(TaskAggregateMomentsMod, mod)
        # if (mod == 'actHfAp_multControls'):
            # HfAp_agg_tasks.insert(2, study.tasks[-1])
        study.add_task(TaskPlotMoments, study.tasks[-1], mod=mod)

# Active devices only, fixed direction, fixed value moment arms 
if ((study.momentArms == 'fixed_direction') and 
    (study.whichDevices == 'active_only') and
    (study.fixMomentArms == '0.05')):
    actXxXxXx_fixed = get_exotopology_flags(momentArms='fixed_direction', 
        whichDevices='active_only', fixMomentArms='0.05')[0]
    study.add_task(TaskAggregatePeakPower, mods=actXxXxXx_fixed, 
        suffix='actXxXxXx_fixed')
    study.add_task(TaskAggregateMetabolicRate, mods=actXxXxXx_fixed, 
        suffix='actXxXxXx_fixed')
    study.add_task(TaskPlotDeviceMetabolicRankings, mods=actXxXxXx_fixed, 
        suffix='actXxXxXx_fixed')
    study.add_task(TaskPlotMetabolicReductionVsPeakPower, mods=actXxXxXx_fixed, 
         suffix='actXxXxXx_fixed')

# # All active, free moment arm device solutions
if ((study.momentArms == 'free') and 
    (study.whichDevices == 'active_only') and
    (study.fixMomentArms == '[]')):
    actXXX = get_exotopology_flags(momentArms='free', 
        whichDevices='active_only')[0]
    study.add_task(TaskAggregatePeakPower, mods=actXXX, suffix='actXXX')
    study.add_task(TaskAggregateMetabolicRate, mods=actXXX, suffix='actXXX')
    study.add_task(TaskPlotDeviceMetabolicRankings, mods=actXXX, 
        suffix='actXXX')
    study.add_task(TaskPlotMetabolicReductionVsPeakPower, mods=actXXX, 
        suffix='actXXX')

# Active hip-ankle device, with every passive device combination 
# (free moment arms) 
if ((study.momentArms == 'free') and 
    (study.whichDevices == 'all') and
    (study.fixMomentArms == '[]')):
    actHA_passXXX = get_exotopology_flags(momentArms='free', 
        whichDevices='passive_only', act_combo='actHA')[0]
    study.add_task(TaskAggregatePeakPower, mods=actHA_passXXX, 
        suffix='actHA_passXXX')
    study.add_task(TaskAggregateMetabolicRate, mods=actHA_passXXX, 
        suffix='actHA_passXXX')
    study.add_task(TaskPlotDeviceMetabolicRankings, mods=actHA_passXXX, 
        suffix='actHA_passXXX')

# All variations on hip flexion, ankle plantarflexion strategies
# mods_HfAp1 = ['actHfAp', 'actHfAp_scaledID', 'actHfAp_fixed', 'actHfAp_exp']
# study.add_task(TaskAggregatePeakPower, mods=mods_HfAp1, suffix='actHfAp')
# study.add_task(TaskAggregateMetabolicRate, mods=mods_HfAp1, suffix='actHfAp')
# study.add_task(TaskPlotDeviceMetabolicRankings, mods=mods_HfAp1, 
#     suffix='actHfAp')
# study.add_task(TaskPlotMetabolicReductionVsPeakPower, mods=mods_HfAp1, 
#         suffix='actHfAp')
# mods_HfAp2 = ['actHfAp_scaledID', 'actHfAp_exp']
# study.add_task(TaskAggregateMuscleActivityMod, mods=mods_HfAp2, suffix='actHfAp')
# study.add_task(TaskPlotMuscleActivity, study.tasks[-1], suffix='actHfAp')
# study.add_task(TaskAggregateTorqueParameters, mods=mods_HfAp2, suffix='actHfAp')
# study.add_task(TaskPlotTorqueParameters, mods=mods_HfAp2, suffix='actHfAp')
# for mod in mods_HfAp2:
#         study.add_task(TaskAggregateMomentsMod, mod)
#         # Ordering for WCB plot
#         if (mod == 'actHfAp_scaledID'):
#             HfAp_agg_tasks.insert(0, study.tasks[-1])

#         study.add_task(TaskPlotMoments, study.tasks[-1], mod=mod)
# study.add_task(TaskPlotHipFlexAnklePFMomentComparison, 
#     mod_agg_tasks=HfAp_agg_tasks)


# # Multiple controls post tasks
# actXxXxXx_multControls = ['actHfAp_multControls', 'actHeKe_multControls',
#                           'actKfAp_multControls']


# Parameterized control post tasks
actXxXxXx_paramControls = ['actHfAp_paramControls', 'actHeKe_paramControls',
                           'actKfAp_paramControls']
study.add_task(TaskAggregateMetabolicRate, mods=actXxXxXx_paramControls, 
        suffix='actXxXxXx_paramControls')
study.add_task(TaskPlotDeviceMetabolicRankings, mods=actXxXxXx_paramControls, 
        suffix='actXxXxXx_paramControls')
for mod in actXxXxXx_paramControls:
    study.add_task(TaskAggregateMomentsMod, mod)
    study.add_task(TaskPlotMoments, study.tasks[-1], mod=mod)

## Device comparisons
subjects = ['subject01', 'subject04']

# Hip flexion, ankle plantarflexion
device_list = ['actHfAp_multControls', 'actHfAp']
label_list = ['ind. controls', 'same control, ind. gains']
color_list = ['red', 'blue', 'green']

plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
folder = 'actHfAp'
study.add_task(TaskAggregateMuscleActivityMod, mods=device_list, 
        suffix=folder)
study.add_task(TaskAggregateMetabolicRate, mods=device_list, 
        suffix=folder)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)

# Hip extension, knee extension
device_list = ['actHeKe_multControls', 'actHeKe']
label_list = ['ind. controls', 'same control, ind. gains', 'parameterized']
color_list = ['red', 'blue', 'green']

plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
folder = 'actHeKe'
study.add_task(TaskAggregateMuscleActivityMod, mods=device_list, 
        suffix=folder)
study.add_task(TaskAggregateMetabolicRate, mods=device_list, 
        suffix=folder)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)

# Knee flexion, ankle plantarflexion 
device_list = ['actKfAp_multControls', 'actKfAp']
label_list = ['ind. controls', 'same control, ind. gains']
color_list = ['red', 'blue', 'green']

plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
folder = 'actKfAp'
study.add_task(TaskAggregateMuscleActivityMod, mods=device_list, 
        suffix=folder)
study.add_task(TaskAggregateMetabolicRate, mods=device_list, 
        suffix=folder)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)

# Hip flexion, ankle plantarflexion -- parameterized
device_list = ['actHfAp_multControls', 'actHfAp', 'actHfAp_paramControls']
label_list = ['ind. controls', 'same control, ind. gains', 'parameterized']
color_list = ['red', 'blue', 'green']

plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
folder = 'actHfAp_param'
study.add_task(TaskAggregateMuscleActivityMod, mods=device_list, 
        suffix=folder)
study.add_task(TaskAggregateMetabolicRate, mods=device_list, 
        suffix=folder)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)

solution_list = list()
results_exp_path = os.path.join(study.config['results_path'], 'experiments')
fname_list = ['experiment_walk2_activations.csv',
              'experiment_walk2_activations_Met_mrs_genericMTparams.csv', 
              'experiment_walk2_activations_Met.csv']
for fname in fname_list:
    solution_list.append(os.path.join(results_exp_path, fname))

label_list = ['activations squared', 
              'metabolic energy (generic parameters)',
              'metabolic energy (calibrated parameters)']
color_list = ['red', 'blue', 'green']

plot_lists = dict()
plot_lists['solution_list'] = solution_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
folder = 'muscle_activity_comparison'
study.add_task(TaskAggregateMuscleActivityExperiment, 
    alt_tool_name='mrs_genericMTparams', subjects=subjects)
study.add_task(TaskPlotMuscleActivityComparison, plot_lists, folder, 
    subjects=['subject04'])