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
study.momentArms = 'fixed_direction'
study.whichDevices = 'active_only'

# Flag whether or not to include fixed moment arm trials
study.fixMomentArms = True

# 'Default' (activations squared) or 'Met'. Default is assumed if omitted.
study.costFunction = 'Met' 
study.mod_names = get_exotopology_flags(study)[0]
mod_names_exp = [s.replace('act','exp') for s in study.mod_names]
study.mod_names = study.mod_names + mod_names_exp
if study.fixMomentArms:
    mod_names_fixed = [s + '_fixed' for s in study.mod_names]
    study.mod_names = study.mod_names + mod_names_fixed


study.dof_names = ['hip_flexion_r','knee_angle_r','ankle_angle_r']
study.muscle_names = ['bifemsh_r', 'med_gas_r', 'glut_max2_r', 'psoas_r',
                      'rect_fem_r','semimem_r','soleus_r','tib_ant_r',
                      'vas_int_r']
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
study.add_task(TaskCopyEMGData)
study.add_task(TaskCopyGenericModelFilesToResults)

# Validation
# ----------
study.add_task(TaskValidateAgainstEMG)

# Analysis
# --------

# Experiment results
study.add_task(TaskAggregateMuscleDataExperiment)
study.add_task(TaskPlotMuscleData, study.tasks[-1])
study.add_task(TaskAggregateMomentsExperiment)
study.add_task(TaskPlotMoments, study.tasks[-1])


## Device comparisons
subjects = ['subject01','subject04', 'subject18']
master_device_list = list()

# Hip flexion, knee flexion, ankle plantarflexion
device_list = ['mrsmod_actHfKfAp_multControls',
               'mrsmod_actHfKfAp',
               'mrsmod_actHfAp',
               'mrsmod_actHfKf',
               'mrsmod_actKfAp', 
               'mrsmod_actHf_fixed', 
               'mrsmod_actKf_fixed', 
               'mrsmod_actAp_fixed']
master_device_list  = list(set().union(master_device_list, device_list))
label_list = ['hip flex. + knee flex. + ankle pl. (ind)',
              'hip flex. + knee flex. + ankle pl.',
              'hip flex. + ankle pl.',
              'hip flex. + knee flex.',
              'knee flex. + ankle pl.',
              'hip flex.', 
              'knee flex.',
              'ankle pl.']
color_list = ['saddlebrown','darkolivegreen', 'darkslateblue', 'darkseagreen', 
              'darkkhaki','darkred', 'darkorange','gold']

plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
folder = 'HfKfAp'
study.add_task(TaskAggregateDevicePower, device_list, suffix=folder)
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)

# FitReOpt: hip flexion, knee flexion, ankle plantarflexion
device_list_group2 = ['fitreopt_zhang2017_actHf_fixed', 
                      'fitreopt_zhang2017_actKf_fixed', 
                      'fitreopt_zhang2017_actAp_fixed',
                      'fitreopt_zhang2017_actHfKf',
                      'fitreopt_zhang2017_actKfAp',
                      'fitreopt_zhang2017_actHfAp',
                      'fitreopt_zhang2017_actHfKfAp']
master_device_list  = list(set().union(master_device_list, device_list_group2))
label_list_group2 = ['hip flex.', 
                     'knee flex.',
                     'ankle pl.',
                     'hip flex. + \nknee flex.',
                     'knee flex. + \nankle pl.',
                     'hip flex. + \nankle pl.',
                     'hip flex. + \nknee flex. + \nankle pl.']
color_list_group2 = ['darkred', 'darkorange', 'gold', 'darkseagreen', 
  'darkkhaki', 'darkslateblue', 'darkolivegreen']

plot_lists = dict()
plot_lists['device_list'] = device_list_group2
plot_lists['label_list'] = label_list_group2
plot_lists['color_list'] = color_list_group2
folder = 'fitreopt_HfKfAp'
study.add_task(TaskAggregateDevicePower, device_list_group2, suffix=folder)
study.add_task(TaskAggregateMetabolicRate, device_list_group2, suffix=folder)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)
study.add_task(TaskPlotMetabolicsVsParameters, plot_lists, folder,
    subjects=subjects)

# Hip extension, knee extension
device_list = ['mrsmod_actHeKe_multControls',
               'mrsmod_actHeKe', 
               'mrsmod_actHe_fixed', 
               'mrsmod_actKe_fixed']
master_device_list  = list(set().union(master_device_list, device_list))
label_list = ['hip ext. + knee ext. (ind)', 
              'hip ext. + knee ext.', 
              'hip ext.', 
              'knee ext.']
color_list = ['deeppink','darkorchid', 'darksalmon', 'darkcyan']


plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
folder = 'HeKe'
study.add_task(TaskAggregateDevicePower, device_list, suffix=folder)
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)

# FitReOpt: hip extension, knee extension
device_list_group1 = ['fitreopt_zhang2017_actHe_fixed', 
                      'fitreopt_zhang2017_actKe_fixed',
                      'fitreopt_zhang2017_actHeKe']
master_device_list  = list(set().union(master_device_list, device_list_group1))
label_list_group1 = ['hip ext.', 
                     'knee ext.',
                     'hip ext. + knee ext.']
color_list_group1 = ['darksalmon', 'darkcyan', 'darkorchid']

plot_lists = dict()
plot_lists['device_list'] = device_list_group1
plot_lists['label_list'] = label_list_group1
plot_lists['color_list'] = color_list_group1
folder = 'fitreopt_HeKe'
study.add_task(TaskAggregateDevicePower, device_list_group1, suffix=folder)
study.add_task(TaskAggregateMetabolicRate, device_list_group1, suffix=folder)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)
study.add_task(TaskPlotMetabolicsVsParameters, plot_lists, folder,
    subjects=subjects)

# All devices
device_list = device_list_group1 + device_list_group2
device_list = [name.replace('fitreopt_zhang2017', 'mrsmod') 
               for name in device_list]
color_list = color_list_group1 + color_list_group2
label_list = label_list_group1 + color_list_group2
plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
folder = 'all'
study.add_task(TaskAggregateDevicePower, device_list, suffix=folder)
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder)
study.add_task(TaskPlotMetabolicsVsParameters, plot_lists, folder,
    subjects=subjects)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)

device_list = device_list_group1 + device_list_group2
color_list = color_list_group1 + color_list_group2
label_list = label_list_group1 + label_list_group2
plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
folder = 'fitreopt_all'
study.add_task(TaskAggregateDevicePower, device_list, suffix=folder)
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder)
study.add_task(TaskPlotMetabolicsVsParameters, plot_lists, folder,
    subjects=subjects)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)

all_fitreopt_device_list = device_list
all_fitreopt_color_list = color_list
all_fitreopt_label_list = label_list
all_fitreopt_lists = zip(all_fitreopt_device_list, all_fitreopt_color_list,
  all_fitreopt_label_list)

fix_param_tags = ['fix_peak_torque', 'fix_peak_time', 'fix_rise_time', 
  'fix_fall_time', 'fix_all_times', 'fix_all_torques']
def create_param_label_from_tag(tag):
    tag_spaces = tag.replace('_', ' ')
    return '(' + tag_spaces + ')'

for device, color, label in all_fitreopt_lists:
  plot_lists = dict()
  device_list = list()
  color_list = list()
  label_list = list()
  device_list.append(device)
  label_list.append(label)
  color_list.append(color)
  for tag in fix_param_tags:

    param_names, param_bounds = get_parameter_info(study, device)
    if (tag == 'fix_all_torques') and (len(param_names) == 4): continue

    device_list.append(device + '/' + tag)
    label_list.append(label + ' ' + create_param_label_from_tag(tag))
    color_list.append(color)

  plot_lists['device_list'] = device_list
  master_device_list  = list(set().union(master_device_list, device_list))
  plot_lists['label_list'] = label_list
  plot_lists['color_list'] = color_list
  folder = device

  study.add_task(TaskAggregateDevicePower, device_list, suffix=folder)
  study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder)
  study.add_task(TaskPlotMetabolicsVsParameters, plot_lists, folder,
    subjects=subjects)
  study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)

#  Fixed parmeter metabolic changes
device_list = all_fitreopt_device_list
label_list = all_fitreopt_label_list
color_list = all_fitreopt_color_list
fixed_param_list = ['fix_peak_torque', 'fix_peak_time', 'fix_rise_time',
                    'fix_fall_time', 'fix_all_times']
met_device_list = list(device_list)
for device in device_list:
  for param in fixed_param_list:
    met_device_list.append(device + '/' + param)

plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
plot_lists['fixed_param_list'] = fixed_param_list
folder = 'fixed_param'
study.add_task(TaskAggregateMetabolicRate, met_device_list, suffix=folder)
study.add_task(TaskPlotMetabolicsForFixedParameters, plot_lists, folder, 
  subjects=subjects)

#  FitReOpt: temp folder
device_list = [      'fitreopt_zhang2017_actKfAp',
                      'fitreopt_zhang2017_actHfAp', 
                      'fitreopt_zhang2017_actHfKfAp']
label_list = [  'knee flex. + ankle pl.',
                     'hip flex. + ankle pl.',
                     'hip flex. + knee flex. + ankle pl.']
color_list = ['darkkhaki',
                     'darkslateblue',
                    'darkolivegreen' ]

plot_lists = dict()
plot_lists['device_list'] = device_list
plot_lists['label_list'] = label_list
plot_lists['color_list'] = color_list
folder = 'fitreopt_temp'
study.add_task(TaskAggregateDevicePower, device_list, suffix=folder)
study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder)
study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)
study.add_task(TaskPlotMetabolicsVsParameters, plot_lists, folder,
    subjects=subjects)

for mod in master_device_list:
    study.add_task(TaskAggregateMomentsMod, mod)
    study.add_task(TaskPlotMoments, study.tasks[-1])
    study.add_task(TaskAggregateMuscleDataMod, mod)
    study.add_task(TaskPlotMuscleData, study.tasks[-1])
    if 'fitreopt' in mod:
      study.add_task(TaskAggregateTorqueParameters, mod)
      study.add_task(TaskPlotTorqueParameters, study.tasks[-1])


################## CODE TO BE REFACTORED ######################

# # FitOpt: hermite-simpson / hip flexion, ankle plantarflexion
# num_params = 8
# nodes_list = range(2, num_params+1)
# device_list = ['fitopt_zhang2017_actHfAp_fixed/params_4'] + \
#               ['fitopt_hermite_actHfAp_fixed/params_%s' 
#                 % n for n in nodes_list] + \
#               ['mrsmod_actHfAp_fixed']
# # master_device_list  = list(set().union(master_device_list, device_list))
# label_list = ['Zhang2017'] + \
#              ['%s nodes' % n for n in nodes_list] + ['optimized']
# from matplotlib import cm
# color_list = ['red'] + \
#              [cm.viridis(rgb) for rgb 
#               in np.linspace(0, 1, len(nodes_list))] + \
#              ['pink']

# plot_lists = dict()
# plot_lists['device_list'] = device_list
# plot_lists['label_list'] = label_list
# plot_lists['color_list'] = color_list
# folder = 'fitopt_hermite_actHfAp_fixed'
# study.add_task(TaskAggregateDevicePower, device_list, suffix=folder)
# study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder)
# study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)

# # FitOpt: legendre / hip flexion, ankle plantarflexion
# num_params = 10
# nodes_list = range(4, num_params+1)
# device_list = ['fitopt_zhang2017_actHfAp_fixed/params_4'] + \
#               ['fitopt_legendre_actHfAp_fixed/params_%s' % n for n in nodes_list] + \
#               ['mrsmod_actHfAp_fixed']
# # master_device_list  = list(set().union(master_device_list, device_list))
# label_list = ['Zhang2017'] + ['%s nodes' % n for n in nodes_list] + ['optimized']
# from matplotlib import cm
# color_list = ['red'] + \
#              [cm.viridis(rgb) for rgb in np.linspace(0, 1, len(nodes_list))] + \
#              ['pink']

# plot_lists = dict()
# plot_lists['device_list'] = device_list
# plot_lists['label_list'] = label_list
# plot_lists['color_list'] = color_list
# folder = 'fitopt_legendre_actHfAp_fixed'
# study.add_task(TaskAggregateDevicePower, device_list, suffix=folder)
# study.add_task(TaskAggregateMetabolicRate, device_list, suffix=folder)
# study.add_task(TaskPlotDeviceComparison, plot_lists, folder, subjects=subjects)


# Active devices only, fixed direction, free value moment arms
# if ((study.momentArms == 'fixed_direction') and 
#     (study.whichDevices == 'active_only')):
#     actXxXxXx = get_exotopology_flags(study)[0]
#     study.add_task(TaskAggregateDevicePower, actXxXxXx, suffix='actXxXxXx')
#     study.add_task(TaskAggregateMetabolicRate, actXxXxXx, suffix='actXxXxXx')
#     study.add_task(TaskPlotDeviceMetabolicRankings, actXxXxXx, 
#         suffix='actXxXxXx')
#     study.add_task(TaskPlotMetabolicReductionVsPeakPower, actXxXxXx, 
#         suffix='actXxXxXx')
#     actXxXxXx_multControls = get_mult_controls_mod_names(study)
#     study.add_task(TaskAggregateDevicePower, actXxXxXx_multControls, 
#         suffix='actXxXxXx_multControls')
#     study.add_task(TaskAggregateMetabolicRate, actXxXxXx_multControls, 
#         suffix='actXxXxXx_multControls')
#     study.add_task(TaskPlotDeviceMetabolicRankings, actXxXxXx_multControls, 
#         suffix='actXxXxXx_multControls')
#     study.add_task(TaskPlotMetabolicReductionVsPeakPower, 
#         actXxXxXx_multControls, 
#         suffix='actXxXxXx_multControls')
    # study.add_task(TaskAggregateMuscleDataMod, actXxXxXx_multControls, 
    #     suffix='actXxXxXx_multControls')
    # study.add_task(TaskPlotMuscleData, study.tasks[-1], 
    #     suffix='actXxXxXx_multControls')
    # study.add_task(TaskAggregateTorqueParameters, actXxXxXx_multControls, 
    #     suffix='actXxXxXx_multControls')
    # study.add_task(TaskPlotTorqueParameters, actXxXxXx_multControls, 
    #     suffix='actXxXxXx_multControls')
    # for mod in actXxXxXx_multControls:
    #     study.add_task(TaskAggregateMomentsMod, mod)
    #     study.add_task(TaskPlotMoments, study.tasks[-1], mod=mod)

# Active devices only, fixed direction, fixed value moment arms 
# if ((study.momentArms == 'fixed_direction') and 
#     (study.whichDevices == 'active_only')):
#     actXxXxXx_fixed = get_exotopology_flags(study)[0]
#     study.add_task(TaskAggregateDevicePower, actXxXxXx_fixed, 
#         suffix='actXxXxXx_fixed')
#     study.add_task(TaskAggregateMetabolicRate, actXxXxXx_fixed, 
#         suffix='actXxXxXx_fixed')
#     study.add_task(TaskPlotDeviceMetabolicRankings, actXxXxXx_fixed, 
#         suffix='actXxXxXx_fixed')
#     study.add_task(TaskPlotMetabolicReductionVsPeakPower, actXxXxXx_fixed, 
#          suffix='actXxXxXx_fixed')

# Parameterized control post tasks
# actXxXxXx_paramControls = ['mrsmod_actHfAp_paramControls', 
#                            'mrsmod_actHeKe_paramControls',
#                            'mrsmod_actKfAp_paramControls']
# study.add_task(TaskAggregateMetabolicRate, actXxXxXx_paramControls, 
#         suffix='actXxXxXx_paramControls')
# study.add_task(TaskPlotDeviceMetabolicRankings, actXxXxXx_paramControls, 
#         suffix='actXxXxXx_paramControls')
# for mod in actXxXxXx_paramControls:
#     study.add_task(TaskAggregateMomentsMod, mod)
#     study.add_task(TaskPlotMoments, study.tasks[-1], mod=mod)

# # # All active, free moment arm device solutions
# if ((study.momentArms == 'free') and 
#     (study.whichDevices == 'active_only')):
#     actXXX = get_exotopology_flags(study)[0]
#     study.add_task(TaskAggregateDevicePower, actXXX, suffix='actXXX')
#     study.add_task(TaskAggregateMetabolicRate, actXXX, suffix='actXXX')
#     study.add_task(TaskPlotDeviceMetabolicRankings, actXXX, 
#         suffix='actXXX')
#     study.add_task(TaskPlotMetabolicReductionVsPeakPower, actXXX, 
#         suffix='actXXX')


# # Active hip-ankle device, with every passive device combination 
# # (free moment arms) 
# if ((study.momentArms == 'free') and 
#     (study.whichDevices == 'all')):
#     actHA_passXXX = get_exotopology_flags(study)[0]
#     study.add_task(TaskAggregateDevicePower, actHA_passXXX, 
#         suffix='actHA_passXXX')
#     study.add_task(TaskAggregateMetabolicRate, actHA_passXXX, 
#         suffix='actHA_passXXX')
#     study.add_task(TaskPlotDeviceMetabolicRankings, actHA_passXXX, 
#         suffix='actHA_passXXX')