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
study.fixMomentArms = '[]'
study.mod_names = get_exotopology_flags(momentArms=study.momentArms, 
    whichDevices=study.whichDevices, fixMomentArms=study.fixMomentArms)[0]
study.muscle_names = ['bifemsh_r', 'med_gas_r', 'glut_max2_r', 'psoas_r',
                      'rect_fem_r','semimem_r','soleus_r','tib_ant_r',
                      'vas_int_r']

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
 
study.add_task(TaskCopyEMGData)

study.add_task(TaskCopyGenericModelFilesToResults)

# Validation
# ----------
study.add_task(TaskValidateAgainstEMG)

# Analysis
# --------
HfAp_agg_tasks = list()

# Global results
study.add_task(TaskAggregateMomentsExperiment)
study.add_task(TaskPlotMoments, study.tasks[-1])
for mod in study.mod_names:
    study.add_task(TaskAggregateMomentsMod, mod)
    if mod == 'actHfAp':
        HfAp_agg_tasks.insert(1, study.tasks[-1])

    study.add_task(TaskPlotMoments, study.tasks[-1], mod=mod)
study.add_task(TaskAggregateMuscleActivity)
study.add_task(TaskPlotMuscleActivity, study.tasks[-1])
study.add_task(TaskAggregateTorqueParameters)
study.add_task(TaskPlotTorqueParameters)

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
    study.add_task(TaskAggregateMuscleActivity, mods=actXxXxXx_multControls, 
        suffix='actXxXxXx_multControls')
    study.add_task(TaskPlotMuscleActivity, study.tasks[-1], 
        suffix='actXxXxXx_multControls')
    study.add_task(TaskAggregateTorqueParameters, mods=actXxXxXx_multControls, 
        suffix='actXxXxXx_multControls')
    study.add_task(TaskPlotTorqueParameters, mods=actXxXxXx_multControls, 
        suffix='actXxXxXx_multControls')
    for mod in actXxXxXx_multControls:
        study.add_task(TaskAggregateMomentsMod, mod)
        if (mod == 'actHfAp_multControls'):
            HfAp_agg_tasks.insert(2, study.tasks[-1])
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
mods_HfAp1 = ['actHfAp', 'actHfAp_scaledID', 'actHfAp_fixed', 'actHfAp_exp']
study.add_task(TaskAggregatePeakPower, mods=mods_HfAp1, suffix='actHfAp')
study.add_task(TaskAggregateMetabolicRate, mods=mods_HfAp1, suffix='actHfAp')
study.add_task(TaskPlotDeviceMetabolicRankings, mods=mods_HfAp1, 
    suffix='actHfAp')
study.add_task(TaskPlotMetabolicReductionVsPeakPower, mods=mods_HfAp1, 
        suffix='actHfAp')
mods_HfAp2 = ['actHfAp_scaledID', 'actHfAp_exp']
study.add_task(TaskAggregateMuscleActivity, mods=mods_HfAp2, suffix='actHfAp')
study.add_task(TaskPlotMuscleActivity, study.tasks[-1], suffix='actHfAp')
study.add_task(TaskAggregateTorqueParameters, mods=mods_HfAp2, suffix='actHfAp')
study.add_task(TaskPlotTorqueParameters, mods=mods_HfAp2, suffix='actHfAp')
for mod in mods_HfAp2:
        study.add_task(TaskAggregateMomentsMod, mod)
        # Ordering for WCB plot
        if (mod == 'actHfAp_scaledID'):
            HfAp_agg_tasks.insert(0, study.tasks[-1])

        study.add_task(TaskPlotMoments, study.tasks[-1], mod=mod)
study.add_task(TaskPlotHipFlexAnklePFMomentComparison, 
    mod_agg_tasks=HfAp_agg_tasks)



