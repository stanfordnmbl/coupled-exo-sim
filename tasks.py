import os
import time

import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

import osimpipeline as osp
from osimpipeline import utilities as util
from osimpipeline import postprocessing as pp
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

class working_directory():
    """Use this to temporarily run code with some directory as a working
    directory and to then return to the original working directory::

        with working_directory('<dir>'):
            pass
    """
    def __init__(self, path):
        self.path = path
        self.original_working_dir = os.getcwd()
    def __enter__(self):
        os.chdir(self.path)
    def __exit__(self, *exc_info):
        os.chdir(self.original_working_dir)

class TaskCopyMotionCaptureData(osp.TaskCopyMotionCaptureData):
    REGISTRY = []
    def __init__(self, study, walk100=None, walk125=None, walk150=None,
        walk175=None, run200=None, run300=None, run400=None, run500=None):
        regex_replacements = list()

        default_args = dict()
        default_args['walk100'] = walk100
        default_args['walk125'] = walk125
        default_args['walk150'] = walk150
        default_args['walk175'] = walk175
        default_args['run200'] = run200
        default_args['run300'] = run300
        default_args['run400'] = run400
        default_args['run500'] = run500

        for subject in study.subjects:

            cond_args = subject.cond_args            
            if 'walk100' in cond_args: walk100 = cond_args['walk100']
            else: walk100 = default_args['walk100']

            if 'walk125' in cond_args: walk125 = cond_args['walk125']
            else: walk125 = default_args['walk125']

            if 'walk150' in cond_args: walk150 = cond_args['walk150']
            else: walk150 = default_args['walk150']

            if 'walk175' in cond_args: walk175 = cond_args['walk175']
            else: walk175 = default_args['walk175']

            if 'run200' in cond_args: run200 = cond_args['run200']
            else: run200 = default_args['run200']

            if 'run300' in cond_args: run300 = cond_args['run300']
            else: run300 = default_args['run300']

            if 'run400' in cond_args: run400 = cond_args['run400']
            else: run400 = default_args['run400']

            if 'run500' in cond_args: run500 = cond_args['run500']
            else: run500 = default_args['run500']

            for datastr, condname, arg in [
                    ('Walk_100', 'walk1', walk100),
                    ('Walk_125', 'walk2', walk125),
                    ('Walk_150', 'walk3', walk150),
                    ('Walk_175', 'walk4', walk175),
                    ('Run_200', 'run1', run200),
                    ('Run_300', 'run2', run300),
                    ('Run_400', 'run3', run400),
                    ('Run_500', 'run4', run500)]:
                # Marker trajectories.
                regex_replacements.append(
                    (
                        os.path.join(subject.name, 'Data',
                            '%s %02i.trc' % (datastr, arg[0])).replace('\\',
                            '\\\\'),
                        os.path.join('experiments',
                            subject.name, condname, 'expdata', 
                            'marker_trajectories.trc').replace('\\','\\\\')
                        ))
                # Ground reaction.
                regex_replacements.append(
                    (
                        os.path.join(subject.name, 'Data',
                            '%s %02i%s.mot' % (datastr, arg[0],arg[1])).replace(
                                '\\','\\\\'),
                        os.path.join('experiments', subject.name, condname,
                            'expdata','ground_reaction_orig.mot').replace(
                                '\\','\\\\') 
                        )) 
                # EMG
                regex_replacements.append(
                    (
                        os.path.join(subject.name, 'Results', datastr,
                            '%s%02i_gait_controls.sto' % (datastr, arg[0])
                            ).replace('\\','\\\\'),
                        os.path.join('experiments', subject.name,
                            condname, 'expdata', 'emg_with_headers.sto'
                            ).replace('\\','\\\\')
                        ))
            regex_replacements.append((
                        os.path.join(subject.name, 'Data',
                            'Static_FJC.trc').replace('\\','\\\\'),
                        os.path.join('experiments', subject.name, 'static',
                            'expdata',
                            'marker_trajectories.trc').replace('\\','\\\\') 
                        ))

        super(TaskCopyMotionCaptureData, self).__init__(study,
                regex_replacements)

class TaskRemoveEMGFileHeaders(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study):
        super(TaskRemoveEMGFileHeaders, self).__init__(study)
        self.name = '%s_remove_emg_file_headers' % study.name
        self.doc = 'Remove headers from EMG data files.'

        file_dep = list()
        target = list()
        for subject in study.subjects:
            for cond in subject.conditions:
                if cond.name == 'static': continue
                file_dep += [os.path.join(subject.results_exp_path,
                    cond.name, 'expdata', 'emg_with_headers.sto')]
                target += [os.path.join(subject.results_exp_path,
                    cond.name, 'expdata', 'emg.sto')]

        self.add_action(file_dep, target, self.remove_file_headers)

    def remove_file_headers(self, file_dep, target):
        
        for i, fpath in enumerate(file_dep):
            infile = open(fpath, 'r').readlines()
            if os.path.isfile(target[i]):
                print 'File ' + target[i] + ' already exists. Deleting...'
                os.remove(target[i])
            print 'Writing to ' + target[i]
            with open(target[i], 'w') as outfile:
                prev_line = ''
                writing = False
                for index, line in enumerate(infile):
                    if 'endheader' in prev_line:
                        writing = True

                    if writing:
                        outfile.write(line)

                    prev_line = line

class TaskUpdateGroundReactionColumnLabels(osp.TrialTask):
    REGISTRY = []
    def __init__(self, trial):
        super(TaskUpdateGroundReactionColumnLabels, self).__init__(trial)
        self.name = trial.id + '_update_grf_column_labels'
        self.add_action(
                [os.path.join(trial.expdata_path, 'ground_reaction_orig.mot')],
                [trial.ground_reaction_fpath],
                self.dispatch)
    def dispatch(self, file_dep, target):
        from perimysium.dataman import storage2numpy, ndarray2storage
        import re
        data = storage2numpy(file_dep[0])
        new_names = list()
        for name in data.dtype.names:
            if name == 'time':
                new_name = name
            elif name.endswith('_1'):
                new_name = re.sub('ground_(.*)_(.*)_1', 'ground_\\1_l_\\2',
                        name)
            else:
                new_name = re.sub('ground_(.*)_(.*)', 'ground_\\1_r_\\2',
                        name)
            new_names.append(new_name)
        data.dtype.names = new_names
        ndarray2storage(data, target[0])

class TaskScaleMuscleMaxIsometricForce(osp.SubjectTask):
    REGISTRY = []
    def __init__(self, subject):
        super(TaskScaleMuscleMaxIsometricForce, self).__init__(subject)
        self.subject = subject
        self.name = '%s_scale_max_force' % self.subject.name
        self.doc = 'Scale subject muscle Fmax parameters from Handsfield2014'
        self.generic_model_fpath = self.study.source_generic_model_fpath
        self.subject_model_fpath = os.path.join(self.subject.results_exp_path, 
            '%s.osim' % self.subject.name)
        self.scaled_param_model_fpath = os.path.join(
            self.subject.results_exp_path, 
            '%s_scaled_Fmax.osim' % self.subject.name)

        self.add_action([self.generic_model_fpath, self.subject_model_fpath],
                        [self.scaled_param_model_fpath],
                        self.scale_model_parameters)

    def scale_model_parameters(self, file_dep, target):
        """From Handsfields 2014 figure 5a and from Apoorva's muscle properties
       spreadsheet.
       
       v: volume fraction
       V: total volume
       F: max isometric force
       l: optimal fiber length

       F = v * sigma * V / l

       *_g: generic model.
       *_s: subject-specific model.

       F_g = v * sigma * V_g / l_g
       F_s = v * sigma * V_s / l_s

       F_s = (F_g * l_g / V_g) * V_s / l_s
           = F_g * (V_s / V_g) * (l_g / l_s)

        Author: Chris Dembia 
        Borrowed from mrsdeviceopt GitHub repo:
        https://github.com/chrisdembia/mrsdeviceopt          
       """

        print("Muscle force scaling: "
              "total muscle volume and optimal fiber length.")

        def total_muscle_volume_regression(mass):
            return 91.0*mass + 588.0

        generic_TMV = total_muscle_volume_regression(75.337)
        subj_TMV = total_muscle_volume_regression(self.subject.mass)

        import opensim as osm
        generic_model = osm.Model(file_dep[0])
        subj_model = osm.Model(file_dep[1])

        generic_mset = generic_model.getMuscles()
        subj_mset = subj_model.getMuscles()

        for im in range(subj_mset.getSize()):
            muscle_name = subj_mset.get(im).getName()

            generic_muscle = generic_mset.get(muscle_name)
            subj_muscle = subj_mset.get(muscle_name)

            generic_OFL = generic_muscle.get_optimal_fiber_length()
            subj_OFL = subj_muscle.get_optimal_fiber_length()

            scale_factor = (subj_TMV / generic_TMV) * (generic_OFL / subj_OFL)
            print("Scaling '%s' muscle force by %f." % (muscle_name,
                scale_factor))

            generic_force = generic_muscle.getMaxIsometricForce()
            scaled_force = generic_force * scale_factor
            subj_muscle.setMaxIsometricForce(scaled_force)

        subj_model.printToXML(target[0])

class TaskValidateMarkerErrors(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, cond_names=['walk1','walk2','walk3','walk4']):
        super(TaskValidateMarkerErrors, self).__init__(study)
        suffix = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'validate_marker_errors%s' % suffix
        self.doc = 'Compute marker errors across subjects and conditions.'
        self.results_path = study.config['results_path']
        self.validate_path = os.path.join(study.config['validate_path'],
            'marker_errors')
        self.cond_names = cond_names
        self.subjects = study.subjects

        errors_fpaths = list()
        for cond_name in cond_names:
            for subject in study.subjects:
                errors_fpaths.append(os.path.join(self.results_path, 
                    'experiments', subject.name, cond_name, 'ik', 
                    'marker_error.csv'))
                
        val_fname = os.path.join(self.validate_path, 'marker_errors.txt')
        self.add_action(errors_fpaths, [val_fname],
                        self.validate_marker_errors)

    def validate_marker_errors(self, file_dep, target):
        if not os.path.isdir(self.validate_path): os.mkdir(self.validate_path)

        all_errors = pd.DataFrame()
        for file in file_dep:
            df = pd.read_csv(file)
            df = df.drop(columns=['Unnamed: 0', 'time'])
            all_errors = pd.concat([all_errors, df])

        numMarkers = all_errors.shape[1]
        all_errors_sq = all_errors.pow(2)
        all_errors_sq_sum_norm = all_errors_sq.sum(axis=1) / float(numMarkers)
        rms_errors = all_errors_sq_sum_norm.pow(0.5)

        peak_rms = rms_errors.max()
        mean_rms = rms_errors.mean()

        with open(target[0],"w") as f:
            f.write('subjects: ')
            for isubj, subject in enumerate(self.subjects):
                if isubj:
                    f.write(', %s' % subject.name)
                else:
                    f.write('%s' % subject.name)
            f.write('\n')
            f.write('conditions: ')
            for icond, cond_name in enumerate(self.cond_names):
                if icond:
                    f.write(', %s' % subject)
                else:
                    f.write('%s' % cond_name)
            f.write('\n')
            f.write('Peak RMS marker error: %1.3f cm \n' % (100*peak_rms))
            f.write('Mean RMS marker error: %1.3f cm \n' % (100*mean_rms))

class TaskValidateKinetics(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, cond_names=['walk1','walk2','walk3','walk4'],
            gen_forces=['hip_flexion_r_moment', 
                        'knee_angle_r_moment',
                        'ankle_angle_r_moment']):
        super(TaskValidateKinetics, self).__init__(study)
        suffix = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'validate_kinetics%s' % suffix
        self.doc = 'Validate joint moments and residuals across subjects and conditions.'
        self.results_path = study.config['results_path']
        self.validate_path = os.path.join(study.config['validate_path'],
            'kinetics')
        self.cond_names = cond_names
        self.subjects = study.subjects
        self.gen_forces = gen_forces

        colors = plt.cm.jet(np.linspace(0, 1, len(study.subjects)))
        masses = list()
        for subject in study.subjects:
            masses.append(subject.mass)

        for cond_name in cond_names:
            moments_fpaths = list()
            gait_events = list()
            isubjs = list()
            for isubj, subject in enumerate(study.subjects):
                moments_fpaths.append(os.path.join(self.results_path, 
                    'experiments', subject.name, cond_name, 'id', 'results', 
                    '%s_%s_%s_id_solution.sto' % (study.name, subject.name, 
                        cond_name)))
                condition = subject.get_condition(cond_name)
                trial = condition.get_trial(1)
                cycles = trial.get_cycles()
                gait_events_this_cond = list()
                for cycle in cycles:
                    gait_events_this_cond.append((cycle.start, cycle.end))
                gait_events.append(gait_events_this_cond)
                isubjs.append(isubj)

            joint_moments_fname = os.path.join(self.validate_path, 
                    'joint_moments_%s.pdf' % cond_name)
            self.add_action(moments_fpaths, [joint_moments_fname], 
                            self.validate_joint_moments, gait_events, isubjs, 
                            colors, masses)

    def validate_joint_moments(self, file_dep, target, gait_events, isubjs,
            colors, masses):
        if not os.path.isdir(self.validate_path): os.mkdir(self.validate_path)

        fig = pl.figure(figsize=(8, 8))
        ind_axes = list()
        mean_axes = list()
        dfs = list()
        for iforce, gen_force in enumerate(self.gen_forces):
            idx = iforce + 1
            ind_axes.append(fig.add_subplot(len(self.gen_forces), 2, 2*idx-1))
            mean_axes.append(fig.add_subplot(len(self.gen_forces), 2, 2*idx))
            dfs.append(pd.DataFrame())

        for file, gait_events_cond, isubj in zip(file_dep, gait_events, isubjs):
            id = pp.storage2numpy(file)
            time = id['time']
            for gait_event in gait_events_cond:
                start = np.argmin(abs(time-gait_event[0]))
                end = np.argmin(abs(time-gait_event[1]))
                for iforce, gen_force in enumerate(self.gen_forces):
                    force = id[gen_force][start:end] / masses[isubj]
                    new_time = np.linspace(time[start], time[end], 101)
                    force_interp = np.interp(new_time, time[start:end], force)
                    dfs[iforce] = pd.concat([dfs[iforce], 
                            pd.DataFrame(force_interp)], axis=1)

                    pgc = np.linspace(0, 100, 101)
                    ind_axes[iforce].plot(pgc, force_interp, color=colors[isubj])
                    ind_axes[iforce].set_ylabel('%s (N-m/kg)' % gen_force)

        for iforce, gen_force in enumerate(self.gen_forces):
            force_mean = dfs[iforce].mean(axis=1)
            force_std = dfs[iforce].std(axis=1)
            mean_axes[iforce].fill_between(pgc, force_mean-force_std, 
                force_mean+force_std, color='black', alpha=0.2)
            mean_axes[iforce].plot(pgc, force_mean, color='black')
            if iforce == len(self.gen_forces)-1:
                mean_axes[iforce].set_xlabel('gait cycle (%)')
                ind_axes[iforce].set_xlabel('gait cycle (%)')

        fig.tight_layout()
        fig.savefig(target[0])
        pl.close(fig)

def get_muscle_parameters_as_list(fpath):

    import h5py
    hdf_output = h5py.File(fpath, 'r')
    params = hdf_output['OptInfo']['paramCal']

    lMo = list()
    lTs = list()
    alf = list()
    e0 = list()
    musc_names = list()
    for musc in params:
        musc_names.append(musc)
        for param in params[musc]:
            if param == 'optimal_fiber_length':
                lMo.append(params[musc][param][0][0])
            elif param == 'tendon_slack_length':
                lTs.append(params[musc][param][0][0])
            elif param == 'pennation_angle':
                alf.append(params[musc][param][0][0])
            elif param == 'muscle_strain':
                e0.append(params[musc][param][0][0])

    return lMo, lTs, alf, e0, musc_names

def get_muscle_parameters_as_dict(fpath):

    import h5py
    hdf_output = h5py.File(fpath, 'r')
    params = hdf_output['OptInfo']['paramCal']

    lMo = dict()
    lTs = dict()
    alf = dict()
    e0 = dict()
    musc_names = list()
    for musc in params:
        musc_names.append(musc)
        for param in params[musc]:
            if param == 'optimal_fiber_length':
                lMo[musc] = params[musc][param][0][0]
            elif param == 'tendon_slack_length':
                lTs[musc] = params[musc][param][0][0]
            elif param == 'pennation_angle':
                alf[musc] = params[musc][param][0][0]
            elif param == 'muscle_strain':
                e0[musc] = params[musc][param][0][0]

    return lMo, lTs, alf, e0, musc_names

class TaskCalibrateParametersMultiPhaseSetup(osp.SetupTask):
    REGISTRY = []
    # TODO: the specific trial is ignored in this task, since this is 
    # really a subject task. 
    def __init__(self, trial, conditions, cycles, param_dict, cost_dict, 
            passive_precalibrate=False, **kwargs):
        super(TaskCalibrateParametersMultiPhaseSetup, self).__init__(
            'calibrate_multiphase', trial, **kwargs)

        self.doc = "Create a setup file for a mulit-phase parameter " \
                   "calibration tool."
        self.path = os.path.join(self.subject.results_exp_path, 'calibrate')
        self.name = '%s_calibrate_multiphase_setup' % self.subject.name

        self.kinematics_files = list()
        self.rel_kinematics_files = list()
        self.kinetics_files = list()
        self.rel_kinetics_files = list()
        self.emg_files = list()
        self.rel_emg_files = list()
        self.init_times = list()
        self.final_times = list()
        self.cycle_ids = list()
        self.trial_results_exp_paths = list()

        for cond in self.subject.conditions:
            if not (cond.name in conditions): continue
            # Assume only trial per condition
            trial = cond.trials[0]
            for cycle in trial.cycles:
                if not(cycle.name in cycles): continue

                kinematics_file = os.path.join(trial.results_exp_path, 'ik', 
                    '%s_%s_ik_solution.mot' % (self.study.name, trial.id))
                self.kinematics_files.append(kinematics_file)
                self.rel_kinematics_files.append(
                        os.path.relpath(kinematics_file, self.path)
                    )

                kinetics_file = os.path.join(trial.results_exp_path, 'id', 
                    'results', '%s_%s_id_solution.sto' % (self.study.name, 
                        trial.id))
                self.kinetics_files.append(kinetics_file)
                self.rel_kinetics_files.append(
                        os.path.relpath(kinetics_file, self.path)
                    )

                emg_file = os.path.join(trial.results_exp_path, 'expdata', 
                    'emg.sto')
                self.emg_files.append(emg_file)
                self.rel_emg_files.append(
                        os.path.relpath(emg_file, self.path)
                    )
                self.init_times.append(cycle.start)
                self.final_times.append(cycle.end)

                self.cycle_ids.append(cycle.id)
                self.trial_results_exp_paths.append(trial.results_exp_path)

        self.results_setup_fpath = os.path.join(self.path, 'setup.m')
        self.results_output_fpath = os.path.join(self.path, 
            '%s_%s_calibrate.mat' % (self.study.name, self.subject.name))
        self.param_dict = param_dict
        self.cost_dict = cost_dict
        self.passive_precalibrate = passive_precalibrate

        # Fill out setup.m template and write to results directory
        self.create_setup_action()

    def create_setup_action(self): 
        if not os.path.exists(self.path): os.makedirs(self.path)
        self.add_action(
                    ['templates/%s/setup.m' % self.tool],
                    [self.results_setup_fpath],
                    self.fill_setup_template,       
                    )

    def fill_setup_template(self, file_dep, target):
        with open(file_dep[0]) as ft:
            content = ft.read()

            possible_params = ['optimal_fiber_length', 'tendon_slack_length',
                               'pennation_angle', 'muscle_strain']
            paramstr = ''
            for param in possible_params:
                if param in self.param_dict:
                    paramstr += param + ' = true;\n'
                else:
                    paramstr += param + ' = false;\n'

            possible_costs = ['emg']
            coststr = ''
            for cost in possible_costs:
                if cost in self.cost_dict:
                    coststr += cost + ' = true;\n'
                else:
                    coststr += cost + ' = false;\n'


            pass_cal = ''
            if self.passive_precalibrate:
                pass_cal = 'Misc.passive_precalibrate = true;\n'

            content = content.replace('Misc = struct();',
                'Misc = struct();\n' + paramstr + coststr + pass_cal + '\n')

            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.subject.name)
            # TODO should this be an RRA-adjusted model?
            content = content.replace('@MODEL@', os.path.relpath(
                self.subject.scaled_model_fpath, self.path))
            content = content.replace('@REL_PATH_TO_TOOL@', os.path.relpath(
                self.study.config['optctrlmuscle_path'], self.path))
            # TODO provide slop on either side? start before the cycle_start?
            # end after the cycle_end?
            content = content.replace('@INIT_TIMES@',
                    ','.join('%0.5f' % x for x in self.init_times))
            content = content.replace('@FINAL_TIMES@', 
                    ','.join('%0.5f' % x for x in self.final_times))
            content = content.replace('@IK_SOLUTIONS@',
                    ','.join(self.rel_kinematics_files))
            content = content.replace('@ID_SOLUTIONS@',
                    ','.join(self.rel_kinetics_files))
            content = content.replace('@SIDE@',
                    self.trial.primary_leg[0])
            content = content.replace('@EMG_PATHS@', 
                    ','.join(self.rel_emg_files))
            content = content.replace('@CYCLE_IDS@',
                    ','.join(self.cycle_ids))
            if 'optimal_fiber_length' in self.param_dict:
                content = content.replace('@lMo_MUSCLES@',
                        ','.join(self.param_dict['optimal_fiber_length']))
            if 'tendon_slack_length' in self.param_dict:
                content = content.replace('@lTs_MUSCLES@',
                        ','.join(self.param_dict['tendon_slack_length']))
            if 'pennation_angle' in self.param_dict:
                content = content.replace('@alf_MUSCLES@',
                        ','.join(self.param_dict['pennation_angle']))
            if 'muscle_strain' in self.param_dict:
                content = content.replace('@e0_MUSCLES@',
                        ','.join(self.param_dict['muscle_strain']))
            if 'emg' in self.cost_dict:
                content = content.replace('@emg_MUSCLES@',
                        ','.join(self.cost_dict['emg']))

        with open(target[0], 'w') as f:
            f.write(content)

class TaskCalibrateParametersMultiPhase(osp.ToolTask):
    REGISTRY = []
    def __init__(self, trial, calibrate_setup_task, **kwargs):
        super(TaskCalibrateParametersMultiPhase, self).__init__(
            calibrate_setup_task, trial, opensim=False, **kwargs)
        self.doc = "Run multi-phase parameter calibration tool via DeGroote " \
                   "MRS solver."
        self.name = '%s_calibrate_multiphase' % self.subject.name
        self.results_setup_fpath = calibrate_setup_task.results_setup_fpath
        self.results_output_fpath = calibrate_setup_task.results_output_fpath

        self.file_dep += [
                self.results_setup_fpath,
                self.subject.scaled_model_fpath,
                ] + calibrate_setup_task.emg_files 

                # + calibrate_setup_task.kinematics_files \
                # + calibrate_setup_task.kinetics_files \

        self.actions += [
                self.run_parameter_calibration,
                # self.delete_muscle_analysis_results,
                ]

        self.targets += [
                self.results_output_fpath
                ]

    def run_parameter_calibration(self):
        with util.working_directory(self.path):
            # On Mac, CmdAction was causing MATLAB ipopt with GPOPS output to
            # not display properly.
            
            import subprocess
            status = subprocess.call('matlab %s -logfile matlab_log.txt -wait ' 
                    '-r "try, '
                    "run('%s'); disp('SUCCESS'); "
                    'catch ME; disp(getReport(ME)); exit(2), end, exit(0);"\n'
                    % ('-automation' if os.name == 'nt' else '',
                        self.results_setup_fpath)
                    )
            if status != 0:
                # print 'Non-zero exist status. Continuing....'
                raise Exception('Non-zero exit status.')

            # Wait until output mat file exists to finish the action
            import time
            while True:
                time.sleep(3.0)

                mat_exists = os.path.isfile(self.results_output_fpath)
                if mat_exists:
                    break

    def delete_muscle_analysis_results(self):
        if os.path.exists(os.path.join(self.path, 'results')):
            import shutil
            shutil.rmtree(os.path.join(self.path, 'results'))

class TaskCalibrateParametersMultiPhasePost(osp.PostTask):
    REGISTRY = []
    def __init__(self, trial, calibrate_setup_task, **kwargs):
        super(TaskCalibrateParametersMultiPhasePost, self).__init__(
            calibrate_setup_task, trial, **kwargs)
        self.doc = 'Postprocessing of multi-phase parameter calibration results.'
        self.name = '%s_calibrate_multiphase_post' % self.subject.name
        self.setup_task = calibrate_setup_task
        self.results_output_fpath = self.setup_task.results_output_fpath

        numPhases = len(calibrate_setup_task.kinematics_files)
        for i in range(numPhases):
            trial_results_exp_path = \
                calibrate_setup_task.trial_results_exp_paths[i]
            cycle_id = calibrate_setup_task.cycle_ids[i]
            init_time = calibrate_setup_task.init_times[i]
            final_time = calibrate_setup_task.final_times[i]

            emg_fpath = os.path.join(trial_results_exp_path, 'expdata', 
                'emg_with_headers.sto')

            self.add_action([emg_fpath, self.results_output_fpath],
                [os.path.join(self.path, '%s_muscle_activity' % cycle_id),
                 os.path.join(self.path, '%s_reserve_activity.pdf' % cycle_id)],
                self.plot_muscle_and_reserve_activity,
                cycle_id, init_time, final_time)

        self.add_action([self.results_output_fpath],
            [os.path.join(self.path, 'optimal_fiber_length.pdf'),
             os.path.join(self.path, 'tendon_slack_length.pdf'),
             os.path.join(self.path, 'pennation_angle.pdf'),
             os.path.join(self.path, 'muscle_strain.pdf')],
            self.plot_muscle_parameters)

    def plot_muscle_and_reserve_activity(self, file_dep, target, cycle_id, 
        init_time, final_time):

        emg = util.storage2numpy(file_dep[0])
        time = emg['time']

        def min_index(vals):
            idx, val = min(enumerate(vals), key=lambda p: p[1])
            return idx

        start_idx = min_index(abs(time-init_time))
        end_idx = min_index(abs(time-final_time))

        # Load mat file fields
        muscle_names = util.hdf2list(file_dep[1], 'MuscleNames', type=str)
        df_exc = util.hdf2pandas(file_dep[1], 'MExcitation/%s' % cycle_id, 
            labels=muscle_names)
        df_act = util.hdf2pandas(file_dep[1], 'MActivation/%s' % cycle_id, 
            labels=muscle_names)
        dof_names = util.hdf2list(file_dep[1], 'DOFNames', 
            type=str)

        pgc_emg = np.linspace(0, 100, len(time[start_idx:end_idx]))
        pgc_exc = np.linspace(0, 100, len(df_exc.index))
        pgc_act = np.linspace(0, 100, len(df_act.index))

        muscles = self.study.muscle_names
        fig = pl.figure(figsize=(12, 12))
        nice_muscle_names_dict = self.study.nice_muscle_names_dict


        emg_map = {
                'med_gas_r': 'gasmed_r',
                'glut_max2_r': 'glmax2_r',
                'rect_fem_r': 'recfem_r',
                'semimem_r': 'semimem_r',
                'soleus_r': 'soleus_r',
                'tib_ant_r': 'tibant_r',
                'vas_int_r': 'vasmed_r', 
        }

        emg_muscles = ['bflh_r', 'gaslat_r', 'gasmed_r', 'glmax1_r', 'glmax2_r',
                       'glmax3_r', 'glmed1_r', 'glmed2_r', 'glmed3_r', 
                       'recfem_r', 'semimem_r', 'semiten_r', 'soleus_r',
                       'tibant_r', 'vaslat_r', 'vasmed_r']

        for imusc, musc_name in enumerate(muscles):
            side_len = np.ceil(np.sqrt(len(muscles)))
            ax = fig.add_subplot(side_len, side_len, imusc + 1)
            ax.axhline(color='k', linewidth=0.5, zorder=0)
            y_exc = df_exc[musc_name]
            y_act = df_act[musc_name]
            exc_plot, = ax.plot(pgc_exc, y_exc, color='blue', 
                linestyle='--')
            act_plot, = ax.plot(pgc_act, y_act, color='red', 
                linestyle='--')
            handles = [exc_plot, act_plot   ]
            labels = ['%s exc.' % nice_muscle_names_dict[musc_name],
                      '%s act.' % nice_muscle_names_dict[musc_name]]
            ax.legend(handles, labels)
            
            if emg_map.get(musc_name):
                y_emg = emg[emg_map[musc_name]]
                ax.plot(pgc_emg, y_emg[start_idx:end_idx], color='black', 
                    linestyle='-')

            # ax.legend(frameon=False, fontsize=6)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.0)
            ax.set_title(nice_muscle_names_dict[musc_name])
            ax.set_xlabel('time (% gait cycle)')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        fig.tight_layout()
        fig.savefig(target[0]+'.pdf')
        fig.savefig(target[0]+'.png', dpi=600)
        pl.close(fig)

        # Plot reserve activity
        df_res = util.hdf2pandas(file_dep[1],'RActivation/%s' % cycle_id, 
            labels=dof_names)
        pp.plot_reserve_activity(target[1], df_res)

    def plot_muscle_parameters(self, file_dep, target):

        lMo, lTs, alf, e0, musc_names = get_muscle_parameters_as_dict(
            file_dep[0])

        # Plot muscle parameters
        def param_barplot(param_name, param, output_fpath):

            fig = pl.figure(figsize=(11,6))
            ax = fig.add_subplot(1,1,1)
            musc_names = list(param.keys())
            pos = np.arange(len(musc_names))

            param_to_plot = [(val - 1.0)*100 for val in param.values()]
            bar = ax.bar(pos, param_to_plot, color='green')
            ax.set_xticks(pos)
            ax.set_xticklabels(musc_names, fontsize=10)
            ax.set_ylabel('Percent Change in ' + param_name, fontsize=12)
            ax.set_ylim([-30, 30])
            ax.set_yticks(np.linspace(-30, 30, 13))
            ax.grid(which='both', axis='both', linestyle='--')
            ax.set_axisbelow(True)

            fig.tight_layout()
            fig.savefig(output_fpath)
            pl.close(fig)

        if lMo:
            param_barplot('Optimal Fiber Length', lMo, target[0])
            fpath = os.path.join(self.subject.results_exp_path, 
                    os.path.basename(target[0]))
            param_barplot('Optimal Fiber Length', lMo, fpath)
            df_lMo = pd.DataFrame.from_dict(lMo, orient='index')
            df_lMo.to_csv(fpath.replace('.pdf', '.csv'), header=False)

        if lTs:
            param_barplot('Tendon Slack Length', lTs, target[1])
            fpath = os.path.join(self.subject.results_exp_path, 
                    os.path.basename(target[1]))
            param_barplot('Tendon Slack Length', lTs, fpath)
            df_lTs = pd.DataFrame.from_dict(lTs, orient='index')
            df_lTs.to_csv(fpath.replace('.pdf', '.csv'), header=False)

        if alf:
            param_barplot('Pennation Angle', alf, target[2])
            fpath = os.path.join(self.subject.results_exp_path, 
                    os.path.basename(target[2]))
            param_barplot('Pennation Angle', alf, fpath)
            df_alf = pd.DataFrame.from_dict(alf, orient='index')
            df_alf.to_csv(fpath.replace('.pdf', '.csv'), header=False)

        if e0:
            param_barplot('Muscle Strain', e0, target[3])
            fpath = os.path.join(self.subject.results_exp_path, 
                    os.path.basename(target[3]))
            param_barplot('Muscle Strain', e0, fpath)
            df_e0 = pd.DataFrame.from_dict(e0, orient='index')
            df_e0.to_csv(fpath.replace('.pdf', '.csv'), header=False)

class TaskMRSDeGrooteSetup(osp.TaskMRSDeGrooteSetup):
    REGISTRY = []
    def __init__(self, trial, param_dict, cost='Default', **kwargs):
        super(TaskMRSDeGrooteSetup, self).__init__(trial, 
            cost=cost, **kwargs)
        self.param_dict = param_dict

        if 'optimal_fiber_length' in self.param_dict:
            self.lMo_modifiers_fpath = os.path.join(
                self.subject.results_exp_path, 'optimal_fiber_length.csv')
            self.lMo_modifiers_relpath = os.path.relpath(
                self.lMo_modifiers_fpath, self.path)
            self.file_dep += [self.lMo_modifiers_fpath]

        if 'tendon_slack_length' in self.param_dict:
            self.lTs_modifiers_fpath = os.path.join(
                self.subject.results_exp_path, 'tendon_slack_length.csv')
            self.lTs_modifiers_relpath = os.path.relpath(
                self.lTs_modifiers_fpath, self.path)
            self.file_dep += [self.lTs_modifiers_fpath]

        if 'pennation_angle' in self.param_dict:
            self.alf_modifiers_fpath = os.path.join(
                self.subject.results_exp_path, 'pennation_angle.csv')
            self.alf_modifiers_relpath = os.path.relpath(
                self.alf_modifiers_fpath, self.path)
            self.file_dep += [self.alf_modifiers_fpath]

        if 'muscle_strain' in self.param_dict:
            self.e0_modifiers_fpath = os.path.join(
                self.subject.results_exp_path, 'muscle_strain.csv')
            self.e0_modifiers_relpath = os.path.relpath(
                self.e0_modifiers_fpath, self.path)
            self.file_dep += [self.e0_modifiers_fpath]

        self.speed = trial.condition.metadata['walking_speed']

    def fill_setup_template(self, file_dep, target,
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()

            possible_params = ['optimal_fiber_length', 'tendon_slack_length',
                               'pennation_angle', 'muscle_strain']
            paramstr = ''
            for param in possible_params:
                if param in self.param_dict:
                    paramstr += param + ' = true;\n'
                else:
                    paramstr += param + ' = false;\n'

            content = content.replace('Misc = struct();',
                'Misc = struct();\n' + paramstr + '\n')

            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            # TODO should this be an RRA-adjusted model?
            content = content.replace('@MODEL@', os.path.relpath(
                self.subject.scaled_model_fpath, self.path))
            content = content.replace('@REL_PATH_TO_TOOL@', os.path.relpath(
                self.study.config['optctrlmuscle_path'], self.path))
            # TODO provide slop on either side? start before the cycle_start?
            # end after the cycle_end?
            content = content.replace('@INIT_TIME@',
                    '%.5f' % init_time)
            content = content.replace('@FINAL_TIME@', 
                    '%.5f' % final_time)
            content = content.replace('@IK_SOLUTION@',
                    self.rel_kinematics_file)
            content = content.replace('@ID_SOLUTION@',
                    self.rel_kinetics_file)
            content = content.replace('@SIDE@',
                    self.trial.primary_leg[0])
            content = content.replace('@COST@', self.cost)
            content = content.replace('@ACTDYN@', self.actdyn)
            content = content.replace('@SPEED@', '%.5f' % self.speed)

            if 'optimal_fiber_length' in self.param_dict:
                content = content.replace('@lMo_MODIFIERS@', 
                        self.lMo_modifiers_relpath)
            if 'tendon_slack_length' in self.param_dict:
                content = content.replace('@lTs_MODIFIERS@', 
                        self.lTs_modifiers_relpath)
            if 'pennation_angle' in self.param_dict:
                content = content.replace('@alf_MODIFIERS@', 
                        self.alf_modifiers_relpath)
            if 'muscle_strain' in self.param_dict:
                content = content.replace('@e0_MODIFIERS@', 
                        self.e0_modifiers_relpath)

        with open(target[0], 'w') as f:
            f.write(content)

class TaskMRSDeGrooteMod(osp.TaskMRSDeGrooteMod):
    def __init__(self, trial, mrs_setup_task, mod_name, description,
        mrsflags, **kwargs):
        super(TaskMRSDeGrooteMod, self).__init__(trial, mrs_setup_task, 
            mod_name, description, mrsflags, **kwargs)

        self.mrs_setup_task = mrs_setup_task
        # This makes this task cycle-specifc
        self.cycle = mrs_setup_task.cycle
        self.cost = mrs_setup_task.cost

        if 'optimal_fiber_length' in self.mrs_setup_task.param_dict:
            self.lMo_modifiers_fpath = \
                self.mrs_setup_task.lMo_modifiers_fpath
            self.lMo_modifiers_relpath = os.path.relpath(
                self.lMo_modifiers_fpath, self.path)
            self.file_dep += [self.lMo_modifiers_fpath]

        if 'tendon_slack_length' in self.mrs_setup_task.param_dict:
            self.lTs_modifiers_fpath = \
                self.mrs_setup_task.lTs_modifiers_fpath
            self.lTs_modifiers_relpath = os.path.relpath(
                self.lTs_modifiers_fpath, self.path)
            self.file_dep += [self.lTs_modifiers_fpath]

        if 'pennation_angle' in self.mrs_setup_task.param_dict:
            self.alf_modifiers_fpath = \
                self.mrs_setup_task.alf_modifiers_fpath
            self.alf_modifiers_relpath = os.path.relpath(
                self.alf_modifiers_fpath, self.path)
            self.file_dep += [self.alf_modifiers_fpath]

        if 'muscle_strain' in self.mrs_setup_task.param_dict:
            self.e0_modifiers_fpath = \
                self.mrs_setup_task.e0_modifiers_fpath
            self.e0_modifiers_relpath = os.path.relpath(
                self.e0_modifiers_fpath, self.path)
            self.file_dep += [self.e0_modifiers_fpath]

        self.speed = trial.condition.metadata['walking_speed']

    def fill_setup_template(self, file_dep, target, 
                            init_time=None, final_time=None):
        with open(self.setup_template_fpath) as ft:
            content = ft.read()

            if type(self.mrsflags) is list:
                list_of_flags = self.mrsflags 
            else:
                list_of_flags = self.mrsflags(self.cycle)

            # Copy mod name and add this to the mod name flag later, in case
            # the flag passed to MATLAB needs to be slightly different than 
            # self.mod_name.
            mod_name = self.mod_name

            # Append mod_name flag
            list_of_flags.append("mod_name='%s'" % mod_name)
            list_of_flags.append("mrs_path='%s'" % 
                os.path.relpath(self.mrs_setup_task.results_output_fpath, 
                    self.path))

            # Insert flags for the mod.
            flagstr = ''
            for flag in list_of_flags:
                flagstr += 'Misc.%s;\n' % flag

            possible_params = ['optimal_fiber_length', 'tendon_slack_length',
                               'pennation_angle', 'muscle_strain']
            paramstr = ''
            for param in possible_params:
                if param in self.mrs_setup_task.param_dict:
                    paramstr += param + ' = true;\n'
                else:
                    paramstr += param + ' = false;\n'

            content = content.replace('Misc = struct();',
                    'Misc = struct();\n' +
                    flagstr + paramstr + '\n' +
                    # In case the description has multiple lines, add comment
                    # symbol in front of every line.
                    '% ' + self.description.replace('\n', '\n% ') + '\n')

            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            # TODO should this be an RRA-adjusted model?
            content = content.replace('@MODEL@', os.path.relpath(
                self.subject.scaled_model_fpath, self.path))
            content = content.replace('@REL_PATH_TO_TOOL@', os.path.relpath(
                self.study.config['optctrlmuscle_path'], self.path))
            # TODO provide slop on either side? start before the cycle_start?
            # end after the cycle_end?
            content = content.replace('@INIT_TIME@',
                    '%.5f' % init_time)
            content = content.replace('@FINAL_TIME@', 
                    '%.5f' % final_time)

            content = content.replace('@IK_SOLUTION@',
                    os.path.relpath(self.kinematics_fpath, self.path))
            content = content.replace('@ID_SOLUTION@',
                    os.path.relpath(self.kinetics_fpath, self.path))
            content = content.replace('@SIDE@',
                    self.trial.primary_leg[0])
            content = content.replace('@COST@', self.cost)
            content = content.replace('@ACTDYN@', self.actdyn)
            content = content.replace('@SPEED@', '%.5f' % self.speed)
            if 'optimal_fiber_length' in self.mrs_setup_task.param_dict:
                content = content.replace('@lMo_MODIFIERS@', 
                        self.lMo_modifiers_relpath)
            if 'tendon_slack_length' in self.mrs_setup_task.param_dict:
                content = content.replace('@lTs_MODIFIERS@', 
                        self.lTs_modifiers_relpath)
            if 'pennation_angle' in self.mrs_setup_task.param_dict:
                content = content.replace('@alf_MODIFIERS@', 
                        self.alf_modifiers_relpath)
            if 'muscle_strain' in self.mrs_setup_task.param_dict:
                content = content.replace('@e0_MODIFIERS@', 
                        self.e0_modifiers_relpath)            

        with open(self.setup_fpath, 'w') as f:
            f.write(content)

class TaskMRSDeGrooteModPost(osp.TaskMRSDeGrooteModPost):
    REGISTRY = []
    def __init__(self, trial, mrsmod_task, **kwargs):
        super(TaskMRSDeGrooteModPost, self).__init__(trial, mrsmod_task, 
            **kwargs)
        # original, "no-mod" solution
        self.mrs_results_output_fpath = \
            mrsmod_task.mrs_setup_task.results_output_fpath

        self.add_action([self.results_output_fpath],
                        [os.path.join(self.path,
                            'device_moment_arms.pdf')],
                        self.plot_device_moment_arms)

        self.add_action([self.mrs_results_output_fpath,
                         self.results_output_fpath],
                         [os.path.join(self.path, 'metabolic_reductions.pdf')],
                         self.plot_metabolic_reductions)

        self.add_action([self.mrs_results_output_fpath,
                         self.results_output_fpath],
                         [os.path.join(self.path, 
                            'muscle_activity_reductions.pdf')],
                         self.plot_muscle_activity_reductions)

    def plot_joint_moment_breakdown(self, file_dep, target):

        # Load mat file fields
        muscle_names = util.hdf2list(file_dep[0], 'MuscleNames', type=str)
        dof_names = util.hdf2list(file_dep[0],'DatStore/DOFNames',type=str)
        num_dofs = len(dof_names)
        num_muscles = len(muscle_names)
        joint_moments_exp = util.hdf2numpy(file_dep[0], 'DatStore/T_exp')
        tendon_forces = util.hdf2numpy(file_dep[0], 'TForce')
        exp_time = util.hdf2numpy(file_dep[0], 'DatStore/time').transpose()[0]
        time = util.hdf2numpy(file_dep[0], 'Time').transpose()[0]
        moment_arms_exp = util.hdf2numpy(file_dep[0], 'DatStore/dM').transpose()

        # Clip large tendon forces at final time
        from warnings import warn
        for imusc in range(len(muscle_names)):
            tendon_force = tendon_forces[:,imusc]
            if (tendon_force[-1] > 10*tendon_force[-2]):
                tendon_force[-1] = tendon_force[-2]
                tendon_forces[:,imusc] = tendon_force
                warn('WARNING: large %s tendon force at final time. '
                    'Clipping...' % muscle_names[imusc])

        # Get device torques
        device_torques = list()
        device_names = list()
        device_colors = list()
        if 'device' in self.mrsmod_task.mod_name:
            act_torques = util.hdf2pandas(file_dep[0], 
                'DatStore/ExoTorques', labels=dof_names)
            device_torques.append(act_torques)
            device_names.append('device')
            device_colors.append('green')

        # Interpolate to match solution time
        from scipy.interpolate import interp1d
        ma_shape = (len(time), moment_arms_exp.shape[1], 
            moment_arms_exp.shape[2])
        moment_arms = np.empty(ma_shape)
        for i in range(moment_arms_exp.shape[2]):
            func_moment_arms_interp = interp1d(exp_time, 
                moment_arms_exp[:,:,i].squeeze(), axis=0)
            moment_arms[:,:,i] = func_moment_arms_interp(time)

        func_joint_moments_interp = interp1d(exp_time, joint_moments_exp,
            axis=0)
        joint_moments = func_joint_moments_interp(time)

        # Generate plots
        pp.plot_joint_moment_breakdown(time, joint_moments, tendon_forces,
            moment_arms, dof_names, muscle_names, target[0], target[1],
            mass=self.subject.mass, ext_moments=device_torques,
            ext_names=device_names, ext_colors=device_colors)

    def plot_device_moment_arms(self, file_dep, target):

        dof_names = util.hdf2list(file_dep[0],'DatStore/DOFNames',type=str)
        num_dofs = len(dof_names)

        # Get device moment arms
        device_mom_arms = np.array([[0.0, 0.0, 0.0]])
        if 'device' in self.mrsmod_task.mod_name:
            device_mom_arms = util.hdf2numpy(file_dep[0],
                'DatStore/MomentArms')

        # Plot moment arms
        fig = pl.figure(figsize=(11,8.5))
        ax = fig.add_subplot(1,1,1)
        pos = np.arange(num_dofs)
        width = 0.4

        bar1 = ax.bar(pos, device_mom_arms[0], width, color='green')
        ax.set_xticks(pos + width / 2)
        ax.set_xticklabels(dof_names, fontsize=10)
        # ax.set_yticks(np.arange(-100,105,5))
        # for label in ax.yaxis.get_ticklabels()[1::2]:
        #     label.set_visible(False)
        ax.set_ylabel('Moment Arms', fontsize=12)
        ax.grid(which='both', axis='both', linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(bar1, 'device')

        fig.tight_layout()
        fig.savefig(target[0])
        pl.close(fig)

    def plot_passive_device_information(self, file_dep, target):

        dof_names = util.hdf2list(file_dep[0],'DatStore/DOFNames',type=str)
        num_dofs = len(dof_names)
        time = util.hdf2numpy(file_dep[0], 'Time').transpose()[0]

        pass_force = util.hdf2pandas(file_dep[0],
            'DatStore/passiveForce')
        # slack_var = util.hdf2pandas(file_dep[0],
        #     'DatStore/passiveSlackVar')
        path_length = util.hdf2pandas(file_dep[0],
            'DatStore/pathLength')
        joint_angles = util.hdf2numpy(file_dep[0],
            'DatStore/jointAngles')
        slack_length = util.hdf2numpy(file_dep[0],
            'DatStore/slackLength')
        pass_mom_arms = util.hdf2numpy(file_dep[0],
                'DatStore/MomentArms_Pass')

        # Plot passive force information
        fig = pl.figure(figsize=(8.5, 11))
        for idof in range(num_dofs):
            dof_name = dof_names[idof]
            ax = fig.add_subplot(3, 2, 2*(idof+1))
            ax.plot(time, joint_angles[:, idof], label=dof_name, color='black')
            ax.set_title(dof_name)
            ax.set_ylabel('angle (rad)')
            ax.set_xlabel('time')
            ax.grid(which='both', axis='both', linestyle='--')
            pl.text(time[0], max(joint_angles[:, idof]), 
                'moment arm = %0.3f' % pass_mom_arms[0][idof],
                fontsize=10, weight='bold')

        ax = fig.add_subplot(3, 2, 1)
        ax.plot(time, pass_force, color='blue')
        ax.set_title('passive force')
        ax.set_ylabel('force (N)')
        ax.set_xlabel('time')
        ax.grid(which='both', axis='both', linestyle='--')

        ax = fig.add_subplot(3, 2, 3)
        ax.plot(time, path_length)
        ax.set_title('path length')
        ax.set_ylabel('length (m)')
        ax.set_xlabel('time')
        ax.grid(which='both', axis='both', linestyle='--')
        pl.text(time[0], max(path_length[0]), 
            'slack length = %0.3f' % slack_length[0][0],
            fontsize=10, weight='bold')

        fig.tight_layout()
        fig.savefig(target[0])
        pl.close(fig)

    def plot_metabolic_reductions(self, file_dep, target):

        # Load mat file fields from original, "no-mod" solution
        muscle_names = util.hdf2list(file_dep[0], 'MuscleNames', type=str)
        num_muscles = len(muscle_names)

        mrs_whole_body_metabolic_rate = util.hdf2pandas(file_dep[0], 
            'DatStore/MetabolicInfo/UmbergerKoelewijn2018/average_wholebody_metabolic_rate')
        mrs_muscle_metabolic_rates = util.hdf2pandas(file_dep[0],
            'DatStore/MetabolicInfo/UmbergerKoelewijn2018/muscle_average_metabolic_rates', 
            labels=muscle_names)

        # Assume symmetry and add basal rate
        mrs_whole_body_metabolic_rate *= 2
        mrs_muscle_metabolic_rates *= 2
        mrs_whole_body_metabolic_rate += 1.2

        # Load mat file fields from modified solution
        muscle_names = util.hdf2list(file_dep[1], 'MuscleNames', type=str)
        num_muscles = len(muscle_names)

        mrsmod_whole_body_metabolic_rate = util.hdf2pandas(file_dep[1], 
            'DatStore/MetabolicInfo/UmbergerKoelewijn2018/average_wholebody_metabolic_rate')
        mrsmod_muscle_metabolic_rates = util.hdf2pandas(file_dep[1],
            'DatStore/MetabolicInfo/UmbergerKoelewijn2018/muscle_average_metabolic_rates', 
            labels=muscle_names)

        # Assume symmetry and add basal rate
        mrsmod_whole_body_metabolic_rate *= 2
        mrsmod_muscle_metabolic_rates *= 2
        mrsmod_whole_body_metabolic_rate += 1.2

        reductions = list()
        reduc_names = list()
        colors = list()
        for musc in muscle_names:
            muscle_reduction = 100.0 * ((mrsmod_muscle_metabolic_rates[musc][0] -
                mrs_muscle_metabolic_rates[musc][0]) / 
                mrs_whole_body_metabolic_rate[0][0])
            reductions.append(muscle_reduction)
            reduc_names.append(musc)
            colors.append('b')

        whole_body_reduction = 100.0 * ((mrsmod_whole_body_metabolic_rate - 
            mrs_whole_body_metabolic_rate) / mrs_whole_body_metabolic_rate)
        reductions.append(whole_body_reduction[0][0])
        reduc_names.append('whole_body')
        colors.append('r')

        # import pdb
        # pdb.set_trace()

        # Plot metabolic reductions
        fig = pl.figure(figsize=(11,8.5))
        ax = fig.add_subplot(1,1,1)
        pos = np.arange(len(muscle_names)+1)
        
        ax.bar(pos, reductions, align='center', color=colors)
        ax.set_xticks(pos)
        ax.set_xticklabels(reduc_names, fontsize=10)
        ax.set_yticks(np.arange(-60,65,5))
        for label in ax.yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        ax.set_ylabel('Percent Change in Metabolic Rate', fontsize=12)
        ax.grid(which='both', axis='both', linestyle='--')
        ax.set_axisbelow(True)

        fig.tight_layout()
        fig.savefig(target[0])
        pl.close(fig)

    def plot_muscle_activity_reductions(self, file_dep, target):

        # Load mat file fields from original, "no-mod" solution
        muscle_names = util.hdf2list(file_dep[0], 'MuscleNames', type=str)
        num_muscles = len(muscle_names)

        mrs_excitations = util.hdf2pandas(file_dep[0], 
            'MExcitation', labels=muscle_names)
        mrs_activations = util.hdf2pandas(file_dep[0],
            'MActivation', labels=muscle_names)

        # Load mat file fields from modified solution
        muscle_names = util.hdf2list(file_dep[1], 'MuscleNames', type=str)
        num_muscles = len(muscle_names)

        mrsmod_excitations = util.hdf2pandas(file_dep[1], 
            'MExcitation', labels=muscle_names)
        mrsmod_activations = util.hdf2pandas(file_dep[1],
            'MActivation', labels=muscle_names)

        exc_reductions = list()
        act_reductions = list()
        reduc_names = list()
        exc_colors = list()
        act_colors = list()

        from matplotlib import colors as mcolors
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

        # Individual muscles
        for musc in muscle_names:
            reduc_names.append(musc)

            mrs_exc = mrs_excitations[musc]
            mrsmod_exc = mrsmod_excitations[musc]
            diff_exc = mrsmod_exc - mrs_exc 
            reduc_exc = 100.0 * (sum(diff_exc) / sum(mrs_exc))
            exc_reductions.append(reduc_exc)
            exc_colors.append(colors['khaki'])

            mrs_act = mrs_activations[musc]
            mrsmod_act = mrsmod_activations[musc]
            diff_act = mrsmod_act - mrs_act
            reduc_act = 100.0 * (sum(diff_act) / sum(mrs_act))
            act_reductions.append(reduc_act)
            act_colors.append(colors['palegreen'])

        # Whole body
        reduc_names.append('whole_body')
        whole_reduc_exc = sum(exc_reductions)
        exc_reductions.append(whole_reduc_exc)
        exc_colors.append('gold')

        whole_reduc_act = sum(act_reductions)
        act_reductions.append(whole_reduc_act)
        act_colors.append('seagreen')

        # Plot activity reductions
        fig = pl.figure(figsize=(11,8.5))
        ax = fig.add_subplot(1,1,1)
        pos = np.arange(len(muscle_names)+1)
        width = 0.4

        bar1 = ax.bar(pos, exc_reductions, width, color=exc_colors)
        bar2 = ax.bar(pos + width, act_reductions, width, color=act_colors)
        ax.set_xticks(pos + width / 2)
        ax.set_xticklabels(reduc_names, fontsize=10)
        # ax.set_yticks(np.arange(-100,105,5))
        # for label in ax.yaxis.get_ticklabels()[1::2]:
        #     label.set_visible(False)
        ax.set_ylabel('Percent Change in Muscle Activity', fontsize=12)
        ax.grid(which='both', axis='both', linestyle='--')
        ax.set_axisbelow(True)
        ax.legend([bar1, bar2], ['excitations', 'activations'])

        fig.tight_layout()
        fig.savefig(target[0])
        pl.close(fig)

def construct_multiindex_tuples(study, subjects, conditions, 
    muscle_level=False, dof_level=False):
    ''' Construct multiindex tuples and list of cycles for DataFrame indexing.
    '''
    
    multiindex_tuples = list()
    cycles = list()

    for subject in study.subjects:
        if not subject.num in subjects: continue
        for cond_name in conditions:
            cond = subject.get_condition(cond_name)
            if not cond: continue
            # We know there is only one overground trial, but perhaps it
            # has not yet been added for this subject.
            assert len(cond.trials) <= 1
            if len(cond.trials) == 1:
                trial = cond.trials[0]
                for cycle in trial.cycles:
                    if study.cycles_to_plot:
                        if not (cycle.name in study.cycles_to_plot): continue
                    cycles.append(cycle)
                    if (not muscle_level) and (not dof_level):
                        multiindex_tuples.append((
                            cycle.subject.name,
                            cycle.condition.name,
                            # This must be the full ID, not just the cycle
                            # name, because 'cycle01' from subject 1 has
                            # nothing to do with 'cycle01' from subject 2
                            # (whereas the 'walk2' condition for subject 1 is
                            # related to 'walk2' for subject 2).
                            cycle.id))
                    elif muscle_level and (not dof_level):
                        for mname in study.muscle_names:
                            multiindex_tuples.append((
                                cycle.subject.name,
                                cycle.condition.name,
                                cycle.id,
                                mname))
                    elif dof_level and (not muscle_level):
                        for dofname in study.dof_names:
                            multiindex_tuples.append((
                                cycle.subject.name,
                                cycle.condition.name,
                                cycle.id,
                                dofname))
                    elif muscle_level and dof_level:
                        Exception('Cannot have levels for muscles and DOFs.')

    return multiindex_tuples, cycles

class TaskAggregateMetabolicRate(osp.StudyTask):
    """Aggregate metabolic rate without and with mods across all subjects and
    gait cycles for each condition provided."""
    REGISTRY = []
    def __init__(self, study, mods, subjects=None, 
            conditions=['walk1','walk2','walk3','walk4'], suffix=''):
        super(TaskAggregateMetabolicRate, self).__init__(study)
        self.mod_names = list()
        self.mod_dirs = list()
        for mod in mods:
            self.mod_dirs.append(mod.replace('/','\\\\'))
            if len(mod.split('/')) > 1:
                self.mod_names.append('_'.join(mod.split('/')))
            else:
                self.mod_names.append(mod)
        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction
        self.name = 'aggregate_metabolic_rate%s' % suffix
        self.whole_fpath = os.path.join(self.suffix_path, 
            'whole_body_metabolic_rates%s.csv' % suffix)
        self.muscs_fpath = os.path.join(self.suffix_path, 
            'muscle_metabolic_rates%s.csv' % suffix)   
        self.doc = 'Aggregate metabolic rate.'
        self.study = study

        if subjects == None:
            subjects = [s.num for s in study.subjects]

        # Get multiindex tuples for DataFrame indexing for both whole body,
        # and muscle level metabolic rate. Also get cycles list.
        self.multiindex_tuples, cycles = construct_multiindex_tuples(study, 
            subjects, conditions, muscle_level=False)
        self.multiindex_tuples_musc, _ = construct_multiindex_tuples(study, 
            subjects, conditions, muscle_level=True)

        self.mod_for_file_dep = list()
        deps = list()

        # Prepare for processing simulations of experiments.
        for cycle in cycles:
            if study.cycles_to_plot:
                if not (cycle.name in study.cycles_to_plot): continue
            self.mod_for_file_dep.append('experiment')
            deps.append(os.path.join(
                    cycle.trial.results_exp_path, 'mrs', cycle.name,
                    self.costdir, '%s_%s_mrs.mat' % (study.name, cycle.id))
                    )

        # Prepare for processing simulations of mods.
        for mod_name, mod_dir in zip(self.mod_names, self.mod_dirs):
            for cycle in cycles:
                if study.cycles_to_plot:
                    if not (cycle.name in study.cycles_to_plot): continue
                self.mod_for_file_dep.append(mod_name)
                deps.append(os.path.join(
                        self.study.config['results_path'],
                        mod_dir, cycle.trial.rel_path, 'mrs', 
                        cycle.name, self.costdir,
                        '%s_%s_mrs.mat' % (study.name, cycle.id))
                    )

        self.add_action(deps,
                [os.path.join(study.config['analysis_path'], self.whole_fpath)],
                self.aggregate_metabolic_rate)

        self.add_action(deps,
                [os.path.join(study.config['analysis_path'], self.muscs_fpath)],
                self.aggregate_metabolic_rate_muscles)

    def aggregate_metabolic_rate(self, file_dep, target):
        import numpy as np
        from collections import OrderedDict
        metabolic_rate = OrderedDict()
        for ifile, fpath in enumerate(file_dep):
            df = util.hdf2pandas(fpath, 
                'DatStore/MetabolicInfo/UmbergerKoelewijn2018/average_wholebody_metabolic_rate')
            this_mod = self.mod_for_file_dep[ifile]
            if not this_mod in metabolic_rate:
                metabolic_rate[this_mod] = list()
            metabolic_rate[this_mod].append(df[0][0])
       
        # http://pandas.pydata.org/pandas-docs/stable/advanced.html#advanced-hierarchical
        index = pd.MultiIndex.from_tuples(self.multiindex_tuples,
                names=['subject', 'condition', 'cycle'])
        df = pd.DataFrame(metabolic_rate, index=index)

        target_dir = os.path.dirname(target[0])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(target[0], 'w') as f:
            f.write('# columns contain whole body metabolic rate normalized by '
                    'subject mass (W/kg)\n')
            df.to_csv(f)

    def aggregate_metabolic_rate_muscles(self, file_dep, target):
        import numpy as np
        from collections import OrderedDict
        metabolic_rate = OrderedDict()
        for ifile, fpath in enumerate(file_dep):
            df = util.hdf2pandas(fpath, 
                'DatStore/MetabolicInfo/UmbergerKoelewijn2018/muscle_average_metabolic_rates',
                labels=self.study.muscle_names)
            this_mod = self.mod_for_file_dep[ifile]
            if not this_mod in metabolic_rate:
                metabolic_rate[this_mod] = list()
            for muscle in self.study.muscle_names:
                metabolic_rate[this_mod].append(df[muscle][0])
       
        # http://pandas.pydata.org/pandas-docs/stable/advanced.html#advanced-hierarchical
        index = pd.MultiIndex.from_tuples(self.multiindex_tuples_musc,
                names=['subject', 'condition', 'cycle', 'muscle'])

        df = pd.DataFrame(metabolic_rate, index=index)

        target_dir = os.path.dirname(target[0])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(target[0], 'w') as f:
            f.write('# columns contain muscle metabolic rates normalized by '
                    'subject mass (W/kg)\n')
            df.to_csv(f)

class TaskAggregateDevicePower(osp.StudyTask):
    """Aggregate peak instantaneous and average device power for assisted cases 
    across all subjects and gait cycles for each provided condition."""
    REGISTRY = []
    def __init__(self, study, mods, subjects=None, 
            conditions=['walk1','walk2','walk3','walk4'], suffix=''):
        super(TaskAggregateDevicePower, self).__init__(study)
        self.mod_names = list()
        self.mod_dirs = list()
        for mod in mods:
            self.mod_dirs.append(mod.replace('/', '\\'))
            if len(mod.split('/')) > 1:
                self.mod_names.append('_'.join(mod.split('/')))
            else:
                self.mod_names.append(mod)
        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction
        self.name = 'aggregate_device_power%s' % suffix
        self.peak_power_fpath = os.path.join(self.suffix_path, 
            'peak_power%s.csv' % suffix)
        self.avg_pos_power_fpath = os.path.join(self.suffix_path, 
            'avg_pos_power%s.csv' % suffix)
        self.avg_neg_power_fpath = os.path.join(self.suffix_path, 
            'avg_neg_power%s.csv' % suffix)
        self.total_power_fpath = os.path.join(self.suffix_path, 
            'total_power%s.csv' % suffix)
        self.doc = """Aggregate peak instantaneous and average negative and '
                   'positive power normalized by subject mass."""
        self.study = study

        if subjects == None:
            subjects = [s.num for s in study.subjects]

        # Get multiindex tuples and cycles list
        self.multiindex_tuples, all_cycles = construct_multiindex_tuples(study, 
            subjects, conditions)

        self.mod_for_file_dep = list()
        self.subject_masses = list()
        deps = list()
        cycles = list()
        for mod_name, mod_dir in zip(self.mod_names, self.mod_dirs):
            for cycle in all_cycles:
                if study.cycles_to_plot:
                    if not (cycle.name in study.cycles_to_plot): 
                        continue
                self.mod_for_file_dep.append(mod_name)
                self.subject_masses.append(cycle.subject.mass)
                cycles.append(cycle)
                deps.append(os.path.join(
                        self.study.config['results_path'],
                        mod_dir, cycle.trial.rel_path, 'mrs', 
                        cycle.name, self.costdir,
                        '%s_%s_mrs.mat' % (study.name, cycle.id))
                    )
        self.add_action(deps,
                [os.path.join(study.config['analysis_path'], 
                    self.peak_power_fpath),
                os.path.join(study.config['analysis_path'], 
                    self.avg_pos_power_fpath),
                os.path.join(study.config['analysis_path'], 
                    self.avg_neg_power_fpath),
                os.path.join(study.config['analysis_path'], 
                    self.total_power_fpath)], 
                self.aggregate_device_power, 
                cycles)

    def aggregate_device_power(self, file_dep, target, cycles):
        import numpy as np
        from scipy.interpolate import interp1d
        from collections import OrderedDict
        peak_norm_power = OrderedDict()
        avg_pos_norm_power = OrderedDict()
        avg_neg_norm_power = OrderedDict()
        device_array = list()
        subject_array = list()
        cond_array = list()
        cycle_array = list()
        dof_array = list()
        all_P = list()
        num_time_points = 400
        pgc = np.linspace(0, 100, num_time_points) # Percent gait cycle.
        for ifile, fpath in enumerate(file_dep):
            time = util.hdf2pandas(fpath, 'Time').round(4)
            time_exp = util.hdf2pandas(fpath, 'DatStore/time').round(4)
            df_Texo = util.hdf2pandas(fpath, 'DatStore/ExoTorques', 
                labels=self.study.dof_names)
            df_q_deg = util.hdf2pandas(fpath, 'DatStore/q_exp',
                labels=self.study.dof_names)
            import math
            df_q_rad = (math.pi / 180.0) * df_q_deg
            #df_q_reidx = df_q_rad.reindex(df_q_rad.index.union(time[0]))

            # Interpolate joint angles to match solution time domain
            f = interp1d(time_exp[0], df_q_rad, kind='cubic', axis=0)
            df_q = pd.DataFrame(f(time[0]), columns=self.study.dof_names)
            
            # Get angular velocities
            df_dq = df_q.diff().fillna(method='backfill')
            dt = time.diff().fillna(method='backfill')
            dt[1] = dt[0]
            dt[2] = dt[0]
            dt.columns = self.study.dof_names
            df_dqdt = df_dq / dt

            # Compute active device power (assuming constant moment arms)
            # P = F*v
            # l = l0 - sum{ri*qi}
            # v = dl/dt = -sum{ri*(dq/dt)i}
            # P = F*(-sum{ri*(dq/dt)i})
            # P = -sum{Mi*(dq/dt)i}
            df_P = df_Texo.multiply(df_dqdt, axis='index')

            # Get max value and normalize to subject mass
            Pmax_norm = df_P.max().sum() / self.subject_masses[ifile]

            # Get average positive power and normalize to subject mass
            df_Ppos = df_P.copy(deep=True)
            df_Ppos[df_Ppos < 0] = 0
            Pavg_pos_norm = df_Ppos.mean().sum() / self.subject_masses[ifile]

            # Get average negative power and normalize to subject mass
            df_Pneg = df_P.copy(deep=True)
            df_Pneg[df_Pneg > 0] = 0
            Pavg_neg_norm = df_Pneg.mean().sum() / self.subject_masses[ifile]

            this_mod = self.mod_for_file_dep[ifile]

            if not this_mod in peak_norm_power:
                peak_norm_power[this_mod] = list()
            peak_norm_power[this_mod].append(Pmax_norm)

            if not this_mod in avg_pos_norm_power:
                avg_pos_norm_power[this_mod] = list()
            avg_pos_norm_power[this_mod].append(Pavg_pos_norm)

            if not this_mod in avg_neg_norm_power:
                avg_neg_norm_power[this_mod] = list()
            avg_neg_norm_power[this_mod].append(Pavg_neg_norm)

            cycle = cycles[ifile]
            for dof_name in self.study.dof_names:
                device_array.append(this_mod)
                subject_array.append(cycle.subject.name)
                cond_array.append(cycle.condition.name)
                cycle_array.append(cycle.id)
                dof_array.append(dof_name)
                time_sqz = time.squeeze()
                new_time = np.linspace(time_sqz.iloc[0], time_sqz.iloc[-1], 
                    num_time_points)
                power = np.interp(new_time, time_sqz, df_P[dof_name].squeeze())
                all_P.append(power)

        index = pd.MultiIndex.from_tuples(self.multiindex_tuples,
                names=['subject', 'condition', 'cycle'])

        df_peak = pd.DataFrame(peak_norm_power, index=index)
        target_dir = os.path.dirname(target[0])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(target[0], 'w') as f:
            f.write('peak instantaneous positive power normalized by subject '
                'mass (W/kg)\n')
            df_peak.to_csv(f)

        df_avg_pos = pd.DataFrame(avg_pos_norm_power, index=index)
        target_dir = os.path.dirname(target[1])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(target[1], 'w') as f:
            f.write('average positive power normalized by subject mass (W/kg)\n')
            df_avg_pos.to_csv(f)

        df_avg_neg = pd.DataFrame(avg_neg_norm_power, index=index)
        target_dir = os.path.dirname(target[2])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(target[2], 'w') as f:
            f.write('average negative power normalized by subject mass (W/kg)\n')
            df_avg_neg.to_csv(f)

        # Convert from (n_cycles * n_dofs * n_muscles) x n_times
        #         to   n_times x (n_cycles * n_dofs * n_muscles)
        all_P_array = np.array(all_P).transpose()

        multiindex_arrays = [device_array, subject_array, cond_array, 
                             cycle_array, dof_array]
        columns = pd.MultiIndex.from_arrays(multiindex_arrays,
                names=['device','subject', 'condition', 'cycle', 'DOF'])

        all_P_df = pd.DataFrame(all_P_array, columns=columns, index=pgc)

        target_dir = os.path.dirname(target[3])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(target[3], 'w') as f:
            f.write('# all columns are device power normalized by subject '
                    'mass (N-m/kg).\n')
            all_P_df.to_csv(f)
        # How to read this in: df.read_csv(..., index_col=0, header=[0, 1, 2, 3],
        #                                  skiprows=1)

class TaskPlotMetabolicReductionVsPeakPower(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, mods, suffix=''):
        super(TaskPlotMetabolicReductionVsPeakPower, self).__init__(study)
        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction
        self.name = 'plot_metabolics_versus_power%s' % suffix
        self.mods = mods
        self.mods_act1DOF = list()
        self.mods_act2DOF = list()
        self.mods_act3DOF = list()
        self.met_fpath = os.path.join(study.config['analysis_path'],
            self.suffix_path, 'whole_body_metabolic_rates%s.csv' % suffix)
        self.power_fpath = os.path.join(study.config['analysis_path'],
            self.suffix_path, 'peak_power%s.csv' % suffix)

        self.actions += [self.create_device_lists_by_dof]

        self.add_action(
                [self.met_fpath, self.power_fpath],
                [os.path.join(study.config['analysis_path'],
                              self.suffix_path, 
                              'metabolics_versus_power_all%s.pdf' % suffix), 
                 os.path.join(study.config['analysis_path'],
                              self.suffix_path, 
                              'metabolics_versus_power_all%s.png' % suffix)],
                self.plot_metabolics_versus_power,
                self.mods
                )

        self.add_action(
                [self.met_fpath, self.power_fpath],
                [os.path.join(study.config['analysis_path'],
                              self.suffix_path, 
                              'metabolics_versus_power_1DOF%s.pdf' % suffix), 
                 os.path.join(study.config['analysis_path'],
                              self.suffix_path, 
                              'metabolics_versus_power_1DOF%s.png' % suffix)],
                self.plot_metabolics_versus_power,
                self.mods_act1DOF
                )

        self.add_action(
                [self.met_fpath, self.power_fpath],
                [os.path.join(study.config['analysis_path'],
                              self.suffix_path, 
                              'metabolics_versus_power_2DOF%s.pdf' % suffix), 
                 os.path.join(study.config['analysis_path'],
                              self.suffix_path, 
                              'metabolics_versus_power_2DOF%s.png' % suffix)],
                self.plot_metabolics_versus_power,
                self.mods_act2DOF
                )

        self.add_action(
                [self.met_fpath, self.power_fpath],
                [os.path.join(study.config['analysis_path'],
                              self.suffix_path, 
                              'metabolics_versus_power_3DOF%s.pdf' % suffix), 
                 os.path.join(study.config['analysis_path'],
                              self.suffix_path, 
                              'metabolics_versus_power_3DOF%s.png' % suffix)],
                self.plot_metabolics_versus_power,
                self.mods_act3DOF
                )

    def plot_metabolics_versus_power(self, file_dep, target, mods_list):

        # Process metabolic rate
        df_met = pd.read_csv(file_dep[0], index_col=[0, 1, 2], skiprows=1)
        df_met_walk2 = df_met.xs('walk2', level='condition')
        df_met_relchange = df_met_walk2.subtract(df_met_walk2['experiment'],
                axis='index').divide(df_met_walk2['experiment'], axis='index')
        df_met_relchange.drop('experiment', axis='columns', inplace=True)
        df_met_by_subjs = df_met_relchange.groupby(level='subject').mean()
        met_mean = df_met_by_subjs.mean()[self.mods] * 100
        met_std = df_met_by_subjs.std()[self.mods] * 100

        # Process positive instantaneous peak power
        df_power = pd.read_csv(file_dep[1], index_col=[0, 1, 2], skiprows=1)
        df_power_walk2 = df_power.xs('walk2', level='condition')
        df_power_by_subjs = df_power_walk2.groupby(level='subject').mean()
        power_mean = df_power_by_subjs.mean()[self.mods]
        power_std = df_power_by_subjs.std()[self.mods]

        fig = pl.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(1, 1, 1)
        for mod in mods_list:
            if 'all' in target[0]:
                ax.scatter(power_mean[mod], -met_mean[mod], color='black')
            else:
                ax.errorbar(power_mean[mod], -met_mean[mod], xerr=power_std[mod], 
                        yerr=met_std[mod], color='black', fmt='o')
            ax.text(power_mean[mod]+0.07, -met_mean[mod]+0.2, mod)

        ax.set_ylabel('reduction in average whole-body metabolic rate (%)')
        ax.set_xlabel('peak instantaneous positive device power (W/kg)')
        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[1], dpi=600)
        pl.close(fig)

    def create_device_lists_by_dof(self):

        for mod in self.mods:

            # List of possible DOFs that device assists
            dof_list = ['H','K','A']
            # Check that at least one DOF is assisted. Raise exception if not.
            for dof in dof_list:
                if dof in mod:
                    DOF_flag = 1
                    # Remove from list, DOFs only assisted once per device.
                    dof_list.remove(dof)
                    break

            if len(dof_list) == 3:
                raise Exception('3 DOF left in list, should have found '
                                'at least one. Mod: %s' % mod)

            # Check if two DOFs are assisted
            for dof in dof_list:
                if dof in mod:
                    DOF_flag = 2
                    dof_list.remove(dof)
                    break

            # Check if three DOFs are assisted
            for dof in dof_list:
                if dof in mod:
                    DOF_flag = 3

            # Append devices to appropriate lists
            if DOF_flag == 1:
                self.mods_act1DOF.append(mod)
            elif DOF_flag == 2:
                self.mods_act2DOF.append(mod)
            elif DOF_flag == 3:
                self.mods_act3DOF.append(mod)

class TaskPlotDeviceMetabolicRankings(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, mods, suffix=''):
        super(TaskPlotDeviceMetabolicRankings, self).__init__(study)
        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'plot_device_metabolic_rankings%s' % suffix
        self.mods = mods

        self.met_fpath = os.path.join(study.config['analysis_path'],
            self.suffix_path, 'whole_body_metabolic_rates%s.csv' % suffix)
        self.pdf_fpath = os.path.join(study.config['analysis_path'],
                self.suffix_path, 'device_metabolic_rankings%s.pdf' % suffix)
        self.png_fpath = os.path.join(study.config['analysis_path'],
                self.suffix_path, 'device_metabolic_rankings%s.png' % suffix)

        self.add_action(
                [self.met_fpath],
                [self.pdf_fpath, self.png_fpath],
                self.plot_device_metabolic_rankings,
                )

    def plot_device_metabolic_rankings(self, file_dep, target):
        
        # Process metabolic rate
        # -----------------------
        # The first three "columns" form a MultiIndex.
        # Skip the first line, which has comments.
        df = pd.read_csv(file_dep[0], index_col=[0, 1, 2], skiprows=1)

        # Using walk2 condition to start. TODO: update if adding 
        # more conditions.
        # "xs" stands for "cross section"
        df_walk2 = df.xs('walk2', level='condition')

        # Subtract the no-assist cost from all other columns.
        df_relchange = df_walk2.subtract(df_walk2['experiment'],
                axis='index').divide(df_walk2['experiment'], axis='index')

        # Delete the 'experiments' column, whicih is no longer needed.
        df_relchange.drop('experiment', axis='columns', inplace=True)

        # Average over cycles.
        df_by_subjs = df_relchange.groupby(level='subject').mean()

        # Relative change in metabolic cost
        met_relchange_pcent_mean = df_by_subjs.mean()[self.mods] * 100
        met_relchange_pcent_std = df_by_subjs.std()[self.mods] * 100

        met_relchange_pcent_mean_sort = met_relchange_pcent_mean.sort_values(0)
        mods_sort = list()
        for key in met_relchange_pcent_mean_sort.keys():
            mods_sort.append(key)

        met_relchange_pcent_std_sort = met_relchange_pcent_std[mods_sort]

        # Plot changes in metabolic rate
        fig = pl.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(1, 1, 1)
        ax.axvline(color='k', linewidth=1.5, zorder=1)
   
        y_pos = np.arange(len(mods_sort))

        ax.barh(y_pos, met_relchange_pcent_mean_sort, 
            xerr=met_relchange_pcent_std_sort, align='center',
            color=colors['darkslateblue'], ecolor='black')
        for i, v in enumerate(met_relchange_pcent_mean_sort):
            color = 'green' if (v < 0) else 'red'
            shift = -6 if (v < 0) else 3
            ax.text(v + shift, i, '%.2f' % v, va='center',
                color=color, fontweight='bold')
        textstr = 'Subjects included: \n'
        for subj in list(df_by_subjs.index):
            textstr += '  ' + subj + '\n'
        
        props = dict(boxstyle='round', facecolor='wheat')
        ax.text(0.10, 0.25, textstr, transform=ax.transAxes, fontsize=14,
            bbox=props)

        ax.set_xticks(np.linspace(-50,15,14))
        ax.set_xticklabels(np.linspace(-50,15,14))
        ax.set_yticks(y_pos)
        ax.set_ylim(y_pos[0]-1, y_pos[-1]+1)
        ax.invert_yaxis()
        ax.set_yticklabels(mods_sort)
        ax.set_title('Percent Change in Metabolic Rate')
        ax.grid()
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[1], dpi=600)
        pl.close(fig)

def aggregate_moments(file_dep, target, cond_name, cycles):

    ## Percent gait cycle.
    num_time_points = 400
    pgc = np.linspace(0, 100, num_time_points)

    muscle_names = None

    subject_array = list()
    cycle_array = list()
    dof_array = list()
    muscle_array = list()
    all_data = list()
    for icycle, fpath in enumerate(file_dep):
        cycle = cycles[cond_name][icycle]
        #for cycle_info in cycles:
        cycle_df = pd.read_csv(fpath, index_col=0, header=[0, 1], skiprows=1)
        # Convert time to percent gait cycle.
        x = np.linspace(cycle.gl.cycle_start,
                cycle.gl.cycle_end, num_time_points)
        for column in cycle_df.columns:
            dof, actuator = column
            subject_array.append(cycle.subject.name)
            cycle_array.append(cycle.id)
            dof_array.append(dof)
            muscle_array.append(actuator)
            moment = np.interp(pgc, cycle_df.index, cycle_df[column])
            all_data.append(moment)
    # Convert from (n_cycles * n_dofs * n_muscles) x n_times
    #         to   n_times x (n_cycles * n_dofs * n_muscles)
    all_data_array = np.array(all_data).transpose()

    multiindex_arrays = [subject_array, cycle_array, dof_array, muscle_array]
    columns = pd.MultiIndex.from_arrays(multiindex_arrays,
            names=['subject', 'cycle', 'DOF', 'actuator'])

    all_data_df = pd.DataFrame(all_data_array, columns=columns, index=pgc)

    target_dir = os.path.dirname(target[0])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with file(target[0], 'w') as f:
        f.write('# all columns are moments normalized by subject '
                'mass (N-m/kg).\n')
        all_data_df.to_csv(f)
    # How to read this in: df.read_csv(..., index_col=0, header=[0, 1, 2, 3],
    #                                  skiprows=1)

class TaskAggregateMomentsExperiment(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects=None, 
            cond_names=['walk1','walk2','walk3','walk4']):
        super(TaskAggregateMomentsExperiment, self).__init__(study)
        self.name = 'aggregate_moments_experiment'
        self.doc = 'Aggregate no-mod actuator moments into a data file.'

        suffix = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 

        if subjects == None:
            subjects = [s.num for s in study.subjects]

        self.cycles = dict()
        for cond_name in cond_names:
            self.cycles[cond_name] = list()
            deps = []
            for subject in study.subjects:
                if not subject.num in subjects: continue
                cond = subject.get_condition(cond_name)
                if not cond: continue
                # We know there is only one overground trial, but perhaps it
                # has not yet been added for this subject.
                assert len(cond.trials) <= 1
                if len(cond.trials) == 1:
                    trial = cond.trials[0]
                    for cycle in trial.cycles:
                        if study.cycles_to_plot:
                            if not (cycle.name in study.cycles_to_plot): continue
                        self.cycles[cond_name].append(cycle)

                        # Moment file paths
                        fpath = os.path.join(trial.results_exp_path, 'mrs',
                            cycle.name, self.costdir,
                             '%s_%s_mrs_moments.csv' % (study.name, cycle.id))
                        deps.append(fpath)

            self.add_action(deps,
                    [
                        os.path.join(study.config['results_path'], 
                            'experiments',
                            'experiment_%s_moments%s.csv' % (cond_name, suffix)),
                        ],
                    aggregate_moments, cond_name, self.cycles)

class TaskAggregateMomentsMod(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, mod, subjects=None, 
            cond_names=['walk1','walk2','walk3','walk4']):
        super(TaskAggregateMomentsMod, self).__init__(study)
        self.mod_dir = mod.replace('/','\\\\')
        if len(mod.split('/')) > 1:
            self.mod_name = '_'.join(mod.split('/'))
        else:
            self.mod_name = mod

        suffix = '_' + self.mod_name
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 

        self.name = 'aggregate_moments%s' % suffix
        self.doc = 'Aggregate actuator moments into a data file.'
        self.cond_names = cond_names

        if subjects == None:
            subjects = [s.num for s in study.subjects]

        self.cycles = dict()
        for cond_name in cond_names:
            self.cycles[cond_name] = list()
            deps = []
            for subject in study.subjects:
                if not subject.num in subjects: continue
                cond = subject.get_condition(cond_name)
                if not cond: continue
                # We know there is only one overground trial, but perhaps it
                # has not yet been added for this subject.
                assert len(cond.trials) <= 1
                if len(cond.trials) == 1:
                    trial = cond.trials[0]
                    for cycle in trial.cycles:
                        if study.cycles_to_plot:
                            if not (cycle.name in study.cycles_to_plot): continue
                        self.cycles[cond_name].append(cycle)
                        deps.append(os.path.join(
                                self.study.config['results_path'],
                                self.mod_dir,
                                trial.rel_path, 'mrs', cycle.name, self.costdir,
                                '%s_%s_mrs_moments.csv' % (study.name,
                                    cycle.id))
                                )

            self.add_action(deps,
                    [os.path.join(study.config['results_path'], 
                        self.mod_dir,
                        '%s_%s_moments_%s.csv' % (self.mod_name,
                            cond_name, self.costdir)),
                        ],
                    aggregate_moments, cond_name, self.cycles)

class TaskPlotMoments(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, agg_task):
        super(TaskPlotMoments, self).__init__(study)
        self.agg_task = agg_task
        self.task_name = self.agg_task.mod_name if hasattr(self.agg_task, 
            'mod_name') else 'experiment' 
        suffix = '_' + self.task_name
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'plot_moment_breakdown%s' % suffix
        self.doc = 'Plot joint moment breakdown by muscle and device moments'

        for icond, agg_target in enumerate(agg_task.targets):
            # This assumes csv_task.targets and csv_task.cycles hold cycles in
            # the same order.
            self.add_action([], [], 
                    # [agg_target.replace('.csv', '.pdf')],
                    self.plot_joint_moment_breakdown, agg_target)

    def plot_joint_moment_breakdown(self, file_dep, target, agg_target):

        df_all = pd.read_csv(agg_target, index_col=0,
                header=[0, 1, 2, 3], skiprows=1)

        # Average over cycles.
        # axis=1 for columns (not rows).
        df_by_subj_dof_musc = df_all.groupby(
                level=['subject', 'DOF', 'actuator'], axis=1).mean()
        df_by_subj_dof_musc *= -1
        df_mean = df_by_subj_dof_musc.groupby(level=['DOF', 'actuator'],
                axis=1).mean()
        df_std = df_by_subj_dof_musc.groupby(level=['DOF', 'actuator'],
                axis=1).std()

        pgc = df_mean.index

        import seaborn.apionly as sns
        palette = sns.color_palette('muted', 9)

        muscles = ['glut_max2_r', 'psoas_r', 'semimem_r', 'rect_fem_r',
                   'bifemsh_r', 'vas_int_r', 'med_gas_r', 'soleus_r', 
                   'tib_ant_r']
        colors = {muscles[i]: palette[i] for i in range(9)}
        colors['net'] = 'black' #(0.7,) * 3 # light gray
        colors['device'] = 'green'

        fig = pl.figure(figsize=(9, 3.75))
        dof_names = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
        ylabels = ['hip extension', 'knee extension', 'ankle plantarflexion']
        nice_muscle_names_dict = self.study.nice_muscle_names_dict
        act_dict = {'net': 'net',
                    'device': 'device'}

        nice_act_names = dict()
        nice_act_names.update(nice_muscle_names_dict)
        nice_act_names.update(act_dict)

        act_names = df_mean.columns.levels[1]
        def plot(column_key, act_name):
            if column_key in df_mean.columns:
                y_mean = df_mean[column_key]
                y_std = df_std[column_key]
                if act_name == 'device':
                    ax.plot(pgc, y_mean, color=colors[act_name],
                            label=nice_act_names[act_name],
                            linestyle='--')
                    ax.fill_between(pgc, y_mean-y_std, y_mean+y_std,
                        color=colors[act_name], alpha=0.3)
                else:
                    ax.plot(pgc, y_mean, color=colors[act_name],
                        label=nice_act_names[act_name])

        for idof, dof_name in enumerate(dof_names):
            ax = fig.add_subplot(1, len(dof_names), idof + 1)
            ax.axhline(color='k', linewidth=0.5, zorder=0)
            plot((dof_name, 'net'), 'net')
            for iact, act_name in enumerate(act_names):
                if act_name == 'net': continue
                # Not all DOFs have each actuator column (e.g., the hip flexion
                # DOF does not have a soleus column.
                column_key = (dof_name, act_name)
                plot(column_key, act_name)
            ax.legend(frameon=False, fontsize=6)
            ax.set_xlim(0, 100)
            ax.set_ylim(-1.1, 2.0)
            if idof > 0:
                ax.set_yticklabels([])
            ax.set_ylabel('%s (N-m/kg)' % ylabels[idof])
            ax.set_xlabel('time (% gait cycle)')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        fig.tight_layout()
        fig.savefig(agg_target.replace('.csv', '.pdf'))
        fig.savefig(agg_target.replace('.csv', '.png'), dpi=600)
        pl.close(fig)

def aggregate_muscle_data(file_dep, target, cond_name, cycles):

    num_time_points = 400
    pgc = np.linspace(0, 100, num_time_points)

    muscle_names = None

    subject_array = list()
    cycle_array = list()
    muscle_array = list()
    all_exc = list()
    all_act = list()
    all_fce = list()
    all_fpe = list()
    all_lMtilde = list()
    all_vMtilde = list()
    all_Pce = list() # active fiber power
    all_pce = list() # normalized active fiber power
    for icycle, fpath in enumerate(file_dep):
        cycle = cycles[cond_name][icycle]

        muscle_names = util.hdf2list(fpath, 'MuscleNames', type=str)
        exc_df = util.hdf2pandas(fpath, 'MExcitation', labels=muscle_names)
        act_df = util.hdf2pandas(fpath, 'MActivation', labels=muscle_names)
        fce_df = util.hdf2pandas(fpath, 'MuscleData/fce', labels=muscle_names)
        Fce_df = util.hdf2pandas(fpath, 'MuscleData/Fce', labels=muscle_names)
        fpe_df = util.hdf2pandas(fpath, 'MuscleData/fpe', labels=muscle_names)
        lMtilde_df = util.hdf2pandas(fpath, 'MuscleData/lMtilde',
            labels=muscle_names)
        lM_df = util.hdf2pandas(fpath, 'MuscleData/lM', labels=muscle_names)
        # Negate fiber velocities so:
        #   positive <==> shortening
        #   negative <==> lengthening
        # This matches the convention in Arnold et al. 2013
        vMtilde_df = -util.hdf2pandas(fpath, 'MuscleData/vMtilde',
            labels=muscle_names)
        vM_df = -util.hdf2pandas(fpath, 'MuscleData/vM', labels=muscle_names)
        pce_df = Fce_df.multiply(vM_df) / cycle.subject.mass
        Pce_df = Fce_df.multiply(vM_df)

        exc_index = np.linspace(0, 100, len(exc_df.index.values))
        act_index = np.linspace(0, 100, len(act_df.index.values))
        fce_index = np.linspace(0, 100, len(fce_df.index.values))
        fpe_index = np.linspace(0, 100, len(fpe_df.index.values))
        lMtilde_index = np.linspace(0, 100, len(lMtilde_df.index.values))
        vMtilde_index = np.linspace(0, 100, len(vMtilde_df.index.values))
        pce_index = np.linspace(0, 100, len(pce_df.index.values))
        Pce_index = np.linspace(0, 100, len(Pce_df.index.values))
        for muscle in exc_df.columns:
            subject_array.append(cycle.subject.name)
            cycle_array.append(cycle.id)
            muscle_array.append(muscle)

            exc = np.interp(pgc, exc_index, exc_df[muscle])
            act = np.interp(pgc, act_index, act_df[muscle])
            fce = np.interp(pgc, fce_index, fce_df[muscle])
            fpe = np.interp(pgc, fpe_index, fpe_df[muscle])
            lMtilde = np.interp(pgc, lMtilde_index, lMtilde_df[muscle])
            vMtilde = np.interp(pgc, vMtilde_index, vMtilde_df[muscle])
            Pce = np.interp(pgc, pce_index, Pce_df[muscle])
            pce = np.interp(pgc, pce_index, pce_df[muscle])

            all_exc.append(exc)
            all_act.append(act)
            all_fce.append(fce)
            all_fpe.append(fpe)
            all_lMtilde.append(lMtilde)
            all_vMtilde.append(vMtilde)
            all_Pce.append(Pce)
            all_pce.append(pce)

    all_exc_array = np.array(all_exc).transpose()
    all_act_array = np.array(all_act).transpose()
    all_fce_array = np.array(all_fce).transpose()
    all_fpe_array = np.array(all_fpe).transpose()
    all_lMtilde_array = np.array(all_lMtilde).transpose()
    all_vMtilde_array = np.array(all_vMtilde).transpose()
    all_pce_array = np.array(all_pce).transpose()
    all_Pce_array = np.array(all_Pce).transpose()

    multiindex_arrays = [subject_array, cycle_array, muscle_array]
    columns = pd.MultiIndex.from_arrays(multiindex_arrays,
            names=['subject', 'cycle', 'muscle'])

    all_exc_df = pd.DataFrame(all_exc_array, columns=columns, index=pgc)
    target_dir = os.path.dirname(target[0])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with file(target[0], 'w') as f:
        f.write('# all columns are muscle excitations.\n')
        all_exc_df.to_csv(f)

    all_act_df = pd.DataFrame(all_act_array, columns=columns, index=pgc)
    target_dir = os.path.dirname(target[1])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with file(target[1], 'w') as f:
        f.write('# all columns are muscle activations.\n')
        all_act_df.to_csv(f)

    all_fce_df = pd.DataFrame(all_fce_array, columns=columns, index=pgc)
    target_dir = os.path.dirname(target[2])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with file(target[2], 'w') as f:
        f.write('# all columns are normalized muscle active fiber forces.\n')
        all_fce_df.to_csv(f)

    all_fpe_df = pd.DataFrame(all_fpe_array, columns=columns, index=pgc)
    target_dir = os.path.dirname(target[3])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with file(target[3], 'w') as f:
        f.write('# all columns are normalized muscle passive fiber forces.\n')
        all_fpe_df.to_csv(f)

    all_lMtilde_df = pd.DataFrame(all_lMtilde_array, columns=columns, index=pgc)
    target_dir = os.path.dirname(target[4])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with file(target[4], 'w') as f:
        f.write('# all columns are normalized muscle fiber lengths.\n')
        all_lMtilde_df.to_csv(f)

    all_vMtilde_df = pd.DataFrame(all_vMtilde_array, columns=columns, index=pgc)
    target_dir = os.path.dirname(target[5])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with file(target[5], 'w') as f:
        f.write('# all columns are normalized muscle fiber velocities.\n')
        all_vMtilde_df.to_csv(f)

    all_pce_df = pd.DataFrame(all_pce_array, columns=columns, index=pgc)
    target_dir = os.path.dirname(target[6])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with file(target[6], 'w') as f:
        f.write('# all columns are normalized muscle active fiber powers.\n')
        all_pce_df.to_csv(f)

    all_Pce_df = pd.DataFrame(all_Pce_array, columns=columns, index=pgc)
    target_dir = os.path.dirname(target[7])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with file(target[7], 'w') as f:
        f.write('# all columns are muscle active fiber powers.\n')
        all_Pce_df.to_csv(f)
    # How to read this in: df.read_csv(..., index_col=0, header=[0, 1, 2],
    #                                  skiprows=1)

class TaskAggregateMuscleDataExperiment(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects=None, 
            cond_names=['walk1','walk2','walk3','walk4'],
            alt_tool_name=None):
        super(TaskAggregateMuscleDataExperiment, self).__init__(study)
        suffix = ''
        cost = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            cost = '_' + study.costFunction
            self.costdir = study.costFunction

        if alt_tool_name:
            alt_tool_tag = '_%s' % alt_tool_name
            suffix += alt_tool_tag
            tool = alt_tool_name
        else: 
            alt_tool_tag = ''
            tool = 'mrs'

        self.name = 'aggregate_muscle_data_experiment%s' % suffix
        self.doc = 'Aggregate muscle data into a data file.'

        if subjects == None:
            subjects = [s.name for s in study.subjects]

        self.cycles = dict()
        for cond_name in cond_names:
            self.cycles[cond_name] = list()
            deps = []
            for subject in study.subjects:
                if not subject.name in subjects: continue
                cond = subject.get_condition(cond_name)
                if not cond: continue
                # We know there is only one overground trial, but perhaps it
                # has not yet been added for this subject.
                assert len(cond.trials) <= 1
                if len(cond.trials) == 1:
                    trial = cond.trials[0]
                    for cycle in trial.cycles:
                        if study.test_cycles:
                            if not (cycle.name in study.test_cycles): 
                                continue
                        self.cycles[cond_name].append(cycle)

                        # Results MAT file paths
                        fpath = os.path.join(study.config['results_path'], 
                            'experiments', subject.name, cond.name, tool, 
                            cycle.name, self.costdir,
                            '%s_%s_mrs.mat' % (study.name, cycle.id))
                        deps.append(fpath)

            self.add_action(deps,
                    [os.path.join(study.config['results_path'], 'experiments',
                        'experiment_%s_excitations%s.csv' % (
                            cond_name, suffix)),
                    os.path.join(study.config['results_path'], 'experiments',
                        'experiment_%s_activations%s.csv' % (
                            cond_name, suffix)),
                    os.path.join(study.config['results_path'], 'experiments',
                        'experiment_%s_norm_act_fiber_forces%s.csv' % (
                            cond_name, suffix)),
                    os.path.join(study.config['results_path'], 'experiments',
                        'experiment_%s_norm_pass_fiber_forces%s.csv' % (
                            cond_name, suffix)),
                    os.path.join(study.config['results_path'], 'experiments',
                        'experiment_%s_norm_fiber_lengths%s.csv' % (
                            cond_name, suffix)),
                    os.path.join(study.config['results_path'], 'experiments',
                        'experiment_%s_norm_fiber_velocities%s.csv' % (
                            cond_name, suffix)),
                    os.path.join(study.config['results_path'], 'experiments',
                        'experiment_%s_norm_fiber_powers%s.csv' % (
                            cond_name, suffix)),
                    os.path.join(study.config['results_path'], 'experiments',
                        'experiment_%s_fiber_powers%s.csv' % (
                            cond_name, suffix)),
                    ],
                    aggregate_muscle_data, cond_name, self.cycles)

class TaskAggregateMuscleDataMod(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, mod, subjects=None, 
            cond_names=['walk1','walk2','walk3','walk4'], 
            alt_tool_name=None):
        super(TaskAggregateMuscleDataMod, self).__init__(study)
        self.mod_dir = mod.replace('/','\\\\')
        if len(mod.split('/')) > 1:
            self.mod_name = '_'.join(mod.split('/'))
        else:
            self.mod_name = mod

        suffix = '_' + self.mod_name
        cost = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            cost = '_' + study.costFunction
            self.costdir = study.costFunction

        if alt_tool_name:
            alt_tool_tag = '_%s' % alt_tool_name
            suffix += alt_tool_tag
            tool = alt_tool_name
        else: 
            alt_tool_tag = ''
            tool = 'mrs'

        self.name = 'aggregate_muscle_data%s' % suffix
        self.doc = 'Aggregate muscle data into a data file.'

        if subjects == None:
            subjects = [s.name for s in study.subjects]

        self.cycles = dict()
        for cond_name in cond_names:
            self.cycles[cond_name] = list()
            deps = []
            for subject in study.subjects:
                if not subject.name in subjects: continue
                cond = subject.get_condition(cond_name)
                if not cond: continue
                # We know there is only one overground trial, but perhaps it
                # has not yet been added for this subject.
                assert len(cond.trials) <= 1
                if len(cond.trials) == 1:
                    trial = cond.trials[0]
                    for cycle in trial.cycles:
                        if study.test_cycles:
                            if not (cycle.name in study.test_cycles): 
                                continue
                        self.cycles[cond_name].append(cycle)

                        # Results MAT file paths
                        fpath = os.path.join(study.config['results_path'], 
                            self.mod_dir, subject.name, cond.name, tool, 
                            cycle.name, self.costdir,
                            '%s_%s_mrs.mat' % (study.name, cycle.id))
                        deps.append(fpath)

            self.add_action(deps,
                    [os.path.join(study.config['results_path'], 
                        self.mod_dir,'%s%s_%s_excitations%s.csv' % (
                            self.mod_name, alt_tool_tag, cond_name, cost)),
                    os.path.join(study.config['results_path'], 
                        self.mod_dir,'%s%s_%s_activations%s.csv' % (
                            self.mod_name, alt_tool_tag, cond_name, cost)),
                    os.path.join(study.config['results_path'], 
                        self.mod_dir,'%s%s_%s_norm_act_fiber_forces%s.csv' % (
                            self.mod_name, alt_tool_tag, cond_name, cost)),
                    os.path.join(study.config['results_path'], 
                        self.mod_dir,'%s%s_%s_norm_pass_fiber_forces%s.csv' % (
                            self.mod_name, alt_tool_tag, cond_name, cost)),
                    os.path.join(study.config['results_path'], 
                        self.mod_dir,'%s%s_%s_norm_fiber_lengths%s.csv' % (
                            self.mod_name, alt_tool_tag, cond_name, cost)),
                    os.path.join(study.config['results_path'], 
                        self.mod_dir,'%s%s_%s_norm_fiber_velocities%s.csv' % (
                            self.mod_name, alt_tool_tag, cond_name, cost)),
                    os.path.join(study.config['results_path'], 
                        self.mod_dir,'%s%s_%s_norm_fiber_powers%s.csv' % (
                            self.mod_name, alt_tool_tag, cond_name, cost)),
                    os.path.join(study.config['results_path'], 
                        self.mod_dir,'%s%s_%s_fiber_powers%s.csv' % (
                            self.mod_name, alt_tool_tag, cond_name, cost)),
                    ],
                    aggregate_muscle_data, cond_name, self.cycles)

class TaskPlotMuscleData(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, agg_task):
        super(TaskPlotMuscleData, self).__init__(study)
        self.agg_task = agg_task
        self.task_name = self.agg_task.mod_name if hasattr(self.agg_task, 
            'mod_name') else 'experiment' 
        suffix = '_' + self.task_name
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'plot_muscle_data%s' % suffix
        self.doc = 'Plot muscle data for experiment and mod tasks'

        for icond, agg_target in enumerate(agg_task.targets):
            # This assumes csv_task.targets and csv_task.cycles hold cycles in
            # the same order.
            # self.agg_target = agg_target
            # self.actions += [self.plot_muscle_data]
            # print agg_target
            self.add_action([],[],
                            self.plot_muscle_data,
                            agg_target)

    def plot_muscle_data(self, file_dep, target, agg_target):

        df_agg = pd.read_csv(agg_target, index_col=0, header=[0, 1, 2], 
            skiprows=1)
        # Average over cycles.
        df_agg_by_subj_musc = df_agg.groupby(level=['subject', 'muscle'], 
            axis=1).mean()
        df_agg_mean = df_agg_by_subj_musc.groupby(level=['muscle'], 
            axis=1).mean()
        df_agg_std = df_agg_by_subj_musc.groupby(level=['muscle'], axis=1).std()

        if not (self.task_name == 'experiment'):
            agg_fname = os.path.basename(agg_target)
            agg_dir = os.path.dirname(agg_target)
            exp_fname = agg_fname.replace(self.task_name, 'experiment')
            exp_dir = agg_dir.replace(self.task_name, 'experiments')

            exp_fpath = os.path.join(exp_dir, exp_fname)

            df_exp = pd.read_csv(exp_fpath, index_col=0, header=[0, 1, 2], 
                skiprows=1)
            # Average over cycles.
            df_exp_by_subj_musc = df_exp.groupby(level=['subject', 'muscle'], 
                axis=1).mean()
            df_exp_mean = df_exp_by_subj_musc.groupby(level=['muscle'], 
                axis=1).mean()
            df_exp_std = df_exp_by_subj_musc.groupby(level=['muscle'], 
                axis=1).std()
            exp_pgc = df_exp_mean.index

        agg_pgc = df_agg_mean.index
        muscles = self.study.muscle_names
        fig = pl.figure(figsize=(8, 8))
        nice_muscle_names_dict = self.study.nice_muscle_names_dict

        for imusc, musc_name in enumerate(muscles):
            side_len = np.ceil(np.sqrt(len(muscles)))
            ax = fig.add_subplot(side_len, side_len, imusc + 1)
            ax.axhline(color='k', linewidth=0.5, zorder=0)
            agg_mean_musc = df_agg_mean[musc_name]
            agg_std_musc = df_agg_std[musc_name]
            ax.plot(agg_pgc, agg_mean_musc, color='blue', linestyle='-',
                label='assisted', zorder=2)
            ax.fill_between(agg_pgc, agg_mean_musc-agg_std_musc, 
                agg_mean_musc+agg_std_musc, color='blue', alpha=0.3, zorder=2)
            if not (self.task_name == 'experiment'):
                exp_mean_musc = df_exp_mean[musc_name]
                exp_std_musc = df_exp_std[musc_name]
                ax.plot(exp_pgc, exp_mean_musc, color='red', linestyle='-',
                    label='unassisted', zorder=1)
                ax.fill_between(exp_pgc, exp_mean_musc-exp_std_musc, 
                    exp_mean_musc+exp_std_musc, color='red', alpha=0.3,
                    zorder=1)

                ax.legend(frameon=False, fontsize=6)

            ax.set_xlim(0, 100)
            if 'norm_fiber_lengths' in agg_target:
                ax.set_ylim(0, 2.0)
                ax.set_yticks(np.linspace(0, 2, 9))
                ax.axhline(y=0.5, color='k', linewidth=0.5, ls='--', zorder=0)
                ax.axhline(y=1.0, color='k', linewidth=0.5, ls='--', zorder=0)
                ax.axhline(y=1.5, color='k', linewidth=0.5, ls='--', zorder=0)
            elif 'norm_fiber_velocities' in agg_target:
                ax.set_ylim(-0.75, 0.75)
                ax.set_yticks(np.linspace(-0.75, 0.75, 7))
                ax.axhline(y=-0.25, color='k', linewidth=0.5, ls='--', zorder=0)
                ax.axhline(y=0.25, color='k', linewidth=0.5, ls='--', zorder=0)
            elif 'norm_fiber_powers' in agg_target:
                ax.set_ylim(-1.25, 1.25)
                ax.set_yticks(np.linspace(-1.25, 1.25, 11))
                ax.axhline(y=-0.25, color='k', linewidth=0.5, ls='--', zorder=0)
                ax.axhline(y=0.25, color='k', linewidth=0.5, ls='--', zorder=0)
            elif any(s in agg_target for s in ['activ', 'exc', 'forces']):
                ax.set_ylim(0, 1.0)
            ax.set_title(nice_muscle_names_dict[musc_name])
            ax.set_xlabel('time (% gait cycle)')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        fig.tight_layout()
        fig.savefig(agg_target.replace('.csv', '.pdf'))
        fig.savefig(agg_target.replace('.csv', '.png'), dpi=600)
        pl.close(fig)

class TaskCopyEMGData(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study):
        super(TaskCopyEMGData, self).__init__(study)
        self.name = '%s_copy_emg_data' % study.name
        self.data_path = self.study.config['motion_capture_data_path']  
        self.results_path = self.study.config['results_path']
        self.cond_map = {
                    'walk1' : 'Walk_100',
                    'walk2' : 'Walk_125',
                    'walk3' : 'Walk_150',
                    'walk4' : 'Walk_175',
                    'run1' : 'Run_200',
                    'run2' : 'Run_300',
                    'run3' : 'Run_400',
                    'run4' : 'Run_500'
                    }
        self.cond_map2 = {
                    'walk1' : 'walk100',
                    'walk2' : 'walk125',
                    'walk3' : 'walk150',
                    'walk4' : 'walk175',
                    'run1' : 'run200',
                    'run2' : 'run300',
                    'run3' : 'run400',
                    'run4' : 'run500'
                    }


        self.actions += [self.copy_emg_data]

    def copy_emg_data(self):
        for subject in self.study.subjects:
            for cond in subject.conditions:

                if cond.name=='static': continue

                if self.cond_map2[cond.name] in subject.cond_args:
                    args = subject.cond_args[self.cond_map2[cond.name]]
                    num_tag = args[0]
                else:
                    num_tag = 2

                emg_fpath = os.path.join(self.data_path, subject.name, 
                    'Results', self.cond_map[cond.name], 
                    '%s%02i_gait_controls.sto' % (self.cond_map[cond.name],
                        num_tag))
                states_fpath = os.path.join(self.data_path, subject.name, 
                    'Results', self.cond_map[cond.name], 
                    '%s%02i_gait_states.sto' % (self.cond_map[cond.name],
                        num_tag))

                emg = util.storage2numpy(emg_fpath)
                names = emg.dtype.names
                muscle_names = [m for m in names if not m=='time']
                states = util.storage2numpy(states_fpath)
                time = states['time']
                knee_angle_r = states['knee_angle_r']

                r_strikes, l_strikes, r_offs, l_offs = \
                    util.gait_landmarks_from_grf(states_fpath,
                        right_grfy_column_name='knee_angle_r',
                        left_grfy_column_name='knee_angle_l',
                        threshold=0.075)

                from scipy.signal import argrelmin
                idxs = argrelmin(knee_angle_r)[0]                
                if subject.name=='subject01':
                    strikes = [time[i] for i in idxs if knee_angle_r[i] < 0.075]
                elif subject.name=='subject02':
                    strikes = [time[i] for i in idxs if knee_angle_r[i] > 0.13]
                elif subject.name=='subject04':
                    strikes = [0.272, 1.379, 2.528, 3.620]
                elif subject.name=='subject18':
                    strikes = [0.284, 1.381, 2.453, 3.517]
                elif subject.name=='subject19':
                    strikes = [0.581, 1.637, 2.696, 3.759]

                cycle_array = list()
                muscle_array = list()
                emg_data = list()
                for i, cycle in enumerate(cond.trials[0].cycles):
                    for muscle in muscle_names:
                        cycle_array.append(cycle.id)
                        muscle_array.append(muscle)
                        x = np.linspace(strikes[i], strikes[i+1], 400)
                        emg_interp = np.interp(x, time, emg[muscle])
                        pgc = np.linspace(0, 100, 400)
                        emg_interp = np.interp(pgc, pgc, emg_interp)
                        emg_data.append(emg_interp)

                emg_data_array = np.array(emg_data).transpose()

                multiindex_arrays = [cycle_array, muscle_array]
                columns = pd.MultiIndex.from_arrays(multiindex_arrays,
                names=['cycle', 'muscle'])

                all_exc_df = pd.DataFrame(emg_data_array, columns=columns, 
                    index=pgc)
                emg_target_fpath = os.path.join(self.results_path,
                    'experiments', subject.name, cond.name, 'expdata',
                    'processed_emg.csv')
                target_dir = os.path.dirname(emg_target_fpath)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                with file(emg_target_fpath, 'w') as f:
                    f.write('# all columns are processed EMG data.\n')
                    all_exc_df.to_csv(f)

class TaskValidateMuscleActivity(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, cond_names=['walk1','walk2','walk3','walk4']):
        super(TaskValidateMuscleActivity, self).__init__(study)
        suffix = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'validate_muscle_activity%s' % suffix
        self.doc = 'Plot muscle activity from simulation against EMG data.'
        self.results_path = study.config['results_path']
        self.validate_path = os.path.join(study.config['validate_path'], 
            'muscle_activity')

        for cond_name in cond_names:
            for subject in study.subjects:
                emg_fpath = os.path.join(self.results_path, 'experiments',
                    subject.name, cond_name, 'expdata', 'emg_with_headers.sto')
                exc_fpath = os.path.join(self.results_path, 'experiments',
                    'experiment_%s_excitations%s.csv' % (cond_name, suffix))
                act_fpath = os.path.join(self.results_path, 'experiments',
                    'experiment_%s_activations%s.csv' % (cond_name, suffix))

                val_fname = os.path.join(self.validate_path, 
                    '%s_%s_emg_validation%s' % (subject.name, cond_name, 
                        suffix))
                
                self.add_action([emg_fpath, exc_fpath, act_fpath],
                                [val_fname],
                                self.validate_muscle_activity,
                                cond_name, subject)

    def validate_muscle_activity(self, file_dep, target, cond_name, subject):

        muscles = self.study.muscle_names
        fig = pl.figure(figsize=(12, 12))
        nice_muscle_names_dict = self.study.nice_muscle_names_dict

        emg_map = {
                'bflh_r': [],
                'gaslat_r': [],
                'gasmed_r': 'med_gas_r',
                'glmax1_r': [],
                'glmax2_r': 'glut_max2_r',
                'glmax3_r': [],
                'glmed1_r': [],
                'glmed2_r': [],
                'glmed3_r': [],
                'recfem_r': 'rect_fem_r',
                'semimem_r': 'semimem_r',
                'semiten_r': 'semimem_r',
                'soleus_r': 'soleus_r',
                'tibant_r': 'tib_ant_r',
                'vaslat_r': 'vas_int_r',
                'vasmed_r': 'vas_int_r', 
        }

        emg_muscles = ['bflh_r', 'gaslat_r', 'gasmed_r', 'glmax1_r', 'glmax2_r',
                       'glmax3_r', 'glmed1_r', 'glmed2_r', 'glmed3_r', 
                       'recfem_r', 'semimem_r', 'semiten_r', 'soleus_r',
                       'tibant_r', 'vaslat_r', 'vasmed_r']

        # df_emg = pd.read_csv(file_dep[0], index_col=0, header=[0, 1], 
        #     skiprows=1)

        emg = util.storage2numpy(file_dep[0])
        time = emg['time']
        pgc_emg = np.linspace(0, 100, 400)

        def min_index(vals):
            idx, val = min(enumerate(vals), key=lambda p: p[1])
            return idx

        condition = subject.get_condition(cond_name)
        trial = condition.get_trial(1) # only one trial per condition
        cycle_array = list()
        muscle_array = list()
        emg_data = list()
        for cycle_num in self.study.test_cycles_num:
            cycle = trial.get_cycle(cycle_num)
            start_idx = min_index(abs(time-cycle.start))
            end_idx = min_index(abs(time-cycle.end))

            for emg_name in emg_muscles:

                emg_interp = np.interp(pgc_emg, 
                    np.linspace(0, 100, len(time[start_idx:end_idx])), 
                    emg[emg_name][start_idx:end_idx])
                emg_data.append(emg_interp)
                cycle_array.append(cycle.id)
                muscle_array.append(emg_name)

        # Convert from (n_cycles * n_muscles) x n_times
        #         to   n_times x (n_cycles * n_muscles)
        emg_data_array = np.array(emg_data).transpose()
        columns = pd.MultiIndex.from_arrays([cycle_array, muscle_array],
                names=['cycle', 'muscle'])
        df_emg = pd.DataFrame(emg_data_array, columns=columns, index=pgc_emg)
        df_emg_mean = df_emg.groupby(level=['muscle'], axis=1).mean()
        df_emg_std = df_emg.groupby(level=['muscle'], axis=1).std()

        df_exc_all = pd.read_csv(file_dep[1], index_col=0, header=[0, 1, 2], 
            skiprows=1)
        df_exc = df_exc_all[subject.name]
        df_exc_mean = df_exc.groupby(level=['muscle'], axis=1).mean()
        df_exc_std = df_exc.groupby(level=['muscle'], axis=1).std()

        df_act_all = pd.read_csv(file_dep[2], index_col=0, header=[0, 1, 2], 
            skiprows=1)
        df_act = df_act_all[subject.name]
        df_act_mean = df_act.groupby(level=['muscle'], axis=1).mean()
        df_act_std = df_act.groupby(level=['muscle'], axis=1).std()

        pgc_exc = df_exc_mean.index
        pgc_act = df_act_mean.index


        for iemg, emg_name in enumerate(emg_muscles):
            side_len = np.ceil(np.sqrt(len(emg_muscles)))
            ax = fig.add_subplot(side_len, side_len, iemg + 1)
            ax.axhline(color='k', linewidth=0.5, zorder=0)
            y_emg_mean = df_emg_mean[emg_name]
            y_emg_std = df_emg_std[emg_name]
            ax.plot(pgc_emg, y_emg_mean, color='black', linestyle='-')
            ax.fill_between(pgc_emg, y_emg_mean-y_emg_std, y_emg_mean+y_emg_std,
                    color='black', alpha=0.5)
            if emg_map.get(emg_name):
                # y_exc_mean = df_exc_mean[emg_map[emg_name]]
                # y_exc_std = df_exc_std[emg_map[emg_name]]
                y_act_mean = df_act_mean[emg_map[emg_name]]
                y_act_std = df_act_std[emg_map[emg_name]]
                # exc_plot, = ax.plot(pgc_exc, y_exc_mean, color='blue', 
                #     linestyle='--')
                # ax.fill_between(pgc_exc, y_exc_mean-y_exc_std, 
                #     y_exc_mean+y_exc_std, color='blue', alpha=0.25)
                act_plot, = ax.plot(pgc_act, y_act_mean, color='red', 
                    linestyle='--')
                ax.fill_between(pgc_act, y_act_mean-y_act_std, 
                    y_act_mean+y_act_std, color='red', alpha=0.25)
                handles = [act_plot]
                labels = ['%s act.' % nice_muscle_names_dict[emg_map[emg_name]]]
                ax.legend(handles, labels)

            # ax.legend(frameon=False, fontsize=6)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.0)
            ax.set_title(emg_name)
            ax.set_xlabel('% gait cycle')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        fig.tight_layout()
        fig.savefig(target[0]+'.pdf')
        fig.savefig(target[0]+'.png', dpi=600)
        pl.close(fig)

def plot_dof(ax, df, dof, actuator, label, color, ls=None, drop_subjects=None,
        fill_between=False, zorder=0, lw=2):
    # Average over cycles.
    # axis=1 for columns (not rows).
    for subj in drop_subjects:
        df.drop(subj, axis='columns', inplace=True)

    df_by_subj_dof_musc = df.groupby(
            level=['subject', 'DOF', 'actuator'], axis=1).mean()
    df_by_subj_dof_musc *= -1

    df_mean = df_by_subj_dof_musc.groupby(
        level=['DOF', 'actuator'], axis=1).mean()
    df_std = df_by_subj_dof_musc.groupby(
        level=['DOF', 'actuator'], axis=1).std()
    pgc = df_mean.index

    if actuator == 'muscles': 
        mask1 = df_by_subj_dof_musc.columns.get_level_values('subject').isin(
            df_by_subj_dof_musc.columns.get_level_values('subject'))
        dofs = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
        mask2 = df_by_subj_dof_musc.columns.get_level_values('DOF').isin(dofs)
        actu_names = list(df_by_subj_dof_musc.columns.get_level_values(
                'actuator'))
        actu_names[:] = [a for a in actu_names if not (a in ['net', 'device'])]
        mask3 = df_by_subj_dof_musc.columns.get_level_values(
                'actuator').isin(actu_names)

        df_muscs_only = df_by_subj_dof_musc.loc[:, mask1 & mask2 & mask3]
        df_sum_muscs = df_muscs_only.groupby(
                level=['DOF', 'subject'], axis=1).sum()

        df_muscs_mean = df_sum_muscs.groupby(level='DOF', axis=1).mean()
        df_muscs_std = df_sum_muscs.groupby(level='DOF', axis=1).std()
        if dof in df_muscs_mean.columns:
            y_mean = df_muscs_mean[dof]
            y_std = df_muscs_std[dof]
            ls = ls if ls else '-'
            ax.plot(pgc, y_mean, color=color, label=label, linestyle=ls, 
                    linewidth=lw, zorder=zorder)
            if fill_between:
                ax.fill_between(pgc, y_mean-y_std, y_mean+y_std, facecolor=color, 
                    alpha=0.3, zorder=zorder)
    else:
        if (dof, actuator) in df_mean.columns:
            y_mean = df_mean[(dof, actuator)]
            y_std = df_std[(dof, actuator)]
            ls = ls if ls else '-'
            ax.plot(pgc, y_mean, color=color, label=label, linestyle=ls, 
                    linewidth=lw, zorder=zorder)
            if fill_between:
                ax.fill_between(pgc, y_mean-y_std, y_mean+y_std, facecolor=color, 
                    alpha=0.3, zorder=zorder)

def set_axis(ax, idof, dof, dof_labels, legend_dof=None, label_as_title=False):
    # ylabel_coordx = -0.
    # ylabel_coordy = 0.5
    if dof == 'hip_flexion_r':
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_ylim(-1.5, 2.0)
        # ax.yaxis.set_label_coords(ylabel_coordx, ylabel_coordy)
        ax.tick_params(labelsize=5)
        # ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])

    elif dof == 'knee_angle_r':
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_ylim(-1.5, 2.0)
        # ax.yaxis.set_label_coords(ylabel_coordx, ylabel_coordy)
        ax.tick_params(labelsize=5)
        # ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])

    elif dof == 'ankle_angle_r':
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])                
        ax.set_ylim(-1.5, 2.0)
        # ax.yaxis.set_label_coords(ylabel_coordx, ylabel_coordy)
        ax.tick_params(labelsize=5)
        # ax.set_yticks([-0.5, 0, 0.5, 1, 1.5, 2.0])

    # if idof < len(self.dof_names)-1:
    #     ax.set_xticklabels([])
    if label_as_title:
        ax.set_title('%s' % dof_labels[idof], fontsize=4)
    else: 
        ax.set_ylabel('%s \n moment (N-m/kg)' % dof_labels[idof], fontsize=6)
    ax.set_xlabel('% gait cycle', fontsize=5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if dof == legend_dof:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, frameon=False, fontsize=7, 
            loc="best")
        ax.get_legend().get_title().set_fontsize(8)
        # ax.get_legend().get_title().set_fontstyle('italic')
        ax.get_legend().get_title().set_fontweight('bold')

# Get whole body metabolic percent reductions.
def get_percent_reductions(dep, device_names, plot_case, 
        drop_subjects=None, add_basal_rate=False, assume_symmetry=False):

    df_met = pd.read_csv(dep, index_col=[0, 1, 2], skiprows=1)
    if assume_symmetry:
        df_met *= 2

    if add_basal_rate:
        df_met += 1.2 # W/kg

    for subj in drop_subjects:
        df_met.drop(subj, axis='index', inplace=True)

    if plot_case == 'all':
        df_met_cond = df_met.copy(deep=True)
    else:
        df_met_cond = df_met.xs(plot_case, level='condition')

    df_met_change = df_met_cond.subtract(df_met_cond['experiment'], 
        axis='index')
    df_met_relchange = df_met_change.divide(df_met_cond['experiment'], 
        axis='index')
    df_met_relchange.drop('experiment', axis='columns', inplace=True)

    df_met_by_subjs = df_met_relchange.groupby(level='subject').mean()
    met_mean = df_met_by_subjs.mean()[device_names] * 100
    met_std = df_met_by_subjs.std()[device_names] * 100

    return met_mean, met_std, df_met_change

# Get individual muscle metabolic percent reductions, normalized by whole body
# metabolic rate.
def get_percent_reductions_muscles(muscles_dep, whole_body_dep, device_names,
        plot_case, drop_subjects=None, add_basal_rate=False, 
        assume_symmetry=False):

    df_met_muscles = pd.read_csv(muscles_dep, index_col=[0, 1, 2, 3], skiprows=1)
    df_met_whole_body = pd.read_csv(whole_body_dep, index_col=[0, 1, 2], 
        skiprows=1)
    for subj in drop_subjects:
        df_met_muscles.drop(subj, axis='index', inplace=True)
        df_met_whole_body.drop(subj, axis='index', inplace=True)

    if assume_symmetry:
        df_met_muscles *= 2
        df_met_whole_body *= 2

    if add_basal_rate:
        df_met_whole_body += 1.2 # W/kg
        
    if plot_case == 'all':
        df_met_cond_muscles = df_met_muscles.copy(deep=True)
        df_met_cond_whole_body = df_met_whole_body.copy(deep=True)
    else:
        df_met_cond_muscles = df_met_muscles.xs(plot_case, level='condition', 
            drop_level=True)
        df_met_cond_whole_body = df_met_whole_body.xs(plot_case, 
            level='condition', drop_level=True)

    df_met_change_whole_body = df_met_cond_whole_body.subtract(
                df_met_cond_whole_body['experiment'], axis='index')
    df_met_relchange_whole_body = df_met_change_whole_body.copy(deep=True)
    df_met_relchange_whole_body.drop('experiment', axis='columns', inplace=True)

    df_met_change_muscles = df_met_cond_muscles.subtract(
        df_met_cond_muscles['experiment'], axis='index')
    df_met_relchange_muscles = df_met_change_muscles.copy(deep=True)
    for i in np.arange(len(df_met_relchange_muscles.index.levels)):
        if df_met_relchange_muscles.index.levels[i].name == 'cycle':
            cycles = df_met_relchange_muscles.index.levels[i]
            break

    for cycle in cycles:
        cycle_split = cycle.split('_')
        subject = cycle_split[0]
        cond_name = cycle_split[1]

        if plot_case == 'all':
            Wkg_exp = df_met_cond_whole_body.loc[subject]['experiment']
            for cond_name, cond_df in Wkg_exp.groupby(level='condition'):
                if not (cond_name in cycle): continue

                Wkg_exp_this_cond = cond_df.loc[(cond_name, cycle),]
                df_met_relchange_muscles.loc[(subject, cond_name, cycle), :] = \
                    df_met_relchange_muscles.loc[(subject, cond_name, 
                        cycle), :].values / Wkg_exp_this_cond
        else:
            # If condition specified, we've already indexed values for that
            # condition out
            if not (plot_case == cond_name): continue

            Wkg_exp = df_met_cond_whole_body.loc[(subject, 
                cycle),]['experiment']
            
            df_met_relchange_muscles.loc[(subject, cycle),:] = \
                df_met_relchange_muscles.loc[(subject, cycle),:].values / Wkg_exp

    df_met_relchange_muscles.drop('experiment', axis='columns', inplace=True)

    df_met_by_subjs = df_met_relchange_muscles.groupby(
        level=['subject', 'muscle']).mean()
    met_mean = df_met_by_subjs.groupby(
        level='muscle').mean()[device_names]
    met_std = df_met_by_subjs.groupby(
        level='muscle').std()[device_names]
    
    return met_mean, met_std, df_met_change_muscles

class TaskPlotDeviceComparison(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, plot_lists, folder, 
            cond_names=['walk1','walk2','walk3','walk4'], subjects=None,
            max_metabolic_reduction=35, fig_width=6, fig_height=6):
        super(TaskPlotDeviceComparison, self).__init__(study)
        self.name = 'plot_device_comparison_%s' % folder
        self.doc = 'Plot to compare assistive moments across devices.'
        self.output_path = os.path.join(study.config['analysis_path'], folder)
        if not os.path.exists(self.output_path): os.mkdir(self.output_path)
        self.cond_names = cond_names
        self.max_metabolic_reduction = max_metabolic_reduction
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.device_list = plot_lists['device_list']
        self.device_names = list()
        self.device_dirs = list()
        for device in self.device_list:
            self.device_dirs.append(device.replace('/', '\\'))
            if len(device.split('/')) > 1:
                self.device_names.append('_'.join(device.split('/')))
            else:
                self.device_names.append(device)
        self.label_list = plot_lists['label_list']
        self.color_list = plot_lists['color_list']
        if 'linestyle_list' in plot_lists:
            self.linestyle_list = plot_lists['linestyle_list']

        self.folder = folder

        if ('He' in folder) or ('Hf' in folder):
            self.legend_dof = 'hip_flexion_r'
        elif ('Ke' in folder) or ('Kf' in folder):
            self.legend_dof = 'knee_angle_r'
        else:
            self.legend_dof = 'ankle_angle_r'

        self.dof_names = list()
        self.dof_labels = list()
        for device_name in self.device_names:
            if ('He' in device_name) or ('Hf' in device_name):
                if 'hip_flexion_r' in self.dof_names: continue
                self.dof_names.append('hip_flexion_r')
                self.dof_labels.append('hip extension')
            if ('Ke' in device_name) or ('Kf' in device_name):
                if 'knee_angle_r' in self.dof_names: continue
                self.dof_names.append('knee_angle_r')
                self.dof_labels.append('knee extension')
            if ('Ap' in device_name) or ('Ad' in device_name):
                if 'ankle_angle_r' in self.dof_names: continue
                self.dof_names.append('ankle_angle_r')
                self.dof_labels.append('ankle plantarflexion')

        if study.costFunction:
            if not (study.costFunction == 'Default'):
                cost = '_' + study.costFunction
            else:
                cost = ''
        else:
            cost = ''
        self.cost = cost

        if subjects == None:
            self.subjects = [s.name for s in study.subjects]
        else:
            self.subjects = subjects 
        self.study_subjects = [s.name for s in self.study.subjects]
        self.drop_subjects = list()
        for study_subj in self.study_subjects:
            if not (study_subj in self.subjects):
                self.drop_subjects.append(study_subj)

        plot_cases = list()
        plot_cases += self.cond_names
        plot_cases.append('all')
        moment_deps_dict = dict()
        for plot_case in plot_cases:
            subfolder = ''
            if plot_case in cond_names:
                subfolder = plot_case
                subdir = os.path.join(self.output_path, subfolder)
                if not os.path.exists(subdir): os.mkdir(subdir)
        
            self.add_action([os.path.join(self.output_path, 
                    'whole_body_metabolic_rates_%s%s.csv' % (folder, cost)),
                os.path.join(self.output_path, 
                    'peak_power_%s%s.csv' % (folder, cost)),
                os.path.join(study.config['analysis_path'], folder, 
                    'avg_pos_power_%s%s.csv' % (folder, cost))], 
                [os.path.join(self.output_path, subfolder,
                    '%s_%s_metabolics_per_reduction.pdf' % (folder, plot_case)),
                 os.path.join(self.output_path, subfolder,
                    '%s_%s_metabolics_peak_power_norm.pdf' % (folder, plot_case)),
                 os.path.join(self.output_path, subfolder,
                    '%s_%s_metabolics_avg_power_norm.pdf' % (folder, plot_case))],
                self.plot_metabolics, plot_case)

            self.add_action([os.path.join(study.config['analysis_path'], folder,
                    'muscle_metabolic_rates_%s%s.csv' % (folder, cost)),
                 os.path.join(self.output_path, 
                    'whole_body_metabolic_rates_%s%s.csv' % (folder, cost))],
                [os.path.join(self.output_path, subfolder,
                    '%s_%s_muscle_metabolics_per_reduction.pdf' 
                     % (folder, plot_case))],
                self.plot_metabolics_muscles, plot_case)

            if plot_case in self.cond_names:
                moment_deps = list()
                for device_name, device_dir in zip(self.device_names, 
                        self.device_dirs):
                    fname = '%s_%s_moments%s.csv' % (device_name, plot_case, 
                        cost)
                    moment_deps.append(os.path.join(study.config['results_path'], 
                        device_dir, fname))
                self.add_action(moment_deps, 
                                [], 
                                self.plot_moments, plot_case)
                moment_deps_dict[plot_case] = moment_deps

                # musc_data_names = ['activations', 'norm_act_fiber_forces', 
                #                    'norm_fiber_lengths', 'norm_fiber_velocities',
                #                    'norm_pass_fiber_forces', 'norm_fiber_powers',
                #                    'fiber_powers']
                musc_data_names = ['activations', 'fiber_powers', 
                                   'norm_fiber_lengths']
                for mdname in  musc_data_names:
                    md_deps = list()
                    fname ='experiment_%s_%s%s.csv' % (plot_case, mdname, cost)
                    md_deps.append(os.path.join(study.config['results_path'], 
                        'experiments', fname))
                    for device_name, device_dir in zip(self.device_names, 
                            self.device_dirs):
                        fname = '%s_%s_%s%s.csv' % (device_name, plot_case, 
                            mdname, cost)
                        md_deps.append(os.path.join(study.config['results_path'], 
                            device_dir, fname))                            

                    self.add_action(md_deps,
                                    [os.path.join(self.output_path, subfolder,
                                        '%s_%s_%s.pdf' % (folder, plot_case, 
                                        mdname))],
                                    self.plot_muscle_data)

                # muscle_data_names = ['activations', 'norm_fiber_powers', 
                #                      'norm_act_fiber_forces']
                # muscles_to_plot = ['psoas_r', 'soleus_r', 'glut_max2_r', 
                #                    'tib_ant_r','med_gas_r','semimem_r',
                #                    'bifemsh_r','vas_int_r','rect_fem_r']
                # for mdname in muscle_data_names:
                #     for musc_name in muscles_to_plot:
                #         md_deps = list()
                #         fname ='experiment_%s_%s%s.csv' % (plot_case, mdname, 
                #             cost)
                #         md_deps.append(os.path.join(study.config['results_path'], 
                #             'experiments', fname))
                #         for device_name, device_dir in zip(self.device_names, 
                #                 self.device_dirs):
                #             fname = '%s_%s_%s%s.csv' % (device_name, plot_case, 
                #                 mdname, cost)
                #             md_deps.append(os.path.join(
                #                 study.config['results_path'], device_dir, fname))
                #         self.add_action(md_deps,
                #                        [os.path.join(self.output_path, subfolder,
                #                             '%s_%s_%s_%s.pdf' % (folder, 
                #                                 plot_case, mdname, musc_name))],
                #                         self.plot_data_one_muscle, musc_name)

        # Overview plots
        self.add_action([os.path.join(self.output_path, 
                    'whole_body_metabolic_rates_%s%s.csv' % (folder, cost))],
                [os.path.join(self.output_path,
                    '%s_metabolics_per_cycle_overview.pdf' % folder),
                 os.path.join(self.output_path,
                    '%s_metabolics_and_moments_overview.pdf' % folder)],
                 self.plot_overview, moment_deps_dict)
        
    def plot_moments(self, file_dep, target, plot_case):

        from matplotlib import gridspec        
        plot_iterate = zip(file_dep, self.device_names, self.label_list, 
            self.color_list)

        for not_first, (dep, device, label, color) in enumerate(plot_iterate):

            ls = None
            if hasattr(self, 'linestyle_list'):
                ls = self.linestyle_list[not_first]

            # Get dataframes
            df = pd.read_csv(dep, index_col=0,
                            header=[0, 1, 2, 3], skiprows=1)
            # Skip compare dataframe on first iteration, since they are the same
            if not_first:
                df_compare = pd.read_csv(file_dep[0], index_col=0,
                            header=[0, 1, 2, 3], skiprows=1)
                label_compare = self.label_list[0]
                color_compare = self.color_list[0]
                ls_compare = None
                if hasattr(self, 'linestyle_list'):
                    ls_compare = self.linestyle_list[0]

            fig = pl.figure(figsize=(3, 0.8*3*len(self.dof_names)))
            handles = set()
            labels = set()
            for idof, dof in enumerate(self.dof_names):

                gs = gridspec.GridSpec(len(self.dof_names), 1) 
                ax = fig.add_subplot(gs[idof])
                # ax = fig.add_subplot(2, len(dof_names), idof + 1)
                ax.axhline(color='k', linewidth=0.5, zorder=0)
                plot_dof(ax, df, dof, 'net', 'net joint moment', 'darkgray',
                    ls='-', drop_subjects=self.drop_subjects)
                plot_dof(ax, df, dof, 'device', label, color, ls=ls,
                    drop_subjects=self.drop_subjects)
                if not_first:
                    plot_dof(ax, df_compare, dof, 'device', label_compare, 
                        color_compare, ls=ls_compare,
                        drop_subjects=self.drop_subjects)
                set_axis(ax, idof, dof, self.dof_labels)

            fig.tight_layout()
            fname = '%s_moments' % device
            fig.savefig(os.path.join(self.output_path, plot_case, 
                fname + '.pdf'))
            fig.savefig(os.path.join(self.output_path, plot_case, 
                fname + '.png'), ppi=1800)
            fig.savefig(os.path.join(self.output_path, plot_case, 
                fname + '.svg'), format='svg', dpi=1000)
            pl.close(fig)


        fig = pl.figure(figsize=(3, 0.8*3*len(self.dof_names)))
        for idof, dof in enumerate(self.dof_names):
            gs = gridspec.GridSpec(len(self.dof_names), 1)
            ax = fig.add_subplot(gs[idof])
            ax.axhline(color='k', linewidth=0.5, zorder=0)
            for idep, (dep, device, label, color) in enumerate(plot_iterate):
                df = pd.read_csv(dep, index_col=0,
                            header=[0, 1, 2, 3], skiprows=1)
                if not idep:
                    plot_dof(ax, df, dof, 'net', 'net joint moment', 'darkgray',
                        ls='-', drop_subjects=self.drop_subjects)
                if hasattr(self, 'linestyle_list'):
                    ls = self.linestyle_list[idep]

                plot_dof(ax, df, dof, 'device', label, color, ls=ls,
                    drop_subjects=self.drop_subjects)

            set_axis(ax, idof, dof, self.dof_labels)

        fig.tight_layout()
        fname = '%s_all_device_moments' % self.folder
        fig.savefig(os.path.join(self.output_path, plot_case, fname + '.pdf'))
        fig.savefig(os.path.join(self.output_path, plot_case, fname + '.png'), 
            ppi=1800)
        fig.savefig(os.path.join(self.output_path, plot_case, fname + '.svg'),
            format='svg', dpi=1000)
        pl.close(fig)

    def plot_torque_parameters(self, file_dep, target, plot_case):

        plot_iterate = zip(file_dep, self.device_names, self.label_list, 
            self.color_list)

        fig = pl.figure(figsize=(6, 3*len(plot_iterate)))
        for i, (dep, device, label, color) in enumerate(plot_iterate):

            df = pd.read_csv(dep, index_col=0, header=[0, 1], skiprows=1)

            param_names, param_bounds = \
                get_parameter_info(self.study, device)

            # Normalize within [0,1]
            for ip, param_name in enumerate(param_names):
                norm_factor = param_bounds['upper'][ip] - param_bounds['lower'][ip]
                df.loc[param_name] -= param_bounds['lower'][ip]
                df.loc[param_name] /= norm_factor

            param_data = list()
            for folder_param_name in self.folder_param_names:
                if folder_param_name in df.index:
                    param_data.append(df.loc[folder_param_name])
                else:
                    param_data.append(np.zeros(df.index.size))
                        
            ax = fig.add_subplot(len(plot_iterate), 1, i+1)
            bplot = ax.boxplot(param_data, patch_artist=True)
            for patch in bplot['boxes']:
                patch.set_facecolor(color)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks(np.arange(0.0, 1.1, 0.1))
            ax.set_ylabel(label)    
            if i == (len(plot_iterate)-1):
                ax.set_xticklabels(self.folder_param_names, rotation=45, 
                    ha='right')
            else:
                ax.set_xticklabels([])
                

        fig.tight_layout()
        fname = '%s_%s_torque_parameters' % (self.folder, plot_case)
        fig.savefig(os.path.join(self.output_path, plot_case, fname + '.pdf'))
        fig.savefig(os.path.join(self.output_path, plot_case, fname + '.png'), 
            ppi=1800)
        fig.savefig(os.path.join(self.output_path, plot_case, fname + '.svg'),
            format='svg', dpi=1000)
        pl.close(fig)

    def plot_metabolics(self, file_dep, target, plot_case):

        fig = pl.figure(figsize=(self.fig_width, self.fig_height))
        ax = fig.add_subplot(1,1,1)
        ind = np.arange(len(self.device_names))

        met_mean, met_std, df_met_change = get_percent_reductions(file_dep[0], 
            self.device_names, plot_case, drop_subjects=self.drop_subjects,
            add_basal_rate=True, assume_symmetry=True)

        met_mean.to_csv(os.path.join(self.output_path, 
            'total_whole_body_reductions.csv'))

        barlist = ax.bar(ind, met_mean, yerr=met_std)
        for barelt, color in zip(barlist, self.color_list):
            barelt.set_color(color)
            barelt.set_linewidth(2)

        # ax.axhline(y=met_mean[0], linewidth=2, ls='--', 
        #     color=self.color_list[0])

        ax.set_xticks(ind)
        ax.set_xticklabels(self.label_list)
        #ax.set_xticklabels(self.label_list, ha='left', fontsize=8)

        ax.set_ylabel('change in gross average whole-body metabolic rate (%)')
        ax.set_ylim(-self.max_metabolic_reduction, 0)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('top')

        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[0].replace('.pdf','.png'), ppi=1800)
        fig.savefig(target[0].replace('.pdf','.svg'))
        fig.savefig(target[0].replace('.pdf','.eps'), metadata='eps')
        pl.close(fig)

        # Plot normalized absolute metabolic reduction by peak positive device 
        # power
        df_peak_power = pd.read_csv(file_dep[1], index_col=[0, 1, 2], skiprows=1)
        for subj in self.study_subjects:
            if not (subj in self.subjects):
                df_peak_power.drop(subj, axis='index', inplace=True)

        if plot_case in self.cond_names:
            df_peak_power_cond = df_peak_power.xs(plot_case, level='condition')
        else:
            df_peak_power_cond = df_peak_power.copy(deep=True)

        df_eff = -df_met_change.divide(df_peak_power_cond)
        df_eff_by_subjs = df_eff.groupby(level='subject').mean()
        eff_mean = df_eff_by_subjs.mean()[self.device_names]
        eff_std = df_eff_by_subjs.std()[self.device_names]

        fig = pl.figure(figsize=(4*1.2, 6*1.2))
        ax = fig.add_subplot(1,1,1)
        ind = np.arange(len(self.device_names))

        barlist = ax.bar(ind, eff_mean, yerr=eff_std)
        for barelt, color in zip(barlist, self.color_list):
            barelt.set_color(color)

        ax.set_xticks(ind)
        ax.set_xticklabels(self.label_list, rotation=45, ha='right')
        ax.set_ylabel('device efficiency (met rate / peak device power)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')

        fig.tight_layout()
        fig.savefig(target[1])
        fig.savefig(target[1].replace('.pdf','.png'), ppi=1800)
        fig.savefig(target[1].replace('.pdf','.svg'), format='svg', dpi=1000)

        pl.close(fig)

        # Plot normalized absolute metabolic reduction by peak positive device 
        # power
        df_avg_power = pd.read_csv(file_dep[2], index_col=[0, 1, 2], skiprows=1)
        for subj in self.study_subjects:
            if not (subj in self.subjects):
                df_avg_power.drop(subj, axis='index', inplace=True)

        if plot_case in self.cond_names:
            df_avg_power_cond = df_avg_power.xs(plot_case, level='condition')
        else:
            df_avg_power_cond = df_avg_power.copy(deep=True)

        df_perfidx = -df_met_change.divide(df_avg_power_cond)
        df_perfidx_by_subjs = df_perfidx.groupby(level='subject').mean()
        perfidx_mean = df_perfidx_by_subjs.mean()[self.device_names]
        perfidx_std = df_perfidx_by_subjs.std()[self.device_names]

        fig = pl.figure(figsize=(4*1.2, 6*1.2))
        ax = fig.add_subplot(1,1,1)
        ind = np.arange(len(self.device_names))

        barlist = ax.bar(ind, perfidx_mean, yerr=perfidx_std)
        for barelt, color in zip(barlist, self.color_list):
            barelt.set_color(color)

        ax.set_xticks(ind)
        ax.set_xticklabels(self.label_list, rotation=45, ha='right')
        ax.set_ylabel('performance index (met rate / avg device power)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')

        fig.tight_layout()
        fig.savefig(target[2])
        fig.savefig(target[2].replace('.pdf','.png'), ppi=1800)
        fig.savefig(target[2].replace('.pdf','.svg'), format='svg', dpi=1000)
        pl.close(fig)

    def plot_metabolics_muscles(self, file_dep, target, plot_case):

        met_mean, met_std, df_met_change = \
            get_percent_reductions_muscles(file_dep[0], file_dep[1],
            self.device_names, plot_case, drop_subjects=self.drop_subjects,
            add_basal_rate=True, assume_symmetry=True)

        fig = pl.figure(figsize=(len(met_mean.index)*0.6, 3))
        ax = fig.add_subplot(1,1,1)
        ind = np.arange(len(met_mean.index))
        met_mean_copy = met_mean.copy(deep=True)
        met_mean_copy.plot.bar(ax=ax, color=self.color_list, legend=False)

        nice_muscle_names_dict = self.study.nice_muscle_names_dict
        nice_muscle_names = \
            [nice_muscle_names_dict[name] for name in met_mean.index]

        ax.set_xticks(ind)
        ax.set_xticklabels(nice_muscle_names, rotation=45, ha='left')
        ax.set_ylabel('contribution to reduction in metabolic cost (%)')
        # ax.set_ylim(-30, 0)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('top')

        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[0].replace('.pdf','.png'), ppi=1800)
        fig.savefig(target[0].replace('.pdf','.svg'), format='svg', dpi=1000)
        pl.close(fig)

    def plot_muscle_data(self, file_dep, target):

        device_names = ['experiment'] + self.device_names
        label_list = ['unassisted'] + self.label_list
        color_list = ['darkgray'] + self.color_list
        if hasattr(self, 'linestyle_list'):
            linestyle_list = ['-'] + self.linestyle_list
        plot_iterate = zip(file_dep, device_names, label_list, color_list)

        fig = pl.figure(figsize=(5, 3))
        muscles = self.study.muscle_names
        nice_muscle_names_dict = self.study.nice_muscle_names_dict

        mean_list = list()
        std_list = list()
        pgc_list = list()
        for dep in file_dep:
            df = pd.read_csv(dep, index_col=0,
                    header=[0, 1, 2], skiprows=1)
            for subj in self.study_subjects:
                if not (subj in self.subjects):
                    df.drop(subj, axis='columns', inplace=True)

            # Average over cycles.
            df_by_subj_musc = df.groupby(level=['subject', 'muscle'], 
                axis=1).mean()
            df_mean = df_by_subj_musc.groupby(level=['muscle'], axis=1).mean()
            df_std = df_by_subj_musc.groupby(level=['muscle'], axis=1).std()
            pgc = df_mean.index

            mean_list.append(df_mean)
            std_list.append(df_std)
            pgc_list.append(pgc)

        for imusc, musc_name in enumerate(muscles):
            side_len = np.ceil(np.sqrt(len(muscles)))
            ax = fig.add_subplot(side_len, side_len, imusc + 1)
            ax.axhline(color='k', linewidth=0.5, zorder=0)

            for i, (dep, device, label, color) in enumerate(plot_iterate):
                df_mean = mean_list[i]
                pgc = pgc_list[i]

                ls = '-'
                if hasattr(self, 'linestyle_list'):
                    ls = linestyle_list[i]

                ax.plot(pgc, df_mean[musc_name], label=label, color=color, ls=ls,
                        linewidth=2.5)
                ax.set_xlim(0, 100)
                ax.set_xticks([0, 50, 100])
                if 'norm_fiber_lengths' in dep:
                    ax.set_ylim(0, 2.0)
                    ax.set_yticks(np.linspace(0, 2, 9))
                    ax.axhline(y=0.5, color='k', linewidth=0.5, ls='--', 
                        zorder=0)
                    ax.axhline(y=1.0, color='k', linewidth=0.5, ls='--', 
                        zorder=0)
                    ax.axhline(y=1.5, color='k', linewidth=0.5, ls='--', 
                        zorder=0)
                elif 'norm_fiber_velocities' in dep:
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_yticks(np.linspace(-0.5, 0.5, 5))
                    ax.axhline(y=-0.25, color='k', linewidth=0.5, ls='--', 
                        zorder=0)
                    ax.axhline(y=0.25, color='k', linewidth=0.5, ls='--', 
                        zorder=0)
                elif ('norm_fiber_powers' in dep) or ('fiber_powers' in dep):
                    # this line is here just to skip settings for these cases
                    ax.axhline(color='k', linewidth=0.5, zorder=0)
                else:
                    ax.set_ylim(0, 0.5)
                    ax.set_yticks([0, 0.5])
                ax.set_title(nice_muscle_names_dict[musc_name])
                if imusc > len(muscles)-4:
                    ax.set_xlabel('% gait cycle')
                else:
                    ax.set_xticklabels([])
                if not imusc % 3:
                    ax.set_ylabel('activation')
                else:
                    ax.set_yticklabels([])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')

            # if not imusc:
            #     handles, labels = ax.get_legend_handles_labels()
            #     ax.legend(handles, labels, frameon=False, fontsize=7, 
            #         loc="best")
            #     ax.get_legend().get_title().set_fontsize(8)
            #     ax.get_legend().get_title().set_fontweight('bold')

        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[0].replace('.pdf','.png'), dpi=600)
        fig.savefig(target[0].replace('.pdf','.svg'))
        fig.savefig(target[0].replace('.pdf','.eps'), metadata='eps')
        pl.close(fig)

    def plot_data_one_muscle(self, file_dep, target, musc_name):

        device_names = ['experiment'] + self.device_names
        label_list = ['unassisted'] + self.label_list
        color_list = ['black'] + self.color_list
        plot_iterate = zip(file_dep, device_names, label_list, color_list)

        fig = pl.figure(figsize=(8, 4))
        muscles = self.study.muscle_names
        nice_muscle_names_dict = self.study.nice_muscle_names_dict

        df_musc_list = list()
        mean_list = list()
        std_list = list()
        pgc_list = list()
        for dep in file_dep:
            df = pd.read_csv(dep, index_col=0,
                    header=[0, 1, 2], skiprows=1)
            for subj in self.study_subjects:
                if not (subj in self.subjects):
                    df.drop(subj, axis='columns', inplace=True)

            # Average over cycles.
            df_by_subj_musc = df.groupby(level=['subject', 'muscle'], 
                axis=1).mean()
            df_musc = df_by_subj_musc.xs(musc_name, level='muscle', axis=1)
            df_mean = df_by_subj_musc.groupby(level=['muscle'], axis=1).mean()
            df_std = df_by_subj_musc.groupby(level=['muscle'], axis=1).std()
            pgc = df_mean.index

            df_musc_list.append(df_musc)
            mean_list.append(df_mean)
            std_list.append(df_std)
            pgc_list.append(pgc)

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.axhline(color='k', linewidth=0.5, zorder=0)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.axhline(color='k', linewidth=0.5, zorder=0)

        for i, (dep, device, label, color) in enumerate(plot_iterate):
            df_musc = df_musc_list[i]
            df_mean = mean_list[i]
            pgc = pgc_list[i]

            # from scipy import signal
            # b, a = signal.butter(4, 0.075)
            # y_mean = signal.filtfilt(b, a, df_mean[musc_name])
            ax1.plot(pgc, df_mean[musc_name], label=label, color=color, linestyle='-')
            ax1.set_xlim(0, 100)
            ax1.set_xlabel('time (% gait cycle)')
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.xaxis.set_ticks_position('bottom')
            ax1.yaxis.set_ticks_position('left')

            if 'activation' in dep:
                ax1.set_ylabel('activation')
                ax2.set_ylabel('avg. activation')

                df_mean_subjs = df_musc.mean()
                y = df_mean_subjs.mean()
                yerr = df_mean_subjs.std()
                ax2.bar(i, y, color=color)
                ax2.errorbar(i, y, yerr=yerr, color=color)

            if 'norm_fiber_powers' in dep:
                ax1.set_ylabel('norm. fiber power (W/kg)')
                ax2.set_ylabel('avg. norm. fiber power (W/kg)')

                df_musc_pos = df_musc.copy(deep=True)
                df_musc_pos[df_musc_pos < 0] = 0
                df_mean_subjs_pos = df_musc_pos.mean()
                y_pos = df_mean_subjs_pos.mean()
                yerr_pos = df_mean_subjs_pos.std()
                ax2.bar(i, y_pos, color=color)
                ax2.errorbar(i, y_pos, yerr=yerr_pos, color=color)

                df_musc_neg = df_musc.copy(deep=True)
                df_musc_neg[df_musc_pos > 0] = 0
                df_mean_subjs_neg = df_musc_neg.mean()
                y_neg = df_mean_subjs_neg.mean()
                yerr_neg = df_mean_subjs_neg.std()
                ax2.bar(i, y_neg, color=color)
                ax2.errorbar(i, y_neg, yerr=yerr_neg, color=color)

                ax2.axhline(color='k', linewidth=0.5, zorder=10)

            if 'norm_act_fiber_forces' in dep:
                ax1.set_ylabel('norm. fiber force')
                ax2.set_ylabel('avg. norm fiber force')

                df_mean_subjs = df_musc.mean()
                y = df_mean_subjs.mean()
                yerr = df_mean_subjs.std()
                ax2.bar(i, y, color=color)
                ax2.errorbar(i, y, yerr=yerr, color=color)

            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.xaxis.set_ticks_position('bottom')
            ax2.set_xticks(range(len(device_names)))
            ax2.set_xticklabels([])
            ax2.yaxis.set_ticks_position('left')

        fig.suptitle(nice_muscle_names_dict[musc_name] + '\n')
        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[0].replace('.pdf','.png'), dpi=600)
        pl.close(fig)

    def plot_overview(self, file_dep, target, moments_dep_dict):

        # Metabolics for each cycle overview plot
        df_met = pd.read_csv(file_dep[0], index_col=[0, 1, 2], skiprows=1)
        for subj in self.study_subjects:
            if not (subj in self.subjects):
                df_met.drop(subj, axis='index', inplace=True)

        df_met_change = df_met.subtract(df_met['experiment'], axis='index')
        df_met_relchange = df_met_change.divide(df_met['experiment'], 
            axis='index')
        df_met_relchange.drop('experiment', axis='columns', inplace=True)
        max_reduction = df_met_relchange.min().min() * 100

        fig = pl.figure(figsize=(4*1.2, 6*1.2))
        ax = fig.add_subplot(1,1,1)
        df_met_relchange.plot.barh(ax=ax, color=self.color_list, legend=False)

        fig.tight_layout()
        fig.savefig(target[0])

        # Metabolics + moment trends overview plot
        fig = pl.figure(figsize=(8.5, 11))
        from matplotlib import gridspec        
        gs = gridspec.GridSpec(4, 4)

        # Top row: metabolics plots
        for icond, cond_name in enumerate(self.cond_names):
            ax = fig.add_subplot(gs[0, icond])

            df_met_cond = df_met.xs(cond_name, level='condition')
            df_met_change = df_met_cond.subtract(df_met_cond['experiment'], 
                axis='index')
            df_met_relchange = df_met_change.divide(df_met_cond['experiment'], 
                axis='index')
            df_met_relchange.drop('experiment', axis='columns', inplace=True)

            df_met_by_subjs = df_met_relchange.groupby(level='subject').mean()
            met_mean = df_met_by_subjs.mean()[self.device_names] * 100
            met_std = df_met_by_subjs.std()[self.device_names] * 100

            ind = np.arange(len(self.device_names))
            barlist = ax.bar(ind, met_mean, yerr=met_std)
            for barelt, color in zip(barlist, self.color_list):
                barelt.set_color(color)

            ax.set_xticks(ind)
            ax.set_xticklabels(self.label_list, rotation=75, ha='left')
            ax.set_ylabel('reduction in metabolic cost (%)')
            ymin = max_reduction
            ymax = 0.0
            ax.set_ylim(ymin, ymax)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_ticks_position('top')

            moments_dep = moments_dep_dict[cond_name]
            plot_iterate = zip(moments_dep, self.device_names, self.label_list, 
                self.color_list)

            # Remaining rows: moment plots
            ls = None
            for idof, dof in enumerate(self.dof_names):
                ax = fig.add_subplot(gs[idof+1, icond])
                ax.axhline(color='k', linewidth=0.5, zorder=0)
                for idep, (dep, device, label, color) in enumerate(plot_iterate):
                    df = pd.read_csv(dep, index_col=0,
                                header=[0, 1, 2, 3], skiprows=1)
                    if not idep:
                        plot_dof(ax, df, dof, 'net', 'net joint moment',
                            'darkgray', ls='--',
                             drop_subjects=self.drop_subjects)
                    if hasattr(self, 'linestyle_list'):
                        ls = self.linestyle_list[idep]
                    plot_dof(ax, df, dof, 'device', label, color, ls=ls,
                        drop_subjects=self.drop_subjects)

                set_axis(ax, idof, dof, self.dof_labels)

        fig.tight_layout()
        fig.savefig(target[1])

    def plot_overview_torque_parameters(self, file_dep, target, 
            param_device_dep_dict):

        # Torque parameters overview plot
        from matplotlib.backends.backend_pdf import PdfPages
        # from matplotlib import gridspec      
        # gs = gridspec.GridSpec(4, 4)

        plot_iterate = zip(self.device_names, self.label_list, 
            self.color_list)

        device1 = self.device_names[0]
        cond_name1 = self.cond_names[0]
        all_conds_df_dict_norm = dict()
        all_conds_df_dict = dict()

        with PdfPages(target[0]) as pdf:

            for iplot, (device, label, color) in enumerate(plot_iterate):
                fig = pl.figure(figsize=(6, 3*len(plot_iterate)))
                all_conds_df_dict[device] = pd.DataFrame()
                all_conds_df_dict_norm[device] = pd.DataFrame()

                
                for icond, cond_name in enumerate(self.cond_names):

                    dep = param_device_dep_dict[device][cond_name]
                    df = pd.read_csv(dep, index_col=0, header=[0, 1], 
                        skiprows=1)

                    param_names, param_bounds = \
                        get_parameter_info(self.study, device)

                    # Append condition to all conds dataframe to use later
                    df_join = all_conds_df_dict[device].join(df, how='outer')
                    all_conds_df_dict[device] = df_join

                    # Normalize within [0,1]
                    for ip, param_name in enumerate(param_names):
                        norm_factor = \
                            param_bounds['upper'][ip] - param_bounds['lower'][ip]
                        df.loc[param_name] -= param_bounds['lower'][ip]
                        df.loc[param_name] /= norm_factor

                    # Append condition to all conds dataframe to use later (norm)
                    df_join_norm = all_conds_df_dict_norm[device].join(df, 
                        how='outer')
                    all_conds_df_dict_norm[device] = df_join_norm
                                
                    ax = fig.add_subplot(len(self.cond_names), 1, icond+1)
                    bplot = ax.boxplot(df.values.transpose(), patch_artist=True)
                    for patch in bplot['boxes']:
                        patch.set_facecolor(color)
                    ax.set_ylim(0.0, 1.0)
                    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
                    ax.set_ylabel(cond_name)    
                    if icond == (len(self.cond_names)-1):
                        ax.set_xticklabels(param_names, rotation=45, ha='right',
                            fontsize=14)
                    else:
                        ax.set_xticklabels([])
                    if icond == 0:
                        ax.set_title(label, fontsize=14)
                
                fig.tight_layout()
                pdf.savefig(fig)
                pl.close()

        # Torque parameter breakdown collapsed across conditions
        df_mean_params = pd.DataFrame()
        with PdfPages(target[1]) as pdf:
            for iplot, (device, label, color) in enumerate(plot_iterate):
                fig = pl.figure(figsize=(6, 6))

                param_names, param_bounds = \
                        get_parameter_info(self.study, device)

                df_norm = all_conds_df_dict_norm[device]

                ax = fig.add_subplot(111)
                bplot = ax.boxplot(df_norm.values.transpose(), patch_artist=True)

                for patch in bplot['boxes']:
                        patch.set_facecolor(color)
                ax.set_ylim(0.0, 1.0)
                ax.set_yticks(np.arange(0.0, 1.1, 0.1))
                ax.set_ylabel(cond_name)    
                ax.set_xticklabels(param_names, rotation=45, ha='right',
                        fontsize=14)
                ax.set_title(label, fontsize=14)

                fig.tight_layout()
                pdf.savefig(fig)
                pl.close()

                # Average torque parameters across all gait cycles and store
                # in DataFrame
                df = all_conds_df_dict[device]
                df_mean = df.mean(axis=1)
                df_mean = df_mean.to_frame()
                df_mean.columns = [device]
                df_mean_params = df_mean_params.join(df_mean, how='outer')

        # Print mean torque parameters (unnormalized) to file
        target_dir = os.path.dirname(target[2])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(target[2], 'w') as f:
            f.write('# all columns are torque parameter means.\n')
            df_mean_params.to_csv(f)

class TaskPlotMetabolicReductions(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, plot_lists, folder, 
            cond_names=['walk1','walk2','walk3','walk4'], subjects=None,
            max_metabolic_reduction=35, fig_width=8, fig_height=6, 
            bar_width=0.4):
        super(TaskPlotMetabolicReductions, self).__init__(study)
        self.name = 'plot_metabolic_reductions_%s' % folder
        self.doc = 'Plots of metabolic reductions across devices.'
        self.output_path = os.path.join(study.config['analysis_path'], folder)
        if not os.path.exists(self.output_path): os.mkdir(self.output_path)
        self.cond_names = cond_names
        self.max_metabolic_reduction = max_metabolic_reduction
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.bar_width = bar_width
        self.device_list = plot_lists['device_list']
        self.device_names = list()
        self.device_dirs = list()
        for device in self.device_list:
            self.device_dirs.append(device.replace('/', '\\'))
            if len(device.split('/')) > 1:
                self.device_names.append('_'.join(device.split('/')))
            else:
                self.device_names.append(device)
        self.label_list = plot_lists['label_list']
        self.color_list = plot_lists['color_list']
        self.index_list = plot_lists['index_list']
        self.xticks = plot_lists['xticks']
        if 'linestyle_list' in plot_lists:
            self.linestyle_list = plot_lists['linestyle_list']

        self.folder = folder

        if ('He' in folder) or ('Hf' in folder):
            self.legend_dof = 'hip_flexion_r'
        elif ('Ke' in folder) or ('Kf' in folder):
            self.legend_dof = 'knee_angle_r'
        else:
            self.legend_dof = 'ankle_angle_r'

        self.dof_names = list()
        self.dof_labels = list()
        for device_name in self.device_names:
            if ('He' in device_name) or ('Hf' in device_name):
                if 'hip_flexion_r' in self.dof_names: continue
                self.dof_names.append('hip_flexion_r')
                self.dof_labels.append('hip extension')
            if ('Ke' in device_name) or ('Kf' in device_name):
                if 'knee_angle_r' in self.dof_names: continue
                self.dof_names.append('knee_angle_r')
                self.dof_labels.append('knee extension')
            if ('Ap' in device_name) or ('Ad' in device_name):
                if 'ankle_angle_r' in self.dof_names: continue
                self.dof_names.append('ankle_angle_r')
                self.dof_labels.append('ankle plantarflexion')

        if study.costFunction:
            if not (study.costFunction == 'Default'):
                cost = '_' + study.costFunction
            else:
                cost = ''
        else:
            cost = ''
        self.cost = cost

        if subjects == None:
            self.subjects = [s.name for s in study.subjects]
        else:
            self.subjects = subjects 
        self.study_subjects = [s.name for s in self.study.subjects]
        self.drop_subjects = list()
        for study_subj in self.study_subjects:
            if not (study_subj in self.subjects):
                self.drop_subjects.append(study_subj)

        self.add_action(
            [os.path.join(self.output_path, 
                'whole_body_metabolic_rates_%s%s.csv' % (folder, cost))], 
            [os.path.join(self.output_path,
                '%s_metabolics_per_reduction.pdf' % folder)],
            self.plot_metabolics)

    def plot_metabolics(self, file_dep, target):

        fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        ax = fig.add_subplot(1,1,1)

        met_mean, met_std, df_met_change = get_percent_reductions(file_dep[0], 
            self.device_names, 'all', drop_subjects=self.drop_subjects,
            add_basal_rate=True, assume_symmetry=True)

        met_mean.to_csv(os.path.join(self.output_path, 
            'total_whole_body_reductions.csv'))
        
        handles = list()
        labels = list()
        singleFlag = True
        coupledFlag = True
        uncoupledFlag = True
        for device, index in zip(self.device_names, self.index_list):
            # color = self.color_list[index]
            if 'multControls' in device:
                index += self.bar_width / 2.0
                color = 'blue'
            elif 'fixed' in device:
                color = 'dimgray' 
            else: 
                index -= self.bar_width / 2.0
                color = 'orange'
            
            hbar = ax.bar(index, met_mean[device], color=color,
                   ls='-', width=self.bar_width)

            if ('multControls' in device) and uncoupledFlag:
                handles.append(hbar)
                labels.append('multi-joint, uncoupled')
                uncoupledFlag = False
            elif ('fixed' in device) and singleFlag:
                handles.append(hbar)
                labels.append('single-joint') 
                singleFlag = False
            elif ('fixed' not in device) and coupledFlag: 
                handles.append(hbar)
                labels.append('multi-joint, coupled')
                coupledFlag = False
                
            yerr = met_std[device]
            plotline, caplines, barlinecols = ax.errorbar(
                index, met_mean[device], yerr=yerr, color='black',
                capsize=0, solid_capstyle='projecting', lw=1, uplims=True)
            caplines[0].set_marker('_')
            caplines[0].set_markersize(8)

        ax.set_xticks(self.xticks )
        ax.set_xticklabels(self.label_list)
        ax.margins(0.2, None)
        ax.set_ylabel(r'change in gross average whole-body metabolic rate (%)', 
                fontsize=10)
        ax.set_ylim(-self.max_metabolic_reduction, 0)
        ax.set_xlim(-0.1-(self.bar_width/2), 
                    self.index_list[-1]+self.bar_width+0.1)  
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('top')
        ax.legend(handles, labels, loc=(0.3, 0.25))
        # for i, label in enumerate(ax.get_xmajorticklabels()):
        #     if i < 5:
        #         label.set_rotation(45)
        #         label.set_horizontalalignment('left')

        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[0].replace('.pdf','.png'), ppi=1800)
        fig.savefig(target[0].replace('.pdf','.svg'), format='svg', dpi=1000)
        pl.close(fig)

class TaskPlotDeviceShowcase(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, name, plot_lists, cond_name, subjects, act_muscs,
            met_ylim=[-20, 5], 
            met_yticks=[-20, -15, -10, -5, 0, 5],
            moment_ylim=[-1, 2],
            moment_yticks=[-1, 0, 1, 2],
            legend_loc=(0.55, 0.7),
            include_activations=True):
        super(TaskPlotDeviceShowcase, self).__init__(study)
        self.device_list = plot_lists['device_list']
        self.name = 'plot_%s' % name
        self.doc = 'Showcase the results from a single device.'
        self.label_list = plot_lists['label_list']
        self.color_list = plot_lists['color_list']
        self.cond_name = cond_name
        self.act_muscs = act_muscs
        self.folder = 'showcase_%s' % self.device_list[-1]
        self.output_path = os.path.join(study.config['analysis_path'], 
                self.folder)
        self.met_ylim = met_ylim
        self.met_yticks = met_yticks
        self.moment_ylim = moment_ylim
        self.moment_yticks = moment_yticks
        self.legend_loc = legend_loc
        self.include_activations = include_activations

        if ('He' in self.folder) or ('Hf' in self.folder):
            self.legend_dof = 'hip_flexion_r'
        elif ('Ke' in self.folder) or ('Kf' in self.folder):
            self.legend_dof = 'knee_angle_r'
        else:
            self.legend_dof = 'ankle_angle_r'

        self.dof_names = list()
        self.dof_labels = list()
        for device in self.device_list:
            if ('H' in device) or ('Hf' in device):
                if 'hip_flexion_r' in self.dof_names: continue
                self.dof_names.append('hip_flexion_r')
                self.dof_labels.append('hip extension')
            if ('K' in device) or ('Kf' in device):
                if 'knee_angle_r' in self.dof_names: continue
                self.dof_names.append('knee_angle_r')
                self.dof_labels.append('knee extension')
            if ('A' in device) or ('Ad' in device):
                if 'ankle_angle_r' in self.dof_names: continue
                self.dof_names.append('ankle_angle_r')
                self.dof_labels.append('ankle plantarflexion')

        self.fill_between = True # if len(self.device_list) == 1 else False

        if study.costFunction:
            if not (study.costFunction == 'Default'):
                cost = '_' + study.costFunction
            else:
                cost = ''
        else:
            cost = ''
        self.cost = cost

        self.subjects = subjects 
        self.study_subjects = [s.name for s in self.study.subjects]
        self.drop_subjects = list()
        for study_subj in self.study_subjects:
            if not (study_subj in self.subjects):
                self.drop_subjects.append(study_subj)

        # File dependencies
        deps = list()
        deps.append(os.path.join(self.output_path,
                'muscle_metabolic_rates_%s%s.csv' % (self.folder, cost)))
        deps.append(os.path.join(self.output_path, 
                'whole_body_metabolic_rates_%s%s.csv' % (self.folder, cost)))

        for device in self.device_list:
            deps.append(os.path.join(study.config['results_path'], device,
                '%s_%s_moments%s.csv' % (device, cond_name, cost)))

        deps.append(os.path.join(study.config['results_path'], 'experiments', 
                'experiment_%s_activations%s.csv' % (cond_name, cost)))

        for device in self.device_list:
            deps.append(os.path.join(study.config['results_path'], device, 
                '%s_%s_activations%s.csv' % (device, cond_name, cost)))
            
        # Plot action.
        self.add_action(deps,
            [os.path.join(self.output_path, '%s_%s_showcase.pdf' 
                 % (self.device_list[-1], cond_name))],
            self.plot_device_showcase)

    def plot_device_showcase(self, file_dep, target):

        unassisted_color = 'dimgray'

        scale = 2.2
        aspect = 1.35 if self.include_activations else 1.1
        fig = pl.figure(figsize=(scale, scale*aspect))
        dpi_scale_trans = fig.dpi_scale_trans
        pl.subplots_adjust(left=0.22, top=0.9, bottom=0.1, right=0.95)
        from matplotlib import gridspec
        if self.include_activations:
            heights = [0.8, 1.0, 0.5]
            nrows = 3
        else: 
            heights = [0.8, 1.0]
            nrows = 2
        gs = gridspec.GridSpec(ncols=144, nrows=nrows, height_ratios=heights,
            hspace=0.6)

        # https://stackoverflow.com/questions/28615887/how-to-move-a-ticks-label-in-matplotlib
        import matplotlib.transforms
        def shift_xticks(ax, dy):
            dy = dy/72.0 
            offset = matplotlib.transforms.ScaledTranslation(0, dy, 
                dpi_scale_trans)
            # apply offset transform to all x ticklabels.
            for label in ax.xaxis.get_majorticklabels():
                label.set_transform(label.get_transform() + offset)

        def shift_yticks(ax, dx):
            dx = dx/72.0 
            offset = matplotlib.transforms.ScaledTranslation(dx, 0, 
                dpi_scale_trans)
            # apply offset transform to all x ticklabels.
            for label in ax.yaxis.get_majorticklabels():
                label.set_transform(label.get_transform() + offset)

        # Plot moments
        # ------------
        num_devices = len(self.device_list)
        moment_deps = file_dep[2:num_devices+2]
        moment_iterate = zip(moment_deps, self.device_list, self.label_list, 
            self.color_list)

        import matplotlib.lines as mlines
        for idof, dof in enumerate(self.dof_names):
            spacing = 144 / len(self.dof_names) - 6
            gap = (144 - spacing*len(self.dof_names)) / (len(self.dof_names)-1)
            col = idof*(spacing+gap)
            # import pdb
            # pdb.set_trace()
            ax = fig.add_subplot(gs[0, col:col+spacing])
            pl.axhline(zorder=0, color='darkgray', linewidth=0.25, linestyle='--')
            
            handles = list()
            labels = list()
            for i, (dep, device, label, color) in enumerate(moment_iterate):
                df = pd.read_csv(dep, index_col=0, header=[0, 1, 2, 3], 
                        skiprows=1)
                if not i:
                   plot_dof(ax, df, dof, 'net', 'net joint moment', unassisted_color,
                        lw=0.7, ls='-', drop_subjects=self.drop_subjects, zorder=0,
                        fill_between=self.fill_between)
                   h = mlines.Line2D([], [], ls='-', color=unassisted_color, lw=1)
                   handles.append(h)
                   labels.append('net joint moment')

                plot_dof(ax, df, dof, 'device', label, color, 
                    ls='-', lw=0.7, drop_subjects=self.drop_subjects, 
                    fill_between=self.fill_between, zorder=1)
                h = mlines.Line2D([], [], ls='-', color=color, lw=1)
                handles.append(h)       
                labels.append(label)

            set_axis(ax, idof, dof, self.dof_labels, label_as_title=True)
            ax.set_ylim(self.moment_ylim[0], self.moment_ylim[1])
            ax.set_yticks(self.moment_yticks)
            if not idof: 
                ax.set_ylabel('moment (N-m/kg)', fontsize=5)
                leg = ax.legend(handles, labels, loc=self.legend_loc, 
                    fontsize=2.5, fancybox=False, edgecolor='black', 
                    borderpad=0.8, frameon=False)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=3.5)
            ax.set_xlabel('% gait cycle', fontsize=4)
            shift_xticks(ax, 2)
            shift_yticks(ax, 2)

        # Metabolic breakdown
        # -------------------
        ax = fig.add_subplot(gs[1, :])

        met_mean, met_std, df_met_change = \
            get_percent_reductions_muscles(file_dep[0], file_dep[1],
            self.device_list, self.cond_name, drop_subjects=self.drop_subjects,
            add_basal_rate=True, assume_symmetry=True)

        muscle_index = ['psoas_r', 'glut_max2_r', 'rect_fem_r', 'semimem_r', 
                        'vas_int_r', 'bifemsh_r', 'med_gas_r', 'soleus_r', 
                        'tib_ant_r']
        met_mean = 100 * met_mean.reindex(muscle_index)
        met_std = 100 * met_std.reindex(muscle_index)
        # df_met_change.reindex(muscle_index) 

        met_mean_copy = met_mean.copy(deep=True)
        met_mean_copy.plot.bar(ax=ax, color=self.color_list, legend=False)
        ax.axes.get_xaxis().get_label().set_visible(False)

        # for index, musc in enumerate(muscle_index):
        #     # met_mean_musc = met_mean[musc]
        #     import pdb
        #     pdb.set_trace()

        #     plotline, caplines, barlinecols = ax.errorbar(
        #             index, met_mean[device], yerr=yerr, color='black',
        #             capsize=0, solid_capstyle='projecting', lw=1, uplims=True)
        #     caplines[0].set_marker('_')
        #     caplines[0].set_markersize(8)

        nice_muscle_names_dict = self.study.nice_muscle_names_dict
        nice_muscle_names = \
            [nice_muscle_names_dict[name] for name in met_mean.index]

        # import pdb
        # pdb.set_trace()
        ax.set_xticks(np.arange(len(met_mean.index)))
        ax.set_xticklabels(nice_muscle_names, rotation=45, ha='left',
                fontsize=5, rotation_mode='anchor')
        ax.set_ylabel('change in average\nmuscle metabolic rate (%)',
                fontsize=5)
        ax.set_ylim(self.met_ylim[0], self.met_ylim[1])
        ax.set_yticks(self.met_yticks)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(labelsize=3.5)
        pl.axhline(zorder=0, color='darkgray', linewidth=0.25, linestyle='--')
        shift_xticks(ax, -2)
        shift_yticks(ax, 2)


        dy = -4/72. 
        box = ax.get_position()
        box.y0 = box.y0 + dy
        box.y1 = box.y1 + dy
        ax.set_position(box)

        # Activations
        # -----------
        if self.include_activations:
            device_list = ['experiment'] + self.device_list
            label_list = ['unassisted'] + self.label_list
            color_list = [unassisted_color] + self.color_list
            act_deps = file_dep[(num_devices+2):(2*num_devices+4)]
            plot_iterate = zip(act_deps, device_list, label_list, color_list)

            mean_list = list()
            std_list = list()
            pgc_list = list()
            for dep in act_deps:
                df = pd.read_csv(dep, index_col=0, header=[0, 1, 2], skiprows=1)
                for subj in self.drop_subjects:
                    df.drop(subj, axis='columns', inplace=True)

                df_by_subj_musc = df.groupby(level=['subject', 'muscle'], 
                        axis=1).mean()
                df_mean = df_by_subj_musc.groupby(level=['muscle'], axis=1).mean()
                df_std = df_by_subj_musc.groupby(level=['muscle'], axis=1).std() 
                pgc = df_mean.index

                mean_list.append(df_mean)
                std_list.append(df_std)
                pgc_list.append(pgc)

            from math import floor
            for imusc, musc in enumerate(self.act_muscs):
                spacing = 144 / len(self.act_muscs) - 6
                gap = (144 - spacing*len(self.act_muscs)) / (len(self.act_muscs) - 1)
                col = int(imusc*(spacing+gap))
                ax = fig.add_subplot(gs[2, col:col+int(spacing)])
                ax.axhline(color='k', linewidth=0.5, zorder=0)

                for i, (dep, device, label, color) in enumerate(plot_iterate):
                    df_mean = mean_list[i]
                    df_std = std_list[i]
                    pgc = pgc_list[i]
                    lw = 0.7 if device == 'experiment' else 0.7
                    color = unassisted_color if device == 'experiment' else color
                    ax.plot(pgc, df_mean[musc], label=label, color=color, 
                        linestyle='-', linewidth=lw)
                    # ax.fill_between(pgc, df_mean[musc]-df_std[musc], 
                    #     df_mean[musc]+df_std[musc], facecolor=color, 
                    #     alpha=0.3, zorder=1)
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 0.5)

                # ax.set_title(nice_muscle_names[musc])
                ax.set_xlabel('% gait cycle', fontsize=4)
                ax.set_xticks([0, 50, 100])
                ax.set_yticks([0, 0.5])
                if imusc:
                    ax.set_yticklabels([])
                ax.tick_params(axis='both', which='major', labelsize=3)
                if not imusc: ax.set_ylabel('activation', fontsize=5)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                ax.set_title(nice_muscle_names_dict[musc], fontsize=4)
                shift_xticks(ax, 2)
                shift_yticks(ax, 2) 

        # Generate figure
        # fig.tight_layout(h_pad=0.3)
        # fig.tight_layout()
        # gs.tight_layout(fig)
        fig_name = '%s_showcase' % self.device_list[0]
        fig.savefig(os.path.join(self.output_path, fig_name + '.pdf'))
        fig.savefig(os.path.join(self.output_path, fig_name + '.png'), ppi=1800)
        fig.savefig(os.path.join(self.output_path, fig_name + '.svg'))
        fig.savefig(os.path.join(self.output_path, fig_name + '.eps'), 
            metadata='eps')
        pl.close(fig)

class TaskPlotMetabolicSummary(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, plot_lists, subjects):
        super(TaskPlotMetabolicSummary, self).__init__(study)
        self.name = 'plot_metabolic_summary'
        self.doc = 'Showcase the results from a single device.'
        self.output_path = os.path.join(study.config['analysis_path'], 
            'metabolic_summary')
        self.device_list = plot_lists['device_list']
        self.num_dof_list = plot_lists['num_dof_list']
        self.label_list = plot_lists['label_list']
        self.color_list = plot_lists['color_list']
        self.mult_device_list = plot_lists['mult_device_list']

        self.subjects = subjects 
        self.study_subjects = [s.name for s in study.subjects]
        self.drop_subjects = list()
        for study_subj in self.study_subjects:
            if not (study_subj in self.subjects):
                self.drop_subjects.append(study_subj)

        if study.costFunction:
            if not (study.costFunction == 'Default'):
                cost = '_' + study.costFunction
            else:
                cost = ''
        else:
            cost = ''
        self.cost = cost

        # Plot action.
        self.input_fpath = os.path.join(self.output_path, 
                'whole_body_metabolic_rates_%s%s.csv' % ('metabolic_summary', 
                    cost))

        for plot_level in range(len(self.device_list)+1):
            self.add_action([], [], self.plot_metabolic_summary, 
                os.path.join(self.output_path, 
                    'metabolic_summary_%i.pdf' % plot_level),
                plot_level, False)

        for plot_level in range(len(self.device_list)+1):
            self.add_action([], [], self.plot_metabolic_summary, 
                os.path.join(self.output_path, 
                    'metabolic_summary_with_shift_%i.pdf' % plot_level),
                plot_level, True)

    def plot_metabolic_summary(self, file_dep, target, output_fpath, plot_level,
            plot_shift):

        import matplotlib.patches as mpatches

        width = 5
        fig = pl.figure(figsize=(width, 3))
        ax = fig.add_subplot(1,1,1)
        x_arrow_pad = 1.0 / width
        y_arrow_pad = 0
        mutation_scale = 10
        arrowstyle = '->'

        met_mean, met_std, df_met_change = get_percent_reductions(
            self.input_fpath, self.device_list + self.mult_device_list, 'all', 
            drop_subjects=self.drop_subjects, add_basal_rate=True, 
            assume_symmetry=True)

        plot_device_list = [False] * len(self.device_list)

        for i in range(plot_level):
            plot_device_list[i] = True


        plot_iterate = zip(self.device_list, self.num_dof_list, self.label_list, 
            self.color_list, plot_device_list)
        for i, (device, num_dof, label, color, plot_device) in enumerate(plot_iterate):
            if not plot_device: continue

            if num_dof == 1:
                ax.plot(4, met_mean[device], color=color, marker='o', 
                    markersize=12)
            elif num_dof == 2:
                param5_mean = met_mean[device]
                param8_mean = met_mean[device + '_multControls']
                if plot_shift or ('deviceHfAp' in device):
                    ax.plot(5, param5_mean, color=color, marker='o', 
                        markersize=12)
                    ax.plot(8, param8_mean, color=color, 
                        marker='o', markersize=12, alpha=0.3)
                    arrow = mpatches.FancyArrowPatch(
                        (8-x_arrow_pad, param8_mean), 
                        (5+x_arrow_pad, param5_mean+y_arrow_pad), 
                        color=color, mutation_scale=mutation_scale,
                        arrowstyle=arrowstyle)
                    ax.add_patch(arrow)
                else:
                    ax.plot(8, param8_mean, color=color, 
                        marker='o', markersize=12)
                # ax.plot([5, 8], [param5_mean, param8_mean], color=color, 
                #     linestyle='-', linewidth=2)
            elif num_dof == 3:
                param6_mean = met_mean[device]
                param12_mean = met_mean[device + '_multControls']
                if plot_shift:
                    ax.plot(6, param6_mean, color=color, marker='o', 
                        markersize=12)
                    ax.plot(12, param12_mean, color=color, 
                        marker='o', markersize=12, alpha=0.3)
                    arrow = mpatches.FancyArrowPatch(
                        (12-x_arrow_pad, param12_mean), 
                        (6+x_arrow_pad, param6_mean+y_arrow_pad), 
                        color=color, mutation_scale=mutation_scale,
                        arrowstyle=arrowstyle)
                    ax.add_patch(arrow)
                else:
                    ax.plot(12, param12_mean, color=color, 
                        marker='o', markersize=12)
                # ax.plot([6, 12], [param6_mean, param12_mean], color=color, 
                #     linestyle='-', linewidth=2)

        ax.set_xlabel('number of control parameters')
        ax.set_xticks(np.arange(4, 13))
        ax.set_xticklabels(['4','5','6','', '8','','','','12'])
        ax.set_xlim(3, 13)
        ax.set_ylabel('change in metabolic cost (%)')
        ax.set_ylim(-35, 0)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        # pl.grid(which='both', color='silver', linestyle='--')
        ax.set_axisbelow(True)

        fig.tight_layout()
        fig.savefig(output_fpath)
        fig.savefig(output_fpath.replace('.pdf','.png'), ppi=1800)
        fig.savefig(output_fpath.replace('.pdf','.svg'), format='svg', 
            dpi=1000)
        pl.close(fig)