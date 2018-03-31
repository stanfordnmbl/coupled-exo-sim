import os

import numpy as np
import pylab as pl
import pandas as pd
import pdb

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

class TaskCalibrateParametersSetup(osp.SetupTask):
    REGISTRY = []
    def __init__(self, trial, **kwargs):
        super(TaskCalibrateParametersSetup, self).__init__('calibrate', trial,
            **kwargs)
        self.doc = "Create a setup file for a parameter calibration tool."
        self.kinematics_file = os.path.join(self.trial.results_exp_path, 'ik',
                '%s_%s_ik_solution.mot' % (self.study.name, self.trial.id))
        self.rel_kinematics_file = os.path.relpath(self.kinematics_file,
                self.path)
        self.kinetics_file = os.path.join(self.trial.results_exp_path,
                'id', 'results', '%s_%s_id_solution.sto' % (self.study.name,
                    self.trial.id))
        self.rel_kinetics_file = os.path.relpath(self.kinetics_file,
                self.path)
        self.emg_file = os.path.join(self.trial.results_exp_path, 
                'expdata', 'emg.sto')
        self.rel_emg_file = os.path.relpath(self.emg_file, self.path)
        self.results_setup_fpath = os.path.join(self.path, 'setup.m')
        self.results_output_fpath = os.path.join(self.path, 
            '%s_%s_calibrate.mat' % (self.study.name, self.tricycle.id))

        # Fill out setup.m template and write to results directory
        self.create_setup_action()

    def create_setup_action(self): 
        self.add_action(
                    ['templates/%s/setup.m' % self.tool],
                    [self.results_setup_fpath],
                    self.fill_setup_template,  
                    init_time=self.init_time,
                    final_time=self.final_time,      
                    )


    def fill_setup_template(self, file_dep, target,
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
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
            content = content.replace('@EMG_PATH@', self.rel_emg_file)

        with open(target[0], 'w') as f:
            f.write(content)

class TaskCalibrateParameters(osp.ToolTask):
    REGISTRY = []
    def __init__(self, trial, calibrate_setup_task, **kwargs):
        super(TaskCalibrateParameters, self).__init__(calibrate_setup_task, 
            trial, opensim=False, **kwargs)
        self.doc = "Run parameter calibration tool via DeGroote MRS solver."
        self.results_setup_fpath = calibrate_setup_task.results_setup_fpath
        self.results_output_fpath = calibrate_setup_task.results_output_fpath

        self.file_dep += [
                self.results_setup_fpath,
                self.subject.scaled_model_fpath,
                calibrate_setup_task.kinematics_file,
                calibrate_setup_task.kinetics_file,
                calibrate_setup_task.emg_file,
                ]

        self.actions += [
                self.run_parameter_calibration,
                self.delete_muscle_analysis_results,
                ]

        self.targets += [
                self.results_output_fpath
                ]

    def run_parameter_calibration(self):
        with util.working_directory(self.path):
            # On Mac, CmdAction was causing MATLAB ipopt with GPOPS output to
            # not display properly.

            status = os.system('matlab %s -logfile matlab_log.txt -wait -r "try, '
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

class TaskCalibrateParametersPost(osp.PostTask):
    REGISTRY = []
    def __init__(self, trial, calibrate_setup_task, **kwargs):
        super(TaskCalibrateParametersPost, self).__init__(calibrate_setup_task,
            trial, **kwargs)
        self.doc = 'Postprocessing of parameter calibration results.'
        self.setup_task = calibrate_setup_task
        self.results_output_fpath = self.setup_task.results_output_fpath

        self.emg_fpath = os.path.join(trial.results_exp_path, 'expdata', 
            'emg_with_headers.sto')

        self.add_action([self.emg_fpath,
                         self.results_output_fpath],
                        [os.path.join(self.path, 'muscle_activity'),
                         os.path.join(self.path, 'reserve_activity.pdf')],
                        self.plot_muscle_and_reserve_activity)

    def plot_muscle_and_reserve_activity(self, file_dep, target):

        emg = util.storage2numpy(file_dep[0])
        time = emg['time']

        def min_index(vals):
            idx, val = min(enumerate(vals), key=lambda p: p[1])
            return idx

        start_idx = min_index(abs(time-self.setup_task.cycle.start))
        end_idx = min_index(abs(time-self.setup_task.cycle.end))

        # Load mat file fields
        muscle_names = util.hdf2list(file_dep[1], 'MuscleNames', type=str)
        df_exc = util.hdf2pandas(file_dep[1], 'MExcitation', labels=muscle_names)
        df_act = util.hdf2pandas(file_dep[1], 'MActivation', labels=muscle_names)
        dof_names = util.hdf2list(file_dep[1], 'DatStore/DOFNames', 
            type=str)

        pgc_emg = np.linspace(0, 100, len(time[start_idx:end_idx]))
        pgc_exc = np.linspace(0, 100, len(df_exc.index))
        pgc_act = np.linspace(0, 100, len(df_act.index))

        muscles = self.study.muscle_names
        fig = pl.figure(figsize=(12, 12))
        nice_act_names = {
                'glut_max2_r': 'glut. max.',
                'psoas_r': 'iliopsoas',
                'semimem_r': 'hamstrings',
                'rect_fem_r': 'rect. fem.',
                'bifemsh_r': 'bi. fem. s.h.',
                'vas_int_r': 'vasti',
                'med_gas_r': 'gastroc.',
                'soleus_r': 'soleus',
                'tib_ant_r': 'tib. ant.',
                }

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
            labels = ['%s exc.' % nice_act_names[musc_name],
                      '%s act.' % nice_act_names[musc_name]]
            ax.legend(handles, labels)
            
            if emg_map.get(musc_name):
                y_emg = emg[emg_map[musc_name]]
                ax.plot(pgc_emg, y_emg[start_idx:end_idx], color='black', 
                    linestyle='-')

            # ax.legend(frameon=False, fontsize=6)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.0)
            ax.set_title(nice_act_names[musc_name])
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
        df_res = util.hdf2pandas(file_dep[1],'RActivation', labels=dof_names)
        pp.plot_reserve_activity(target[1], df_res)

class TaskMRSDeGrooteModPost(osp.TaskMRSDeGrooteModPost):
    REGISTRY = []
    def __init__(self, trial, mrsmod_task, **kwargs):
        super(TaskMRSDeGrooteModPost, self).__init__(trial, mrsmod_task, 
            **kwargs)
        # original, "no-mod" solution
        self.mrs_results_output_fpath = \
            mrsmod_task.mrs_setup_task.results_output_fpath

        if (('scaledID' not in self.mrsmod_task.mod_name) and
            ('exp' not in self.mrsmod_task.mod_name)):
            self.add_action([self.results_output_fpath],
                            [os.path.join(self.path,
                                'device_moment_arms.pdf')],
                            self.plot_device_moment_arms)

        if 'pass' in self.mrsmod_task.mod_name:
            self.add_action([self.results_output_fpath],
                            [os.path.join(self.path,
                                'passive_device_info.pdf')],
                            self.plot_passive_device_information)

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
        if 'act' in self.mrsmod_task.mod_name:
            act_torques = util.hdf2pandas(file_dep[0], 
                'DatStore/ExoTorques_Act', labels=dof_names)
            device_torques.append(act_torques)
            device_names.append('active')
            device_colors.append('green')

        if 'pass' in self.mrsmod_task.mod_name:
            pass_torques = util.hdf2pandas(file_dep[0], 
                'DatStore/ExoTorques_Pass', labels=dof_names)
            device_torques.append(pass_torques)
            device_names.append('passive')
            device_colors.append('blue')

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
        act_mom_arms = np.array([[0.0, 0.0, 0.0]])
        if 'act' in self.mrsmod_task.mod_name:
            act_mom_arms = util.hdf2numpy(file_dep[0],
                'DatStore/MomentArms_Act')

        pass_mom_arms = np.array([[0.0, 0.0, 0.0]])
        if 'pass' in self.mrsmod_task.mod_name:
            pass_mom_arms = util.hdf2numpy(file_dep[0],
                'DatStore/MomentArms_Pass')

        # Plot moment arms
        fig = pl.figure(figsize=(11,8.5))
        ax = fig.add_subplot(1,1,1)
        pos = np.arange(num_dofs)
        width = 0.4

        bar1 = ax.bar(pos, act_mom_arms[0], width, color='green')
        bar2 = ax.bar(pos + width, pass_mom_arms[0], width, color='blue')
        ax.set_xticks(pos + width / 2)
        ax.set_xticklabels(dof_names, fontsize=10)
        # ax.set_yticks(np.arange(-100,105,5))
        # for label in ax.yaxis.get_ticklabels()[1::2]:
        #     label.set_visible(False)
        ax.set_ylabel('Moment Arms', fontsize=12)
        ax.grid(which='both', axis='both', linestyle='--')
        ax.set_axisbelow(True)
        ax.legend([bar1, bar2], ['active', 'passive'])

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
            'DatStore/MetabolicRate/whole_body')
        mrs_muscle_metabolic_rates = util.hdf2pandas(file_dep[0],
            'DatStore/MetabolicRate/individual_muscles', labels=muscle_names)

        # Load mat file fields from modified solution
        muscle_names = util.hdf2list(file_dep[1], 'MuscleNames', type=str)
        num_muscles = len(muscle_names)

        mrsmod_whole_body_metabolic_rate = util.hdf2pandas(file_dep[1], 
            'DatStore/MetabolicRate/whole_body')
        mrsmod_muscle_metabolic_rates = util.hdf2pandas(file_dep[1],
            'DatStore/MetabolicRate/individual_muscles', labels=muscle_names)

        reductions = list()
        reduc_names = list()
        colors = list()
        for musc in muscle_names:
            muscle_reduction = 100.0 * ((mrsmod_muscle_metabolic_rates[musc] -
                mrs_muscle_metabolic_rates[musc]) / 
                mrs_muscle_metabolic_rates[musc])
            reductions.append(muscle_reduction)
            reduc_names.append(musc)
            colors.append('b')

        whole_body_reduction = 100.0 * (mrsmod_whole_body_metabolic_rate - 
            mrs_whole_body_metabolic_rate) / mrs_whole_body_metabolic_rate
        reductions.append(whole_body_reduction[0])
        reduc_names.append('whole_body')
        colors.append('r')

        # Plot metabolic reductions
        fig = pl.figure(figsize=(11,8.5))
        ax = fig.add_subplot(1,1,1)
        pos = np.arange(len(muscle_names)+1)
        
        ax.bar(pos, reductions, align='center', color=colors)
        ax.set_xticks(pos)
        ax.set_xticklabels(reduc_names, fontsize=10)
        ax.set_yticks(np.arange(-100,105,5))
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
    muscle_level=False):
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
                    cycles.append(cycle)
                    if not muscle_level:
                        multiindex_tuples.append((
                            cycle.subject.name,
                            cycle.condition.name,
                            # This must be the full ID, not just the cycle
                            # name, because 'cycle01' from subject 1 has
                            # nothing to do with 'cycle01' from subject 2
                            # (whereas the 'walk2' condition for subject 1 is
                            # related to 'walk2' for subject 2).
                            cycle.id))
                    if muscle_level:
                        for mname in study.muscle_names:
                            multiindex_tuples.append((
                                cycle.subject.name,
                                cycle.condition.name,
                                cycle.id,
                                mname))

    return multiindex_tuples, cycles

class TaskAggregateMetabolicRate(osp.StudyTask):
    """Aggregate metabolic rate without and with mods across all subjects and
    gait cycles for each condition provided."""
    REGISTRY = []
    def __init__(self, study, mods=None, subjects=None, 
            conditions=['walk2'], suffix=''):
        super(TaskAggregateMetabolicRate, self).__init__(study)
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

        if mods == None:
            mods = study.mod_names
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
            self.mod_for_file_dep.append('experiment')
            deps.append(os.path.join(
                    cycle.trial.results_exp_path, 'mrs', cycle.name,
                    self.costdir, '%s_%s_mrs.mat' % (study.name, cycle.id))
                    )

        # Prepare for processing simulations of mods.
        for mod in mods:
            for cycle in cycles:
                self.mod_for_file_dep.append(mod)
                deps.append(os.path.join(
                        self.study.config['results_path'],
                        'mrsmod_%s' % mod, cycle.trial.rel_path, 'mrs', 
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
            df = util.hdf2pandas(fpath, 'DatStore/MetabolicRate/whole_body')
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
                'DatStore/MetabolicRate/individual_muscles',
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

class TaskAggregatePeakPower(osp.StudyTask):
    """Aggregate peak instantaneous power for assisted cases across all 
    subjects and gait cycles for each provided condition."""
    REGISTRY = []
    def __init__(self, study, mods=None, subjects=None, 
            conditions=['walk2'], suffix=''):
        super(TaskAggregatePeakPower, self).__init__(study)
        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction
        self.name = 'aggregate_peak_power%s' % suffix
        self.power_fpath = os.path.join(self.suffix_path, 
            'peak_power%s.csv' % suffix)
        self.doc = 'Aggregate peak instantaneous positive power normalized by '
        'subject mass.'
        self.study = study

        if mods == None:
            mods = study.mod_names
        if subjects == None:
            subjects = [s.num for s in study.subjects]

        # Get multiindex tuples and cycles list
        self.multiindex_tuples, cycles = construct_multiindex_tuples(study, 
            subjects, conditions)

        self.mod_for_file_dep = list()
        self.subject_masses = list()
        deps = list()

        for mod in mods:
            for cycle in cycles:
                self.mod_for_file_dep.append(mod)
                self.subject_masses.append(cycle.subject.mass)
                deps.append(os.path.join(
                        self.study.config['results_path'],
                        'mrsmod_%s' % mod, cycle.trial.rel_path, 'mrs', 
                        cycle.name, self.costdir,
                        '%s_%s_mrs.mat' % (study.name, cycle.id))
                    )
        self.add_action(deps,
                [os.path.join(study.config['analysis_path'], self.power_fpath)], 
                self.aggregate_peak_power)

    def aggregate_peak_power(self, file_dep, target):
        import numpy as np
        from scipy.interpolate import interp1d
        from collections import OrderedDict
        peak_norm_power = OrderedDict()
        for ifile, fpath in enumerate(file_dep):
            time = util.hdf2pandas(fpath, 'Time').round(4)
            time_exp = util.hdf2pandas(fpath, 'DatStore/time').round(4)
            df_Texo = util.hdf2pandas(fpath, 'DatStore/ExoTorques_Act')
            df_q_deg = util.hdf2pandas(fpath, 'DatStore/q_exp')
            import math
            df_q_rad = (math.pi / 180.0) * df_q_deg
            #df_q_reidx = df_q_rad.reindex(df_q_rad.index.union(time[0]))

            # Interpolate joint angles to match solution time domain
            f = interp1d(time_exp[0], df_q_rad, kind='cubic', axis=0)
            df_q = pd.DataFrame(f(time[0]))
            
            # Get angular velocities
            df_dq = df_q.diff().fillna(method='backfill')
            dt = time.diff().fillna(method='backfill')
            dt[1] = dt[0]
            dt[2] = dt[0]
            df_dqdt = df_dq / dt

            # Compute active device power
            # P = F*v
            # l = l0 - sum{ri*qi}
            # v = dl/dt = -sum{ri*(dq/dt)i}
            # P = F*(-sum{ri*(dq/dt)i})
            # P = -sum{Mi*(dq/dt)i}
            df_P = df_Texo.multiply(df_dqdt, axis='index').sum(axis=1)

            # Get max value and normalize to subject mass
            Pmax_norm = df_P.max() / self.subject_masses[ifile]

            this_mod = self.mod_for_file_dep[ifile]
            if not this_mod in peak_norm_power:
                peak_norm_power[this_mod] = list()
            peak_norm_power[this_mod].append(Pmax_norm)

        index = pd.MultiIndex.from_tuples(self.multiindex_tuples,
                names=['subject', 'condition', 'cycle'])
        df = pd.DataFrame(peak_norm_power, index=index)

        target_dir = os.path.dirname(target[0])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(target[0], 'w') as f:
            f.write('peak instantaneous positive power normalized by subject '
                'mass (W/kg)\n')
            df.to_csv(f)

class TaskPlotMetabolicReductionVsPeakPower(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, suffix='', mods=[]):
        super(TaskPlotMetabolicReductionVsPeakPower, self).__init__(study)
        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction
        self.name = 'plot_metabolics_versus_power%s' % suffix
        self.mods = mods if mods else self.study.mod_names
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
    def __init__(self, study, suffix='', mods=[]):
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
        mods = self.mods if self.mods else self.study.mod_names
        met_relchange_pcent_mean = df_by_subjs.mean()[mods] * 100
        met_relchange_pcent_std = df_by_subjs.std()[mods] * 100

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
    def __init__(self, study, subjects=None, conditions=['walk2']):
        super(TaskAggregateMomentsExperiment, self).__init__(study)
        self.name = 'aggregate_experiment_moments'
        self.doc = 'Aggregate no-mod actuator moments into a data file.'

        suffix = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 

        if subjects == None:
            subjects = [s.num for s in study.subjects]

        self.cycles = dict()
        for cond_name in conditions:
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
    def __init__(self, study, mod_name, conditions=['walk2'], subjects=None):
        super(TaskAggregateMomentsMod, self).__init__(study)
        self.name = 'aggregate_mod_moments_%s' % mod_name
        self.doc = 'Aggregate actuator moments into a data file.'
        self.mod_name = mod_name
        self.conditions = conditions

        suffix = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 

        if subjects == None:
            subjects = [s.num for s in study.subjects]

        self.cycles = dict()
        for cond_name in conditions:
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
                        self.cycles[cond_name].append(cycle)
                        deps.append(os.path.join(
                                self.study.config['results_path'],
                                'mrsmod_%s' % mod_name,
                                trial.rel_path, 'mrs', cycle.name, self.costdir,
                                '%s_%s_mrs_moments.csv' % (study.name,
                                    cycle.id))
                                )

            self.add_action(deps,
                    [os.path.join(study.config['results_path'], 
                        'mrsmod_%s' % mod_name,
                        'mod_%s_%s_moments%s.csv' % (mod_name,
                            cond_name, suffix)),
                        ],
                    aggregate_moments, cond_name, self.cycles)

class TaskPlotMoments(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, agg_task, conditions=['walk2'], mod=None, 
            subjects=None):
        super(TaskPlotMoments, self).__init__(study)
        task_name = 'experiment' if mod==None else mod
        self.name = 'plot_moment_breakdown_%s' % task_name
        self.doc = 'Plot joint moment breakdown by muscle and device moments'

        for icond, agg_target in enumerate(agg_task.targets):
            # This assumes csv_task.targets and csv_task.cycles hold cycles in
            # the same order.
            self.agg_target = agg_target
            # self.add_action([agg_target], [], 
            #         # [agg_target.replace('.csv', '.pdf')],
            #         self.plot_joint_moment_breakdown)

            self.actions += [self.plot_joint_moment_breakdown]

    def plot_joint_moment_breakdown(self):

        df_all = pd.read_csv(self.agg_target, index_col=0,
                header=[0, 1, 2, 3], skiprows=1)

        # Average over cycles.
        # axis=1 for columns (not rows).
        df_by_subj_dof_musc = df_all.groupby(
                level=['subject', 'DOF', 'actuator'], axis=1).mean()
        df_mean = df_by_subj_dof_musc.groupby(level=['DOF', 'actuator'],
                axis=1).mean()
        df_std = df_by_subj_dof_musc.groupby(level=['DOF', 'actuator'],
                axis=1).std()

        pgc = df_mean.index

        import seaborn.apionly as sns
        palette = sns.color_palette('muted', 9)

        muscles = ['glut_max2_r', 'psoas_r', 'semimem_r', 'rect_fem_r',
                'bifemsh_r', 'vas_int_r', 'med_gas_r', 'soleus_r', 'tib_ant_r']
        colors = {muscles[i]: palette[i] for i in range(9)}
        colors['net'] = 'black' #(0.7,) * 3 # light gray
        colors['active'] = 'green'
        colors['passive'] = 'blue'

        fig = pl.figure(figsize=(9, 3.75))
        dof_names = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
        ylabels = ['hip extension', 'knee extension', 'ankle plantarflexion']
        nice_act_names = {
                'glut_max2_r': 'glut. max.',
                'psoas_r': 'iliopsoas',
                'semimem_r': 'hamstrings',
                'rect_fem_r': 'rect. fem.',
                'bifemsh_r': 'bi. fem. s.h.',
                'vas_int_r': 'vasti',
                'med_gas_r': 'gastroc.',
                'soleus_r': 'soleus',
                'tib_ant_r': 'tib. ant.',
                'net': 'net',
                'active': 'active',
                'passive': 'passive',
                }

        act_names = df_mean.columns.levels[1]
        def plot(column_key, act_name):
            if column_key in df_mean.columns:
                y_mean = -df_mean[column_key]
                y_std = -df_std[column_key]
                if act_name == 'active' or act_name == 'passive':
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
        fig.savefig(self.agg_target.replace('.csv', '.pdf'))
        fig.savefig(self.agg_target.replace('.csv', '.png'), dpi=600)
        pl.close(fig)

class TaskAggregateMuscleActivity(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, mods=None, 
            subjects=None, conditions=['walk2'], suffix=''):
        super(TaskAggregateMuscleActivity, self).__init__(study)
        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'aggregate_muscle_activity%s' % suffix
        self.doc = 'Aggregate muscle activity into a data file.'

        # TODO: this is kind of a mess, find a better solution
        # Create new lists so original doesn't get changed by reference
        if mods == None:
            mods = list(study.mod_names)
        else:
            mods = list(mods)

        # Prefix all mods so directory is correct
        mods_fpath = ['mrsmod_' + mod for mod in mods]

        # Add experiment case is no mods specified. Otherwise, assumed that
        # experiment case already has a task for it
        if mods == None:
            mods.insert(0, 'experiment')
            mods_fpath.insert(0, 'experiments')

        if subjects == None:
            subjects = [s.num for s in study.subjects]

        self.cycles = dict()
        for mod, mod_fpath in zip(mods, mods_fpath):
            for cond_name in conditions:
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
                            self.cycles[cond_name].append(cycle)

                            # Results MAT file paths
                            fpath = os.path.join(study.config['results_path'], 
                                mod_fpath, subject.name, cond.name, 'mrs', 
                                cycle.name, self.costdir,
                                '%s_%s_mrs.mat' % (study.name, cycle.id))
                            deps.append(fpath)

                self.add_action(deps,
                        [
                            os.path.join(study.config['results_path'], 
                                mod_fpath,'%s_%s_excitations%s.csv' % (
                                    mod, cond_name, suffix)),
                            os.path.join(study.config['results_path'], 
                                mod_fpath,'%s_%s_activations%s.csv' % (
                                    mod, cond_name, suffix)),
                            ],
                        self.aggregate_muscle_activity, cond_name, self.cycles)

    def aggregate_muscle_activity(self, file_dep, target, cond_name, cycles):

        num_time_points = 400
        pgc = np.linspace(0, 100, num_time_points)

        muscle_names = None

        subject_array = list()
        cycle_array = list()
        muscle_array = list()
        all_exc = list()
        all_act = list()
        for icycle, fpath in enumerate(file_dep):
            cycle = cycles[cond_name][icycle]

            muscle_names = util.hdf2list(fpath, 'MuscleNames', type=str)
            exc_df = util.hdf2pandas(fpath,
                'MExcitation', labels=muscle_names)
            act_df = util.hdf2pandas(fpath,
                'MActivation', labels=muscle_names)

            exc_index = np.linspace(0, 100, len(exc_df.index.values))
            act_index = np.linspace(0, 100, len(act_df.index.values))
            for muscle in exc_df.columns:
                subject_array.append(cycle.subject.name)
                cycle_array.append(cycle.id)
                muscle_array.append(muscle)
                exc = np.interp(pgc, exc_index, exc_df[muscle])
                act = np.interp(pgc, act_index, act_df[muscle])
                all_exc.append(exc)
                all_act.append(act)

        all_exc_array = np.array(all_exc).transpose()
        all_act_array = np.array(all_act).transpose()

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
        # How to read this in: df.read_csv(..., index_col=0, header=[0, 1, 2],
        #                                  skiprows=1)

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

class TaskPlotMuscleActivity(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, agg_task, conditions=['walk2'], mod=None, 
            subjects=None, suffix=''):
        super(TaskPlotMuscleActivity, self).__init__(study)
        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'plot_muscle_activity%s' % suffix
        self.doc = 'Plot muscle activity for experiment and mod tasks'

        for icond, agg_target in enumerate(agg_task.targets):
            # This assumes csv_task.targets and csv_task.cycles hold cycles in
            # the same order.
            # self.agg_target = agg_target
            # self.actions += [self.plot_muscle_activity]
            # print agg_target
            self.add_action([],[],
                            self.plot_muscle_activity,
                            agg_target)

    def plot_muscle_activity(self, file_dep, target, agg_target):

        df_all = pd.read_csv(agg_target, index_col=0,
                header=[0, 1, 2], skiprows=1)

        # Average over cycles.
        df_by_subj_musc = df_all.groupby(
                level=['subject', 'muscle'], axis=1).mean()
        df_mean = df_by_subj_musc.groupby(level=['muscle'],
                axis=1).mean()
        df_std = df_by_subj_musc.groupby(level=['muscle'],
                axis=1).std()

        pgc = df_mean.index
        muscles = self.study.muscle_names
        fig = pl.figure(figsize=(8, 8))
        nice_act_names = {
                'glut_max2_r': 'glut. max.',
                'psoas_r': 'iliopsoas',
                'semimem_r': 'hamstrings',
                'rect_fem_r': 'rect. fem.',
                'bifemsh_r': 'bi. fem. s.h.',
                'vas_int_r': 'vasti',
                'med_gas_r': 'gastroc.',
                'soleus_r': 'soleus',
                'tib_ant_r': 'tib. ant.',
                }

        for imusc, musc_name in enumerate(muscles):
            side_len = np.ceil(np.sqrt(len(muscles)))
            ax = fig.add_subplot(side_len, side_len, imusc + 1)
            ax.axhline(color='k', linewidth=0.5, zorder=0)
            y_mean = df_mean[musc_name]
            y_std = df_std[musc_name]
            ax.plot(pgc, y_mean, color='blue', linestyle='-')
            ax.fill_between(pgc, y_mean-y_std, y_mean+y_std,
                    color='blue', alpha=0.3)
            # ax.legend(frameon=False, fontsize=6)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.0)
            ax.set_title(nice_act_names[musc_name])
            ax.set_xlabel('time (% gait cycle)')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        fig.tight_layout()
        fig.savefig(agg_target.replace('.csv', '.pdf'))
        fig.savefig(agg_target.replace('.csv', '.png'), dpi=600)
        pl.close(fig)

class TaskValidateAgainstEMG(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, conditions=['walk2']):
        super(TaskValidateAgainstEMG, self).__init__(study)
        suffix = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'validate_against_emg%s' % suffix
        self.doc = 'Plot muscle activity from simulation against EMG data.'
        self.subjects = ['subject01', 'subject02', 'subject04', 'subject18',
                         'subject19']
        self.results_path = study.config['results_path']
        self.validate_path = study.config['validate_path']

        for cond in conditions:
            for subject in self.subjects:
                emg_fpath = os.path.join(self.results_path, 'experiments',
                    subject, cond, 'expdata', 'processed_emg.csv')
                exc_fpath = os.path.join(self.results_path, 'experiments',
                    'experiment_%s_excitations%s.csv' % (cond, suffix))
                act_fpath = os.path.join(self.results_path, 'experiments',
                    'experiment_%s_activations%s.csv' % (cond, suffix))

                val_fname = os.path.join(self.validate_path, 
                    '%s_%s_emg_validation%s' % (subject, cond, suffix))
                
                self.add_action([emg_fpath, exc_fpath, act_fpath],
                                [val_fname],
                                self.validate_against_emg,
                                cond, subject)

    def validate_against_emg(self, file_dep, target, cond, subject):

        df_emg = pd.read_csv(file_dep[0], index_col=0, header=[0, 1], 
            skiprows=1)
        df_emg_mean = df_emg.groupby(level=['muscle'], axis=1).mean()
        df_emg_std = df_emg.groupby(level=['muscle'], axis=1).std()

        df_exc_all = pd.read_csv(file_dep[1], index_col=0, header=[0, 1, 2], 
            skiprows=1)
        df_exc = df_exc_all[subject]
        df_exc_mean = df_exc.groupby(level=['muscle'], axis=1).mean()
        df_exc_std = df_exc.groupby(level=['muscle'], axis=1).std()

        df_act_all = pd.read_csv(file_dep[2], index_col=0, header=[0, 1, 2], 
            skiprows=1)
        df_act = df_act_all[subject]
        df_act_mean = df_act.groupby(level=['muscle'], axis=1).mean()
        df_act_std = df_act.groupby(level=['muscle'], axis=1).std()

        pgc_emg = df_emg_mean.index
        pgc_exc = df_exc_mean.index
        pgc_act = df_act_mean.index
        muscles = self.study.muscle_names
        fig = pl.figure(figsize=(12, 12))
        nice_act_names = {
                'glut_max2_r': 'glut. max.',
                'psoas_r': 'iliopsoas',
                'semimem_r': 'hamstrings',
                'rect_fem_r': 'rect. fem.',
                'bifemsh_r': 'bi. fem. s.h.',
                'vas_int_r': 'vasti',
                'med_gas_r': 'gastroc.',
                'soleus_r': 'soleus',
                'tib_ant_r': 'tib. ant.',
                }

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
                y_exc_mean = df_exc_mean[emg_map[emg_name]]
                y_exc_std = df_exc_std[emg_map[emg_name]]
                y_act_mean = df_act_mean[emg_map[emg_name]]
                y_act_std = df_act_std[emg_map[emg_name]]
                exc_plot, = ax.plot(pgc_exc, y_exc_mean, color='blue', 
                    linestyle='--')
                ax.fill_between(pgc_exc, y_exc_mean-y_exc_std, 
                    y_exc_mean+y_exc_std, color='blue', alpha=0.25)
                act_plot, = ax.plot(pgc_act, y_act_mean, color='red', 
                    linestyle='--')
                ax.fill_between(pgc_act, y_act_mean-y_act_std, 
                    y_act_mean+y_act_std, color='red', alpha=0.25)
                handles = [exc_plot, act_plot   ]
                labels = ['%s exc.' % nice_act_names[emg_map[emg_name]],
                          '%s act.' % nice_act_names[emg_map[emg_name]]]
                ax.legend(handles, labels)

            # ax.legend(frameon=False, fontsize=6)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.0)
            ax.set_title(emg_name)
            ax.set_xlabel('time (% gait cycle)')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        fig.tight_layout()
        fig.savefig(target[0]+'.pdf')
        fig.savefig(target[0]+'.png', dpi=600)
        pl.close(fig)

class TaskPlotHipFlexAnklePFMomentComparison(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, mod_agg_tasks, conditions=['walk2'],
            subjects=None):
        super(TaskPlotHipFlexAnklePFMomentComparison, self).__init__(study)
        self.name = 'plot_HfAp_moment_comparison'
        self.doc = 'Plot to compare assistive moments across HfAp devices.'
        self.mod_agg_tasks = mod_agg_tasks
        self.actions += [self.plot_joint_moment_breakdown]

        self.actHfAp_pdf_path = os.path.join(study.config['analysis_path'], 
            'actHfAp', 'device_moments_comparison_actHfAp.pdf')
        self.actHfAp_png_path = os.path.join(study.config['analysis_path'], 
            'actHfAp', 'device_moments_comparison_actHfAp.png')


    def plot_joint_moment_breakdown(self):

        def add_subplot_axes(ax,rect,axisbg='w'):
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            box = ax.get_position()
            width = box.width
            height = box.height
            inax_position  = ax.transAxes.transform(rect[0:2])
            transFigure = fig.transFigure.inverted()
            infig_position = transFigure.transform(inax_position)    
            x = infig_position[0]
            y = infig_position[1]
            width *= rect[2]
            height *= rect[3]  # <= Typo was here
            subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
            x_labelsize = subax.get_xticklabels()[0].get_size()
            y_labelsize = subax.get_yticklabels()[0].get_size()
            x_labelsize *= rect[2]**0.5
            y_labelsize *= rect[3]**0.5
            subax.xaxis.set_tick_params(labelsize=x_labelsize)
            subax.yaxis.set_tick_params(labelsize=y_labelsize)
            return subax

        fig = pl.figure(figsize=(6*1.2, 2.50*1.4))
        dof_names = ['hip_flexion_r', 'ankle_angle_r']
        ylabels = ['hip extension', 'ankle plantarflexion']
        for idof, dof_name in enumerate(dof_names):

            from matplotlib import gridspec
            gs = gridspec.GridSpec(2, 2, height_ratios=[6, 1]) 
            ax = fig.add_subplot(gs[idof])
            # ax = fig.add_subplot(2, len(dof_names), idof + 1)
            ax.axhline(color='k', linewidth=0.5, zorder=0)

            for iagg, agg_task in enumerate(self.mod_agg_tasks):
                for icond, agg_target in enumerate(agg_task.targets):

                    df_all = pd.read_csv(agg_target, index_col=0,
                            header=[0, 1, 2, 3], skiprows=1)
                    # Average over cycles.
                    # axis=1 for columns (not rows).
                    df_by_subj_dof_musc = df_all.groupby(
                            level=['subject', 'DOF', 'actuator'], axis=1).mean()
                    df_mean = df_by_subj_dof_musc.groupby(
                        level=['DOF', 'actuator'], axis=1).mean()
                    df_std = df_by_subj_dof_musc.groupby(
                        level=['DOF', 'actuator'], axis=1).std()
                    pgc = df_mean.index


                    def plot(column_key, act_name):
                        if column_key in df_mean.columns:
                            y_mean = -df_mean[column_key]
                            y_std = -df_std[column_key]
                            if act_name == 'net':
                                ax.plot(pgc, y_mean, color='black',
                                    label='net joint moment')
                            else:
                                if act_name == 'actHfAp_scaledID':
                                    label = 'original underactuated (8.0% avg. metabolic ' \
                                            'reduction)'
                                    color = 'blue'
                                elif act_name == 'actHfAp': 
                                    label = 'optimized underactuated ' \
                                            '(16.4%)'
                                    color = 'red'
                                elif act_name == 'actHfAp_multControls':
                                    label = 'independently actuated (39.7%)'
                                    color = colors['green']

                                ax.plot(pgc, y_mean, color=color,
                                        label=label,
                                        linestyle='--')
                                ax.fill_between(pgc, y_mean-y_std, y_mean+y_std,
                                    color=color, alpha=0.3)

                    if iagg == 0:
                        plot((dof_name, 'net'), 'net')


                    column_key = (dof_name, 'active')
                    plot(column_key, agg_task.mod_name)

            if dof_name == 'hip_flexion_r':
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[1:], labels[1:], frameon=False, fontsize=7, 
                          loc="best",
                          title="assistive device torques")
                ax.get_legend().get_title().set_fontsize(8)
                # ax.get_legend().get_title().set_fontstyle('italic')
                ax.get_legend().get_title().set_fontweight('bold')

                ax.text(16, 0.5, 'net joint moment', fontweight='bold')

            if dof_name == 'ankle_angle_r':
                ax.text(55, 1.6, 'net joint moment', fontweight='bold')


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

        # from matplotlib import rc
        # rc('text', usetex=True)
        txt = r'$\bf{Fig. 1:}$ Device torques for each assistive strategy ' \
               '(dashed colored lines) and net joint moments (solid black ' \
               'lines).'
        fig.text(.5, .1, txt, ha='center')
        fig.tight_layout()
        fig.savefig(self.actHfAp_pdf_path)
        fig.savefig(self.actHfAp_png_path, ppi=150)
        pl.close(fig)

class TaskAggregateTorqueParameters(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, mods=None, conditions=['walk2'],
            subjects=None, suffix=''):
        super(TaskAggregateTorqueParameters, self).__init__(study)
        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'aggregate_torque_parameters%s' % suffix
        self.doc = 'Aggregate parameters for active control signals.'

         # Create new lists so original doesn't get changed by reference
        if mods == None:
            mods = list(study.mod_names)
        else:
            mods = list(mods)

        for mod in mods:
            for cond_name in conditions:
                file_dep = os.path.join(
                        self.study.config['results_path'],
                        'mrsmod_%s' % mod,  
                        'mod_%s_%s_moments%s.csv' % (mod, cond_name, suffix))
                target = os.path.join(
                        self.study.config['results_path'], 
                        'mrsmod_%s' % mod, 
                        'mod_%s_%s_parameters%s.csv' % (mod, cond_name, suffix))
                self.add_action([file_dep],
                                [target], 
                                self.aggregate_torque_parameters,
                                cond_name, mod)

    def aggregate_torque_parameters(self, file_dep, target, cond_name, mod):

        df = pd.read_csv(file_dep[0], index_col=0, header=[0, 1, 2, 3], 
            skiprows=1)

        muscle_names = None
        subject_array = list()
        cycle_array = list()
        dof_array = list()
        muscle_array = list()
        all_data = list()

        def calc_torque_parameters(pgc, torque):


            params = list()

            import operator
            peak_idx, peak_torque = max(enumerate(torque), 
                key=operator.itemgetter(1))

            # peak torque
            params.append(peak_torque) # N-m/kg

            # peak time
            peak_time = pgc[peak_idx]
            params.append(peak_time) # percent GC

            # rise time
            for i in np.arange(peak_idx, -1, -1):
                if torque[pgc[i]] <= 0.01:
                    rise_idx = i
                    break
                rise_idx = i

            rise_time = pgc[peak_idx] - pgc[rise_idx]
            params.append(rise_time) # percent GC

            # fall time
            for i in np.arange(peak_idx, len(torque), 1):
                if torque[pgc[i]] <= 0.01:
                    fall_idx = i
                    break
                fall_idx = i


            fall_time = pgc[fall_idx] - pgc[peak_idx]
            params.append(fall_time) # percent GC

            return params

        for col in df.columns:
            subject, cycle, dof, actuator = col
            if actuator == 'active':
                act_torque = df[subject][cycle][dof][actuator]

                if ((('He' in mod) and ('hip' in dof)) or 
                    (('Ke' in mod) and ('knee' in dof)) or
                    (('Ap' in mod) and ('ankle' in dof))):
                    act_torque = -act_torque

                params = calc_torque_parameters(df.index, act_torque)

                subject_array.append(subject)
                cycle_array.append(cycle)
                dof_array.append(dof)
                all_data.append(params)

        #  n_params x (n_subjects * n_cycles * n_dofs)  
        all_data_array = np.array(all_data).transpose()

        multiindex_arrays = [subject_array, cycle_array, dof_array]
        columns = pd.MultiIndex.from_arrays(multiindex_arrays,
            names=['subject', 'cycle', 'DOF'])

        params_idx = ['peak_torque', 'peak_time', 'rise_time', 'fall_time']
        all_data_df = pd.DataFrame(all_data_array, columns=columns, 
            index=params_idx)
        target_dir = os.path.dirname(target[0])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(target[0], 'w') as f:
            f.write('torque active control parameters in units (N-m/kg) for '
                    'peak_torque and (percent g.c.) for times .\n')
            all_data_df.to_csv(f)
        # How to read this in: df.read_csv(..., index_col=0, header=[0, 1, 2, 3],
        #                                  skiprows=1)

class TaskPlotTorqueParameters(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, mods=None, conditions=['walk2'],
            subjects=None, suffix=''):
        super(TaskPlotTorqueParameters, self).__init__(study)
        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'plot_torque_parameters%s' % suffix
        self.doc = 'Aggregate parameters for active control signals.'

         # Create new lists so original doesn't get changed by reference
        if mods == None:
            mods = list(study.mod_names)
        else:
            mods = list(mods)


        for mod in mods:
            for cond_name in conditions:
                file_dep = os.path.join(
                        self.study.config['results_path'],
                        'mrsmod_%s' % mod,  
                        'mod_%s_%s_parameters%s.csv' % (mod, cond_name, suffix))
                target0 = os.path.join(
                        self.study.config['results_path'], 
                        'mrsmod_%s' % mod, 
                        'mod_%s_%s_parameters%s.pdf' % (mod, cond_name, suffix))
                target1 = os.path.join(
                        self.study.config['results_path'], 
                        'mrsmod_%s' % mod, 
                        'mod_%s_%s_parameters%s.png' % (mod, cond_name, suffix))

                self.add_action([file_dep],
                                [target0, target1], 
                                self.plot_torque_parameters,
                                cond_name)

    def plot_torque_parameters(self, file_dep, target, cond_name):

        df = pd.read_csv(file_dep[0], index_col=0, header=[0, 1, 2], 
            skiprows=1)

        fig = pl.figure(figsize=(9, 3.75))

        # Get relevant DOFs
        col_labels = df.columns.values
        dof_labels = [label[2] for label in col_labels]
        dof_names = list(set(dof_labels))
        param_names = ['peak_torque', 'peak_time', 'rise_time', 'fall_time']
        for idof, dof_name in enumerate(dof_names):

            df_DOF = df.xs(dof_name, level='DOF', axis=1)
            peak_torque = df_DOF.loc['peak_torque']
            peak_time = df_DOF.loc['peak_time']
            rise_time = df_DOF.loc['rise_time']
            fall_time = df_DOF.loc['fall_time']

            # Normalize and concatenate data
            all_data = [peak_torque / max(peak_torque), 
                        peak_time / 100.0, 
                        rise_time / 100.0, 
                        fall_time / 100.0]

            ax = fig.add_subplot(1, len(dof_names), idof + 1)
            ax.boxplot(all_data)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks(np.arange(0.0, 1.1, 0.1))
            ax.set_title(dof_name)
            ax.set_xticklabels(param_names)

        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[1], dpi=600)
        pl.close(fig)