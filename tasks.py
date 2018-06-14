import os
import time

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
    def __init__(self, trial, param_dict, cost_dict, passive_precalibrate=False,
            **kwargs):
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

        self.param_dict = param_dict
        self.cost_dict = cost_dict
        self.passive_precalibrate = passive_precalibrate

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

        self.add_action([self.results_output_fpath],
                        [os.path.join(self.path, 'optimal_fiber_length.pdf'),
                         os.path.join(self.path, 'tendon_slack_length.pdf'),
                         os.path.join(self.path, 'pennation_angle.pdf'),
                         os.path.join(self.path, 'muscle_strain.pdf')],
                        self.plot_muscle_parameters)

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

        if lTs:
            param_barplot('Tendon Slack Length', lTs, target[1])

        if alf:
            param_barplot('Pennation Angle', alf, target[2])

        if e0:
            param_barplot('Muscle Strain', e0, target[3])

def construct_multiindex_tuples_for_subject(subject, conditions, 
    muscles=None, cycles_to_exclude=None):
    ''' Construct multiindex tuples and list of cycles for DataFrame indexing.
    '''
    
    multiindex_tuples = list()
    cycles = list()

    for cond_name in conditions:
        cond = subject.get_condition(cond_name)
        if not cond: continue
        # We know there is only one overground trial, but perhaps it
        # has not yet been added for this subject.
        assert len(cond.trials) <= 1
        if len(cond.trials) == 1:
            trial = cond.trials[0]
            for cycle in trial.cycles:
                if cycle.name in cycles_to_exclude: continue
                cycles.append(cycle)
                if not muscles:
                    multiindex_tuples.append((
                        cycle.condition.name,
                        # This must be the full ID, not just the cycle
                        # name, because 'cycle01' from subject 1 has
                        # nothing to do with 'cycle01' from subject 2
                        # (whereas the 'walk2' condition for subject 1 is
                        # related to 'walk2' for subject 2).
                        cycle.id))
                if muscles:
                    for mname in muscles:
                        multiindex_tuples.append((
                            cycle.condition.name,
                            cycle.id,
                            mname))

    return multiindex_tuples, cycles

class TaskAggregateMuscleParameters(osp.SubjectTask):
    """Aggregate calibrated muscle parameters for a given subject."""
    REGISTRY = []
    def __init__(self, subject, param_dict, 
            conditions=['walk1','walk2','walk3'], 
            cycles_to_exclude=['cycle03']):
        super(TaskAggregateMuscleParameters, self).__init__(subject)
        self.name = '%s_aggregate_muscle_parameters' % self.subject.name
        self.csv_fpaths = list()
        self.csv_params = list()
        self.param_dict = param_dict
        self.conditions = conditions
        self.cycles_to_exclude = cycles_to_exclude
        
        for param in param_dict:
            muscle_names = param_dict[param]
            multiindex_tuples, cycles = construct_multiindex_tuples_for_subject( 
                self.subject, conditions, muscle_names, self.cycles_to_exclude)

            # Prepare for processing simulations of experiments.
            deps = list()
            for cycle in cycles:
                if cycle.name in cycles_to_exclude: continue
                deps.append(os.path.join(
                        cycle.trial.results_exp_path, 'calibrate', cycle.name,
                        '%s_%s_calibrate.mat' % (self.study.name, cycle.id))
                        )

            csv_path = os.path.join(self.subject.results_exp_path, 
                '%s_agg.csv' % param)
            self.csv_params.append(param)
            self.csv_fpaths.append(csv_path)
            self.add_action(deps,
                    [csv_path],
                    self.aggregate_muscle_parameters, param, multiindex_tuples)

    def aggregate_muscle_parameters(self, file_dep, target, param,
            multiindex_tuples):
        from collections import OrderedDict
        muscle_params = OrderedDict()
        for ifile, fpath in enumerate(file_dep):
            lMo, lTs, alf, e0, musc_names = get_muscle_parameters_as_dict(fpath)
            if not param in muscle_params:
                muscle_params[param] = list()
            for musc in self.param_dict[param]:
                if param == 'optimal_fiber_length':
                    muscle_params[param].append(lMo[musc])
                elif param == 'tendon_slack_length':
                    muscle_params[param].append(lTs[musc])
                elif param == 'pennation_angle':
                    muscle_params[param].append(alf[musc])
                elif param == 'muscle_strain':
                    muscle_params[param].append(e0[musc])
       
        # http://pandas.pydata.org/pandas-docs/stable/advanced.html#advanced-hierarchical
        index = pd.MultiIndex.from_tuples(multiindex_tuples,
                names=['condition', 'cycle', 'muscle'])

        df = pd.DataFrame(muscle_params, index=index)

        target_dir = os.path.dirname(target[0])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(target[0], 'w') as f:
            f.write('# columns contain calibrated muscle parameters for ' 
                    '%s \n' % self.subject.name)
            df.to_csv(f)

class TaskPlotMuscleParameters(osp.SubjectTask):
    REGISTRY = []
    def __init__(self, subject, agg_task, cycles_to_exclude=None, **kwargs):
        super(TaskPlotMuscleParameters, self).__init__(subject)
        self.name = '%s_plot_muscle_parameters' % subject.name
        self.doc = 'Plot aggregated muscle parameters from calibration.'
        self.agg_task = agg_task
        self.cycles_to_exclude = cycles_to_exclude

        for csv_fpath, param in zip(agg_task.csv_fpaths, agg_task.csv_params):
            self.add_action([csv_fpath],
                            [os.path.join(self.subject.results_exp_path, param)],
                            self.plot_muscle_parameters, param)

    def plot_muscle_parameters(self, file_dep, target, param):

        # Process muscle parameters
        df = pd.read_csv(file_dep[0], index_col=[0, 1, 2], skiprows=1)
        if self.cycles_to_exclude:
            for cycle in self.cycles_to_exclude:
                for cond in self.agg_task.conditions:
                    cycle_id = '%s_%s_%s' % (self.subject.name, cond, cycle)
                    df.drop(cycle_id, level='cycle', inplace=True)

        df_mean = df.mean(level='muscle')
        df_std = df.std(level='muscle')

        def param_barplot(param_name, param_mean, param_std, musc_names, 
                output_fpath):

            fig = pl.figure(figsize=(11,6))
            ax = fig.add_subplot(1,1,1)
            pos = np.arange(len(musc_names))

            param_mean_to_plot = [(val - 1.0)*100 for val in param_mean]
            param_std_to_plot = [val*100 for val in param_std]
            bar = ax.bar(pos, param_mean_to_plot, yerr=param_std_to_plot,
                color='green')
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

        if param == 'optimal_fiber_length':
            lMo_mean = df_mean['optimal_fiber_length']
            lMo_std = df_std['optimal_fiber_length']
            param_barplot('Optimal Fiber Length', lMo_mean, lMo_std, 
                lMo_mean.index, target[0]+'.pdf')
            lMo_mean.to_csv(target[0]+'.csv')

        if param == 'tendon_slack_length':
            lTs_mean = df_mean['tendon_slack_length']
            lTs_std = df_std['tendon_slack_length']
            param_barplot('Tendon Slack Length', lTs_mean, lTs_std, 
                lTs_mean.index, target[0]+'.pdf')
            lTs_mean.to_csv(target[0]+'.csv')

        if param == 'pennation_angle':
            alf_mean = df_mean['pennation_angle']
            alf_std = df_std['pennation_angle']
            param_barplot('Pennation Angle', alf_mean, alf_std, 
                alf_mean.index, target[0]+'.pdf')
            alf_mean.to_csv(target[0]+'.csv')

        if param == 'muscle_strain':
            e0_mean = df_mean['muscle_strain']
            e0_std = df_std['muscle_strain']
            param_barplot('Muscle Strain', e0_mean, e0_std, 
                e0_mean.index, target[0]+'.pdf')
            e0_mean.to_csv(target[0]+'.csv')

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
                ] + calibrate_setup_task.kinematics_files \
                + calibrate_setup_task.kinetics_files \
                + calibrate_setup_task.emg_files

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

class TaskMRSDeGrooteGenericMTParamsSetup(osp.TaskMRSDeGrooteSetup):
    REGISTRY = []
    def __init__(self, trial, cost='Default', **kwargs):
        super(TaskMRSDeGrooteGenericMTParamsSetup, self).__init__(trial, 
            cost=cost, alt_tool_name='mrs_genericMTparams', **kwargs)

        self.scaled_generic_model = os.path.join(
            self.study.config['results_path'], 'experiments', 
            self.subject.name, '%s.osim' % self.subject.name)

        self.file_dep += [self.scaled_generic_model]

    def fill_setup_template(self, file_dep, target,
                            init_time=None, final_time=None):
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            # TODO should this be an RRA-adjusted model?
            content = content.replace('@MODEL@', os.path.relpath(
                self.scaled_generic_model, self.path))
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

        with open(target[0], 'w') as f:
            f.write(content)

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

    def fill_setup_template(self, file_dep, target, 
                            init_time=None, final_time=None):
        with open(self.setup_template_fpath) as ft:
            content = ft.read()

            if type(self.mrsflags) is list:
                list_of_flags = self.mrsflags 
            else:
             list_of_flags = self.mrsflags(self.cycle)

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

        if not ('fitopt' in self.mrsmod_task.mod_name):
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
        if (('act' in self.mrsmod_task.mod_name) or 
           ('exp' in self.mrsmod_task.mod_name)):
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
        if (('act' in self.mrsmod_task.mod_name) or 
           ('exp' in self.mrsmod_task.mod_name)):
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

class TaskFitOptimizedExoSetup(osp.SetupTask):
    REGISTRY = []
    def __init__(self, trial, mrsmod_task, param_info, fit, **kwargs):
        super(TaskFitOptimizedExoSetup, self).__init__('fitopt', trial, **kwargs)
        self.doc = 'Setup optimized exoskeleton torque fitting.'
        self.mrsmod_task = mrsmod_task
        self.fit = fit
        self.name = '%s_%s_%s_%s_setup_%s' % (trial.id, self.tool, self.fit,
                self.mrsmod_task.mod_name.replace('mrsmod_',''), self.cycle.name)
        if not (self.mrsmod_task.cost == 'Default'):
            self.name += '_%s' % self.mrsmod_task.cost
        self.min_param = param_info['min_param']
        self.max_param = param_info['max_param']
        self.start_time = 0
        if hasattr(self.tricycle, 'fit_start_time'):
            self.start_time = self.tricycle.fit_start_time

        self.mrs_setup_task = self.mrsmod_task.mrs_setup_task
        self.path = os.path.join(self.study.config['results_path'],
            'fitopt_%s_%s' % (self.fit, 
                self.mrsmod_task.mod_name.replace('mrsmod_','')), 
            'fitopt', trial.rel_path,
            self.mrs_setup_task.cycle.name if self.mrs_setup_task.cycle else '', 
            self.mrsmod_task.costdir)
        self.mrsmod_output_fpath = mrsmod_task.results_output_fpath
        self.results_setup_fpath = os.path.join(self.path, 'setup.m')
        self.results_output_fpath = os.path.join(self.path, '%s_%s_fitopt_%s.mat' % 
            (self.study.name, self.tricycle.id, self.fit))

        # Fill out setup.m template and write to results directory
        self.create_setup_action()

    def create_setup_action(self): 
        self.add_action(
                    ['templates/%s/setup.m' % self.tool],
                    [self.results_setup_fpath],
                    self.fill_setup_template,      
                    )

    def fill_setup_template(self, file_dep, target):
        self.add_setup_dir()
        with open(file_dep[0]) as ft:
            content = ft.read()
            content = content.replace('@STUDYNAME@', self.study.name)
            content = content.replace('@NAME@', self.tricycle.id)
            content = content.replace('@REL_PATH_TO_TOOL@', os.path.relpath(
                self.study.config['optctrlmuscle_path'], self.path))
            content = content.replace('@FIT@', self.fit)
            content = content.replace('@MIN_PARAM@', str(self.min_param))
            content = content.replace('@MAX_PARAM@', str(self.max_param))
            content = content.replace('@START_TIME@', str(self.start_time))
            content = content.replace('@MRSMOD_OUTPUT@', 
                self.mrsmod_output_fpath)

        with open(target[0], 'w') as f:
            f.write(content)

    def add_setup_dir(self):
        if not os.path.exists(self.path): os.makedirs(self.path)

class TaskFitOptimizedExo(osp.ToolTask):
    REGISTRY = []
    def __init__(self, trial, fitopt_setup_task, **kwargs):
        super(TaskFitOptimizedExo, self).__init__(fitopt_setup_task, trial, 
            opensim=False, **kwargs)
        self.doc = 'Fit parameterized curve optimized exoskeleton torque curve.'
        self.name = fitopt_setup_task.name.replace('_setup','')
        self.results_setup_fpath = fitopt_setup_task.results_setup_fpath
        self.results_output_fpath = fitopt_setup_task.results_output_fpath

        self.file_dep += [self.results_setup_fpath] 
        self.actions += [self.run_fitting_script]
        self.targets += [self.results_output_fpath]

    def run_fitting_script(self):
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

class TaskMRSFitOptimizedExoSetup(osp.SetupTask):
    REGISTRY = []
    def __init__(self, trial, fitopt_setup_task, mod_name, param_num, mrsflags, 
            **kwargs):
        super(TaskMRSFitOptimizedExoSetup, self).__init__('fitopt_mrs',trial, 
            **kwargs)
        self.mod_name = mod_name
        self.param_num = param_num
        self.mrsflags = mrsflags
        self.fit = fitopt_setup_task.fit
        self.min_param = fitopt_setup_task.min_param
        self.max_param = fitopt_setup_task.max_param
        if (param_num > self.max_param) or (param_num < self.min_param):
            Exception('Parameterized fit not generated for this parameter'
                      'number')
        self.mrsmod_task = fitopt_setup_task.mrsmod_task
        self.fit_output_fpath = fitopt_setup_task.results_output_fpath
        self.mrs_setup_task = self.mrsmod_task.mrs_setup_task
        self.cost = self.mrsmod_task.cost
        self.param_dict = self.mrs_setup_task.param_dict
        self.name = '%s_%s_%s_%s_%s_setup_%s' % (trial.id, self.tool, self.fit,
               str(self.param_num), 
               self.mrsmod_task.mod_name.replace('mrsmod_',''), self.cycle.name)
        if not (self.mrsmod_task.cost == 'Default'):
            self.name += '_%s' % self.cost
        self.doc = """ Create a setup file for the DeGroote Muscle Redundancy 
                       Solver tool, where a curve is prescribed based on a fit
                       to a previous optimized solution. """

        self.path = os.path.join(self.study.config['results_path'],
            'fitopt_%s_%s' % (self.fit, 
                self.mrsmod_task.mod_name.replace('mrsmod_','')), 
            'params_%s' % str(self.param_num), trial.rel_path, 'mrs',
            self.mrs_setup_task.cycle.name if self.mrs_setup_task.cycle else '', 
            self.mrsmod_task.costdir)
        self.kinematics_file = os.path.join(self.trial.results_exp_path, 'ik',
                '%s_%s_ik_solution.mot' % (self.study.name, self.trial.id))
        self.rel_kinematics_file = os.path.relpath(self.kinematics_file,
                self.path)
        self.kinetics_file = os.path.join(self.trial.results_exp_path,
                'id', 'results', '%s_%s_id_solution.sto' % (self.study.name,
                self.trial.id))
        self.rel_kinetics_file = os.path.relpath(self.kinetics_file,
                self.path)
        self.results_setup_fpath = os.path.join(self.path, 'setup.m')
        self.results_output_fpath = os.path.join(self.path, 
                '%s_%s_mrs.mat' % (self.study.name, self.tricycle.id))

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

        self.file_dep += [
            self.kinematics_file,
            self.kinetics_file
        ]

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
        self.add_setup_dir()
        with open(file_dep[0]) as ft:
            content = ft.read()

            if type(self.mrsflags) is list:
                list_of_flags = self.mrsflags 
            else:
                list_of_flags = self.mrsflags(self.cycle)

            # Insert flags for the mod.
            flagstr = ''
            for flag in list_of_flags:
                flagstr += 'Misc.%s;\n' % flag

            possible_params = ['optimal_fiber_length', 'tendon_slack_length',
                               'pennation_angle', 'muscle_strain']
            paramstr = ''
            for param in possible_params:
                if param in self.param_dict:
                    paramstr += param + ' = true;\n'
                else:
                    paramstr += param + ' = false;\n'

            content = content.replace('Misc = struct();',
                'Misc = struct();\n' + flagstr + paramstr + '\n')

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

            content = content.replace('@PARAM_NUM@', str(self.param_num))
            content = content.replace('@FIT@', self.fit)
            content = content.replace('@FIT_PATH@', self.fit_output_fpath)

        with open(target[0], 'w') as f:
            f.write(content)

    def add_setup_dir(self):
        if not os.path.exists(self.path): os.makedirs(self.path)

class TaskMRSFitOptimizedExo(osp.ToolTask):
    REGISTRY = []
    def __init__(self, trial, fitopt_mrs_setup_task, **kwargs):
        super(TaskMRSFitOptimizedExo, self).__init__(fitopt_mrs_setup_task, 
            trial, opensim=False, **kwargs)
        self.doc = """ Run the DeGroote Muscle Redundancy Solver tool, where a
                       curve is prescribed based on a fit to a previous 
                       optimized solution. """
        self.name = fitopt_mrs_setup_task.name.replace('_setup','')
        self.results_setup_fpath = fitopt_mrs_setup_task.results_setup_fpath
        self.results_output_fpath = fitopt_mrs_setup_task.results_output_fpath

        self.file_dep += [self.results_setup_fpath] 
        self.actions += [self.run_muscle_redundancy_solver,
                         self.delete_muscle_analysis_results]
        self.targets += [self.results_output_fpath]

    def run_muscle_redundancy_solver(self):
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
            while True:
                time.sleep(3.0)

                mat_exists = os.path.isfile(self.results_output_fpath)
                if mat_exists:
                    break

    def delete_muscle_analysis_results(self):
        if os.path.exists(os.path.join(self.path, 'results')):
            import shutil
            shutil.rmtree(os.path.join(self.path, 'results'))

class TaskMRSFitOptimizedExoPost(TaskMRSDeGrooteModPost):
    REGISTRY = []
    def __init__(self, trial, fitopt_mrs_setup_task, **kwargs):
        super(TaskMRSFitOptimizedExoPost, self).__init__(trial, 
            fitopt_mrs_setup_task, **kwargs)
        self.doc = """ Plot results from the DeGroote Muscle Redundancy Solver 
                       tool, where a curve is prescribed based on a fit to a  
                       previous optimized solution. """
        self.name = fitopt_mrs_setup_task.name.replace('_setup','_post')

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
            conditions=['walk2'], suffix=''):
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

class TaskAggregateDevicePower(osp.StudyTask):
    """Aggregate peak instantaneous and average device power for assisted cases 
    across all subjects and gait cycles for each provided condition."""
    REGISTRY = []
    def __init__(self, study, mods, subjects=None, 
            conditions=['walk2'], suffix=''):
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
            df_Texo = util.hdf2pandas(fpath, 'DatStore/ExoTorques_Act', 
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
    def __init__(self, study, subjects=None, conditions=['walk2']):
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
    def __init__(self, study, mod, conditions=['walk2'], subjects=None):
        super(TaskAggregateMomentsMod, self).__init__(study)
        self.mod_dir = mod.replace('/','\\\\')
        if len(mod.split('/')) > 1:
            self.mod_name = '_'.join(mod.split('/'))
        else:
            self.mod_name = mod

        self.name = 'aggregate_moments_%s' % self.mod_name
        self.doc = 'Aggregate actuator moments into a data file.'
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
                        '%s_%s_moments%s.csv' % (self.mod_name,
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
                   'bifemsh_r', 'vas_int_r', 'med_gas_r', 'soleus_r', 
                   'tib_ant_r']
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
    all_pce = list() # normalized active fiber power
    for icycle, fpath in enumerate(file_dep):
        cycle = cycles[cond_name][icycle]

        muscle_names = util.hdf2list(fpath, 'MuscleNames', type=str)
        exc_df = util.hdf2pandas(fpath, 'MExcitation', labels=muscle_names)
        act_df = util.hdf2pandas(fpath, 'MActivation', labels=muscle_names)
        fce_df = util.hdf2pandas(fpath, 'MuscleData/fce', labels=muscle_names)
        fpe_df = util.hdf2pandas(fpath, 'MuscleData/fpe', labels=muscle_names)
        lMtilde_df = util.hdf2pandas(fpath, 'MuscleData/lMtilde',
            labels=muscle_names)
        # Negate fiber velocities so:
        #   positive <==> shortening
        #   negative <==> lengthening
        # This matches the convention in Arnold et al. 2013
        vMtilde_df = -util.hdf2pandas(fpath, 'MuscleData/vMtilde',
            labels=muscle_names)
        pce_df = fce_df.multiply(vMtilde_df)

        exc_index = np.linspace(0, 100, len(exc_df.index.values))
        act_index = np.linspace(0, 100, len(act_df.index.values))
        fce_index = np.linspace(0, 100, len(fce_df.index.values))
        fpe_index = np.linspace(0, 100, len(fpe_df.index.values))
        lMtilde_index = np.linspace(0, 100, len(lMtilde_df.index.values))
        vMtilde_index = np.linspace(0, 100, len(vMtilde_df.index.values))
        pce_index = np.linspace(0, 100, len(pce_df.index.values))
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
            pce = np.interp(pgc, pce_index, pce_df[muscle])

            all_exc.append(exc)
            all_act.append(act)
            all_fce.append(fce)
            all_fpe.append(fpe)
            all_lMtilde.append(lMtilde)
            all_vMtilde.append(vMtilde)
            all_pce.append(pce)

    all_exc_array = np.array(all_exc).transpose()
    all_act_array = np.array(all_act).transpose()
    all_fce_array = np.array(all_fce).transpose()
    all_fpe_array = np.array(all_fpe).transpose()
    all_lMtilde_array = np.array(all_lMtilde).transpose()
    all_vMtilde_array = np.array(all_vMtilde).transpose()
    all_pce_array = np.array(all_pce).transpose()

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
    # How to read this in: df.read_csv(..., index_col=0, header=[0, 1, 2],
    #                                  skiprows=1)

class TaskAggregateMuscleDataExperiment(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, subjects=None, conditions=['walk2'], 
            suffix='', alt_tool_name=None):
        super(TaskAggregateMuscleDataExperiment, self).__init__(study)

        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix

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
        for cond_name in conditions:
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
                        if study.cycles_to_plot:
                            if not (cycle.name in study.cycles_to_plot): 
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
                    ],
                    aggregate_muscle_data, cond_name, self.cycles)

class TaskAggregateMuscleDataMod(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, mods, subjects=None, conditions=['walk2'], 
            suffix='', alt_tool_name=None):
        super(TaskAggregateMuscleDataMod, self).__init__(study)
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
        for mod_name, mod_dir in zip(self.mod_names, self.mod_dirs):
            for cond_name in conditions:
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
                            if study.cycles_to_plot:
                                if not (cycle.name in study.cycles_to_plot): 
                                    continue
                            self.cycles[cond_name].append(cycle)

                            # Results MAT file paths
                            fpath = os.path.join(study.config['results_path'], 
                                mod_dir, subject.name, cond.name, tool, 
                                cycle.name, self.costdir,
                                '%s_%s_mrs.mat' % (study.name, cycle.id))
                            deps.append(fpath)

                self.add_action(deps,
                        [os.path.join(study.config['results_path'], 
                            mod_dir,'%s%s_%s_excitations%s.csv' % (
                                mod_name, alt_tool_tag, cond_name, cost)),
                        os.path.join(study.config['results_path'], 
                            mod_dir,'%s%s_%s_activations%s.csv' % (
                                mod_name, alt_tool_tag, cond_name, cost)),
                        os.path.join(study.config['results_path'], 
                            mod_dir,'%s%s_%s_norm_act_fiber_forces%s.csv' % (
                                mod_name, alt_tool_tag, cond_name, cost)),
                        os.path.join(study.config['results_path'], 
                            mod_dir,'%s%s_%s_norm_pass_fiber_forces%s.csv' % (
                                mod_name, alt_tool_tag, cond_name, cost)),
                        os.path.join(study.config['results_path'], 
                            mod_dir,'%s%s_%s_norm_fiber_lengths%s.csv' % (
                                mod_name, alt_tool_tag, cond_name, cost)),
                        os.path.join(study.config['results_path'], 
                            mod_dir,'%s%s_%s_norm_fiber_velocities%s.csv' % (
                                mod_name, alt_tool_tag, cond_name, cost)),
                        os.path.join(study.config['results_path'], 
                            mod_dir,'%s%s_%s_norm_fiber_powers%s.csv' % (
                                mod_name, alt_tool_tag, cond_name, cost)),
                        ],
                        aggregate_muscle_data, cond_name, self.cycles)

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

class TaskPlotMuscleData(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, agg_task, conditions=['walk2'], suffix=''):
        super(TaskPlotMuscleData, self).__init__(study)
        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix
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
            if 'norm_fiber_lengths' in agg_target:
                ax.set_ylim(0, 2.0)
                ax.set_yticks(np.linspace(0, 2, 9))
                ax.axhline(y=0.5, color='k', linewidth=0.5, ls='--', zorder=0)
                ax.axhline(y=1.0, color='k', linewidth=0.5, ls='--', zorder=0)
                ax.axhline(y=1.5, color='k', linewidth=0.5, ls='--', zorder=0)
            elif 'norm_fiber_velocities' in agg_target:
                ax.set_ylim(-0.5, 0.5)
                ax.set_yticks(np.linspace(-0.5, 0.5, 5))
                ax.axhline(y=-0.25, color='k', linewidth=0.5, ls='--', zorder=0)
                ax.axhline(y=0.25, color='k', linewidth=0.5, ls='--', zorder=0)
            else:
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

class TaskPlotDeviceComparison(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, plot_lists, folder, condition='walk2', 
            subjects=None):
        super(TaskPlotDeviceComparison, self).__init__(study)
        self.name = 'plot_device_comparison_%s' % folder
        self.doc = 'Plot to compare assistive moments across devices.'
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
        self.folder = folder
        self.study_subjects = [s.name for s in self.study.subjects]

        if ('He' in folder) or ('Hf' in folder):
            self.legend_dof = 'hip_flexion_r'
        elif ('Ke' in folder) or ('Kf' in folder):
            self.legend_dof = 'knee_angle_r'
        else:
            self.legend_dof = 'ankle_angle_r'

        self.dof_names = list()
        self.dof_labels = list()
        if ('He' in folder) or ('Hf' in folder):
            self.dof_names.append('hip_flexion_r')
            self.dof_labels.append('hip extension')
        if ('Ke' in folder) or ('Kf' in folder):
            self.dof_names.append('knee_angle_r')
            self.dof_labels.append('knee extension')
        if ('Ap' in folder) or ('Ad' in folder):
            self.dof_names.append('ankle_angle_r')
            self.dof_labels.append('ankle plantarflexion')

        if condition:
            cond = '_' + condition
        else:
            cond = ''

        if study.costFunction:
            if not (study.costFunction == 'Default'):
                cost = '_' + study.costFunction
            else:
                cost = ''
        else:
            cost = ''

        if subjects == None:
            self.subjects = [s.name for s in study.subjects]
        else:
            self.subjects = subjects

        output_path = os.path.join(study.config['analysis_path'], folder)
        self.add_action([os.path.join(study.config['analysis_path'], folder, 
                'whole_body_metabolic_rates_%s%s.csv' % (folder, cost)),
            os.path.join(study.config['analysis_path'], folder, 
                'peak_power_%s%s.csv' % (folder, cost)),
            os.path.join(study.config['analysis_path'], folder, 
                'avg_pos_power_%s%s.csv' % (folder, cost))], 
            [os.path.join(output_path,'%s_metabolics_per_reduction' % folder),
             os.path.join(output_path,'%s_metabolics_peak_power_norm' % folder),
             os.path.join(output_path,'%s_metabolics_avg_power_norm' % folder)],
            self.plot_metabolics)

        moment_deps = list()
        for device_name, device_dir in zip(self.device_names, self.device_dirs):
            fname = '%s%s_moments%s.csv' % (device_name, cond, cost)
            moment_deps.append(
                os.path.join(study.config['results_path'], device_dir, fname))
        self.add_action(moment_deps, 
                        [output_path], 
                        self.plot_moments)

        musc_data_names = ['activations', 'norm_act_fiber_forces', 
                           'norm_fiber_lengths', 'norm_fiber_velocities',
                           'norm_pass_fiber_forces', 'norm_fiber_powers']
        for mdname in  musc_data_names:
            md_deps = list()
            fname ='experiment%s_%s%s.csv' % (cond, mdname, cost)
            md_deps.append(os.path.join(study.config['results_path'], 
                'experiments', fname))
            for device_name, device_dir in zip(self.device_names, 
                    self.device_dirs):
                fname = '%s%s_%s%s.csv' % (device_name, cond, mdname, cost)
                md_deps.append(os.path.join(study.config['results_path'], 
                    device_dir, fname))
            self.add_action(md_deps,
                           [os.path.join(output_path, 
                                '%s_%s' % (folder, mdname))],
                            self.plot_muscle_data)

    def plot_moments(self, file_dep, target):

        # import packages and define subroutines for plotting
        from matplotlib import gridspec
        def plot_dof(ax, df, dof, actuator, label, color):
            # Average over cycles.
            # axis=1 for columns (not rows).
            for subj in self.study_subjects:
                if not (subj in self.subjects):
                    df.drop(subj, axis='columns', inplace=True)

            df_by_subj_dof_musc = df.groupby(
                    level=['subject', 'DOF', 'actuator'], axis=1).mean()
            df_mean = df_by_subj_dof_musc.groupby(
                level=['DOF', 'actuator'], axis=1).mean()
            df_std = df_by_subj_dof_musc.groupby(
                level=['DOF', 'actuator'], axis=1).std()
            pgc = df_mean.index

            if (dof, actuator) in df_mean.columns:
                y_mean = -df_mean[(dof, actuator)]
                y_std = -df_std[(dof, actuator)]
                if actuator == 'net':
                    ax.plot(pgc, y_mean, color=color, label=label, 
                        linestyle='-', linewidth=2)
                else:
                    ax.plot(pgc, y_mean, color=color, label=label, 
                        linestyle='--')
                    ax.fill_between(pgc, y_mean-y_std, y_mean+y_std, color=color, 
                        alpha=0.3)

        def set_axis(ax, idof, dof, dof_labels):
            if dof == 'hip_flexion_r':
                ax.set_xlim(0, 100)
                ax.set_ylim(-1.0, 1.0)
                ax.set_yticks(np.linspace(-1.0, 1.0, 5))
            elif dof == 'knee_angle_r':
                ax.set_xlim(0, 100)
                ax.set_ylim(-1.0, 1.0)
                ax.set_yticks(np.linspace(-1.0, 1.0, 5))
            elif dof == 'ankle_angle_r':
                ax.set_xlim(0, 100)
                ax.set_ylim(-0.5, 2.0)

            # if idof < len(self.dof_names)-1:
            #     ax.set_xticklabels([])
            ax.set_ylabel('%s (N-m/kg)' % dof_labels[idof])
            ax.set_xlabel('time (% gait cycle)')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if dof == self.legend_dof:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, frameon=False, fontsize=7, 
                    loc="best")
                ax.get_legend().get_title().set_fontsize(8)
                # ax.get_legend().get_title().set_fontstyle('italic')
                ax.get_legend().get_title().set_fontweight('bold')

        plot_iterate = zip(file_dep, self.device_names, self.label_list, 
            self.color_list)

        for not_first, (dep, device, label, color) in enumerate(plot_iterate):

            # Get dataframes
            df = pd.read_csv(dep, index_col=0,
                            header=[0, 1, 2, 3], skiprows=1)
            # Skip compare dataframe on first iteration, since they are the same
            if not_first:
                df_compare = pd.read_csv(file_dep[0], index_col=0,
                            header=[0, 1, 2, 3], skiprows=1)
                label_compare = self.label_list[0]
                color_compare = self.color_list[0]

            fig = pl.figure(figsize=(3.5, 3.5*len(self.dof_names)))
            handles = set()
            labels = set()
            for idof, dof in enumerate(self.dof_names):

                gs = gridspec.GridSpec(len(self.dof_names), 1) 
                ax = fig.add_subplot(gs[idof])
                # ax = fig.add_subplot(2, len(dof_names), idof + 1)
                ax.axhline(color='k', linewidth=0.5, zorder=0)
                plot_dof(ax, df, dof, 'net', 'net joint moment', 'black')
                plot_dof(ax, df, dof, 'active', label, color)
                if not_first:
                    plot_dof(ax, df_compare, dof, 'active', label_compare, 
                        color_compare)
                set_axis(ax, idof, dof, self.dof_labels)

            fig.tight_layout()
            fname = '%s_moments' % device
            fig.savefig(os.path.join(target[0], fname + '.pdf'))
            fig.savefig(os.path.join(target[0], fname + '.png'), ppi=600)
            pl.close(fig)


        fig = pl.figure(figsize=(3.5, 3.5*len(self.dof_names)))
        for idof, dof in enumerate(self.dof_names):
            gs = gridspec.GridSpec(len(self.dof_names), 1)
            ax = fig.add_subplot(gs[idof])
            ax.axhline(color='k', linewidth=0.5, zorder=0)
            for idep, (dep, device, label, color) in enumerate(plot_iterate):
                df = pd.read_csv(dep, index_col=0,
                            header=[0, 1, 2, 3], skiprows=1)
                if not idep:
                    plot_dof(ax, df, dof, 'net', 'net joint moment', 'black')
                plot_dof(ax, df, dof, 'active', label, color)

            set_axis(ax, idof, dof, self.dof_labels)

        fig.tight_layout()
        fname = '%s_all_moments' % self.folder
        fig.savefig(os.path.join(target[0], fname + '.pdf'))
        fig.savefig(os.path.join(target[0], fname + '.png'), ppi=600)
        pl.close(fig)

    def plot_metabolics(self, file_dep, target):

        # Plot percent reduction
        df_met = pd.read_csv(file_dep[0], index_col=[0, 1, 2], skiprows=1)
        for subj in self.study_subjects:
            if not (subj in self.subjects):
                df_met.drop(subj, axis='index', inplace=True)

        df_met_walk2 = df_met.xs('walk2', level='condition')
        df_met_change = df_met_walk2.subtract(df_met_walk2['experiment'],
                axis='index')
        df_met_relchange = df_met_change.divide(df_met_walk2['experiment'], 
                axis='index')
        df_met_relchange.drop('experiment', axis='columns', inplace=True)
        df_met_by_subjs = df_met_relchange.groupby(level='subject').mean()
        met_mean = df_met_by_subjs.mean()[self.device_names] * 100
        met_std = df_met_by_subjs.std()[self.device_names] * 100

        fig = pl.figure(figsize=(4, 6))
        ax = fig.add_subplot(1,1,1)
        ind = np.arange(len(self.device_names))

        barlist = ax.bar(ind, met_mean, yerr=met_std)
        for barelt, color in zip(barlist, self.color_list):
            barelt.set_color(color)

        ax.set_xticks(ind)
        ax.set_xticklabels(self.label_list, rotation=45)
        ax.set_ylabel('reduction in metabolic cost (%)')
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('top')

        fig.tight_layout()
        fig.savefig(target[0] + '.pdf')
        fig.savefig(target[0] + '.png', ppi=600)
        pl.close(fig)

        # Plot normalized absolute metabolic reduction by peak positive device 
        # power
        df_peak_power = pd.read_csv(file_dep[1], index_col=[0, 1, 2], skiprows=1)
        for subj in self.study_subjects:
            if not (subj in self.subjects):
                df_peak_power.drop(subj, axis='index', inplace=True)

        df_peak_power_walk2 = df_peak_power.xs('walk2', level='condition')
        df_eff = -df_met_change.divide(df_peak_power_walk2)
        df_eff_by_subjs = df_eff.groupby(level='subject').mean()
        eff_mean = df_eff_by_subjs.mean()[self.device_names]
        eff_std = df_eff_by_subjs.std()[self.device_names]

        fig = pl.figure(figsize=(4, 6))
        ax = fig.add_subplot(1,1,1)
        ind = np.arange(len(self.device_names))

        barlist = ax.bar(ind, eff_mean, yerr=eff_std)
        for barelt, color in zip(barlist, self.color_list):
            barelt.set_color(color)

        ax.set_xticks(ind)
        ax.set_xticklabels(self.label_list, rotation=45)
        ax.set_ylabel('device efficiency (met rate / peak device power)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')

        fig.tight_layout()
        fig.savefig(target[1] + '.pdf')
        fig.savefig(target[1] + '.png', ppi=600)
        pl.close(fig)

        # Plot normalized absolute metabolic reduction by peak positive device 
        # power
        df_avg_power = pd.read_csv(file_dep[2], index_col=[0, 1, 2], skiprows=1)
        for subj in self.study_subjects:
            if not (subj in self.subjects):
                df_avg_power.drop(subj, axis='index', inplace=True)

        df_avg_power_walk2 = df_avg_power.xs('walk2', level='condition')
        df_perfidx = -df_met_change.divide(df_avg_power_walk2)
        df_perfidx_by_subjs = df_perfidx.groupby(level='subject').mean()
        perfidx_mean = df_perfidx_by_subjs.mean()[self.device_names]
        perfidx_std = df_perfidx_by_subjs.std()[self.device_names]

        fig = pl.figure(figsize=(4, 6))
        ax = fig.add_subplot(1,1,1)
        ind = np.arange(len(self.device_names))

        barlist = ax.bar(ind, perfidx_mean, yerr=perfidx_std)
        for barelt, color in zip(barlist, self.color_list):
            barelt.set_color(color)

        ax.set_xticks(ind)
        ax.set_xticklabels(self.label_list, rotation=45)
        ax.set_ylabel('performance index (met rate / avg device power)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')

        fig.tight_layout()
        fig.savefig(target[2] + '.pdf')
        fig.savefig(target[2] + '.png', ppi=600)
        pl.close(fig)

    def plot_muscle_data(self, file_dep, target):

        device_names = ['experiment'] + self.device_names
        label_list = ['unassisted'] + self.label_list
        color_list = ['black'] + self.color_list
        plot_iterate = zip(file_dep, device_names, label_list, color_list)

        fig = pl.figure(figsize=(8, 8))
        muscles = self.study.muscle_names
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

                from scipy import signal
                b, a = signal.butter(4, 0.075)
                y_mean = signal.filtfilt(b, a, df_mean[musc_name])
                # y_mean = df_mean[musc_name]
                ax.plot(pgc, y_mean, label=label, color=color, linestyle='-')
                # ax.legend(frameon=False, fontsize=6)
                ax.set_xlim(0, 100)
                if 'norm_fiber_lengths' in dep:
                    ax.set_ylim(0, 2.0)
                    ax.set_yticks(np.linspace(0, 2, 9))
                    ax.axhline(y=0.5, color='k', linewidth=0.5, ls='--', 
                        zorder=0)
                    ax.axhline(y=1.0, color='k', linewidth=0.5, ls='--', 
                        zorder=0)
                    ax.axhline(y=1.5, color='k', linewidth=0.5, ls='--', 
                        zorder=0)
                elif ('norm_fiber_velocities' in dep):
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_yticks(np.linspace(-0.5, 0.5, 5))
                    ax.axhline(y=-0.25, color='k', linewidth=0.5, ls='--', 
                        zorder=0)
                    ax.axhline(y=0.25, color='k', linewidth=0.5, ls='--', 
                        zorder=0)
                elif 'norm_fiber_powers' in dep:
                    ax.set_ylim(-0.06, 0.06)
                    ax.set_yticks(np.linspace(-0.06, 0.06, 13))
                    ax.axhline(y=-0.03, color='k', linewidth=0.5, ls='--', 
                        zorder=0)
                    ax.axhline(y=0.03, color='k', linewidth=0.5, ls='--', 
                        zorder=0)
                else:
                    ax.set_ylim(0, 1.0)
                ax.set_title(nice_act_names[musc_name])
                ax.set_xlabel('time (% gait cycle)')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')

            if not imusc:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, frameon=False, fontsize=7, 
                    loc="best")
                ax.get_legend().get_title().set_fontsize(8)
                ax.get_legend().get_title().set_fontweight('bold')

        fig.tight_layout()
        fig.savefig(target[0] + '.pdf')
        fig.savefig(target[0] + '.png', dpi=600)
        pl.close(fig)

class TaskPlotMuscleActivityComparison(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, plot_lists, folder, condition='walk2', 
            subjects=None):
        super(TaskPlotMuscleDataComparison, self).__init__(study)
        self.name = 'plot_muscle_activity_comparison'
        self.doc = 'Plot to compare muscle activity solutions.'
        self.solution_list = plot_lists['solution_list']
        self.label_list = plot_lists['label_list']
        self.color_list = plot_lists['color_list']
        self.folder = folder
        self.study_subjects = [s.name for s in self.study.subjects]

        if condition:
            cond = '_' + condition
        else:
            cond = ''

        if study.costFunction:
            if not (study.costFunction == 'Default'):
                cost = '_' + study.costFunction
            else:
                cost = ''
        else:
            cost = ''

        if subjects == None:
            self.subjects = [s.name for s in study.subjects]
        else:
            self.subjects = subjects

        # EMG paths
        self.emg_fpaths = list()
        for subject in self.subjects:
            self.emg_fpaths.append(
                os.path.join(study.config['results_path'], 
                    'experiments', subject, 'walk2', 'expdata', 
                    'processed_emg.csv'))

        # actions
        output_path = os.path.join(study.config['analysis_path'], folder)
        self.add_action(self.solution_list,
                   [os.path.join(
                        output_path, 'muscle_activity')],
                    self.plot_muscle_activity)

        

    def plot_muscle_activity(self, file_dep, target):

        df_emg_list = list()
        iterate = zip(self.emg_fpaths, self.subjects)
        for iemg, (emg_fpath, subj) in enumerate(iterate):
            df_emg = pd.read_csv(emg_fpath, index_col=0, header=[0, 1], 
                skiprows=1)
            df_emg.drop('%s_walk2_cycle01' % subj, axis='columns', inplace=True)
            df_emg.drop('%s_walk2_cycle02' % subj, axis='columns', inplace=True)
            # col_name = '%s_walk2_cycle03' % subj
            # df_emg.rename(columns = {col_name : 'walk2_cycle03'}, inplace=True)
            df_emg_list.append(df_emg)

        df_emg_joined = df_emg_list[0]
        for df_emg in df_emg_list[1:]:
            df_emg_joined = df_emg_joined.join(df_emg)

        df_emg_mean = df_emg_joined.groupby(level=['muscle'], axis=1).mean()
        emg_pgc = df_emg_mean.index.values

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

        plot_iterate = zip(file_dep, self.label_list, self.color_list)

        fig = pl.figure(figsize=(8, 8))
        muscles = self.study.muscle_names
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

        mean_list = list()
        std_list = list()
        pgc_list = list()
        for dep in file_dep:
            df = pd.read_csv(dep, index_col=0,
                    header=[0, 1, 2], skiprows=1)
            for subj in self.study_subjects:
                if not (subj in self.subjects) and (subj in df.columns):
                    df.drop(subj, axis='columns', inplace=True)

            # Average over cycles.
            df_by_subj_musc = df.groupby(
                    level=['subject', 'muscle'], axis=1).mean()
            df_mean = df_by_subj_musc.groupby(level=['muscle'],
                    axis=1).mean()
            df_std = df_by_subj_musc.groupby(level=['muscle'],
                    axis=1).std()
            pgc = df_mean.index

            mean_list.append(df_mean)
            std_list.append(df_std)
            pgc_list.append(pgc)

        for imusc, musc_name in enumerate(muscles):
            side_len = np.ceil(np.sqrt(len(muscles)))
            ax = fig.add_subplot(side_len, side_len, imusc + 1)
            ax.axhline(color='k', linewidth=0.5, zorder=0)

            if emg_map.get(musc_name):
                y_emg = df_emg_mean[emg_map[musc_name]]
                ax.plot(emg_pgc, y_emg, label='emg', color='black', 
                    linestyle='--')

            for i, (dep, label, color) in enumerate(plot_iterate):
                df_mean = mean_list[i]
                pgc = pgc_list[i]

                y_mean = df_mean[musc_name]
                ax.plot(pgc, y_mean, label=label, color=color, linestyle='-')
                # ax.legend(frameon=False, fontsize=6)

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.0)
            ax.set_title(nice_act_names[musc_name])
            ax.set_xlabel('time (% gait cycle)')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            if not imusc:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, frameon=False, fontsize=7, 
                    loc="best")
                ax.get_legend().get_title().set_fontsize(8)
                # ax.get_legend().get_title().set_fontstyle('italic')
                ax.get_legend().get_title().set_fontweight('bold')

        fig.tight_layout()
        fig.savefig(target[0] + '.pdf')
        fig.savefig(target[0] + '.png', dpi=600)
        pl.close(fig)

# class TaskAggregateTorqueParameters(osp.StudyTask):
#     REGISTRY = []
#     def __init__(self, study, mods, conditions=['walk2'],
#             subjects=None, suffix=''):
#         super(TaskAggregateTorqueParameters, self).__init__(study)
#         self.suffix_path = suffix
#         if suffix != '':
#             suffix = '_' + suffix
#         self.costdir = ''
#         if not (study.costFunction == 'Default'):
#             suffix += '_%s' % study.costFunction
#             self.costdir = study.costFunction 
#         self.name = 'aggregate_torque_parameters%s' % suffix
#         self.doc = 'Aggregate parameters for active control signals.'

#         # Create new lists so original doesn't get changed by reference
#         mods = list(mods)

#         for mod in mods:
#             for cond_name in conditions:
#                 file_dep = os.path.join(
#                         self.study.config['results_path'],
#                         mod,  
#                         'mod_%s_%s_moments%s.csv' % (mod, cond_name, suffix))
#                 target = os.path.join(
#                         self.study.config['results_path'], 
#                          mod, 
#                         'mod_%s_%s_parameters%s.csv' % (mod, cond_name, suffix))
#                 self.add_action([file_dep],
#                                 [target], 
#                                 self.aggregate_torque_parameters,
#                                 cond_name, mod)

#     def aggregate_torque_parameters(self, file_dep, target, cond_name, mod):

#         df = pd.read_csv(file_dep[0], index_col=0, header=[0, 1, 2, 3], 
#             skiprows=1)

#         muscle_names = None
#         subject_array = list()
#         cycle_array = list()
#         dof_array = list()
#         muscle_array = list()
#         all_data = list()

#         def calc_torque_parameters(pgc, torque):


#             params = list()

#             import operator
#             peak_idx, peak_torque = max(enumerate(torque), 
#                 key=operator.itemgetter(1))

#             # peak torque
#             params.append(peak_torque) # N-m/kg

#             # peak time
#             peak_time = pgc[peak_idx]
#             params.append(peak_time) # percent GC

#             # rise time
#             for i in np.arange(peak_idx, -1, -1):
#                 if torque[pgc[i]] <= 0.01:
#                     rise_idx = i
#                     break
#                 rise_idx = i

#             rise_time = pgc[peak_idx] - pgc[rise_idx]
#             params.append(rise_time) # percent GC

#             # fall time
#             for i in np.arange(peak_idx, len(torque), 1):
#                 if torque[pgc[i]] <= 0.01:
#                     fall_idx = i
#                     break
#                 fall_idx = i


#             fall_time = pgc[fall_idx] - pgc[peak_idx]
#             params.append(fall_time) # percent GC

#             return params

#         for col in df.columns:
#             subject, cycle, dof, actuator = col
#             if actuator == 'active':
#                 act_torque = df[subject][cycle][dof][actuator]

#                 if ((('He' in mod) and ('hip' in dof)) or 
#                     (('Ke' in mod) and ('knee' in dof)) or
#                     (('Ap' in mod) and ('ankle' in dof))):
#                     act_torque = -act_torque

#                 params = calc_torque_parameters(df.index, act_torque)

#                 subject_array.append(subject)
#                 cycle_array.append(cycle)
#                 dof_array.append(dof)
#                 all_data.append(params)

#         #  n_params x (n_subjects * n_cycles * n_dofs)  
#         all_data_array = np.array(all_data).transpose()

#         multiindex_arrays = [subject_array, cycle_array, dof_array]
#         columns = pd.MultiIndex.from_arrays(multiindex_arrays,
#             names=['subject', 'cycle', 'DOF'])

#         params_idx = ['peak_torque', 'peak_time', 'rise_time', 'fall_time']
#         all_data_df = pd.DataFrame(all_data_array, columns=columns, 
#             index=params_idx)
#         target_dir = os.path.dirname(target[0])
#         if not os.path.exists(target_dir):
#             os.makedirs(target_dir)
#         with file(target[0], 'w') as f:
#             f.write('torque active control parameters in units (N-m/kg) for '
#                     'peak_torque and (percent g.c.) for times .\n')
#             all_data_df.to_csv(f)
#         # How to read this in: df.read_csv(..., index_col=0, header=[0, 1, 2, 3],
#         #                                  skiprows=1)

# class TaskPlotTorqueParameters(osp.StudyTask):
#     REGISTRY = []
#     def __init__(self, study, mods, conditions=['walk2'],
#             subjects=None, suffix=''):
#         super(TaskPlotTorqueParameters, self).__init__(study)
#         self.suffix_path = suffix
#         if suffix != '':
#             suffix = '_' + suffix
#         self.costdir = ''
#         if not (study.costFunction == 'Default'):
#             suffix += '_%s' % study.costFunction
#             self.costdir = study.costFunction 
#         self.name = 'plot_torque_parameters%s' % suffix
#         self.doc = 'Aggregate parameters for active control signals.'

#         # Create new lists so original doesn't get changed by reference
#         mods = list(mods)

#         for mod in mods:
#             for cond_name in conditions:
#                 file_dep = os.path.join(
#                         self.study.config['results_path'], mod,  
#                         'mod_%s_%s_parameters%s.csv' % (mod, cond_name, suffix))
#                 target0 = os.path.join(
#                         self.study.config['results_path'], mod, 
#                         'mod_%s_%s_parameters%s.pdf' % (mod, cond_name, suffix))
#                 target1 = os.path.join(
#                         self.study.config['results_path'], mod, 
#                         'mod_%s_%s_parameters%s.png' % (mod, cond_name, suffix))

#                 self.add_action([file_dep],
#                                 [target0, target1], 
#                                 self.plot_torque_parameters,
#                                 cond_name)

#     def plot_torque_parameters(self, file_dep, target, cond_name):

#         df = pd.read_csv(file_dep[0], index_col=0, header=[0, 1, 2], 
#             skiprows=1)

#         fig = pl.figure(figsize=(9, 3.75))

#         # Get relevant DOFs
#         col_labels = df.columns.values
#         dof_labels = [label[2] for label in col_labels]
#         dof_names = list(set(dof_labels))
#         param_names = ['peak_torque', 'peak_time', 'rise_time', 'fall_time']
#         for idof, dof_name in enumerate(dof_names):

#             df_DOF = df.xs(dof_name, level='DOF', axis=1)
#             peak_torque = df_DOF.loc['peak_torque']
#             peak_time = df_DOF.loc['peak_time']
#             rise_time = df_DOF.loc['rise_time']
#             fall_time = df_DOF.loc['fall_time']

#             # Normalize and concatenate data
#             all_data = [peak_torque / max(peak_torque), 
#                         peak_time / 100.0, 
#                         rise_time / 100.0, 
#                         fall_time / 100.0]

#             ax = fig.add_subplot(1, len(dof_names), idof + 1)
#             ax.boxplot(all_data)
#             ax.set_ylim(0.0, 1.0)
#             ax.set_yticks(np.arange(0.0, 1.1, 0.1))
#             ax.set_title(dof_name)
#             ax.set_xticklabels(param_names)

#         fig.tight_layout()
#         fig.savefig(target[0])
#         fig.savefig(target[1], dpi=600)
#         pl.close(fig)