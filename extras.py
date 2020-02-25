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
                # calibrate_setup_task.kinematics_file,
                # calibrate_setup_task.kinetics_file,
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
        if '_multControls' in self.mrsmod_output_fpath:
            self.mrsmod_output_fpath = self.mrsmod_output_fpath.replace(
                '_multControls', '')
        self.results_setup_fpath = os.path.join(self.path, 'setup.m')
        self.results_output_fpath = os.path.join(self.path, '%s_%s_fitopt_%s.mat' % 
            (self.study.name, self.tricycle.id, self.fit))

        # Fill out setup.m template and write to results directory
        self.create_setup_action()

    def create_setup_action(self): 
        self.add_action(
                    ['templates/%s/%s/setup.m' % (self.tool, self.fit)],
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
            if not (self.fit == 'zhang2017'):
                content = content.replace('@MIN_PARAM@', str(self.min_param))
                content = content.replace('@MAX_PARAM@', str(self.max_param))
            content = content.replace('@START_TIME@', str(self.start_time))
            content = content.replace('@MRSMOD_OUTPUT@', 
                self.mrsmod_output_fpath)
            content = content.replace('@MOD_NAME@', self.mrsmod_task.mod_name)
            content = content.replace('@NORM_HIP_MAX_TORQUE@', 
                str(self.study.config['norm_hip_max_torque']))
            content = content.replace('@NORM_KNEE_MAX_TORQUE@', 
                str(self.study.config['norm_knee_max_torque']))
            content = content.replace('@NORM_ANKLE_MAX_TORQUE@', 
                str(self.study.config['norm_ankle_max_torque']))

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

def get_parameter_info(study, mod_name):

    fixed_params = ['fix_peak_torque', 'fix_peak_time', 'fix_rise_time',
                    'fix_fall_time', 'fix_all_times', 'fix_all_torques']
    for fixed_param in fixed_params:
        if fixed_param in mod_name:
            mod_name = mod_name.replace('_' + fixed_param, '')                

    param_bounds_all = study.config['param_bounds']

    param_names = list()
    param_bounds = dict()
    param_bounds['lower'] = list()
    param_bounds['upper'] = list()

    param_names.append('peak torque')
    peak_torque_bounds = param_bounds_all['peak_torque']
    param_bounds['lower'].append(peak_torque_bounds[0])
    param_bounds['upper'].append(peak_torque_bounds[1])

    param_names.append('peak time')
    peak_time_bounds = param_bounds_all['peak_time']
    param_bounds['lower'].append(peak_time_bounds[0])
    param_bounds['upper'].append(peak_time_bounds[1])

    param_names.append('rise time')
    rise_time_bounds = param_bounds_all['rise_time']
    param_bounds['lower'].append(rise_time_bounds[0])
    param_bounds['upper'].append(rise_time_bounds[1])

    param_names.append('fall time')
    fall_time_bounds = param_bounds_all['fall_time']
    param_bounds['lower'].append(fall_time_bounds[0])
    param_bounds['upper'].append(fall_time_bounds[1])

    if 'fitreopt_zhang2017_actHfAp' == mod_name:
        param_names.append('ankle peak torque')
        param_bounds['lower'].append(peak_torque_bounds[0])
        param_bounds['upper'].append(peak_torque_bounds[1])

    elif 'fitreopt_zhang2017_actHfKf' == mod_name:
        param_names.append('knee peak torque')
        param_bounds['lower'].append(peak_torque_bounds[0])
        param_bounds['upper'].append(peak_torque_bounds[1])

    elif 'fitreopt_zhang2017_actKfAp' == mod_name:
        param_names.append('ankle peak torque')
        param_bounds['lower'].append(peak_torque_bounds[0])
        param_bounds['upper'].append(peak_torque_bounds[1])

    elif 'fitreopt_zhang2017_actHfKfAp' == mod_name:
        param_names.append('knee peak torque')
        param_bounds['lower'].append(peak_torque_bounds[0])
        param_bounds['upper'].append(peak_torque_bounds[1])

        param_names.append('ankle peak torque')
        param_bounds['lower'].append(peak_torque_bounds[0])
        param_bounds['upper'].append(peak_torque_bounds[1])

    elif 'fitreopt_zhang2017_actHeKe' == mod_name:
        param_names.append('knee peak torque')
        param_bounds['lower'].append(peak_torque_bounds[0])
        param_bounds['upper'].append(peak_torque_bounds[1])

    if 'fitreopt_zhang2017_actHfAp_multControls' == mod_name:
        param_names.append('peak torque 2')
        peak_torque_bounds = param_bounds_all['peak_torque']
        param_bounds['lower'].append(peak_torque_bounds[0])
        param_bounds['upper'].append(peak_torque_bounds[1])

        param_names.append('peak time 2')
        peak_time_bounds = param_bounds_all['peak_time']
        param_bounds['lower'].append(peak_time_bounds[0])
        param_bounds['upper'].append(peak_time_bounds[1])

        param_names.append('rise time 2')
        rise_time_bounds = param_bounds_all['rise_time']
        param_bounds['lower'].append(rise_time_bounds[0])
        param_bounds['upper'].append(rise_time_bounds[1])

        param_names.append('fall time 2')
        fall_time_bounds = param_bounds_all['fall_time']
        param_bounds['lower'].append(fall_time_bounds[0])
        param_bounds['upper'].append(fall_time_bounds[1])

    elif 'fitreopt_zhang2017_actHfKf_multControls' == mod_name:
        param_names.append('peak torque 2')
        peak_torque_bounds = param_bounds_all['peak_torque']
        param_bounds['lower'].append(peak_torque_bounds[0])
        param_bounds['upper'].append(peak_torque_bounds[1])

        param_names.append('peak time 2')
        peak_time_bounds = param_bounds_all['peak_time']
        param_bounds['lower'].append(peak_time_bounds[0])
        param_bounds['upper'].append(peak_time_bounds[1])

        param_names.append('rise time 2')
        rise_time_bounds = param_bounds_all['rise_time']
        param_bounds['lower'].append(rise_time_bounds[0])
        param_bounds['upper'].append(rise_time_bounds[1])

        param_names.append('fall time 2')
        fall_time_bounds = param_bounds_all['fall_time']
        param_bounds['lower'].append(fall_time_bounds[0])
        param_bounds['upper'].append(fall_time_bounds[1])

    elif 'fitreopt_zhang2017_actKfAp_multControls' == mod_name:
        param_names.append('peak torque 2')
        peak_torque_bounds = param_bounds_all['peak_torque']
        param_bounds['lower'].append(peak_torque_bounds[0])
        param_bounds['upper'].append(peak_torque_bounds[1])

        param_names.append('peak time 2')
        peak_time_bounds = param_bounds_all['peak_time']
        param_bounds['lower'].append(peak_time_bounds[0])
        param_bounds['upper'].append(peak_time_bounds[1])

        param_names.append('rise time 2')
        rise_time_bounds = param_bounds_all['rise_time']
        param_bounds['lower'].append(rise_time_bounds[0])
        param_bounds['upper'].append(rise_time_bounds[1])

        param_names.append('fall time 2')
        fall_time_bounds = param_bounds_all['fall_time']
        param_bounds['lower'].append(fall_time_bounds[0])
        param_bounds['upper'].append(fall_time_bounds[1])

    elif 'fitreopt_zhang2017_actHfKfAp_multControls' == mod_name:
        param_names.append('peak torque 2')
        peak_torque_bounds = param_bounds_all['peak_torque']
        param_bounds['lower'].append(peak_torque_bounds[0])
        param_bounds['upper'].append(peak_torque_bounds[1])

        param_names.append('peak time 2')
        peak_time_bounds = param_bounds_all['peak_time']
        param_bounds['lower'].append(peak_time_bounds[0])
        param_bounds['upper'].append(peak_time_bounds[1])

        param_names.append('rise time 2')
        rise_time_bounds = param_bounds_all['rise_time']
        param_bounds['lower'].append(rise_time_bounds[0])
        param_bounds['upper'].append(rise_time_bounds[1])

        param_names.append('fall time 2')
        fall_time_bounds = param_bounds_all['fall_time']
        param_bounds['lower'].append(fall_time_bounds[0])
        param_bounds['upper'].append(fall_time_bounds[1])

        param_names.append('peak torque 3')
        peak_torque_bounds = param_bounds_all['peak_torque']
        param_bounds['lower'].append(peak_torque_bounds[0])
        param_bounds['upper'].append(peak_torque_bounds[1])

        param_names.append('peak time 3')
        peak_time_bounds = param_bounds_all['peak_time']
        param_bounds['lower'].append(peak_time_bounds[0])
        param_bounds['upper'].append(peak_time_bounds[1])

        param_names.append('rise time 3')
        rise_time_bounds = param_bounds_all['rise_time']
        param_bounds['lower'].append(rise_time_bounds[0])
        param_bounds['upper'].append(rise_time_bounds[1])

        param_names.append('fall time 3')
        fall_time_bounds = param_bounds_all['fall_time']
        param_bounds['lower'].append(fall_time_bounds[0])
        param_bounds['upper'].append(fall_time_bounds[1])

    elif 'fitreopt_zhang2017_actHeKe_multControls' == mod_name:
        param_names.append('peak torque 2')
        peak_torque_bounds = param_bounds_all['peak_torque']
        param_bounds['lower'].append(peak_torque_bounds[0])
        param_bounds['upper'].append(peak_torque_bounds[1])

        param_names.append('peak time 2')
        peak_time_bounds = param_bounds_all['peak_time']
        param_bounds['lower'].append(peak_time_bounds[0])
        param_bounds['upper'].append(peak_time_bounds[1])

        param_names.append('rise time 2')
        rise_time_bounds = param_bounds_all['rise_time']
        param_bounds['lower'].append(rise_time_bounds[0])
        param_bounds['upper'].append(rise_time_bounds[1])

        param_names.append('fall time 2')
        fall_time_bounds = param_bounds_all['fall_time']
        param_bounds['lower'].append(fall_time_bounds[0])
        param_bounds['upper'].append(fall_time_bounds[1])

    return param_names, param_bounds

class TaskAggregateTorqueParameters(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, mod, subjects=None, 
            cond_names=['walk1','walk2','walk3','walk4']):
        super(TaskAggregateTorqueParameters, self).__init__(study)
        self.mod_dir = mod.replace('/','\\\\')
        if len(mod.split('/')) > 1:
            self.mod_name = '_'.join(mod.split('/'))
        else:
            self.mod_name = mod

        if not ('fitreopt' in self.mod_name):
            Exception('Only "fitreopt" tasks accepted for this aggregate task.') 

        suffix = '_' + self.mod_name
        cost = ''
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            cost = '_' + study.costFunction
            self.costdir = study.costFunction

        self.name = 'aggregate_torque_parameters%s' % suffix
        self.doc = 'Aggregate torque parameters into a data file.'

        if subjects == None:
            subjects = [s.name for s in study.subjects]

        # Parameter names and bounds
        self.param_names, self.param_bounds = \
            get_parameter_info(study, self.mod_name)

        self.cycles = dict()
        self.output_fpaths = list()
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
                            self.mod_dir, subject.name, cond.name, 'mrs', 
                            cycle.name, self.costdir,
                            '%s_%s_mrs.mat' % (study.name, cycle.id))
                        deps.append(fpath)

            output_fpath = os.path.join(study.config['results_path'], 
                        self.mod_dir,'%s_%s_torque_parameters%s.csv' % (
                            self.mod_name, cond_name, cost))
            self.output_fpaths += [output_fpath]
            self.add_action([], [], self.aggregate_torque_parameters, cond_name,
                    deps, output_fpath)

    def aggregate_torque_parameters(self, file_dep, target, cond_name, deps,
            output_fpath):

        subject_array = list()
        cycle_array = list()
        param_array = list()
        all_parameters = list()
        lower_bounds = list()
        upper_bounds = list()
        for icycle, fpath in enumerate(deps):
            cycle = self.cycles[cond_name][icycle]

            parametersScaled = util.hdf2numpy(fpath, 
                'OptInfo/result/solution/parameter')
            paramsLowerBound = util.hdf2numpy(fpath, 
                'OptInfo/result/setup/auxdata/paramsLower')
            paramsUpperBound = util.hdf2numpy(fpath, 
                'OptInfo/result/setup/auxdata/paramsUpper')

            # Parameters were scaled between [-1,1] for optimizations. Now
            # rescale them into values within original parameter value ranges.
            # parameters = 0.5*np.multiply(paramsUpperBound-paramsLowerBound, 
            #     parametersScaled + 1) + paramsLowerBound
            parameters = parametersScaled

            subject_array.append(cycle.subject.name)
            cycle_array.append(cycle.id)
            all_parameters.append(parameters[0])
            lower_bounds.append(paramsLowerBound[0])
            upper_bounds.append(paramsUpperBound[0])

        all_parameters_array = np.array(all_parameters).transpose()
        lower_bounds_array = np.array(lower_bounds).transpose()
        upper_bounds_array = np.array(upper_bounds).transpose()

        multiindex_arrays = [subject_array, cycle_array]
        columns = pd.MultiIndex.from_arrays(multiindex_arrays,
                names=['subject', 'cycle'])

        all_parameters_df = pd.DataFrame(all_parameters_array, columns=columns,
            index=self.param_names)
        target_dir = os.path.dirname(output_fpath)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(output_fpath, 'w') as f:
            f.write('# all columns are torque parameter.\n')
            all_parameters_df.to_csv(f)

        lower_bounds_df = pd.DataFrame(lower_bounds_array, columns=columns,
            index=self.param_names)
        lower_bounds_fpath = output_fpath.replace('torque_parameters', 
            'torque_parameters_lower_bound')
        target_dir = os.path.dirname(lower_bounds_fpath)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(lower_bounds_fpath, 'w') as f:
            f.write('# all columns are torque parameter lower bounds.\n')
            lower_bounds_df.to_csv(f)

        upper_bounds_df = pd.DataFrame(upper_bounds_array, columns=columns,
            index=self.param_names)
        upper_bounds_fpath = output_fpath.replace('torque_parameters', 
            'torque_parameters_upper_bound')
        target_dir = os.path.dirname(upper_bounds_fpath)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(upper_bounds_fpath, 'w') as f:
            f.write('# all columns are torque parameter upper bounds.\n')
            upper_bounds_df.to_csv(f)

class TaskPlotTorqueParameters(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, agg_task):
        super(TaskPlotTorqueParameters, self).__init__(study)
        self.agg_task = agg_task
        self.task_name = self.agg_task.mod_name if hasattr(self.agg_task, 
            'mod_name') else 'experiment' 
        suffix = '_' + self.task_name
        self.costdir = ''
        if not (study.costFunction == 'Default'):
            suffix += '_%s' % study.costFunction
            self.costdir = study.costFunction 
        self.name = 'plot_torque_parameters%s' % suffix
        self.doc = 'Plot torque parameters'

        for icond, agg_target in enumerate(agg_task.output_fpaths):
            # This assumes csv_task.targets and csv_task.cycles hold cycles in
            # the same order.
            # self.agg_target = agg_target
            # self.actions += [self.plot_muscle_data]
            # print agg_target
            self.add_action([],[],
                            self.plot_torque_parameters,
                            agg_target, agg_task.param_names,
                            agg_task.param_bounds)

    def plot_torque_parameters(self, file_dep, target, agg_target, 
            param_names, param_bounds):

        df = pd.read_csv(agg_target, index_col=0, header=[0, 1], 
            skiprows=1)
        df_lb = pd.read_csv(agg_target.replace('torque_parameters', 
            'torque_parameters_lower_bound'), index_col=0, header=[0 ,1],
            skiprows=1)
        df_ub = pd.read_csv(agg_target.replace('torque_parameters', 
            'torque_parameters_upper_bound'), index_col=0, header=[0 ,1],
            skiprows=1)

        # Normalize within [0,1]
        # for ip, param_name in enumerate(param_names):
        #     norm_factor = param_bounds['upper'][ip] - param_bounds['lower'][ip]
        #     df.loc[param_name] -= param_bounds['lower'][ip]
        #     df.loc[param_name] /= norm_factor

        for ip, param_name in enumerate(param_names):
            norm_factor = df_ub.loc[param_name] - df_lb.loc[param_name]
            df.loc[param_name] -= df_lb.loc[param_name]
            df.loc[param_name] /= norm_factor

        fig = pl.figure(figsize=(6, 6))
        #df_by_subj = df.groupby(level=['subject'], axis=1).mean()

        ax = fig.add_subplot(1,1,1)
        # import pdb
        # pdb.set_trace()
        ax.scatter(np.matlib.repmat(np.arange(len(df.index)), 6, 1), 
            df.values.transpose())
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        ax.set_xticks(np.arange(len(df.index)))
        ax.set_xticklabels(df.index, rotation=45, ha='right')

        fig.tight_layout()
        fig.savefig(agg_target.replace('.csv', '.pdf'))
        fig.savefig(agg_target.replace('.csv', '.png'), dpi=600)
        pl.close(fig)

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
        self.actdyn = self.mrsmod_task.actdyn
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

        self.speed = trial.condition.metadata['walking_speed']

        self.file_dep += [
            # self.kinematics_file,
            # self.kinetics_file
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

class TaskPlotMetabolicsVsParameters(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, plot_lists, folder,
            cond_names=['walk1','walk2','walk3','walk4'], subjects=None):
        super(TaskPlotMetabolicsVsParameters, self).__init__(study)
        self.name = 'plot_metabolics_vs_parameters_%s' % folder
        self.doc = 'Plot to metabolic reductions vs. number of torque parameters.'
        self.output_path = os.path.join(study.config['analysis_path'])
        self.cond_names = cond_names
        self.device_list = plot_lists['device_list']
        self.device_names = list()
        self.device_dirs = list()
        for device in self.device_list:
            self.device_dirs.append(device.replace('/', '\\'))
            if len(device.split('/')) > 1:
                self.device_names.append('_'.join(device.split('/')))
            else:
                self.device_names.append(device)

            if not ('fitreopt' in device):
                Exception('Only "fitreopt" tasks accepted for '
                          'this aggregate task.') 

        self.label_list = plot_lists['label_list']
        self.color_list = plot_lists['color_list']
        self.folder = folder

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
        self.study_subjects = [s.name for s in self.study.subjects]

        self.add_action([os.path.join(self.output_path, folder, 
                    'whole_body_metabolic_rates_%s%s.csv' % (folder, cost))], 
                [os.path.join(self.output_path, folder, 
                    'metabolics_vs_parameters_%s%s.pdf' % (folder, cost)),
                os.path.join(self.output_path, folder, 
                    'metabolics_vs_parameters_no_error_bars_%s%s.pdf' % (folder, 
                        cost)),
                ],
                self.plot_metabolics_vs_parameters)


    def plot_metabolics_vs_parameters(self, file_dep, target):

        # Plot percent reduction
        df_met = pd.read_csv(file_dep[0], index_col=[0, 1, 2], skiprows=1)
        for subj in self.study_subjects:
            if not (subj in self.subjects):
                df_met.drop(subj, axis='index', inplace=True)

        df_met_change = df_met.subtract(df_met['experiment'], axis='index')
        df_met_relchange = df_met_change.divide(df_met['experiment'], 
            axis='index')
        df_met_relchange.drop('experiment', axis='columns', inplace=True)

        df_met_by_subjs = df_met_relchange.groupby(level='subject').mean()
        met_mean = df_met_by_subjs.mean()[self.device_names] * 100
        met_std = df_met_by_subjs.std()[self.device_names] * 100

        fig = pl.figure(figsize=(4*0.85, 6*0.85))
        ax = fig.add_subplot(1,1,1)

        ind = list()
        for device in self.device_names:
            param_names, param_bounds = \
                get_parameter_info(self.study, device)
            ind.append(len(param_names))

        for i in range(len(ind)):
            plotlist = ax.errorbar(ind[i], met_mean[i], yerr=met_std[i], fmt='o',
                color=self.color_list[i], markersize=12)

        # Temporary, so 6 parameters is always set
        ax.errorbar(6, 0)

        ax.set_xticks(np.arange(4, 7))
        ax.set_xticklabels(np.arange(4, 7))
        ax.margins(0.1)
        ax.set_xlabel('number of parameters')
        ax.set_ylabel('reduction in metabolic cost (%)')
        ax.set_ylim(-25, 0)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[0].replace('.pdf','.png'), ppi=1800)
        fig.savefig(target[0].replace('.pdf','.svg'), format='svg', dpi=1000)
        pl.close(fig)

        # same plot, no error bars
        fig = pl.figure(figsize=(4, 6))
        ax = fig.add_subplot(1,1,1)

        for i in range(len(ind)):
            plotlist = ax.plot(ind[i], met_mean[i], 'o',
                color=self.color_list[i], markersize=12)

        ax.set_xticks(np.arange(4, 7))
        ax.set_xticklabels(np.arange(4, 7))
        ax.margins(0.1)
        ax.set_xlabel('number of parameters')
        ax.set_ylabel('reduction in metabolic cost (%)')
        ax.set_ylim(-25, 0)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        fig.tight_layout()
        fig.savefig(target[1])
        fig.savefig(target[1].replace('.pdf','.png'), ppi=1800)
        fig.savefig(target[1].replace('.pdf','.svg'), format='svg', dpi=1000)
        pl.close(fig)

class TaskPlotMetabolicsForFixedParameters(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, plot_lists, folder,
            cond_names=['walk1','walk2','walk3','walk4'], subjects=None):
        super(TaskPlotMetabolicsForFixedParameters, self).__init__(study)
        self.name = 'plot_metabolics_for_fixed_parameters_%s' % folder
        self.doc = 'Plot metabolic changes for fixed torque parameters.'
        self.output_path = os.path.join(study.config['analysis_path'])
        self.cond_names = cond_names
        self.device_list = plot_lists['device_list']
        self.device_names = list()
        self.device_dirs = list()
        for device in self.device_list:
            self.device_dirs.append(device.replace('/', '\\'))
            if len(device.split('/')) > 1:
                self.device_names.append('_'.join(device.split('/')))
            else:
                self.device_names.append(device)

            if not ('fitreopt' in device):
                Exception('Only "fitreopt" tasks accepted for '
                          'this aggregate task.') 

        self.label_list = plot_lists['label_list']
        self.color_list = plot_lists['color_list']
        self.fixed_param_list = plot_lists['fixed_param_list']
        self.folder = folder

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
        self.study_subjects = [s.name for s in self.study.subjects]

        self.add_action([os.path.join(self.output_path, folder, 
                    'whole_body_metabolic_rates_%s%s.csv' % (folder, cost))], 
                [os.path.join(self.output_path, folder, 
                    'metabolics_for_fixed_parameters_%s%s.pdf' % (folder, cost)),
                ],
                self.plot_metabolics_for_fixed_parameters)


    def plot_metabolics_for_fixed_parameters(self, file_dep, target):

        # Plot percent reduction
        df_met = pd.read_csv(file_dep[0], index_col=[0, 1, 2], skiprows=1)
        for subj in self.study_subjects:
            if not (subj in self.subjects):
                df_met.drop(subj, axis='index', inplace=True)

        df_met_change = df_met.subtract(df_met['experiment'], axis='index')
        df_met_relchange = df_met_change.divide(df_met['experiment'], 
            axis='index')
        df_met_relchange.drop('experiment', axis='columns', inplace=True)

        df_met_by_subjs = df_met_relchange.groupby(level='subject').mean()
        met_mean = df_met_by_subjs.mean()[self.device_names] * 100
        met_std = df_met_by_subjs.std()[self.device_names] * 100

        met_change_fixed_params = list()
        for param in self.fixed_param_list:
            fixed_param_device_names = \
                [n + '_' + param for n in self.device_names]
            met_mean_fixed_param = \
                df_met_by_subjs.mean()[fixed_param_device_names] * 100

            met_change_fixed_params.append(
                met_mean_fixed_param - met_mean.values)

        fig = pl.figure(figsize=(4*0.8, 6*0.8))
        ax = fig.add_subplot(1,1,1)
        ind = np.arange(len(self.fixed_param_list))+1
        met_change_fixed_params_array = np.array(met_change_fixed_params)
        for i in np.arange(len(ind)):
            curr_ind = [ind[i]] * len(met_change_fixed_params_array[i])
            ax.scatter(curr_ind, met_change_fixed_params_array[i], 
                color=self.color_list)

        ax.set_xticks(ind)
        nice_fixed_param_list = [param.replace('_', ' ') for param in 
            self.fixed_param_list]
        ax.set_xticklabels(nice_fixed_param_list, rotation=45, ha='right')
        ax.set_ylim(-1, 5)

        # ax.margins(0.1)
        # ax.set_xlabel('number of parameters')
        ax.set_ylabel('change in reduction in metabolic cost (%)')
        # ax.set_ylim(-25, 0)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.xaxis.set_ticks_position('top')
        # ax.xaxis.set_label_position('top')

        fig.tight_layout()
        fig.savefig(target[0])
        fig.savefig(target[0].replace('.pdf','.png'), ppi=1800)
        fig.savefig(target[0].replace('.pdf','.svg'), format='svg', dpi=1000)
        pl.close(fig)

class TaskPlotMuscleActivityComparison(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, plot_lists, folder, condition='walk2', 
            subjects=None):
        super(TaskPlotMuscleActivityComparison, self).__init__(study)
        self.name = 'plot_muscle_activity_comparison'
        self.doc = 'Plot to compare muscle activity solutions.'
        self.device_list = plot_lists['device_list']
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
        self.add_action(self.device_list,
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
        nice_muscle_names_dict = self.study.nice_muscle_names_dict

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
                ax.plot(emg_pgc, y_emg, label='emg', color='darkgray', 
                    linestyle='--')

            for i, (dep, label, color) in enumerate(plot_iterate):
                df_mean = mean_list[i]
                pgc = pgc_list[i]

                y_mean = df_mean[musc_name]
                ax.plot(pgc, y_mean, label=label, color=color, linestyle='-')
                # ax.legend(frameon=False, fontsize=6)

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1.0)
            ax.set_title(nice_muscle_names_dict[musc_name])
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
#     def __init__(self, study, mod, conditions=['walk1','walk2','walk3','walk4']):
#         super(TaskAggregateTorqueParameters, self).__init__(study)
#         self.mod_dir = mod.replace('/','\\\\')
#         if len(mod.split('/')) > 1:
#             self.mod_name = '_'.join(mod.split('/'))
#         else:
#             self.mod_name = mod

#         suffix = '_%s' % self.mod_name
#         self.cost = ''
#         if not (study.costFunction == 'Default'):
#             suffix += '_%s' % study.costFunction
#             self.cost = study.costFunction 
#         self.name = 'aggregate_torque_parameters%s' % suffix
#         self.doc = 'Aggregate parameters for active control signals.'

#         for cond_name in conditions:
#             file_dep = os.path.join(
#                     self.study.config['results_path'], self.mod_dir,  
#                     '%s_%s_moments_%s.csv' % (self.mod_name, cond_name, 
#                         self.cost))
#             target = os.path.join(
#                     self.study.config['results_path'], self.mod_dir, 
#                     '%s_%s_parameters_%s.csv' % (self.mod_name, cond_name, 
#                         self.cost))
#             self.add_action([file_dep],
#                             [target], 
#                             self.aggregate_torque_parameters,
#                             cond_name, self.mod_name)

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
#     def __init__(self, study, mod, conditions=['walk2']):
#         super(TaskPlotTorqueParameters, self).__init__(study)
#         self.mod_dir = mod.replace('/','\\\\')
#         if len(mod.split('/')) > 1:
#             self.mod_name = '_'.join(mod.split('/'))
#         else:
#             self.mod_name = mod

#         suffix = '_%s' % self.mod_name
#         self.cost = ''
#         if not (study.costFunction == 'Default'):
#             suffix += '_%s' % study.costFunction
#             self.cost = study.costFunction 
#         self.name = 'plot_torque_parameters%s' % suffix
#         self.doc = 'Aggregate parameters for active control signals.'

#         for cond_name in conditions:
#             file_dep = os.path.join(
#                     self.study.config['results_path'], self.mod_dir,  
#                     '%s_%s_parameters_%s.csv' % (self.mod_name, cond_name, 
#                         self.cost))
#             target0 = os.path.join(
#                     self.study.config['results_path'], self.mod_dir, 
#                     '%s_%s_parameters_%s.pdf' % (self.mod_name, cond_name, 
#                         self.cost))
#             target1 = os.path.join(
#                     self.study.config['results_path'], self.mod_dir, 
#                     '%s_%s_parameters_%s.png' % (self.mod_name, cond_name, 
#                         self.cost))

#             self.add_action([file_dep],
#                             [target0, target1], 
#                             self.plot_torque_parameters,
#                             cond_name)

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