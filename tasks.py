import os

import numpy as np
import pylab as pl
import pandas as pd

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

        for subject in study.subjects:

            cond_args = subject.cond_args
            if 'walk100' in cond_args:
                walk100 = cond_args['walk100']
            if 'walk125' in cond_args:
                walk125 = cond_args['walk125']
            if 'walk150' in cond_args:
                walk150 = cond_args['walk150']
            if 'walk175' in cond_args:
                walk175 = cond_args['walk175']
            if 'run200' in cond_args:
                run200 = cond_args['run200']
            if 'run300' in cond_args:
                run300 = cond_args['run300']
            if 'run400' in cond_args:
                run400 = cond_args['run400']
            if 'run500' in cond_args:
                run500 = cond_args['run500']

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
            regex_replacements.append((
                        os.path.join(subject.name, 'Data',
                            'Static_FJC.trc').replace('\\','\\\\'),
                        os.path.join('experiments', subject.name, 'static',
                            'expdata',
                            'marker_trajectories.trc').replace('\\','\\\\') 
                        ))

        super(TaskCopyMotionCaptureData, self).__init__(study,
                regex_replacements)

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
        muscle_names = util.hdf2list(file_dep[0], 'MuscleNames', isString=True)
        dof_names = util.hdf2list(file_dep[0],'DatStore/DOFNames',isString=True)
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

        dof_names = util.hdf2list(file_dep[0],'DatStore/DOFNames',isString=True)
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

        dof_names = util.hdf2list(file_dep[0],'DatStore/DOFNames',isString=True)
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
        muscle_names = util.hdf2list(file_dep[0], 'MuscleNames', isString=True)
        num_muscles = len(muscle_names)

        mrs_whole_body_metabolic_rate = util.hdf2pandas(file_dep[0], 
            'DatStore/MetabolicRate/whole_body')
        mrs_muscle_metabolic_rates = util.hdf2pandas(file_dep[0],
            'DatStore/MetabolicRate/individual_muscles', labels=muscle_names)

        # Load mat file fields from modified solution
        muscle_names = util.hdf2list(file_dep[1], 'MuscleNames', isString=True)
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
        muscle_names = util.hdf2list(file_dep[0], 'MuscleNames', isString=True)
        num_muscles = len(muscle_names)

        mrs_excitations = util.hdf2pandas(file_dep[0], 
            'MExcitation', labels=muscle_names)
        mrs_activations = util.hdf2pandas(file_dep[0],
            'MActivation', labels=muscle_names)

        # Load mat file fields from modified solution
        muscle_names = util.hdf2list(file_dep[1], 'MuscleNames', isString=True)
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

class TaskAggregateMetabolicRate(osp.StudyTask):
    """Aggregate metabolic rate without and with mods across all subjects and
    the provide contditions."""
    REGISTRY = []
    def __init__(self, study, mods=None, subjects=None, 
            conditions=['walk2'], suffix=''):
        super(TaskAggregateMetabolicRate, self).__init__(study)
        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix
        self.name = 'aggregate_metabolic_rate%s' % suffix
        self.whole_fpath = os.path.join(self.suffix_path, 
            'whole_body_metabolic_rates%s.csv' % suffix)
        self.muscs_fpath = os.path.join(self.suffix_path, 
            'muscle_metabolic_rates%s.csv' % suffix)   
        self.doc = 'Aggregate metabolic rate.'
        self.study = study

        cycles = list()
        self.multiindex_tuples = list()
        self.multiindex_tuples2 = list()

        if mods == None:
            mods = study.mod_names
        if subjects == None:
            subjects = [s.num for s in study.subjects]

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
                        self.multiindex_tuples.append((
                            cycle.subject.name,
                            cycle.condition.name,
                            # This must be the full ID, not just the cycle
                            # name, because 'cycle01' from subject 1 has
                            # nothing to do with 'cycle01' from subject 2
                            # (whereas the 'walk2' condition for subject 1 is
                            # related to 'walk2' for subject 2).
                            cycle.id))
                        for mname in study.muscle_names:
                            self.multiindex_tuples2.append((
                                cycle.subject.name,
                                cycle.condition.name,
                                cycle.id,
                                mname))

        self.mod_for_file_dep = list()
        deps = list()
        deps2 = list()

        # Prepare for processing simulations of experiments.
        for cycle in cycles:
            self.mod_for_file_dep.append('experiment')
            deps.append(os.path.join(
                    cycle.trial.results_exp_path, 'mrs', cycle.name,
                    '%s_%s_mrs.mat' % (study.name, cycle.id))
                    )

        # Prepare for processing simulations of mods.
        for mod in mods:
            for cycle in cycles:
                self.mod_for_file_dep.append(mod)
                deps.append(os.path.join(
                        self.study.config['results_path'],
                        'mrsmod_%s' % mod, cycle.trial.rel_path, 'mrs', 
                        cycle.name, '%s_%s_mrs.mat' % (study.name, cycle.id))
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
        index = pd.MultiIndex.from_tuples(self.multiindex_tuples2,
                names=['subject', 'condition', 'cycle', 'muscle'])
        df = pd.DataFrame(metabolic_rate, index=index)

        target_dir = os.path.dirname(target[0])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with file(target[0], 'w') as f:
            f.write('# columns contain muscle metabolic rates normalized by '
                    'subject mass (W/kg)\n')
            df.to_csv(f)

class TaskPlotDeviceMetabolicRankings(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, suffix='', mods=[]):
        super(TaskPlotDeviceMetabolicRankings, self).__init__(study)
        self.suffix_path = suffix
        if suffix != '':
            suffix = '_' + suffix
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

        if subjects == None:
            subjects = [s.num for s in study.subjects]

        self.cycles = dict()
        for cond_name in conditions:
            self.cycles[cond_name] = list()
            deps = []
            deps2 = []
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
                            cycle.name, '%s_%s_mrs_moments.csv' % (study.name,
                            cycle.id))
                        deps.append(fpath)

            self.add_action(deps,
                    [
                        os.path.join(study.config['results_path'], 
                            'experiments',
                            'experiment_%s_moments.csv' % cond_name),
                        ],
                    aggregate_moments, cond_name, self.cycles)

class TaskAggregateMomentsMod(osp.StudyTask):
    REGISTRY = []
    def __init__(self, study, mod_name, conditions=['walk2'], subjects=None):
        super(TaskAggregateMomentsMod, self).__init__(study)
        self.name = 'aggregate_mod_results_%s' % mod_name
        self.doc = 'Aggregate actuator moments into a data file.'
        self.mod_name = mod_name
        self.conditions = conditions

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
                                trial.rel_path, 'mrs', cycle.name,
                                '%s_%s_mrs_moments.csv' % (study.name,
                                    cycle.id))
                                )

            self.add_action(deps,
                    [os.path.join(study.config['results_path'], 
                        'mrsmod_%s' % mod_name,
                        'mod_%s_%s_moments.csv' % (mod_name,
                            cond_name)),
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
        self.name = 'aggregate_muscle_activity%s' % suffix
        self.doc = 'Aggregate muscle activity into a data file.'

        if mods == None:
            mods = study.mod_names

        mods_fpath = ['mrsmod_' + mod for mod in mods]
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
                            fpath = os.path.join(trial.results_exp_path, 'mrs',
                                cycle.name, '%s_%s_mrs.mat' % (study.name,
                                cycle.id))
                            deps.append(fpath)

                self.add_action(deps,
                        [
                            os.path.join(study.config['results_path'], 
                                mod_fpath,'%s_%s_excitations.csv' % (
                                    mod, cond_name)),
                            os.path.join(study.config['results_path'], 
                                mod_fpath,'%s_%s_activations.csv' % (
                                    mod, cond_name)),
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

            muscle_names = util.hdf2list(fpath, 'MuscleNames', isString=True)
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

                emg_fpath = os.path.join(self.data_path, subject.name, 
                    'Results', self.cond_map[cond.name], 
                    '%s_gait_controls.sto' % self.cond_map[cond.name])
                states_fpath = os.path.join(self.data_path, subject.name, 
                    'Results', self.cond_map[cond.name], 
                    '%s_gait_states.sto' % self.cond_map[cond.name])

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
            subjects=None):
        super(TaskPlotMuscleActivity, self).__init__(study)
        self.name = 'plot_muscle_activity'
        self.doc = 'Plot muscle activity for experiment and mod tasks'

        for icond, agg_target in enumerate(agg_task.targets):
            # This assumes csv_task.targets and csv_task.cycles hold cycles in
            # the same order.
            # self.agg_target = agg_target
            # self.actions += [self.plot_muscle_activity]
            print agg_target
            # self.add_action([],[],
            #                 self.plot_muscle_activity,
            #                 agg_target)

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
        self.name = 'validate_against_emg'
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
                    'experiment_%s_excitations.csv' % cond)
                act_fpath = os.path.join(self.results_path, 'experiments',
                    'experiment_%s_activations.csv' % cond)

                val_fname = os.path.join(self.validate_path, 
                    '%s_%s_emg_validation' % (subject, cond))
                
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






