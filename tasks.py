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

        # ax = fig.add_subplot(3, 2, 3)
        # ax.plot(time, slack_var, color='red')
        # ax.set_title('slack variable')
        # ax.set_yticks(np.arange(-0.1*max(pass_force[0]), 1.1*max(pass_force[0]), 
        #     max(pass_force[0])/10.0))
        # ax.set_ylabel('force (N)')
        # ax.set_xlabel('time')
        # ax.grid(which='both', axis='both', linestyle='--')

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

class TaskMetabolicRate(osp.StudyTask):
    """Aggregate metabolic rate without and with mods across all subjects and
    the provide contditions."""
    REGISTRY = []
    def __init__(self, study, mods=None, subjects=None, 
            conditions=['walk2'],
            suffix=''):
        super(TaskMetabolicRate, self).__init__(study)
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

        if mods == None:
            mods = study.mod_names

        cycles = list()
        self.multiindex_tuples = list()
        self.multiindex_tuples2 = list()

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






