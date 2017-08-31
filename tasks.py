import os

import pylab as pl
import pandas as pd

import osimpipeline as osp

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
                            '%s %02i.trc' % (datastr, arg[0])).replace('\\','\\\\'),
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