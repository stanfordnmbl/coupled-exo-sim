import itertools
import osimpipeline as osp
import tasks
import pandas as pd
import os

def get_device_info(study):

    momentArms=study.momentArms
    # fixMomentArms (bool)
    # Flag whether or not to include fixed moment arm tasks
    fixMomentArms=study.fixMomentArms

    # Can moment arms be optimized in either direction or are they restricted
    # to the anterior or posterior side of a joint?
    if momentArms == 'free':
        device_dofs = ['H','K','A']
        # Can't fix moment arms in this case, so override user input
        if not fixMomentArms == True:
            Exception("Moment arms cannot be fixed when optimizing in both "
                "directions across a DOF. Setting flag to free moment arm "
                " optimization.")

    elif momentArms == 'fixed_direction':
        device_dofs = ['Hf','He','Kf','Ke','Ap']

    elif momentArms == 'BiLLEE':
        device_dofs = ['H','K','Ap']

    return device_dofs

def get_exotopology_flags(study):

    device_dofs_all = get_device_info(study)

    mod_names = list()
    deviceDOFs_list = list()

    for L in range(1, len(device_dofs_all)+1):
        for subset in itertools.combinations(device_dofs_all, L):

            mod_name = ''

            # Only one type of assistance allowed at a time at each DOF
            if (('H' in subset) and ('Hf' in subset) or
                ('H' in subset) and ('He' in subset) or
                ('Hf' in subset) and ('He' in subset)):
                continue

            if (('K' in subset) and ('Kf' in subset) or
                ('K' in subset) and ('Ke' in subset) or
                ('Kf' in subset) and ('Ke' in subset)):
                continue

            if (('A' in subset) and ('Ap' in subset) or
                ('A' in subset) and ('Ap' in subset) or
                ('Ap' in subset) and ('Ad' in subset)):
                continue

            device_dofs = list()
            if 'H' in subset:
                device_dofs.append('hip')
                mod_name = 'deviceH'
            elif 'Hf' in subset:
                device_dofs.append('hip/flex')
                mod_name = 'deviceHf'
            elif 'He' in subset:
                device_dofs.append('hip/ext')
                mod_name = 'deviceHe'

            if 'K' in subset:
                device_dofs.append('knee')
                if not mod_name:
                    mod_name = 'deviceK'
                else:
                    mod_name += 'K'
            elif 'Kf' in subset:
                device_dofs.append('knee/flex')
                if not mod_name:
                    mod_name = 'deviceKf'
                else:
                    mod_name += 'Kf'
            elif 'Ke' in subset:
                device_dofs.append('knee/ext')
                if not mod_name:
                    mod_name = 'deviceKe'
                else:
                    mod_name += 'Ke'

            if 'A' in subset:
                device_dofs.append('ankle')
                if not mod_name:
                    mod_name = 'deviceA'
                else:
                    mod_name += 'A'
            elif 'Ap' in subset:
                device_dofs.append('ankle/plantar')
                if not mod_name:
                    mod_name = 'deviceAp'
                else:
                    mod_name += 'Ap'
            elif 'Ad' in subset:
                device_dofs.append('ankle/dorsi')
                if not mod_name:
                    mod_name = 'deviceAd'
                else:
                    mod_name += 'Ad'

            if len(device_dofs)==0:
                deviceDOFs = ''
            elif len(device_dofs)==1:
                deviceDOFs = "'%s'" % device_dofs[0]
            elif len(device_dofs)==2:
                deviceDOFs = "'%s','%s'" % (device_dofs[0], device_dofs[1])
            elif len(device_dofs)==3:
                deviceDOFs = "'%s','%s','%s'" % (device_dofs[0], device_dofs[1],
                    device_dofs[2])

            mod_names.append(mod_name)
            deviceDOFs_list.append(deviceDOFs)

    return mod_names, deviceDOFs_list

def generate_main_tasks(trial):

    # inverse kinematics
    ik_setup_task = trial.add_task(osp.TaskIKSetup)
    trial.add_task(osp.TaskIK, ik_setup_task)
    trial.add_task(osp.TaskIKPost, ik_setup_task, 
        error_markers=trial.study.error_markers)

    # inverse dynamics
    id_setup_task = trial.add_task(osp.TaskIDSetup, ik_setup_task)
    trial.add_task(osp.TaskID, id_setup_task)

    trial.add_task(osp.TaskIDPost, id_setup_task)

    # muscle redundancy solver
    mrs_setup_tasks = trial.add_task_cycles(tasks.TaskMRSDeGrooteSetup,
        trial.study.param_dict,
        cost=trial.study.costFunction,
        use_filtered_id_results=False,
        actdyn=trial.study.actdyn)
    trial.add_task_cycles(osp.TaskMRSDeGroote, 
        setup_tasks=mrs_setup_tasks)
    trial.add_task_cycles(osp.TaskMRSDeGrootePost,
        setup_tasks=mrs_setup_tasks)

    return mrs_setup_tasks

def generate_exotopology_tasks(trial, mrs_setup_tasks):

    device_dofs = get_device_info(trial.study)[0]
    mod_names, deviceDOFs_list = get_exotopology_flags(trial.study)

    for mod_name, deviceDOFs in itertools.izip(mod_names, deviceDOFs_list):

        # Optimized torque control profiles
        mrsflags = [
            "study='SoftExosuitDesign/Topology'",
            "deviceDOFs={%s}" % deviceDOFs,
            "fixMomentArms=[]" ,
            ]

        mrsmod_opt_tasks = trial.add_task_cycles(
            tasks.TaskMRSDeGrooteMod,
            'mrsmod_%s' % mod_name,
            'ExoTopology: multiarticular device optimization',
            mrsflags,
            setup_tasks=mrs_setup_tasks
            )

        trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
            setup_tasks=mrsmod_opt_tasks)

        if trial.study.fixMomentArms:
            # Optimized torque control profiles
            mrsflags = [
                "study='SoftExosuitDesign/Topology'",
                "deviceDOFs={%s}" % deviceDOFs,
                "fixMomentArms=1.0",
                ]

            mrsmod_fixed_opt_tasks = trial.add_task_cycles(
                tasks.TaskMRSDeGrooteMod,
                'mrsmod_%s_fixed' % mod_name,
                'ExoTopology: multiarticular device optimization',
                mrsflags,
                setup_tasks=mrs_setup_tasks
                )

            trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
                setup_tasks=mrsmod_fixed_opt_tasks)

def get_mult_controls_mod_names(study):

    device_dofs = get_device_info(study)[0]
    mod_names, deviceDOFs_list = get_exotopology_flags(study)

    mult_controls_mod_names = list()
    for mod_name in mod_names:
        mult_controls_mod_names.append('%s_multControls' % mod_name)

    return mult_controls_mod_names

def generate_mult_controls_tasks(trial, mrs_setup_tasks):

    device_dofs = get_device_info(trial.study)[0]
    mod_names, deviceDOFs_list = get_exotopology_flags(trial.study)

    for mod_name, deviceDOFs in itertools.izip(mod_names, deviceDOFs_list):

        mrsflags = [
            "study='SoftExosuitDesign/Topology'",
            "deviceDOFs={%s}" % deviceDOFs,
            "fixMomentArms=1.0",
            "mult_controls=true",
            ]

        mrsmod_tasks = trial.add_task_cycles(
            tasks.TaskMRSDeGrooteMod,
            'mrsmod_%s_multControls' % mod_name,
            'ExoTopology: multiarticular device optimization',
            mrsflags,
            setup_tasks=mrs_setup_tasks
            )

        mrsmod_post_tasks = trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
            setup_tasks=mrsmod_tasks)