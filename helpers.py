import itertools
import osimpipeline as osp
import tasks

def get_device_info(momentArms='free', whichDevices='all', fixMomentArms='[]'):

    # fixMomentArms
    # Set value as a string. To use default range [0.01 0.10] for moment arms
    # set to empty brackets, i.e.'[]'

    # Can moment arms be optimized in either direction or are they restricted
    # to the anterior or posterior side of a joint?
    if momentArms == 'free':
        act_dofs_ref = ['actH','actK','actA']
        pass_dofs_ref = ['passH','passK','passA']
        # Can't fix moment arms in this case, so override user input
        if not fixMomentArms == '[]':
            import warnings
            warnings.warn("Moment arms cannot be fixed when optimizing in both "
                "directions across a DOF. Setting flag to free moment arm "
                " optimization.")
            fixMomentArms = '[]' 

    elif momentArms == 'fixed_direction':
        act_dofs_ref = ['actHf','actHe','actKf','actKe','actAd','actAp']
        pass_dofs_ref = ['passHf','passHe','passKf','passKe','passAd','passAp']

    # Which set of devices to choose from?
    if whichDevices == 'active_only':
        device_dofs = act_dofs_ref

    elif whichDevices == 'passive_only':
        device_dofs = pass_dofs_ref

    elif whichDevices == 'all':
        device_dofs = act_dofs_ref + pass_dofs_ref

    return device_dofs, fixMomentArms, act_dofs_ref, pass_dofs_ref

def get_exotopology_flags(momentArms='free', whichDevices='all', 
    fixMomentArms='[]', act_combo=None, pass_combo=None):

    device_dofs, fixMomentArms, act_dofs_ref, pass_dofs_ref = get_device_info(
        momentArms=momentArms, whichDevices=whichDevices, 
        fixMomentArms=fixMomentArms)

    mod_names = list()
    activeDOFs_list = list()
    passiveDOFs_list = list()
    subcases = list()

    for L in range(1, len(device_dofs)+1):
        for subset in itertools.combinations(device_dofs, L):

            isAct = (any(word in subset for word in act_dofs_ref) or act_combo)
            isPass = (any(word in subset for word in pass_dofs_ref) 
                or pass_combo)
        
            mod_name = ''

            # Only one type of active assistance allowed at a time at each DOF
            if (('actH' in subset) and ('actHf' in subset) or
                ('actH' in subset) and ('actHe' in subset) or
                ('actHf' in subset) and ('actHe' in subset)):
                continue

            if (('actK' in subset) and ('actKf' in subset) or
                ('actK' in subset) and ('actKe' in subset) or
                ('actKf' in subset) and ('actKe' in subset)):
                continue

            if (('actA' in subset) and ('actAp' in subset) or
                ('actA' in subset) and ('actAp' in subset) or
                ('actAp' in subset) and ('actAd' in subset)):
                continue

            # Active DOFs
            act_dofs = list()
            if 'actH' in subset:
                act_dofs.append('hip')
                mod_name = 'actH'
            elif 'actHf' in subset:
                act_dofs.append('hip/flex')
                mod_name = 'actHf'
            elif 'actHe' in subset:
                act_dofs.append('hip/ext')
                mod_name = 'actHe'

            if 'actK' in subset:
                act_dofs.append('knee')
                if not mod_name:
                    mod_name = 'actK'
                else:
                    mod_name += 'K'
            elif 'actKf' in subset:
                act_dofs.append('knee/flex')
                if not mod_name:
                    mod_name = 'actKf'
                else:
                    mod_name += 'Kf'
            elif 'actKe' in subset:
                act_dofs.append('knee/ext')
                if not mod_name:
                    mod_name = 'actKe'
                else:
                    mod_name += 'Ke'

            if 'actA' in subset:
                act_dofs.append('ankle')
                if not mod_name:
                    mod_name = 'actA'
                else:
                    mod_name += 'A'
            elif 'actAp' in subset:
                act_dofs.append('ankle/plantar')
                if not mod_name:
                    mod_name = 'actAp'
                else:
                    mod_name += 'Ap'
            elif 'actAd' in subset:
                act_dofs.append('ankle/dorsi')
                if not mod_name:
                    mod_name = 'actAd'
                else:
                    mod_name += 'Ad'

            if act_combo:
                mod_name = act_combo
                if 'H' in act_combo:
                    act_dofs.append('hip')
                # elif 'Hf' in act_combo:
                #     act_dofs.append('hip/flex')
                # elif 'He' in act_combo:
                #     act_dofs.append('hip/ext')

                if 'K' in act_combo:
                    act_dofs.append('knee')
                # elif 'Kf' in act_combo:
                #     act_dofs.append('knee/flex')
                # elif 'Ke' in act_combo:
                #     act_dofs.append('knee/ext')

                if 'A' in act_combo:
                    act_dofs.append('ankle')
                # elif 'Ap' in act_combo:
                #     act_dofs.append('ankle/plantar')
                # elif 'Ad' in act_combo:
                #     act_dofs.append('ankle/dorsi')

            if len(act_dofs)==0:
                activeDOFs = ''
            elif len(act_dofs)==1:
                activeDOFs = "'%s'" % act_dofs[0]
            elif len(act_dofs)==2:
                activeDOFs = "'%s','%s'" % (act_dofs[0], act_dofs[1])
            elif len(act_dofs)==3:
                activeDOFs = "'%s','%s','%s'" % (act_dofs[0], act_dofs[1],
                    act_dofs[2])

            if isAct and isPass:
                mod_name += '_'

            # Only one type of passive assistance allowed at a time at each DOF
            if (('passH' in subset) and ('passHf' in subset) or
                ('passH' in subset) and ('passHe' in subset) or
                ('passHf' in subset) and ('passHe' in subset)):
                continue

            if (('passK' in subset) and ('passKf' in subset) or
                ('passK' in subset) and ('passKe' in subset) or
                ('passKf' in subset) and ('passKe' in subset)):
                continue

            if (('passA' in subset) and ('passAp' in subset) or
                ('passA' in subset) and ('passAp' in subset) or
                ('passAp' in subset) and ('passAd' in subset)):
                continue

            # Passive DOFs
            pass_dofs = list()
            if 'passH' in subset:
                pass_dofs.append('hip')
                mod_name += 'passH'
            elif 'passHf' in subset:
                pass_dofs.append('hip/flex')
                mod_name += 'passHf'
            elif 'passHe' in subset:
                pass_dofs.append('hip/ext')
                mod_name += 'passHe'

            if 'passK' in subset:
                pass_dofs.append('knee')
                if 'pass' not in mod_name:
                    mod_name += 'passK'
                else:
                    mod_name += 'K'
            elif 'passKf' in subset:
                pass_dofs.append('knee/flex')
                if not mod_name:
                    mod_name += 'passKf'
                else:
                    mod_name += 'Kf'
            elif 'passKe' in subset:
                pass_dofs.append('knee/ext')
                if not mod_name:
                    mod_name += 'passKe'
                else:
                    mod_name += 'Ke'

            if 'passA' in subset:
                pass_dofs.append('ankle')
                if 'pass' not in mod_name:
                    mod_name += 'passA'
                else:
                    mod_name += 'A'
            elif 'passAp' in subset:
                pass_dofs.append('ankle/plantar')
                if not mod_name:
                    mod_name += 'passAp'
                else:
                    mod_name += 'Ap'
            elif 'passAd' in subset:
                pass_dofs.append('ankle/dorsi')
                if not mod_name:
                    mod_name += 'passAd'
                else:
                    mod_name += 'Ad'

            if pass_combo:
                mod_name += pass_combo
                if 'H' in pass_combo:
                    pass_dofs.append('hip')
                # elif 'Hf' in pass_combo:
                #     pass_dofs.append('hip/flex')
                # elif 'He' in pass_combo:
                #     pass_dofs.append('hip/ext')

                if 'K' in pass_combo:
                    pass_dofs.append('knee')
                # elif 'Kf' in pass_combo:
                #     pass_dofs.append('knee/flex')
                # elif 'Ke' in pass_combo:
                #     pass_dofs.append('knee/ext')

                if 'A' in pass_combo:
                    pass_dofs.append('ankle')
                # elif 'Ap' in pass_combo:
                #     pass_dofs.append('ankle/plantar')
                # elif 'Ad' in pass_combo:
                #     pass_dofs.append('ankle/dorsi')

            if len(pass_dofs)==0:
                passiveDOFs = ''
            elif len(pass_dofs)==1:
                passiveDOFs = "'%s'" % pass_dofs[0]
            elif len(pass_dofs)==2:
                passiveDOFs = "'%s','%s'" % (pass_dofs[0], pass_dofs[1])
            elif len(pass_dofs)==3:
                passiveDOFs = "'%s','%s','%s'" % (pass_dofs[0], pass_dofs[1],
                    pass_dofs[2])

            subcase = ''
            if (any(word in subset for word in act_dofs_ref) or act_combo):
                subcase += 'Act'
            if (any(word in subset for word in pass_dofs_ref) or pass_combo):
                subcase += 'Pass'

            if not fixMomentArms == '[]':
                mod_name += '_fixed'

            mod_names.append(mod_name)
            activeDOFs_list.append(activeDOFs)
            passiveDOFs_list.append(passiveDOFs)
            subcases.append(subcase)


    return mod_names, activeDOFs_list, passiveDOFs_list, subcases

def generate_main_tasks(trial):

    # walk2: inverse kinematics
    ik_setup_task = trial.add_task(osp.TaskIKSetup)
    trial.add_task(osp.TaskIK, ik_setup_task)
    trial.add_task(osp.TaskIKPost, ik_setup_task, 
        error_markers=trial.study.error_markers)

    # walk2: inverse dynamics
    id_setup_task = trial.add_task(osp.TaskIDSetup, ik_setup_task)
    trial.add_task(osp.TaskID, id_setup_task)
    trial.add_task(osp.TaskIDPost, id_setup_task)

    # walk2: muscle redundancy solver w/ generic MT parameters
    # mrs_genericMT_setup_tasks = trial.add_task_cycles(
    #     tasks.TaskMRSDeGrooteGenericMTParamsSetup,
    #     cost=trial.study.costFunction)
    # trial.add_task_cycles(osp.TaskMRSDeGroote, 
    #     setup_tasks=mrs_genericMT_setup_tasks)
    # trial.add_task_cycles(osp.TaskMRSDeGrootePost,
    #     setup_tasks=mrs_genericMT_setup_tasks)

    # walk2: parameter calibration
    calibrate_setup_tasks = trial.add_task_cycles(
        tasks.TaskCalibrateParametersSetup, 
        trial.study.param_dict, 
        trial.study.cost_dict)
    trial.add_task_cycles(tasks.TaskCalibrateParameters, 
        setup_tasks=calibrate_setup_tasks)
    trial.add_task_cycles(tasks.TaskCalibrateParametersPost,
        setup_tasks=calibrate_setup_tasks)

    # walk2: muscle redundancy solver
    mrs_setup_tasks = trial.add_task_cycles(tasks.TaskMRSDeGrooteSetup,
        trial.study.param_dict,
        cost=trial.study.costFunction)
    trial.add_task_cycles(osp.TaskMRSDeGroote, 
        setup_tasks=mrs_setup_tasks)
    trial.add_task_cycles(osp.TaskMRSDeGrootePost,
        setup_tasks=mrs_setup_tasks)

    return mrs_setup_tasks

def generate_exotopology_tasks(trial, mrs_setup_tasks):

    device_dofs, fixMomentArms = get_device_info(
        momentArms=trial.study.momentArms,
        whichDevices=trial.study.whichDevices,
        fixMomentArms=trial.study.fixMomentArms)[:2]

    mod_names, activeDOFs_list, passiveDOFs_list, subcases = \
        get_exotopology_flags(momentArms=trial.study.momentArms,
        whichDevices=trial.study.whichDevices,
        fixMomentArms=trial.study.fixMomentArms)

    for mod_name, activeDOFs, passiveDOFs, subcase in  \
        itertools.izip(mod_names, 
            activeDOFs_list, passiveDOFs_list, subcases):

        mrsflags = [
            "study='SoftExosuitDesign/Topology'",
            "activeDOFs={%s}" % activeDOFs,
            "passiveDOFs={%s}" % passiveDOFs,
            "subcase='%s'" % subcase,
            "fixMomentArms=%s" % fixMomentArms,
            ]

        mrsmod_tasks = trial.add_task_cycles(
            tasks.TaskMRSDeGrooteMod,
            mod_name,
            'ExoTopology: multiarticular device optimization',
            mrsflags,
            setup_tasks=mrs_setup_tasks
            )

        trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
            setup_tasks=mrsmod_tasks)

def generate_HfAp_tasks(trial, mrs_setup_tasks):

    # "scaled-ID" assistive ankle-hip strategy
    mrsflags = [
        "study='ISB2017/Quinlivan2017'",
        "shift_exo_peaks=true",
        ]

    mrsmod_tasks = trial.add_task_cycles(
        osp.TaskMRSDeGrooteMod,
        'actHfAp_scaledID',
        'ExoTopology: "scaled-ID" assistive ankle-hip strategy',
        mrsflags,
        setup_tasks=mrs_setup_tasks
        )

    trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
        setup_tasks=mrsmod_tasks)

    # Max reduction case from experiment, assistive ankle-hip strategy
    mrsflags = [
        "study='ISB2017/Quinlivan2017'",
        "shift_exo_peaks=true",
        "exo_force_level=4",
        ]

    mrsmod_tasks = trial.add_task_cycles(
        tasks.TaskMRSDeGrooteMod,
        'actHfAp_exp',
        'ExoTopology: "scaled-ID" assistive ankle-hip strategy',
        mrsflags,
        setup_tasks=mrs_setup_tasks
        )

    trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
        setup_tasks=mrsmod_tasks)

    # multiple control solution, hip-ankle strategy, free moment arms
    mrsflags = [
        "study='SoftExosuitDesign/Topology'",
        "activeDOFs={'hip/flex', 'ankle/plantar'}",
        "passiveDOFs={}",
        "subcase='Act'",
        "fixMomentArms=[]",
        "mult_controls=true",
        ]

def get_mult_controls_mod_names(study):

    device_dofs, fixMomentArms = get_device_info(
        momentArms=study.momentArms,
        whichDevices=study.whichDevices,
        fixMomentArms=study.fixMomentArms)[:2]

    mod_names, activeDOFs_list, passiveDOFs_list, subcases = \
        get_exotopology_flags(momentArms=study.momentArms,
        whichDevices=study.whichDevices,
        fixMomentArms=study.fixMomentArms)

    mult_controls_mod_names = list()

    for mod_name in mod_names:
        mult_controls_mod_names.append('%s_multControls' % mod_name)

    return mult_controls_mod_names

def generate_mult_controls_tasks(trial, mrs_setup_tasks):

    device_dofs, fixMomentArms = get_device_info(
        momentArms=trial.study.momentArms,
        whichDevices=trial.study.whichDevices,
        fixMomentArms=trial.study.fixMomentArms)[:2]

    mod_names, activeDOFs_list, passiveDOFs_list, subcases = \
        get_exotopology_flags(momentArms=trial.study.momentArms,
        whichDevices=trial.study.whichDevices,
        fixMomentArms=trial.study.fixMomentArms)

    for mod_name, activeDOFs, passiveDOFs, subcase in  \
        itertools.izip(mod_names, 
            activeDOFs_list, passiveDOFs_list, subcases):

        mrsflags = [
            "study='SoftExosuitDesign/Topology'",
            "activeDOFs={%s}" % activeDOFs,
            "passiveDOFs={}",
            "subcase='%s'" % subcase,
            "fixMomentArms=[]",
            "mult_controls=true",
            ]

        mrsmod_tasks = trial.add_task_cycles(
            tasks.TaskMRSDeGrooteMod,
            '%s_multControls' % mod_name,
            'ExoTopology: multiarticular device optimization',
            mrsflags,
            setup_tasks=mrs_setup_tasks
            )

        mrsmod_post_tasks = trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
            setup_tasks=mrsmod_tasks)

def generate_param_controls_tasks(trial, mrs_setup_tasks):

    device_dofs, fixMomentArms = get_device_info(
        momentArms=trial.study.momentArms,
        whichDevices=trial.study.whichDevices,
        fixMomentArms=trial.study.fixMomentArms)[:2]

    mod_names, activeDOFs_list, passiveDOFs_list, subcases = \
        get_exotopology_flags(momentArms=trial.study.momentArms,
        whichDevices=trial.study.whichDevices,
        fixMomentArms=trial.study.fixMomentArms)


    param_mods = ['actHfAp', 'actHeKe', 'actKfAp']

    for mod_name, activeDOFs, passiveDOFs, subcase in  \
        itertools.izip(mod_names, 
            activeDOFs_list, passiveDOFs_list, subcases):

        if not (mod_name in param_mods): continue

        if subcase == 'Act':
            subcase = 'ActParam'

        mrsflags = [
            "study='SoftExosuitDesign/Topology'",
            "activeDOFs={%s}" % activeDOFs,
            "passiveDOFs={}",
            "subcase='%s'" % subcase,
            "fixMomentArms=[]",
            ]

        mrsmod_tasks = trial.add_task_cycles(
            tasks.TaskMRSDeGrooteMod,
            '%s_paramControls' % mod_name,
            'ExoTopology: multiarticular device optimization',
            mrsflags,
            setup_tasks=mrs_setup_tasks
            )

        mrsmod_post_tasks = trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
            setup_tasks=mrsmod_tasks)