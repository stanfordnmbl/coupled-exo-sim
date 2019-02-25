import itertools
import osimpipeline as osp
import tasks
import pandas as pd
import os

def get_device_info(study):

    momentArms=study.momentArms
    whichDevices=study.whichDevices
    # fixMomentArms (bool)
    # Flag whether or not to include fixed moment arm tasks
    fixMomentArms=study.fixMomentArms

    # Can moment arms be optimized in either direction or are they restricted
    # to the anterior or posterior side of a joint?
    if momentArms == 'free':
        act_dofs_ref = ['actH','actK','actA']
        pass_dofs_ref = ['passH','passK','passA']
        # Can't fix moment arms in this case, so override user input
        if not fixMomentArms == True:
            Exception("Moment arms cannot be fixed when optimizing in both "
                "directions across a DOF. Setting flag to free moment arm "
                " optimization.")

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

    return device_dofs, act_dofs_ref, pass_dofs_ref

def get_exotopology_flags(study, act_combo=None, pass_combo=None):

    device_dofs, act_dofs_ref, pass_dofs_ref = get_device_info(study)

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

            mod_names.append(mod_name)
            activeDOFs_list.append(activeDOFs)
            passiveDOFs_list.append(passiveDOFs)
            subcases.append(subcase)


    return mod_names, activeDOFs_list, passiveDOFs_list, subcases

def generate_main_tasks(trial):

    # inverse kinematics
    ik_setup_task = trial.add_task(osp.TaskIKSetup)
    trial.add_task(osp.TaskIK, ik_setup_task)
    trial.add_task(osp.TaskIKPost, ik_setup_task, 
        error_markers=trial.study.error_markers)

    # inverse dynamics
    id_setup_task = trial.add_task(osp.TaskIDSetup, ik_setup_task)
    trial.add_task(osp.TaskID, id_setup_task)

    if (trial.subject.name == 'subject04') and (trial.condition.name == 'walk1'):
        # Create Butterworth filter to process net joint moments from ID
        from scipy.signal import butter
        import math     
        grf_sample_freq = 2160 # Hz
        cutoff = 65 # Hz
        nyq = 0.5 * grf_sample_freq
        norm_cutoff = cutoff / nyq
        b, a = butter(6, norm_cutoff, btype='low', analog=False)
        butter_polys = (b, a)

        trial.add_task(osp.TaskIDPost, id_setup_task, 
            butter_polys=butter_polys)

        use_filtered_id_results = True
    else:
        trial.add_task(osp.TaskIDPost, id_setup_task)
        use_filtered_id_results = False

    # muscle redundancy solver w/ generic MT parameters
    # mrs_genericMT_setup_tasks = trial.add_task_cycles(
    #     tasks.TaskMRSDeGrooteGenericMTParamsSetup,
    #     cost=trial.study.costFunction)
    # trial.add_task_cycles(osp.TaskMRSDeGroote, 
    #     setup_tasks=mrs_genericMT_setup_tasks)
    # trial.add_task_cycles(osp.TaskMRSDeGrootePost,
    #     setup_tasks=mrs_genericMT_setup_tasks)

    # parameter calibration
    calibrate_setup_tasks = trial.add_task_cycles(
        tasks.TaskCalibrateParametersSetup, 
        trial.study.param_dict, 
        trial.study.cost_dict,
        passive_precalibrate=True)
    trial.add_task_cycles(tasks.TaskCalibrateParameters, 
        setup_tasks=calibrate_setup_tasks)
    trial.add_task_cycles(tasks.TaskCalibrateParametersPost,
        setup_tasks=calibrate_setup_tasks)

    # muscle redundancy solver
    mrs_setup_tasks = trial.add_task_cycles(tasks.TaskMRSDeGrooteSetup,
        trial.study.param_dict,
        cost=trial.study.costFunction,
        use_filtered_id_results=use_filtered_id_results)
    trial.add_task_cycles(osp.TaskMRSDeGroote, 
        setup_tasks=mrs_setup_tasks)
    trial.add_task_cycles(osp.TaskMRSDeGrootePost,
        setup_tasks=mrs_setup_tasks)

    return mrs_setup_tasks

def generate_exotopology_tasks(trial, mrs_setup_tasks):

    device_dofs = get_device_info(trial.study)[0]
    mod_names, activeDOFs_list, passiveDOFs_list, subcases = \
        get_exotopology_flags(trial.study)

    for mod_name, activeDOFs, passiveDOFs, subcase in  \
        itertools.izip(mod_names, 
            activeDOFs_list, passiveDOFs_list, subcases):

        # Optimized torque control profiles
        mrsflags = [
            "study='SoftExosuitDesign/Topology'",
            "activeDOFs={%s}" % activeDOFs,
            "passiveDOFs={%s}" % passiveDOFs,
            "subcase='%s'" % subcase,
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

        # Tasks to fit the parameterize curve from Zhang et al. 2017 to
        # optimized exoskeleton torque profiles
        param_info = dict()
        param_info['min_param'] = 4
        param_info['max_param'] = 4
        fitopt_setup_tasks = trial.add_task_cycles(
            tasks.TaskFitOptimizedExoSetup,
            param_info,
            'zhang2017', 
            setup_tasks=mrsmod_opt_tasks,
            )
        trial.add_task_cycles(tasks.TaskFitOptimizedExo,
            setup_tasks=fitopt_setup_tasks)

        # Re-optimize fitted, parameterized torques.
        fitreopt_mod_name = 'fitreopt_zhang2017_%s' % mod_name
        mrsflags = [
            "study='SoftExosuitDesign/Topology'",
            "activeDOFs={%s}" % activeDOFs,
            "passiveDOFs={}",
            "subcase='%s'" % 'ActParam',
            "fixMomentArms=1.0",
            ]

        mrsmod_tasks = trial.add_task_cycles(
            tasks.TaskMRSDeGrooteMod,
            fitreopt_mod_name,
            'ExoTopology: multiarticular device optimization',
            mrsflags,
            setup_tasks=mrs_setup_tasks
            )

        mrsmod_post_tasks = trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
            setup_tasks=mrsmod_tasks)

        # Re-optimize fitted, parameterized torque, while holding some
        # parameters constant
        analysis_path = trial.study.config['analysis_path']
        param_mean_fpath = os.path.join(analysis_path,
            'fitreopt_all', 'fitreopt_all_torque_parameter_mean.csv')

        
        if os.path.exists(param_mean_fpath):
            df_mean_params = pd.read_csv(param_mean_fpath, index_col=0, 
                skiprows=1)
        
            if fitreopt_mod_name in df_mean_params.columns:
                mod_params = df_mean_params[fitreopt_mod_name]
                from numpy import isnan

                # Fix individual parameters
                for param in mod_params.index:
                    if not isnan(mod_params[param]): 
                        mrsflags = [
                            "study='SoftExosuitDesign/Topology'",
                            "activeDOFs={%s}" % activeDOFs,
                            "passiveDOFs={}",
                            "subcase='%s'" % 'ActParam',
                            "fixMomentArms=1.0",
                            "fixParams.%s=%s" % (param.replace(' ','_'), 
                                mod_params[param]),
                           ]

                        mrsmod_tasks = trial.add_task_cycles(
                            tasks.TaskMRSDeGrooteMod,
                            fitreopt_mod_name + '/fix_%s' % 
                                param.replace(' ', '_'),
                            'ExoTopology: multiarticular device optimization',
                            mrsflags,
                            setup_tasks=mrs_setup_tasks
                            )

                        mrsmod_post_tasks = trial.add_task_cycles(
                            tasks.TaskMRSDeGrooteModPost,
                            setup_tasks=mrsmod_tasks)

                # Fix all torque and scale parameters
                mrsflags = [
                    "study='SoftExosuitDesign/Topology'",
                    "activeDOFs={%s}" % activeDOFs,
                    "passiveDOFs={}",
                    "subcase='%s'" % 'ActParam',
                    "fixMomentArms=1.0",
                   ]

                for param in mod_params.index:
                    if ('torque' in param) and not isnan(mod_params[param]):
                        mrsflags.append("fixParams.%s=%s" % 
                            (param.replace(' ','_'), mod_params[param]))

                mrsmod_tasks = trial.add_task_cycles(
                    tasks.TaskMRSDeGrooteMod,
                    fitreopt_mod_name + '/fix_all_torques',
                    'ExoTopology: multiarticular device optimization',
                    mrsflags,
                    setup_tasks=mrs_setup_tasks
                    )

                mrsmod_post_tasks = trial.add_task_cycles(
                    tasks.TaskMRSDeGrooteModPost,
                    setup_tasks=mrsmod_tasks)

                # Fix all time parameters
                mrsflags = [
                    "study='SoftExosuitDesign/Topology'",
                    "activeDOFs={%s}" % activeDOFs,
                    "passiveDOFs={}",
                    "subcase='%s'" % 'ActParam',
                    "fixMomentArms=1.0",
                   ]

                for param in mod_params.index:
                    if ('time' in param) and not isnan(mod_params[param]):
                        mrsflags.append("fixParams.%s=%s" % 
                            (param.replace(' ','_'), mod_params[param]))

                mrsmod_tasks = trial.add_task_cycles(
                    tasks.TaskMRSDeGrooteMod,
                    fitreopt_mod_name + '/fix_all_times',
                    'ExoTopology: multiarticular device optimization',
                    mrsflags,
                    setup_tasks=mrs_setup_tasks
                    )

                mrsmod_post_tasks = trial.add_task_cycles(
                    tasks.TaskMRSDeGrooteModPost,
                    setup_tasks=mrsmod_tasks)

        # Optimized torque control profiles, shifted based on net joint moments
        # mrsflags = [
        #     "study='SoftExosuitDesign/Topology'",
        #     "activeDOFs={%s}" % activeDOFs,
        #     "passiveDOFs={%s}" % passiveDOFs,
        #     "subcase='%s'" % subcase,
        #     "fixMomentArms=[]",
        #     "shift_exo_peaks=true"
        #     ]

        # mrsmod_shift_opt_tasks = trial.add_task_cycles(
        #     tasks.TaskMRSDeGrooteMod,
        #     'mrsmod_%s_shift' % mod_name,
        #     'ExoTopology: multiarticular device optimization',
        #     mrsflags,
        #     setup_tasks=mrs_setup_tasks
        #     )

        # trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
        #     setup_tasks=mrsmod_shift_opt_tasks)

        # Experimental torque control profile
        # mrsflags = [
        #     "study='SoftExosuitDesign/Topology'",
        #     "activeDOFs={%s}" % activeDOFs,
        #     "passiveDOFs={%s}" % passiveDOFs,
        #     "subcase='%s'" % 'Exp',
        #     "fixMomentArms=[]" ,
        #     ]

        # mrsmod_exp_tasks = trial.add_task_cycles(
        #     tasks.TaskMRSDeGrooteMod,
        #     'mrsmod_%s' % mod_name.replace('act','exp'),
        #     'ExoTopology: multiarticular device optimization',
        #     mrsflags,
        #     setup_tasks=mrs_setup_tasks
        #     )

        # trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
        #     setup_tasks=mrsmod_exp_tasks)

        if trial.study.fixMomentArms:

            # Optimized torque control profiles
            mrsflags = [
                "study='SoftExosuitDesign/Topology'",
                "activeDOFs={%s}" % activeDOFs,
                "passiveDOFs={%s}" % passiveDOFs,
                "subcase='%s'" % subcase,
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

            # Tasks to fit the parameterize curve from Zhang et al. 2017 to
            # optimized exoskeleton torque profiles
            param_info = dict()
            param_info['min_param'] = 4
            param_info['max_param'] = 4
            fitopt_setup_tasks = trial.add_task_cycles(
                tasks.TaskFitOptimizedExoSetup,
                param_info,
                'zhang2017', 
                setup_tasks=mrsmod_fixed_opt_tasks,
                )
            trial.add_task_cycles(tasks.TaskFitOptimizedExo,
                setup_tasks=fitopt_setup_tasks)

            # Re-optimize fitted, parametered torques.
            fitreopt_fixed_mod_name = 'fitreopt_zhang2017_%s_fixed' % mod_name
            mrsflags = [
                "study='SoftExosuitDesign/Topology'",
                "activeDOFs={%s}" % activeDOFs,
                "passiveDOFs={}",
                "subcase='%s'" % 'ActParam',
                "fixMomentArms=1.0",
                ]

            mrsmod_tasks = trial.add_task_cycles(
                tasks.TaskMRSDeGrooteMod,
                fitreopt_fixed_mod_name,
                'ExoTopology: multiarticular device optimization',
                mrsflags,
                setup_tasks=mrs_setup_tasks
                )

            mrsmod_post_tasks = trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
                setup_tasks=mrsmod_tasks)

            if os.path.exists(param_mean_fpath):
                if fitreopt_fixed_mod_name in df_mean_params.columns:
                    mod_params = df_mean_params[fitreopt_fixed_mod_name]
                    from numpy import isnan

                    # Fix individual parameters
                    for param in mod_params.index:
                        if not isnan(mod_params[param]): 
                            mrsflags = [
                                "study='SoftExosuitDesign/Topology'",
                                "activeDOFs={%s}" % activeDOFs,
                                "passiveDOFs={}",
                                "subcase='%s'" % 'ActParam',
                                "fixMomentArms=1.0",
                                "fixParams.%s=%s" % (param.replace(' ', '_'), 
                                    mod_params[param]),
                               ]

                            mrsmod_tasks = trial.add_task_cycles(
                                tasks.TaskMRSDeGrooteMod,
                                fitreopt_fixed_mod_name + '/fix_%s' %
                                     param.replace(' ', '_'),
                                'ExoTopology: multiarticular device optimization',
                                mrsflags,
                                setup_tasks=mrs_setup_tasks
                                )

                            mrsmod_post_tasks = trial.add_task_cycles(
                                tasks.TaskMRSDeGrooteModPost,
                                setup_tasks=mrsmod_tasks)

                    # Fix all torque and scale parameters
                    mrsflags = [
                        "study='SoftExosuitDesign/Topology'",
                        "activeDOFs={%s}" % activeDOFs,
                        "passiveDOFs={}",
                        "subcase='%s'" % 'ActParam',
                        "fixMomentArms=1.0",
                       ]

                    for param in mod_params.index:
                        if ('torque' in param) and not isnan(mod_params[param]):
                            mrsflags.append("fixParams.%s=%s" % 
                                (param.replace(' ','_'), mod_params[param]))

                    mrsmod_tasks = trial.add_task_cycles(
                        tasks.TaskMRSDeGrooteMod,
                        fitreopt_fixed_mod_name + '/fix_all_torques',
                        'ExoTopology: multiarticular device optimization',
                        mrsflags,
                        setup_tasks=mrs_setup_tasks
                        )

                    mrsmod_post_tasks = trial.add_task_cycles(
                        tasks.TaskMRSDeGrooteModPost,
                        setup_tasks=mrsmod_tasks)

                    # Fix all time parameters
                    mrsflags = [
                        "study='SoftExosuitDesign/Topology'",
                        "activeDOFs={%s}" % activeDOFs,
                        "passiveDOFs={}",
                        "subcase='%s'" % 'ActParam',
                        "fixMomentArms=1.0",
                       ]

                    for param in mod_params.index:
                        if ('time' in param) and not isnan(mod_params[param]):
                            mrsflags.append("fixParams.%s=%s" % 
                                (param.replace(' ','_'), mod_params[param]))

                    mrsmod_tasks = trial.add_task_cycles(
                        tasks.TaskMRSDeGrooteMod,
                        fitreopt_fixed_mod_name + '/fix_all_times',
                        'ExoTopology: multiarticular device optimization',
                        mrsflags,
                        setup_tasks=mrs_setup_tasks
                        )

                    mrsmod_post_tasks = trial.add_task_cycles(
                        tasks.TaskMRSDeGrooteModPost,
                        setup_tasks=mrsmod_tasks)

            # # Tasks to fit Hermite-Simpson parameterized curves to optimized 
            # # exoskeleton torque profiles
            # param_info = dict()
            # param_info['min_param'] = 2
            # param_info['max_param'] = 8
            # fitopt_setup_tasks = trial.add_task_cycles(
            #     tasks.TaskFitOptimizedExoSetup,
            #     param_info,
            #     'hermite', 
            #     setup_tasks=mrsmod_fixed_opt_tasks,
            #     )
            # trial.add_task_cycles(tasks.TaskFitOptimizedExo,
            #     setup_tasks=fitopt_setup_tasks)


            # mrsflags = [
            #     "study='SoftExosuitDesign/Topology'",
            #     "activeDOFs={%s}" % activeDOFs,
            #     "passiveDOFs={%s}" % passiveDOFs,
            #     "subcase='%s'" % 'FitOpt',
            #     "fixMomentArms=1.0",
            #     ]

            # for param_num in range(param_info['min_param'], 
            #                        param_info['max_param']+1):
            #     fitopt_mrs_setup_tasks = trial.add_task_cycles(
            #         tasks.TaskMRSFitOptimizedExoSetup,
            #         'fitopt_hermite_%s_%s_fixed' % (str(param_num), mod_name),
            #         param_num,
            #         mrsflags,
            #         setup_tasks=fitopt_setup_tasks,
            #         )
            #     trial.add_task_cycles(tasks.TaskMRSFitOptimizedExo,
            #         setup_tasks=fitopt_mrs_setup_tasks)
            #     trial.add_task_cycles(tasks.TaskMRSFitOptimizedExoPost,
            #          setup_tasks=fitopt_mrs_setup_tasks)

            
            # mrsflags = [
            #     "study='SoftExosuitDesign/Topology'",
            #     "activeDOFs={%s}" % activeDOFs,
            #     "passiveDOFs={%s}" % passiveDOFs,
            #     "subcase='%s'" % 'FitOpt',
            #     "fixMomentArms=1.0",
            #     ]

            # for param_num in range(param_info['min_param'], 
            #                        param_info['max_param']+1):
            #     fitopt_mrs_setup_tasks = trial.add_task_cycles(
            #         tasks.TaskMRSFitOptimizedExoSetup,
            #         'fitopt_zhang2017_%s_%s_fixed' % (str(param_num), mod_name),
            #         param_num,
            #         mrsflags,
            #         setup_tasks=fitopt_setup_tasks,
            #         )
            #     trial.add_task_cycles(tasks.TaskMRSFitOptimizedExo,
            #         setup_tasks=fitopt_mrs_setup_tasks)
            #     trial.add_task_cycles(tasks.TaskMRSFitOptimizedExoPost,
            #          setup_tasks=fitopt_mrs_setup_tasks)

            # # Tasks to fit Legendre polynomial parameterized curves to optimized 
            # # exoskeleton torque profiles
            # param_info = dict()
            # param_info['min_param'] = 4
            # param_info['max_param'] = 10
            # fitopt_setup_tasks = trial.add_task_cycles(
            #     tasks.TaskFitOptimizedExoSetup,
            #     param_info,
            #     'legendre', 
            #     setup_tasks=mrsmod_fixed_opt_tasks,
            #     )
            # trial.add_task_cycles(tasks.TaskFitOptimizedExo,
            #     setup_tasks=fitopt_setup_tasks)

            # mrsflags = [
            #     "study='SoftExosuitDesign/Topology'",
            #     "activeDOFs={%s}" % activeDOFs,
            #     "passiveDOFs={%s}" % passiveDOFs,
            #     "subcase='%s'" % 'FitOpt',
            #     "fixMomentArms=1.0",
            #     ]

            # for param_num in range(param_info['min_param'], 
            #                        param_info['max_param']+1):
            #     fitopt_mrs_setup_tasks = trial.add_task_cycles(
            #         tasks.TaskMRSFitOptimizedExoSetup,
            #         'fitopt_legendre_%s_%s_fixed' % (str(param_num), mod_name),
            #         param_num,
            #         mrsflags,
            #         setup_tasks=fitopt_setup_tasks,
            #         )
            #     trial.add_task_cycles(tasks.TaskMRSFitOptimizedExo,
            #         setup_tasks=fitopt_mrs_setup_tasks)
            #     trial.add_task_cycles(tasks.TaskMRSFitOptimizedExoPost,
            #          setup_tasks=fitopt_mrs_setup_tasks)

            # Experimental torque control profile
            # mrsflags = [
            #     "study='SoftExosuitDesign/Topology'",
            #     "activeDOFs={%s}" % activeDOFs,
            #     "passiveDOFs={%s}" % passiveDOFs,
            #     "subcase='%s'" % 'Exp',
            #     "fixMomentArms=1.0" ,
            #     ]

            # mrsmod_fixed_exp_tasks = trial.add_task_cycles(
            #     tasks.TaskMRSDeGrooteMod,
            #     'mrsmod_%s_fixed' % mod_name.replace('act','exp'),
            #     'ExoTopology: multiarticular device optimization',
            #     mrsflags,
            #     setup_tasks=mrs_setup_tasks
            #     )

            # trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
            #     setup_tasks=mrsmod_fixed_exp_tasks)

def get_mult_controls_mod_names(study):

    device_dofs = get_device_info(study)[0]
    mod_names, activeDOFs_list, passiveDOFs_list, subcases = \
        get_exotopology_flags(study)

    mult_controls_mod_names = list()
    for mod_name in mod_names:
        mult_controls_mod_names.append('%s_multControls' % mod_name)

    return mult_controls_mod_names

def generate_mult_controls_tasks(trial, mrs_setup_tasks):

    device_dofs = get_device_info(trial.study)[0]
    mod_names, activeDOFs_list, passiveDOFs_list, subcases = \
        get_exotopology_flags(trial.study)

    for mod_name, activeDOFs, passiveDOFs, subcase in  \
        itertools.izip(mod_names, 
            activeDOFs_list, passiveDOFs_list, subcases):

        mrsflags = [
            "study='SoftExosuitDesign/Topology'",
            "activeDOFs={%s}" % activeDOFs,
            "passiveDOFs={}",
            "subcase='%s'" % subcase,
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

        # Tasks to fit the parameterize curve from Zhang et al. 2017 to
        # optimized exoskeleton torque profiles
        param_info = dict()
        param_info['min_param'] = 4
        param_info['max_param'] = 4
        fitopt_setup_tasks = trial.add_task_cycles(
            tasks.TaskFitOptimizedExoSetup,
            param_info,
            'zhang2017', 
            setup_tasks=mrsmod_tasks,
            )
        trial.add_task_cycles(tasks.TaskFitOptimizedExo,
            setup_tasks=fitopt_setup_tasks)

        # Re-optimize fitted, parametered torques.
        mrsflags = [
            "study='SoftExosuitDesign/Topology'",
            "activeDOFs={%s}" % activeDOFs,
            "passiveDOFs={}",
            "subcase='%s'" % 'ActParam',
            "fixMomentArms=1.0",
            ]

        mrsmod_tasks = trial.add_task_cycles(
            tasks.TaskMRSDeGrooteMod,
            'fitreopt_zhang2017_%s_multControls' % mod_name,
            'ExoTopology: multiarticular device optimization',
            mrsflags,
            setup_tasks=mrs_setup_tasks
            )

        mrsmod_post_tasks = trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
            setup_tasks=mrsmod_tasks)

def generate_param_controls_tasks(trial, mrs_setup_tasks):

    device_dofs = get_device_info(trial.study)[0]
    mod_names, activeDOFs_list, passiveDOFs_list, subcases = \
        get_exotopology_flags(trial.study)

    param_mods = ['actHfAp', 'actHeKe', 'actKfAp', 'actAp']

    for mod_name, activeDOFs, passiveDOFs, subcase in  \
        itertools.izip(mod_names, 
            activeDOFs_list, passiveDOFs_list, subcases):

        if not (mod_name in param_mods): continue

        if subcase == 'Act': subcase = 'ActParam'

        mrsflags = [
            "study='SoftExosuitDesign/Topology'",
            "activeDOFs={%s}" % activeDOFs,
            "passiveDOFs={}",
            "subcase='%s'" % subcase,
            ]

        mrsmod_tasks = trial.add_task_cycles(
            tasks.TaskMRSDeGrooteMod,
            'mrsmod_%s_paramControls' % mod_name,
            'ExoTopology: multiarticular device optimization',
            mrsflags,
            setup_tasks=mrs_setup_tasks
            )

        mrsmod_post_tasks = trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
            setup_tasks=mrsmod_tasks)

        # mrsflags = [
        #     "study='SoftExosuitDesign/Topology'",
        #     "activeDOFs={%s}" % activeDOFs,
        #     "passiveDOFs={}",
        #     "subcase='%s'" % subcase,
        #     ]

        # power_matches = ['avg_pos', 'avg_net']
        # for match in power_matches:
        #     pmatch_mrs_setup_tasks = trial.add_task_cycles(
        #         tasks.TaskMRSDevicePowerMatchSetup,
        #         'pmatch_%s_%s_paramControls' % (match, mod_name),
        #         match,
        #         mrsflags,
        #         setup_tasks=mrs_setup_tasks
        #         )
        #     trial.add_task_cycles(tasks.TaskMRSDevicePowerMatch,
        #         setup_tasks=pmatch_mrs_setup_tasks)
        #     trial.add_task_cycles(tasks.TaskMRSDevicePowerMatchPost,
        #         setup_tasks=pmatch_mrs_setup_tasks)



