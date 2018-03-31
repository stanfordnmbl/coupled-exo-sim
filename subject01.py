import os

import osimpipeline as osp
import tasks
import helpers

def scale_setup_fcn(util, mset, sset, ikts):
    m = util.Measurement('torso', mset)
    m.add_markerpair('RASI', 'CLAV')
    m.add_markerpair('LASI', 'CLAV')
    m.add_markerpair('LPSI', 'C7')
    m.add_markerpair('RPSI', 'C7')
    m.add_markerpair('RASI',' RACR')
    m.add_markerpair('LASI', 'LACR')
    m.add_bodyscale('torso')

    m = util.Measurement('pelvis_z', mset)
    m.add_markerpair('RPSI', 'LPSI')
    m.add_markerpair('RASI', 'LASI')
    m.add_bodyscale('pelvis', 'Z')

    m = util.Measurement('thigh', mset)
    m.add_markerpair('LHJC', 'LLFC')
    m.add_markerpair('LHJC', 'LMFC')
    m.add_markerpair('RHJC', 'RMFC')
    m.add_markerpair('RHJC', 'RLFC')
    m.add_bodyscale_bilateral('femur')

    m = util.Measurement('shank', mset)
    m.add_markerpair('LLFC', 'LLMAL')
    m.add_markerpair('LMFC', 'LMMAL')
    m.add_markerpair('RLFC', 'RLMAL')
    m.add_markerpair('RMFC', 'RMMAL')
    m.add_bodyscale_bilateral('tibia')

    m = util.Measurement('foot', mset)
    m.add_markerpair('LCAL', 'LMT5')
    m.add_markerpair('LCAL', 'LTOE')
    m.add_markerpair('RCAL', 'RTOE')
    m.add_markerpair('RCAL',' RMT5')
    m.add_bodyscale_bilateral('talus')
    m.add_bodyscale_bilateral('calcn')
    m.add_bodyscale_bilateral('toes')

    m = util.Measurement('humerus', mset)
    m.add_markerpair('LSJC', 'LMEL')
    m.add_markerpair('LSJC', 'LLEL')
    m.add_markerpair('RSJC', 'RLEL')
    m.add_markerpair('RSJC', 'RMEL')
    m.add_bodyscale_bilateral('humerus')

    m = util.Measurement('radius_ulna', mset)
    m.add_markerpair('LLEL', 'LFAradius')
    m.add_markerpair('LMEL', 'LFAulna')
    m.add_markerpair('RMEL', 'RFAulna')
    m.add_markerpair('RLEL', 'RFAradius')
    m.add_bodyscale_bilateral('ulna')
    m.add_bodyscale_bilateral('radius')
    m.add_bodyscale_bilateral('hand')

    m = util.Measurement('pelvis_Y', mset)
    m.add_markerpair('LPSI', 'LHJC')
    m.add_markerpair('RPSI', 'RHJC')
    m.add_markerpair('RASI', 'RHJC')
    m.add_markerpair('LASI', 'LHJC')
    m.add_bodyscale('pelvis', 'Y')

    m = util.Measurement('pelvis_X', mset)
    m.add_markerpair('RASI', 'RPSI')
    m.add_markerpair('LASI', 'LPSI')
    m.add_bodyscale('pelvis', 'X')

    ikts.add_ikmarkertask_bilateral('ASI', True, 15.0)
    ikts.add_ikmarkertask_bilateral('PSI', True, 15.0)
    ikts.add_ikmarkertask_bilateral('LFC', True, 10.0)
    ikts.add_ikmarkertask_bilateral('MFC', True, 10.0)
    ikts.add_ikmarkertask_bilateral('LMAL', True, 5.0)
    ikts.add_ikmarkertask_bilateral('MMAL', True, 5.0)
    ikts.add_ikmarkertask_bilateral('CAL', True, 5.0)
    ikts.add_ikmarkertask_bilateral('TOE', True, 5.0)
    ikts.add_ikmarkertask_bilateral('MT5', True, 5.0)
    ikts.add_ikmarkertask_bilateral('ACR', True, 2.0)
    ikts.add_ikmarkertask_bilateral('ASH', True, 2.0)
    ikts.add_ikmarkertask_bilateral('PSH', True, 2.0)
    ikts.add_ikmarkertask_bilateral('LEL', True, 1.0)
    ikts.add_ikmarkertask_bilateral('MEL', True, 1.0)
    ikts.add_ikmarkertask_bilateral('HJC', True, 20.0)
    ikts.add_ikmarkertask_bilateral('KJC', True, 10.0)
    ikts.add_ikmarkertask_bilateral('AJC', True, 10.0)
    ikts.add_ikmarkertask_bilateral('SJC', True, 1.0)
    ikts.add_ikmarkertask_bilateral('EJC', True, 1.0)
    ikts.add_ikmarkertask_bilateral('FAsuperior', False, 0.0)
    ikts.add_ikmarkertask_bilateral('FAradius', False, 0.0)
    ikts.add_ikmarkertask_bilateral('FAulna', False, 0.0)
    ikts.add_ikmarkertask('CLAV', True, 2.0)
    ikts.add_ikmarkertask('C7', True, 2.0)
    ikts.add_ikmarkertask_bilateral('TH1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA3', False, 0.0)

def add_to_study(study):
    subject = study.add_subject(1, 72.84)

    cond_args = dict()
    subject.cond_args = cond_args

    static = subject.add_condition('static')
    static_trial = static.add_trial(1, omit_trial_dir=True)

    # `os.path.basename(__file__)` should be `subject01.py`.
    scale_setup_task = subject.add_task(osp.TaskScaleSetup,
            init_time=0,
            final_time=0.5, 
            mocap_trial=static_trial,
            edit_setup_function=scale_setup_fcn,
            addtl_file_dep=['dodo.py', os.path.basename(__file__)])

    subject.add_task(osp.TaskScale,
            scale_setup_task=scale_setup_task,
            #scale_max_isometric_force=True,
            )

    subject.add_task(tasks.TaskScaleMuscleMaxIsometricForce)
    subject.scaled_model_fpath = os.path.join(subject.results_exp_path,
        '%s_scaled_Fmax.osim' % subject.name)

    ## walk2 condition
    walk2 = subject.add_condition('walk2', metadata={'walking_speed': 1.25})
    
    # GRF gait landmarks
    # walk2_trial_temp = walk2.add_trial(99, omit_trial_dir=True)
    # walk2_trial_temp.add_task(osp.TaskGRFGaitLandmarks)

    # Trial to use
    gait_events = dict()
    gait_events['right_strikes'] = [1.179, 2.282, 3.361, 4.488] #, 5.572]
    gait_events['left_toeooffs'] = [1.368, 2.471, 3.578] #, 4.680]
    gait_events['left_strikes'] = [1.728, 2.836, 3.943] #, 5.051]
    gait_events['right_toeoffs'] = [1.934, 3.033, 4.137] #, 5.252]

    walk2_trial = walk2.add_trial(1,
            gait_events=gait_events,
            omit_trial_dir=True,
            )
    walk2_trial.add_task(tasks.TaskUpdateGroundReactionColumnLabels)

    # walk2: inverse kinematics
    ik_setup_task = walk2_trial.add_task(osp.TaskIKSetup)
    walk2_trial.add_task(osp.TaskIK, ik_setup_task)
    walk2_trial.add_task(osp.TaskIKPost, ik_setup_task, 
        error_markers=study.error_markers)

    # walk2: inverse dynamics
    id_setup_task = walk2_trial.add_task(osp.TaskIDSetup, ik_setup_task)
    walk2_trial.add_task(osp.TaskID, id_setup_task)
    walk2_trial.add_task(osp.TaskIDPost, id_setup_task)

    # walk2: static optimization
    # so_setup_tasks = walk2_trial.add_task_cycles(osp.TaskSOSetup, ik_setup_task)
    # walk2_trial.add_task_cycles(osp.TaskSO, setup_tasks=so_setup_tasks)
    # walk2_trial.add_task_cycles(osp.TaskSOPost, setup_tasks=so_setup_tasks)

    # walk2: parameter calibration
    calibrate_setup_tasks = walk2_trial.add_task_cycles(
        tasks.TaskCalibrateParametersSetup)
    walk2_trial.add_task_cycles(tasks.TaskCalibrateParameters, 
        setup_tasks=calibrate_setup_tasks)
    walk2_trial.add_task_cycles(tasks.TaskCalibrateParametersPost,
        setup_tasks=calibrate_setup_tasks)

    # walk2: muscle redundancy solver
    mrs_setup_tasks = walk2_trial.add_task_cycles(osp.TaskMRSDeGrooteSetup,
        cost=study.costFunction)
    walk2_trial.add_task_cycles(osp.TaskMRSDeGroote, 
        setup_tasks=mrs_setup_tasks)
    walk2_trial.add_task_cycles(osp.TaskMRSDeGrootePost,
        setup_tasks=mrs_setup_tasks)

    # walk2: muscle redundancy solver Exotopology mods
    helpers.generate_exotopology_tasks(walk2_trial, mrs_setup_tasks)

    # walk2: variations on hip flexion, ankle plantarflexion tasks
    # helpers.generate_HfAp_tasks(walk2_trial, mrs_setup_tasks)

    # walk2: resolve device optimization problems w/ individual controls
    helpers.generate_mult_controls_tasks(walk2_trial, mrs_setup_tasks)

    ## walk1 condition
    walk1 = subject.add_condition('walk1', metadata={'walking_speed': 1.00})

    # GRF gait landmarks
    # walk1_trial_temp = walk1.add_trial(99, omit_trial_dir=True)
    # walk1_trial_temp.add_task(osp.TaskGRFGaitLandmarks)
    # walk1_trial_temp.add_task(tasks.TaskUpdateGroundReactionColumnLabels)
    
    gait_events = dict()
    gait_events['right_strikes'] = [0.487, 1.757, 3.020, 4.302]
    gait_events['left_toeooffs'] = [0.716, 1.980, 3.256]
    gait_events['left_strikes'] = [1.125, 2.393, 3.663]
    gait_events['right_toeoffs'] = [1.351, 2.624, 3.886]

    walk1_trial = walk1.add_trial(1,
            gait_events=gait_events,
            omit_trial_dir=True,
            )

    # walk1: inverse kinematics
    ik_setup_task = walk1_trial.add_task(osp.TaskIKSetup)
    walk1_trial.add_task(osp.TaskIK, ik_setup_task)
    walk1_trial.add_task(osp.TaskIKPost, ik_setup_task, 
        error_markers=study.error_markers)

    # walk1: inverse dynamics
    id_setup_task = walk1_trial.add_task(osp.TaskIDSetup, ik_setup_task)
    walk1_trial.add_task(osp.TaskID, id_setup_task)
    walk1_trial.add_task(osp.TaskIDPost, id_setup_task)

    # walk1: parameter calibration
    calibrate_setup_tasks = walk1_trial.add_task_cycles(
        tasks.TaskCalibrateParametersSetup)
    walk1_trial.add_task_cycles(tasks.TaskCalibrateParameters, 
        setup_tasks=calibrate_setup_tasks)
    walk1_trial.add_task_cycles(tasks.TaskCalibrateParametersPost,
        setup_tasks=calibrate_setup_tasks)

    ## walk3 condition
    walk3 = subject.add_condition('walk3', metadata={'walking_speed': 1.50})

    # GRF gait landmarks
    # walk3_trial_temp = walk3.add_trial(99, omit_trial_dir=True)
    # walk3_trial_temp.add_task(osp.TaskGRFGaitLandmarks)
    # walk3_trial_temp.add_task(tasks.TaskUpdateGroundReactionColumnLabels)
    
    gait_events = dict()
    gait_events['right_strikes'] = [0.911, 1.934, 2.934, 3.978]
    gait_events['left_toeooffs'] = [1.081, 2.089, 3.110]
    gait_events['left_strikes'] = [1.428, 2.440, 3.436]
    gait_events['right_toeoffs'] = [1.598, 2.614, 3.638]

    walk3_trial = walk3.add_trial(1,
            gait_events=gait_events,
            omit_trial_dir=True,
            )

    # walk3: inverse kinematics
    ik_setup_task = walk3_trial.add_task(osp.TaskIKSetup)
    walk3_trial.add_task(osp.TaskIK, ik_setup_task)
    walk3_trial.add_task(osp.TaskIKPost, ik_setup_task, 
        error_markers=study.error_markers)

    # walk3: inverse dynamics
    id_setup_task = walk3_trial.add_task(osp.TaskIDSetup, ik_setup_task)
    walk3_trial.add_task(osp.TaskID, id_setup_task)
    walk3_trial.add_task(osp.TaskIDPost, id_setup_task)

    # walk3: parameter calibration
    calibrate_setup_tasks = walk3_trial.add_task_cycles(
        tasks.TaskCalibrateParametersSetup)
    walk3_trial.add_task_cycles(tasks.TaskCalibrateParameters, 
        setup_tasks=calibrate_setup_tasks)
    walk3_trial.add_task_cycles(tasks.TaskCalibrateParametersPost,
        setup_tasks=calibrate_setup_tasks)