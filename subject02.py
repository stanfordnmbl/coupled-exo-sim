import os

import osimpipeline as osp
import tasks
import helpers

def scale_setup_fcn(pmm, mset, sset, ikts):
    m = pmm.Measurement('torso', mset)
    m.add_markerpair('RASI', 'CLAV')    
    m.add_markerpair('LASI', 'CLAV')
    m.add_markerpair('LASI', 'C7')
    m.add_markerpair('RASI', 'C7')
    m.add_markerpair('RASI',' RACR')
    m.add_markerpair('LASI', 'LACR')
    m.add_bodyscale('torso')

    m = pmm.Measurement('pelvis_z', mset)
    m.add_markerpair('RPSI', 'LPSI')
    m.add_markerpair('RASI', 'LASI')
    m.add_bodyscale('pelvis', 'YZ')

    m = pmm.Measurement('thigh', mset)
    m.add_markerpair('LHJC', 'LLFC')
    m.add_markerpair('LHJC', 'LMFC')
    m.add_markerpair('RHJC', 'RMFC')
    m.add_markerpair('RHJC', 'RLFC')
    m.add_bodyscale_bilateral('femur')

    m = pmm.Measurement('shank', mset)
    m.add_markerpair('LLFC', 'LLMAL')
    m.add_markerpair('LMFC', 'LMMAL')
    m.add_markerpair('RLFC', 'RLMAL')
    m.add_markerpair('RMFC', 'RMMAL')
    m.add_bodyscale_bilateral('tibia', 'XY')

    m = pmm.Measurement('foot', mset)
    m.add_markerpair('LCAL', 'LMT5')
    m.add_markerpair('LCAL', 'LTOE')
    m.add_markerpair('RCAL',' RMT5')
    m.add_markerpair('RCAL', 'RTOE')
    m.add_bodyscale_bilateral('talus')
    m.add_bodyscale_bilateral('calcn')
    m.add_bodyscale_bilateral('toes')

    m = pmm.Measurement('humerus', mset)
    m.add_markerpair('RPSH', 'RLEL')
    m.add_markerpair('RASH', 'RMEL')
    m.add_markerpair('LASH', 'LMEL')
    m.add_markerpair('LPSH', 'LLEL')
    m.add_markerpair('LACR', 'LMEL')
    m.add_markerpair('LACR', 'LLEL')
    m.add_markerpair('RACR', 'RLEL')
    m.add_markerpair('RACR', 'RMEL')
    m.add_bodyscale_bilateral('humerus')

    m = pmm.Measurement('radius_ulna', mset)
    m.add_markerpair('LLEL', 'LFAradius')
    m.add_markerpair('LMEL', 'LFAulna')
    m.add_markerpair('RMEL', 'RFAulna')
    m.add_markerpair('RLEL', 'RFAradius')
    m.add_bodyscale_bilateral('ulna')
    m.add_bodyscale_bilateral('radius')
    m.add_bodyscale_bilateral('hand')

    # Hamner/Arnold defined this measurement but did not use it.
    #m = pmm.Measurement('pelvis_Y', mset)
    #m.add_markerpair('LPSI', 'LHJC')
    #m.add_markerpair('RPSI', 'RHJC')
    #m.add_markerpair('RASI', 'RHJC')
    #m.add_markerpair('LASI', 'LHJC')
    #m.add_bodyscale('pelvis', 'Y')

    m = pmm.Measurement('pelvis_X', mset)
    m.add_markerpair('RASI', 'RPSI')
    m.add_markerpair('LASI', 'LPSI')
    m.add_bodyscale('pelvis', 'X')

    m = pmm.Measurement('shank_width', mset)
    m.add_markerpair('LLMAL', 'LMMAL')
    m.add_markerpair('RMMAL', 'RLMAL')
    m.add_bodyscale_bilateral('tibia', 'Z')

    ikts.add_ikmarkertask_bilateral('ACR', True, 50.0)
    ikts.add_ikmarkertask('C7', True, 100.0)
    ikts.add_ikmarkertask('CLAV', True, 100.0)
    ikts.add_ikmarkertask_bilateral('ASH', True, 10.0)
    ikts.add_ikmarkertask_bilateral('PSH', True, 10.0)
    ikts.add_ikmarkertask_bilateral('LEL', True, 50.0)
    ikts.add_ikmarkertask_bilateral('MEL', True, 50.0)
    ikts.add_ikmarkertask_bilateral('FAradius', True, 50.0)
    ikts.add_ikmarkertask_bilateral('FAulna', True, 50.0)
    ikts.add_ikmarkertask_bilateral('ASI', True, 100.0)
    ikts.add_ikmarkertask_bilateral('PSI', True, 100.0)
    ikts.add_ikmarkertask_bilateral('HJC', True, 1000.0)
    ikts.add_ikmarkertask_bilateral('LFC', True, 100.0)
    ikts.add_ikmarkertask_bilateral('MFC', True, 100.0)
    ikts.add_ikmarkertask_bilateral('LMAL', True, 100.0)
    ikts.add_ikmarkertask_bilateral('MMAL', True, 100.0)
    ikts.add_ikmarkertask_bilateral('CAL', True, 25.0)
    ikts.add_ikmarkertask_bilateral('TOE', True, 25.0)
    ikts.add_ikmarkertask_bilateral('MT5', True, 25.0)

    ikts.add_ikcoordinatetask('pelvis_list', True, 0.0, 1.0)
    ikts.add_ikcoordinatetask_bilateral('hip_flexion', True, 0.0, 10.0)
    ikts.add_ikcoordinatetask_bilateral('hip_rotation', True, 0.0, 1.0)
    ikts.add_ikcoordinatetask_bilateral('knee_angle', True, 0.0, 10.0)
    ikts.add_ikcoordinatetask_bilateral('ankle_angle', True, 0.0, 1.0)

    ikts.add_ikmarkertask_bilateral('FAsuperior', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TH3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('TB3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA1', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA2', False, 0.0)
    ikts.add_ikmarkertask_bilateral('UA3', False, 0.0)
    ikts.add_ikmarkertask_bilateral('AJC', False, 0.0)
    ikts.add_ikmarkertask_bilateral('SJC', False, 0.0)
    ikts.add_ikmarkertask_bilateral('EJC', False, 0.0)
    ikts.add_ikmarkertask_bilateral('KJC', False, 0.0)

def add_to_study(study):    
    subject = study.add_subject(2, 76.4757)

    cond_args = dict()
    cond_args['walk100'] = (5, '_newCOP3')
    cond_args['walk125'] = (4, '_newCOP3')
    subject.cond_args = cond_args

    static = subject.add_condition('static')
    static_trial = static.add_trial(1, omit_trial_dir=True)

    # `os.path.basename(__file__)` should be `subject02.py`.
    scale_setup_task = subject.add_task(osp.TaskScaleSetup,
            init_time=0,
            final_time=0.4, 
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

    ## walk1 condition
    walk1 = subject.add_condition('walk1', metadata={'walking_speed': 1.00})

    # GRF gait landmarks
    # walk1_trial_temp = walk1.add_trial(99, omit_trial_dir=True)
    # walk1_trial_temp.add_task(osp.TaskGRFGaitLandmarks)
    # walk1_trial_temp.add_task(tasks.TaskUpdateGroundReactionColumnLabels)
    
    gait_events = dict()
    gait_events['right_strikes'] = [1.001, 2.268, 3.479, 4.766]
    gait_events['left_toeoffs'] = [1.219, 2.497, 3.750] 
    gait_events['left_strikes'] = [1.625, 2.895, 4.116] 
    gait_events['right_toeoffs'] = [1.864, 3.118, 4.364] 

    walk1_trial = walk1.add_trial(1,
            gait_events=gait_events,
            omit_trial_dir=True,
            )

    # walk1: main study tasks
    mrs_setup_tasks = helpers.generate_main_tasks(walk1_trial)
    helpers.generate_exotopology_tasks(walk1_trial, mrs_setup_tasks)
    helpers.generate_mult_controls_tasks(walk1_trial, mrs_setup_tasks)

    ## walk2 condition
    walk2 = subject.add_condition('walk2', metadata={'walking_speed': 1.25})
    
    # GRF gait landmarks
    # walk2_trial= walk2.add_trial(1, omit_trial_dir=True)
    # walk2_trial.add_task(tasks.TaskUpdateGroundReactionColumnLabels)
    # walk2_trial.add_task(osp.TaskGRFGaitLandmarks, threshold=10)

    # Trial to use
    gait_events = dict()
    gait_events['right_strikes'] = [0.761, 1.921, 3.084, 4.257] #, 5.406]
    gait_events['left_toeoffs'] = [0.939, 2.101, 3.268] #, 4.446]
    gait_events['left_strikes'] = [1.349, 2.516, 3.667] #, 4.851]
    gait_events['right_toeoffs'] = [1.548, 2.709, 3.842] #, 5.031]
    walk2_trial = walk2.add_trial(1,
            gait_events=gait_events,
            omit_trial_dir=True,
            )
    walk2_trial.add_task(tasks.TaskUpdateGroundReactionColumnLabels)

    # Set the time in the gait cycle when to start fitting a parameterization
    # of the optimized exoskeleton torque. 
    # walk2_trial.get_cycle(3).fit_start_time = 3.46
    # walk2_trial.get_cycle(3).peak_torque = 45.749503287345590 # N-m
    # walk2_trial.get_cycle(3).peak_time = 3.770801870141614 # s
    # walk2_trial.get_cycle(3).rise_time = 0.367949905100557 # s
    # walk2_trial.get_cycle(3).fall_time = 0.217250193376961 # s
    
    # walk2: main study tasks
    mrs_setup_tasks = helpers.generate_main_tasks(walk2_trial)
    helpers.generate_exotopology_tasks(walk2_trial, mrs_setup_tasks)
    helpers.generate_mult_controls_tasks(walk2_trial, mrs_setup_tasks)
    helpers.generate_param_controls_tasks(walk2_trial, mrs_setup_tasks)

    ## walk3 condition
    walk3 = subject.add_condition('walk3', metadata={'walking_speed': 1.50})

    # GRF gait landmarks
    # walk3_trial_temp = walk3.add_trial(99, omit_trial_dir=True)
    # walk3_trial_temp.add_task(osp.TaskGRFGaitLandmarks)
    # walk3_trial_temp.add_task(tasks.TaskUpdateGroundReactionColumnLabels)
    
    gait_events = dict()
    gait_events['right_strikes'] = [0.724, 1.789, 2.823, 3.906]
    gait_events['left_toeoffs'] = [0.915, 1.956, 3.000] 
    gait_events['left_strikes'] = [1.245, 2.303, 3.375] 
    gait_events['right_toeoffs'] = [1.431, 2.479, 3.552] 

    walk3_trial = walk3.add_trial(1,
            gait_events=gait_events,
            omit_trial_dir=True,
            )

    # walk3: main study tasks
    mrs_setup_tasks = helpers.generate_main_tasks(walk3_trial)
    helpers.generate_exotopology_tasks(walk3_trial, mrs_setup_tasks)
    helpers.generate_mult_controls_tasks(walk3_trial, mrs_setup_tasks)

    ## walk4 condition
    walk4 = subject.add_condition('walk4', metadata={'walking_speed': 1.75})

    # GRF gait landmarks
    # walk4_trial_temp = walk4.add_trial(99, omit_trial_dir=True)
    # walk4_trial_temp.add_task(osp.TaskGRFGaitLandmarks, threshold=10)
    # walk4_trial_temp.add_task(tasks.TaskUpdateGroundReactionColumnLabels)
    
    gait_events = dict()
    gait_events['right_strikes'] = [0.829, 1.847, 2.857, 3.824] 
    gait_events['left_toeoffs'] = [0.917, 1.983, 2.997] 
    gait_events['left_strikes'] = [1.337, 2.340, 3.337] 
    gait_events['right_toeoffs'] = [1.484, 2.480, 3.472] 

    walk4_trial = walk4.add_trial(1,
            gait_events=gait_events,
            omit_trial_dir=True,
            )

    # walk4: main study tasks
    mrs_setup_tasks = helpers.generate_main_tasks(walk4_trial)
    helpers.generate_exotopology_tasks(walk4_trial, mrs_setup_tasks)
    helpers.generate_mult_controls_tasks(walk4_trial, mrs_setup_tasks)

    # aggregate muscle parameters
    agg_task = subject.add_task(tasks.TaskAggregateMuscleParameters,
        study.param_dict, 
        conditions=['walk1','walk2','walk3','walk4'], 
        cycles_to_exclude=['cycle03'])
    subject.add_task(tasks.TaskPlotMuscleParameters, agg_task, 
        cycles_to_exclude=['cycle03'])

    # multi-phase parameter calibration (trial does not matter)
    calibrate_setup_task = walk3_trial.add_task(
        tasks.TaskCalibrateParametersMultiPhaseSetup,
        ['walk1','walk2','walk3','walk4'],
        ['cycle01'],
        study.param_dict,
        study.cost_dict,
        passive_precalibrate=True)

    walk3_trial.add_task(
        tasks.TaskCalibrateParametersMultiPhase,
        calibrate_setup_task)

    walk3_trial.add_task(
        tasks.TaskCalibrateParametersMultiPhasePost,
        calibrate_setup_task)