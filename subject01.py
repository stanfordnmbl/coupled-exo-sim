import os

import osimpipeline as osp
import tasks

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

    ikts.add_ikmarkertask_bilateral('ASI', True, 5.0)
    ikts.add_ikmarkertask_bilateral('PSI', True, 5.0)
    ikts.add_ikmarkertask_bilateral('LFC', True, 5.0)
    ikts.add_ikmarkertask_bilateral('MFC', True, 5.0)
    ikts.add_ikmarkertask_bilateral('LMAL', True, 5.0)
    ikts.add_ikmarkertask_bilateral('MMAL', True, 5.0)
    ikts.add_ikmarkertask_bilateral('CAL', True, 5.0)
    ikts.add_ikmarkertask_bilateral('TOE', True, 5.0)
    ikts.add_ikmarkertask_bilateral('MT5', True, 5.0)
    ikts.add_ikmarkertask_bilateral('ACR', True, 5.0)
    ikts.add_ikmarkertask_bilateral('ASH', True, 5.0)
    ikts.add_ikmarkertask_bilateral('PSH', True, 5.0)
    ikts.add_ikmarkertask_bilateral('LEL', True, 5.0)
    ikts.add_ikmarkertask_bilateral('MEL', True, 5.0)
    ikts.add_ikmarkertask_bilateral('HJC', True, 5.0)
    ikts.add_ikmarkertask_bilateral('KJC', True, 5.0)
    ikts.add_ikmarkertask_bilateral('AJC', True, 5.0)
    ikts.add_ikmarkertask_bilateral('SJC', True, 5.0)
    ikts.add_ikmarkertask_bilateral('EJC', True, 5.0)
    ikts.add_ikmarkertask_bilateral('FAsuperior', False, 0.0)
    ikts.add_ikmarkertask_bilateral('FAradius', False, 0.0)
    ikts.add_ikmarkertask_bilateral('FAulna', False, 0.0)
    ikts.add_ikmarkertask('CLAV', True, 5.0)
    ikts.add_ikmarkertask('C7', True, 5.0)
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

    marker_suffix = ['ASI','PSI','TH1','TH2','TH3',
                     'TB1','TB2','TB3','CAL','TOE','MT5']
    error_markers = ['*' + marker for marker in marker_suffix] 

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

    ## walk2 condition
    walk2 = subject.add_condition('walk2', metadata={'walking_speed': 1.25})
    gait_events = dict()
    gait_events['right_strikes'] = [1.179, 2.282, 3.361, 4.488]
    gait_events['right_toeoffs'] = [1.934, 3.033, 4.137]
    gait_events['left_strikes'] = [1.728, 2.836, 3.943]
    gait_events['left_toeooffs'] = [1.368, 2.471, 3.578]
    walk2_trial = walk2.add_trial(1,
            gait_events=gait_events,
            omit_trial_dir=True,
            )
    walk2_trial.add_task(tasks.TaskUpdateGroundReactionColumnLabels)

    # walk2: inverse kinematics
    ik_setup_task = walk2_trial.add_task(osp.TaskIKSetup)
    walk2_trial.add_task(osp.TaskIK, ik_setup_task)
    walk2_trial.add_task(osp.TaskIKPost, ik_setup_task, 
        error_markers=error_markers)

    # walk2: inverse dynamics
    id_setup_task = walk2_trial.add_task(osp.TaskIDSetup, ik_setup_task)
    walk2_trial.add_task(osp.TaskID, id_setup_task)
    walk2_trial.add_task(osp.TaskIDPost, id_setup_task)

    # walk2: muscle redundnacy solver
    mrs_setup_tasks = walk2_trial.add_task_cycles(osp.TaskMRSDeGrooteSetup)
    walk2_trial.add_task_cycles(osp.TaskMRSDeGroote, 
        setup_tasks=mrs_setup_tasks)
    walk2_trial.add_task_cycles(osp.TaskMRSDeGrootePost,
        setup_tasks=mrs_setup_tasks)







