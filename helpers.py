def generate_exotopology_tasks(trial, mrs_setup_tasks):

    import itertools
    import osimpipeline as osp
    import tasks

    device_dofs = ['actH','actK','actA','passH','passK','passA']
    for L in range(1, len(device_dofs)+1):
        for subset in itertools.combinations(device_dofs, L):

            mod_name = ''
            descr = ''

            # Active DOFs
            act_dofs = list()
            if 'actH' in subset:
                act_dofs.append('hip')
                mod_name = 'actH'
                descr = 'active hip'

            if 'actK' in subset:
                act_dofs.append('knee')
                if not mod_name:
                    mod_name = 'actK'
                    descr = 'active knee'
                else:
                    mod_name = mod_name + 'K'
                    descr = descr + '/knee'

            if 'actA' in subset:
                act_dofs.append('ankle')
                if not mod_name:
                    mod_name = 'actA'
                    descr = 'active ankle'
                else:
                    mod_name = mod_name + 'A'
                    descr = descr + '/ankle'

            if len(act_dofs)==0:
                activeDOFs = ''
            elif len(act_dofs)==1:
                activeDOFs = "'%s'" % act_dofs[0]
            elif len(act_dofs)==2:
                activeDOFs = "'%s','%s'" % (act_dofs[0], act_dofs[1])
            elif len(act_dofs)==3:
                activeDOFs = "'%s','%s','%s'" % (act_dofs[0], act_dofs[1],
                    act_dofs[2])

            if (any(word in subset for word in ['passH','passK','passA']) and 
                any(word in subset for word in ['actH','actK','actA'])):
                mod_name = mod_name + '_'

            if (('active' in descr) and 
                any(word in subset for word in ['passH','passK','passA'])):
                descr = descr + ' and '

            # Passive DOFs
            pass_dofs = list()
            if 'passH' in subset:
                pass_dofs.append('hip')
                mod_name = mod_name + 'passH'
                descr = descr + 'passive hip'

            if 'passK' in subset:
                pass_dofs.append('knee')
                if 'pass' not in mod_name:
                    mod_name = mod_name + 'passK'
                    descr = descr + 'passive knee'
                else:
                    mod_name = mod_name + 'K'
                    descr = descr + '/knee'

            if 'passA' in subset:
                pass_dofs.append('ankle')
                if 'pass' not in mod_name:
                    mod_name = mod_name + 'passA'
                    descr = descr + 'passive ankle'
                else:
                    mod_name = mod_name + 'A'
                    descr = descr + '/ankle'

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
            if any(word in subset for word in ['actH','actK','actA']):
                subcase = subcase + 'Act'
            if any(word in subset for word in ['passH','passK','passA']):
                subcase = subcase + 'Pass'

            mrsflags = [
                "study='SoftExosuitDesign/Topology'",
                "activeDOFs={%s}" % activeDOFs,
                "passiveDOFs={%s}" % passiveDOFs,
                "subcase='%s'" % subcase,
                ]

            mrsmod_tasks = trial.add_task_cycles(
                osp.TaskMRSDeGrooteMod,
                mod_name,
                'ExoTopology: %s device' % descr,
                mrsflags,
                setup_tasks=mrs_setup_tasks
                )

            trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
                setup_tasks=mrsmod_tasks)