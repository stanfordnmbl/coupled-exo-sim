import itertools
import osimpipeline as osp
import tasks

def get_mod_names():

    mod_names = list()

    # Exotopology mod names
    device_dofs = ['actH','actK','actA','passH','passK','passA']
    exotopology_mod_names = get_exotopology_flags(device_dofs)[0]
    mod_names += exotopology_mod_names

    return mod_names

def generate_exotopology_tasks(trial, mrs_setup_tasks):

    device_dofs = ['actH','actK','actA']
    mod_names, descriptions, activeDOFs_list, passiveDOFs_list, subcases = \
        get_exotopology_flags(device_dofs)

    for mod_name, description, activeDOFs, passiveDOFs, subcase in  \
        itertools.izip(mod_names, descriptions, 
            activeDOFs_list, passiveDOFs_list, subcases):

        mrsflags = [
            "study='SoftExosuitDesign/Topology'",
            "activeDOFs={%s}" % activeDOFs,
            "passiveDOFs={%s}" % passiveDOFs,
            "subcase='%s'" % subcase,
            ]

        mrsmod_tasks = trial.add_task_cycles(
            osp.TaskMRSDeGrooteMod,
            mod_name,
            'ExoTopology: %s device' % description,
            mrsflags,
            setup_tasks=mrs_setup_tasks
            )

        trial.add_task_cycles(tasks.TaskMRSDeGrooteModPost,
            setup_tasks=mrsmod_tasks)

def get_exotopology_flags(device_dofs, act_combo=None, pass_combo=None):

    mod_names = list()
    descriptions = list()
    activeDOFs_list = list()
    passiveDOFs_list = list()
    subcases = list()

    for L in range(1, len(device_dofs)+1):
        for subset in itertools.combinations(device_dofs, L):

            isAct = (any(word in subset for word in ['actH','actK','actA']) 
                    or act_combo)
            isPass = (any(word in subset for word in ['passH','passK','passA'])
                     or pass_combo)
        
            mod_name = ''
            description = ''

            # Active DOFs
            act_dofs = list()
            if 'actH' in subset:
                act_dofs.append('hip')
                mod_name = 'actH'
                description = 'active hip'

            if 'actK' in subset:
                act_dofs.append('knee')
                if not mod_name:
                    mod_name = 'actK'
                    description = 'active knee'
                else:
                    mod_name = mod_name + 'K'
                    description = description + '/knee'

            if 'actA' in subset:
                act_dofs.append('ankle')
                if not mod_name:
                    mod_name = 'actA'
                    description = 'active ankle'
                else:
                    mod_name = mod_name + 'A'
                    description = description + '/ankle'

            if act_combo:
                mod_name = act_combo
                description = ''
                if 'H' in act_combo:
                    act_dofs.append('hip')
                    description = 'active hip'

                if 'K' in act_combo:
                    act_dofs.append('knee')
                    if not description:
                        description = 'active knee'
                    else:
                        description = description + '/knee'

                if 'A' in act_combo:
                    act_dofs.append('ankle')
                    if not description:
                        description = 'active ankle'
                    else:
                        description = description + '/ankle'

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
                description += ' and '

            # Passive DOFs
            pass_dofs = list()
            if 'passH' in subset:
                pass_dofs.append('hip')
                mod_name = mod_name + 'passH'
                description = description + 'passive hip'

            if 'passK' in subset:
                pass_dofs.append('knee')
                if 'pass' not in mod_name:
                    mod_name = mod_name + 'passK'
                    description = description + 'passive knee'
                else:
                    mod_name = mod_name + 'K'
                    description = description + '/knee'

            if 'passA' in subset:
                pass_dofs.append('ankle')
                if 'pass' not in mod_name:
                    mod_name = mod_name + 'passA'
                    description = description + 'passive ankle'
                else:
                    mod_name = mod_name + 'A'
                    description = description + '/ankle'

            if pass_combo:
                mod_name += pass_combo
                if 'H' in pass_combo:
                    pass_dofs.append('hip')
                    description = description + 'passive hip'

                if 'K' in pass_combo:
                    pass_dofs.append('knee')
                    if 'passive' not in description:
                        description = 'passive knee'
                    else:
                        description = description + '/knee'

                if 'A' in pass_combo:
                    pass_dofs.append('ankle')
                    if 'passive' not in description:
                        description = 'passive ankle'
                    else:
                        description = description + '/ankle'

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
            if (any(word in subset for word in ['actH','actK','actA']) 
                or act_combo):
                subcase = subcase + 'Act'
            if (any(word in subset for word in ['passH','passK','passA']) 
                or pass_combo):
                subcase = subcase + 'Pass'

            mod_names.append(mod_name)
            descriptions.append(description)
            activeDOFs_list.append(activeDOFs)
            passiveDOFs_list.append(passiveDOFs)
            subcases.append(subcase)


    return mod_names, descriptions, activeDOFs_list, passiveDOFs_list, subcases