import opensim

apoorvar = opensim.Model('Rajagopal2015.osim')


mset = apoorvar.updMuscles()

muscle_name_map = {
        'addbrev': 'add_brev',
        'addlong': 'add_long',
        'addmagProx': 'add_mag1',
        'addmagMid': 'add_mag2',
        'addmagDist': 'add_mag3',
        'addmagIsch': 'add_mag4',
        'bflh': 'bifemlh',
        'bfsh': 'bifemsh',
        'edl': 'ext_dig',
        'ehl': 'ext_hal',
        'fdl': 'flex_dig',
        'fhl': 'flex_hal',
        'gaslat': 'lat_gas',
        'gasmed': 'med_gas',
        'glmax1': 'glut_max1',
        'glmax2': 'glut_max2',
        'glmax3': 'glut_max3',
        'glmed1': 'glut_med1',
        'glmed2': 'glut_med2',
        'glmed3': 'glut_med3',
        'glmin1': 'glut_min1',
        'glmin2': 'glut_min2',
        'glmin3': 'glut_min3',
        'perbrev': 'per_brev',
        'perlong': 'per_long',
        'piri': 'peri',
        'recfem': 'rect_fem',
        'sart': 'sar',
        'tibant': 'tib_ant',
        'tibpost': 'tib_post',
        'vasint': 'vas_int',
        'vaslat': 'vas_lat',
        'vasmed': 'vas_med'
        }

for k, v in muscle_name_map.items():
    for side in ['r', 'l']:
        mset.get("%s_%s" % (k, side)).setName("%s_%s" % (v, side))

apoorvar.printToXML('Rajagopal2015_muscle_names.osim')
