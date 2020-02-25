# doit subject*_calibrate*multiphase*post*

# Unassisted conditions + validation w/ EMG
doit -n 2 subject*_walk2_mrs_post_cycle02_Met 
doit -n 2 subject*_walk2_mrs_post_cycle03_Met 
# doit aggregate*muscle*data*experiment*
# doit plot*muscle*data*experiment*
# doit validate*

# doit -n 2 subject*_walk2_mrsmod_deviceHe_fixed_post_cycle02_Met
# doit -n 2 subject*_walk2_mrsmod_deviceHe_fixed_post_cycle03_Met
# doit -n 2 subject*_walk2_mrsmod_deviceKe_fixed_post_cycle02_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceKe_fixed_post_cycle03_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceHeKe_post_cycle02_Met
# doit -n 2 subject*_walk2_mrsmod_deviceHeKe_post_cycle03_Met
# doit -n 2 subject*_walk2_mrsmod_deviceHeKe_multControls_post_cycle02_Met
# doit -n 2 subject*_walk2_mrsmod_deviceHeKe_multControls_post_cycle03_Met
# doit -n 2 subject*_walk2_mrsmod_deviceAp_fixed_post_cycle02_Met
# doit -n 2 subject*_walk2_mrsmod_deviceAp_fixed_post_cycle03_Met
# doit -n 2 subject*_walk2_mrsmod_deviceHf_fixed_post_cycle02_Met
# doit -n 2 subject*_walk2_mrsmod_deviceHf_fixed_post_cycle03_Met
# doit -n 2 subject*_walk2_mrsmod_deviceHfAp_post_cycle02_Met
# doit -n 2 subject*_walk2_mrsmod_deviceHfAp_post_cycle03_Met
# doit -n 2 subject*_walk2_mrsmod_deviceHfAp_multControls_post_cycle02_Met
# doit -n 2 subject*_walk2_mrsmod_deviceHfAp_multControls_post_cycle03_Met
# doit -n 2 subject*_walk2_mrsmod_deviceHfKf_post_cycle02_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceHfKf_post_cycle03_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceHfKf_multControls_post_cycle02_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceHfKf_multControls_post_cycle03_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceHfKfAp_post_cycle02_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceHfKfAp_post_cycle03_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceHfKfAp_multControls_post_cycle02_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceHfKfAp_multControls_post_cycle03_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceKf_fixed_post_cycle02_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceKf_fixed_post_cycle03_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceKfAp_post_cycle02_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceKfAp_post_cycle03_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceKfAp_multControls_post_cycle02_Met 
# doit -n 2 subject*_walk2_mrsmod_deviceKfAp_multControls_post_cycle03_Met

# doit -n 2 subject*_walk2_mrsmod_deviceHKAp_multControls_post_cycle02_Met
# doit -n 2 subject*_walk2_mrsmod_deviceHKAp_multControls_post_cycle03_Met

# doit aggregate*muscle*data*mrsmod*deviceHKAp_multControls*
# doit plot*muscle*data*mrsmod*deviceHKAp_multControls*
# doit aggregate*moment*mrsmod*deviceHKAp_multControls*
# doit plot*moment*mrsmod*deviceHKAp_multControls*
# doit plot*device*comparison*mrsmod*HKAp_multControls