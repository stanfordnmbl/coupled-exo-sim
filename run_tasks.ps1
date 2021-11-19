# Scaling, IK, ID, calibration
# ----------------------------
doit subject*scale*
doit subject*adjust*
doit subject*ik*
doit validate*marker*error*
doit subject*id*
doit validate*kinetics*
doit subject*calibrate_multiphase*post*

# Unassisted conditions + validation w/ EMG
# -----------------------------------------
doit -n 2 subject*_walk2_mrs_post_cycle02_Met 
doit -n 2 subject*_walk2_mrs_post_cycle03_Met 
doit aggregate*muscle*data*experiment*
doit plot*muscle*data*experiment*
doit validate*

# Sensitivity analysis
doit -n 2 subject*_walk2_*sensitivity*_post_cycle02_Met
doit -n 2 subject*_walk2_*sensitivity*_post_cycle03_Met

# Main device simulation tasks
# ----------------------------
doit -n 2 subject*_walk2_mrsmod_deviceHe_post_cycle02_Met
doit -n 2 subject*_walk2_mrsmod_deviceHe_post_cycle03_Met
doit -n 2 subject*_walk2_mrsmod_deviceKe_post_cycle02_Met
doit -n 2 subject*_walk2_mrsmod_deviceKe_post_cycle03_Met 
doit -n 2 subject*_walk2_mrsmod_deviceHf_post_cycle02_Met
doit -n 2 subject*_walk2_mrsmod_deviceHf_post_cycle03_Met
doit -n 2 subject*_walk2_mrsmod_deviceKf_post_cycle02_Met 
doit -n 2 subject*_walk2_mrsmod_deviceKf_post_cycle03_Met
doit -n 2 subject*_walk2_mrsmod_deviceAp_post_cycle02_Met
doit -n 2 subject*_walk2_mrsmod_deviceAp_post_cycle03_Met 
doit -n 2 subject*_walk2_mrsmod_deviceHeKe_coupled_post_cycle02_Met
doit -n 2 subject*_walk2_mrsmod_deviceHeKe_coupled_post_cycle03_Met
doit -n 2 subject*_walk2_mrsmod_deviceHeKe_independent_post_cycle02_Met
doit -n 2 subject*_walk2_mrsmod_deviceHeKe_independent_post_cycle03_Met
doit -n 2 subject*_walk2_mrsmod_deviceHfKf_coupled_post_cycle02_Met 
doit -n 2 subject*_walk2_mrsmod_deviceHfKf_coupled_post_cycle03_Met 
doit -n 2 subject*_walk2_mrsmod_deviceHfKf_independent_post_cycle02_Met 
doit -n 2 subject*_walk2_mrsmod_deviceHfKf_independent_post_cycle03_Met
doit -n 2 subject*_walk2_mrsmod_deviceKfAp_coupled_post_cycle02_Met 
doit -n 2 subject*_walk2_mrsmod_deviceKfAp_coupled_post_cycle03_Met 
doit -n 2 subject*_walk2_mrsmod_deviceKfAp_independent_post_cycle02_Met 
doit -n 2 subject*_walk2_mrsmod_deviceKfAp_independent_post_cycle03_Met
doit -n 2 subject*_walk2_mrsmod_deviceHfAp_coupled_post_cycle02_Met
doit -n 2 subject*_walk2_mrsmod_deviceHfAp_coupled_post_cycle03_Met
doit -n 2 subject*_walk2_mrsmod_deviceHfAp_independent_post_cycle02_Met
doit -n 2 subject*_walk2_mrsmod_deviceHfAp_independent_post_cycle03_Met
doit -n 2 subject*_walk2_mrsmod_deviceHfKfAp_coupled_post_cycle02_Met 
doit -n 2 subject*_walk2_mrsmod_deviceHfKfAp_coupled_post_cycle03_Met 
doit -n 2 subject*_walk2_mrsmod_deviceHfKfAp_independent_post_cycle02_Met 
doit -n 2 subject*_walk2_mrsmod_deviceHfKfAp_independent_post_cycle03_Met 

# Plot results
# ------------
doit aggregate*muscle*data*mrsmod*
doit plot*muscle*data*mrsmod*
doit aggregate*moment*mrsmod*
doit plot*moment*mrsmod*
doit plot*device*comparison*mrsmod*
doit plot*metabolic*reduction*
doit plot*showcase*
doit aggregate*device*power*
doit create*power*table*
doit aggregate*device*moment*
doit create*moment*table*
doit plot*muscle*activity*