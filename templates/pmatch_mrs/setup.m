addpath(genpath('@REL_PATH_TO_TOOL@'));
Misc = struct();

Misc.DofNames_Input = {...
                       'hip_flexion_@SIDE@', ...
                       'knee_angle_@SIDE@', ...
                       'ankle_angle_@SIDE@', ...
                       };

Misc.MuscleNames_Input = {};

Misc.costfun = '@COST@';
Misc.tendonStiffnessCoeff = 35;
if exist('optimal_fiber_length', 'var') && optimal_fiber_length
	[lMo_param_vals, lMo_musc_names, ~] = xlsread('@lMo_MODIFIERS@');
	for m = 1:length(lMo_musc_names)
		Misc.optimalFiberLengthModifiers.(lMo_musc_names{m}) = lMo_param_vals(m); 
	end
end
if exist('tendon_slack_length', 'var') && tendon_slack_length
	[lTs_param_vals, lTs_musc_names, ~] = xlsread('@lTs_MODIFIERS@');
	for m = 1:length(lTs_musc_names)
		Misc.tendonSlackLengthModifiers.(lTs_musc_names{m}) = lTs_param_vals(m); 
	end
end
if exist('pennation_angle', 'var') && pennation_angle
	[alf_param_vals, alf_musc_names, ~] = xlsread('@alf_MODIFIERS@');
	for m = 1:length(alf_musc_names)
		Misc.pennationAngleModifiers.(alf_musc_names{m}) = alf_param_vals(m); 
	end
end
if exist('muscle_strain', 'var') && muscle_strain
	[e0_param_vals, e0_musc_names, ~] = xlsread('@e0_MODIFIERS@');
	for m = 1:length(e0_musc_names)
		Misc.muscleStrainModifiers.(e0_musc_names{m}) = e0_param_vals(m); 
	end
end
Misc.tendonStiffnessModifiers.soleus_r = 0.5;
Misc.tendonStiffnessModifiers.med_gas_r = 0.5;

Misc.powerMatchType = '@MATCH@';
Misc.powerMatchValue = @MATCH_VAL@;

tic;
[Time,MExcitation,MActivation,RActivation,TForcetilde,TForce,MuscleNames,MuscleData,OptInfo,DatStore] = ...
SolveMuscleRedundancy_FtildeState_actdyn(...
	'@MODEL@', ... % model_path
	'@IK_SOLUTION@', ... % IK_path
	'@ID_SOLUTION@', ... % ID_path
	[@INIT_TIME@, @FINAL_TIME@], ... % time
	'results', ... % OutPath
	Misc);
	% TODO '', ... % ID_path
toc

filename = ['@STUDYNAME@' '_' '@NAME@' '_mrs'];
save(filename, '-v7.3')

