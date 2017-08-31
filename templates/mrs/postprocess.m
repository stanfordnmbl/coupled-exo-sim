addpath(genpath('@REL_PATH_TO_TOOL@'));
mat = load('@STUDYNAME@_@NAME@_mrs.mat');
timeTable = table(mat.Time, 'VariableNames', {'time'});
table_MExcitation = [timeTable, ...
        array2table(mat.MExcitation, 'VariableNames', mat.MuscleNames)];
writetable(table_MExcitation, '@STUDYNAME@_@NAME@_mrs_MExcitation.csv');
table_MActivation = [timeTable, ...
        array2table(mat.MActivation, 'VariableNames', mat.MuscleNames)];
writetable(table_MActivation, '@STUDYNAME@_@NAME@_mrs_MActivation.csv');
table_RActivation = [timeTable, ...
        array2table(mat.RActivation, 'VariableNames', mat.DatStore.DOFNames)];
writetable(table_RActivation, '@STUDYNAME@_@NAME@_mrs_RActivation.csv');
table_TForcetilde = [timeTable, ...
        array2table(mat.TForcetilde, 'VariableNames', mat.MuscleNames)];
writetable(table_TForcetilde, '@STUDYNAME@_@NAME@_mrs_TForcetilde.csv');
table_TForce = [timeTable, ...
        array2table(mat.TForce, 'VariableNames', mat.MuscleNames)];
writetable(table_TForce, '@STUDYNAME@_@NAME@_mrs_TForce.csv');
table_lMtilde = [timeTable, ...
        array2table(mat.lMtilde, 'VariableNames', mat.MuscleNames)];
writetable(table_lMtilde, '@STUDYNAME@_@NAME@_mrs_lMtilde.csv');

if isfield(mat.OptInfo.result.solution, 'parameter')
    f = fopen('@STUDYNAME@_@NAME@_mrs_parameters.csv', 'w+');
    fprintf(f, '%f\n', mat.OptInfo.result.solution.parameter);
    fclose(f);
end

model = org.opensim.modeling.Model('@MODEL@');
f = fopen('@STUDYNAME@_@NAME@_mrs_metabolic_rate.csv', 'w+');
fprintf(f, '%f\n', calcWholeBodyMetabolicRate(model, mat));
fclose(f);

if isfield(mat.Misc, 'study')
    if any(strfind(mat.Misc.study, 'Quinlivan2017')) && ...
            strcmp(mat.Misc.costfun, 'Exc_Act')
        exoTorques = calcExoTorques_lMtildeISBQuinlivan2017_Exc_Act(...
            mat.OptInfo, mat.DatStore);
    elseif any(strfind(mat.Misc.study, 'Collins2015')) && ...
            strcmp(mat.Misc.costfun, 'Exc_Act')
        exoTorques = calcExoTorques_lMtildeISBCollins2015_Exc_Act(...
            mat.OptInfo, mat.DatStore);
    end
    table_exoTorques = [timeTable, ...
        array2table(exoTorques, 'VariableNames', mat.DatStore.DOFNames)];
    writetable(table_exoTorques, '@STUDYNAME@_@NAME@_mrs_exotorques.csv');
end
