addpath(genpath('@REL_PATH_TO_TOOL@'));
set(groot,'defaultFigureVisible','off')

output = load('@MRSMOD_OUTPUT@');
DatStore = output.DatStore;
OptInfo = output.OptInfo;
Time = output.Time;
q = DatStore.q_exp;
T_exo = DatStore.ExoTorques_Act;
norm_max_torque = @NORM_MAX_TORQUE@;
model_mass = OptInfo.result.setup.auxdata.model_mass;
Tmax = norm_max_torque * model_mass;
startTime = @START_TIME@;
fit = '@FIT@';
mod_name = '@MOD_NAME@';

filename = ['@STUDYNAME@' '_' '@NAME@' '_fitopt_' '@FIT@'];
X0 = [];
output = fitOptimizedExoTorque_Zhang2017(Time, q, T_exo, Tmax, startTime, X0, mod_name);

fig = figure;
ax1 = subplot(2,1,1);
plot(output.time', output.opt.torque(:,1), 'r-', ...
    output.time', output.opt.torque(:,2), 'g-', ...
    output.time', output.opt.torque(:,3), 'b-', ...
    'linewidth', 2)
hold on
plot(output.time', output.fit.torque(:,1), 'r--', ...
    output.time', output.fit.torque(:,2), 'g--', ...
    output.time', output.fit.torque(:,3), 'b--', ...
    'linewidth', 2)
legend({'hip (opt)','knee (opt)','ankle (opt)', ...
    'hip (fit)','knee (fit)','ankle (fit)'}, ...
    'location','best')
title(ax1, 'Exoskeleton Torque')
xlabel('time (s)')
ylabel('torque (normalized')

ax2 = subplot(2,1,2);
plot(output.time, output.opt.P(:,1), 'r-', ...
    output.time, output.opt.P(:,2), 'g-', ...
    output.time, output.opt.P(:,3), 'b-', ...
    'linewidth', 2)
hold on
plot(output.time, output.fit.P(:,1), 'r--', ...
    output.time, output.fit.P(:,2), 'g--', ...
    output.time, output.fit.P(:,3), 'b--', ...
    'linewidth', 2)
title(ax2, 'Exoskeleton Power')
xlabel('time (s)')
ylabel('power (normalized')

print(fig, filename, '-dpng')

timeNorm = (Time - Time(1))/(Time(end)-Time(1));
[~, startIdx] = min(abs(Time-startTime));
startPerGC = timeNorm(startIdx)*100;
output.startPerGC = startPerGC;

save(filename, 'output', '-v7.3')