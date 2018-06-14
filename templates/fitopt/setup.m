addpath(genpath('@REL_PATH_TO_TOOL@'));
set(groot,'defaultFigureVisible','off')

output = load('@MRSMOD_OUTPUT@');
DatStore = output.DatStore;
OptInfo = output.OptInfo;
Time = output.Time;
q = DatStore.q_exp;
T_exo = DatStore.ExoTorques_Act;
Tmax = OptInfo.result.setup.auxdata.Tmax_act;
startTime = @START_TIME@;

fit = '@FIT@';
switch fit
	case 'hermite'
		fitfunc =@(N,X0,orders) fitOptimizedExoTorque_Hermite(Time, q, T_exo, Tmax, startTime, N, X0);
	case 'legendre'
		orders = [];
		fitfunc =@(N,X0,orders) fitOptimizedExoTorque_Legendre(Time, q, T_exo, Tmax, startTime, N, orders);
	case 'zhang2017'
		fitfunc =@(N,X0,orders) fitOptimizedExoTorque_Zhang2017(Time, q, T_exo, Tmax, startTime, X0);
	otherwise 
		error('Fit approach not recognized');
end

minParam = @MIN_PARAM@;
maxParam = @MAX_PARAM@;
filename = ['@STUDYNAME@' '_' '@NAME@' '_fitopt_' '@FIT@'];
X0 = [];
orders = [];

for i = minParam:maxParam
	
	output = fitfunc(i,X0,orders);

	fig = figure;
    ax1 = subplot(2,1,1);
	plot(output.time, output.opt.control, 'k-', 'linewidth', 2)
	hold on
	plot(output.time, output.fit.control, 'r--', 'linewidth', 2)
	if ~strcmp(fit, 'legendre')
		plot(output.nodes.T, output.nodes.Y, 'bo', 'linewidth', 2)
	end
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
    legend({'ankle (opt)','knee (opt)','hip (opt)', ...
            'ankle (fit)','knee (fit)','hip (fit)'}, ...
            'location','best')
    title(ax2, 'Exoskeleton Power')
    xlabel('time (s)')
    ylabel('power (normalized')

    print(fig, [filename '_' num2str(i)], '-dpng')
    paramFits.(['params_' num2str(i)]) = output;
end

timeNorm = (Time - Time(1))/(Time(end)-Time(1));
[~, startIdx] = min(abs(Time-startTime));
startPerGC = timeNorm(startIdx)*100;
paramFits.startPerGC = startPerGC;

save(filename, 'paramFits', '-v7.3')