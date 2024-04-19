function plotter(epoch,plot_freq,loss,y_true,y_pred, y_upper, y_lower, sortflag,x)

arguments
    epoch = 1;
    plot_freq = 10;
    loss = [];
    y_true = [];
    y_pred = [];
    y_upper = [];
    y_lower = [];
    sortflag = "down";
    x = [];

end


persistent hasRun
persistent start_time
persistent line_loss_train_persistent
persistent plotting_timesteps
persistent number_output
persistent oddVector;
persistent evenVector;



if epoch == 1
    hasRun = false;
    start_time = 0;
    line_loss_train_persistent = 0;
    plotting_timesteps = 0;
    number_output = size(y_true);
    number_output = number_output(1);
    oddVector = [];
    evenVector = [];

end

if hasRun == false

    start_time = tic;
    hasRun = true;

    for i=1:number_output*2
        if rem(i,2) == 1
            oddVector = [oddVector,i];
        else
            evenVector = [evenVector,i];
        end
    end


    f = figure;

    %     f.Position(2) = 2*f.Position(2);
    subplot(number_output, 2,oddVector)
    C = colororder;
    line_loss_train_persistent = animatedline(Color=C(2,:));
    ylim([0 inf])
    xlabel("Epoch")
    ylabel("Loss")
    grid on


    plotting_timesteps = max(size(y_true));
    plotting_timesteps = 1:plotting_timesteps;

    


end


if sortflag == "sort"
    x = reshape(x,[1, max(size(x))]);
    [x, idx] = sort(x);
    y_true = y_true(:,idx);
    y_pred = y_pred(:,idx);
    y_upper = y_upper(:,idx);
    y_lower = y_lower(:,idx);
end
    


subplot(number_output, 2,oddVector)

current_loss = double(loss);

line_loss_train = line_loss_train_persistent;
%     line_loss_train = animatedline(Color=C(2,:));

addpoints(line_loss_train,epoch,current_loss);
D = duration(0,0,toc(start_time),Format="hh:mm:ss");
title("Elapsed: " + string(D))
drawnow

if mod(epoch,plot_freq) == 0  || epoch == 1
    for i=1:number_output
        subplot(number_output, 2,evenVector(i))
        plot(plotting_timesteps, y_true(i,:) , "rx");
        hold on
        plot(plotting_timesteps, y_pred(i,:), "bo");
        if ~isempty(y_upper)
            plot(plotting_timesteps, y_upper(i,:));
            plot(plotting_timesteps, y_lower(i,:));
        end
        hold off
    end


end

end