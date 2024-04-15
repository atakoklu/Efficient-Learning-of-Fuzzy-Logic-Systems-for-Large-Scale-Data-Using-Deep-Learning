% commented on 23 JAN 2024 by Ata Koklu
function learnable_parameters = initialize_Glorot_IT2(input_data, fuzzy_set_type, output_data,output_membership_type, number_of_rule,CSCM)


%% centers with Kmean
number_inputs = size(input_data,2);
number_outputs = size(output_data,2);

% data = [input_data output_data];
data = input_data;
data = extractdata(permute(data,[3 2 1]));

for i=1:number_inputs %applying Kmeans clustring for each input
    [~,centers(:,i)] = kmeans(data(:,i),number_of_rule);
end

learnable_parameters.input_centers = centers;
learnable_parameters.input_centers = dlarray(learnable_parameters.input_centers);

%% sigmas

s = std(data); 
s(s == 0) = 1; %to eleminate 0 initialization
s = repmat(s,number_of_rule,1);

learnable_parameters.input_sigmas = s;
learnable_parameters.input_sigmas = dlarray(learnable_parameters.input_sigmas);

if fuzzy_set_type == "H" 

    h = rand(number_of_rule,number_inputs);
    learnable_parameters.input_h = h;
    learnable_parameters.input_h = dlarray(learnable_parameters.input_h);

elseif fuzzy_set_type == "S"

    delta_sigma = rand(number_of_rule,number_inputs)*0.01;
    learnable_parameters.delta_sigmas = delta_sigma;
    learnable_parameters.delta_sigmas = dlarray(learnable_parameters.delta_sigmas);

elseif fuzzy_set_type == "HS"

    h = rand(number_of_rule,number_inputs);
    learnable_parameters.input_h = h;
    learnable_parameters.input_h = dlarray(learnable_parameters.input_h);
    delta_sigmas = rand(number_of_rule,number_inputs)*0.01;
    learnable_parameters.delta_sigmas = delta_sigmas;
    learnable_parameters.delta_sigmas = dlarray(learnable_parameters.delta_sigmas);

end

%%

if output_membership_type == "singleton"

    c = rand(number_of_rule,number_outputs)*0.01;
    learnable_parameters.singleton.c = dlarray(c);

    if CSCM == "BMM" || CSCM == "WNT" || CSCM == "WKM"
        alpha = (rand(1,number_outputs)*2)-1;
        learnable_parameters.singleton.alpha = dlarray(alpha);

    elseif CSCM == "MWNT"
        alpha = (rand(number_of_rule,number_outputs)*2)-1;
        learnable_parameters.singleton.alpha = dlarray(alpha);

    elseif CSCM == "PKM"
        alpha = (rand(number_of_rule,number_outputs)*1)-0.5;
        beta = (rand(number_of_rule,number_outputs)*1)-0.5;
        learnable_parameters.singleton.alpha = dlarray(alpha);
        learnable_parameters.singleton.beta = dlarray(beta);
    end

elseif output_membership_type == "linear"

    a = rand(number_of_rule*number_outputs,number_inputs)*0.01;
    learnable_parameters.linear.a = dlarray(a);

    b = rand(number_of_rule*number_outputs,1)*0.01;
    learnable_parameters.linear.b = dlarray(b);

    if CSCM == "BMM" || CSCM == "WNT" || CSCM == "WKM"
        alpha = (rand(1,number_outputs)*2)-1;
        learnable_parameters.linear.alpha = dlarray(alpha);

    elseif CSCM == "MWNT" 
        alpha = (rand(number_of_rule,number_outputs)*2)-1;
        learnable_parameters.linear.alpha = dlarray(alpha);

    elseif CSCM == "PKM"
        alpha = (rand(number_of_rule,number_outputs)*1)-0.5;
        beta = (rand(number_of_rule,number_outputs)*1)-0.5;
        learnable_parameters.linear.alpha = dlarray(alpha);
        learnable_parameters.linear.beta = dlarray(beta);
    end
end

end