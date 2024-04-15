% commented on 16 JAN 2024 by Ata Koklu
function learnable_parameters = initialize_Glorot_Kmeans(input_data, output_data, number_of_rule, output_type)
%% centers
number_inputs = size(input_data,2);
number_outputs = size(output_data,2);

data = input_data;
data = extractdata(permute(data,[3 2 1]));

for i=1:number_inputs %applying Kmeans clustring for each input
[~,centers(:,i)] = kmeans(data(:,i),number_of_rule);
end

learnable_parameters.input_centers = centers;
learnable_parameters.input_centers = dlarray(learnable_parameters.input_centers);

%% sigmas
% randomly distributed closer to zero
s = std(data); 
s(s == 0) = 1;
s = repmat(s,number_of_rule,1);
learnable_parameters.input_sigmas = s;

learnable_parameters.input_centers = dlarray(learnable_parameters.input_centers);
learnable_parameters.input_sigmas = dlarray(learnable_parameters.input_sigmas);

if output_type == "singleton"

    c = rand(number_of_rule,number_outputs)*0.01;
    learnable_parameters.singleton.c = dlarray(c);

elseif output_type == "linear"

    a = rand(number_of_rule*number_outputs,number_inputs)*0.01;
    learnable_parameters.linear.a = dlarray(a);

    b = rand(number_of_rule*number_outputs,1)*0.01;
    learnable_parameters.linear.b = dlarray(b);
end

end