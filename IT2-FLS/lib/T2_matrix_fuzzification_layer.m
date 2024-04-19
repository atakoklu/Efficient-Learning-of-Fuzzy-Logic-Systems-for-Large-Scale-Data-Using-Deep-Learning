% 20 JULY 2023
% Last changed and commented on 23 JAN 2024 by Ata Koklu
function [output_lower, output_upper] = T2_matrix_fuzzification_layer(x, membership_type,fuzzy_set_type, learnable_parameters, number_of_rules, number_inputs, mbs,tnorm)
%
%       Calculates fuzzified values
%
% @param output 1 -> output_lower
%
%       (mfc,ic,mbs) tensor
%       lower fuzzified value
%       mfc = number of input membership functions (number of rules)
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> lower fuzzified value of the first input with first Membership
%       Function (MF) of that input of first element of the batch
%
% @param output 2 -> output_upper
%
%       (mfc,ic,mbs) tensor
%       upper fuzzified value
%       mfc = number of input membership functions (number of rules)
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> upper fuzzified value of the first input with first Membership
%       Function (MF) of that input of first element of the batch
%
% @param input 1 -> x
%
%       (1,ic,mbs) tensor
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> firts input of the first element of the batch
%
% @param input 2 -> membership_type
%
%       a string
%       anticident MF type
%       it is gaussmf for now but gauss2mf will be added
%
% @param input 3 -> fuzzy_set_type
%
%       a string
%       anticident fuzzy set type
%       3 options are available for now: 
%       "H" , "S" , "HS" 
%
% @param input 4 -> learnable_parameters
%
%       struct
%       consist parameters of antecedent and consequent MFs
%
% @param input 5 -> number_of_rule
%
%       constant
%       number of Membership Function (MF) for inputs
%       In other words number of rules of the FLS
%
% @param input 6 -> number_inputs
%
%       constant
%       number of inputs
%
% @param input 7 -> mbs
%
%       constant
%       mini-batch size
%
% @param input 8 -> tnorm
%
%       a string
%       T-norm operator type
%       3 options are available for now: 
%       "product" , "HTSK2" , "eHTSK2"      
%       only "HTSK2" changes the anticident MF


input_matrix = repmat(x, number_of_rules, 1, 1);

output_upper = zeros(number_of_rules, number_inputs, mbs, "gpuArray");
output_lower = zeros(number_of_rules, number_inputs, mbs, "gpuArray");

if(membership_type == "gaussmf" && fuzzy_set_type == "H" && tnorm ~= "HTSK2")

    h = sigmoid(learnable_parameters.input_h);
    
    output_upper = custom_gaussmf(input_matrix, abs(learnable_parameters.input_sigmas), learnable_parameters.input_centers);
   
    output_lower = output_upper .* h;


elseif(membership_type == "gaussmf" && fuzzy_set_type == "H" && tnorm == "HTSK2")

    h = sigmoid(learnable_parameters.input_h);
    
    output_upper = custom_gaussmf(input_matrix, (sqrt(number_inputs) * abs(learnable_parameters.input_sigmas)), learnable_parameters.input_centers);

    output_lower = output_upper .* h;

elseif(membership_type == "gaussmf" && fuzzy_set_type == "S" && tnorm ~= "HTSK2")

    output_upper = custom_gaussmf(input_matrix, abs(learnable_parameters.input_sigmas) + abs(learnable_parameters.delta_sigmas), learnable_parameters.input_centers);
    output_lower = custom_gaussmf(input_matrix, abs(learnable_parameters.input_sigmas) - abs(learnable_parameters.delta_sigmas), learnable_parameters.input_centers);

elseif(membership_type == "gaussmf" && fuzzy_set_type == "S" && tnorm == "HTSK2")

    output_upper = custom_gaussmf(input_matrix, (sqrt(number_inputs) * (abs(learnable_parameters.input_sigmas) + abs(learnable_parameters.delta_sigmas))), learnable_parameters.input_centers);
    output_lower = custom_gaussmf(input_matrix, (sqrt(number_inputs) * (abs(learnable_parameters.input_sigmas) - abs(learnable_parameters.delta_sigmas))), learnable_parameters.input_centers);

elseif(membership_type == "gaussmf" && fuzzy_set_type == "HS" && tnorm ~= "HTSK2")

    h = sigmoid(learnable_parameters.input_h);

    output_upper = custom_gaussmf(input_matrix, abs(learnable_parameters.input_sigmas) + abs(learnable_parameters.delta_sigmas), learnable_parameters.input_centers);
    output_lower = custom_gaussmf(input_matrix, abs(learnable_parameters.input_sigmas) - abs(learnable_parameters.delta_sigmas), learnable_parameters.input_centers);
    output_lower = output_lower .* h;

elseif(membership_type == "gaussmf" && fuzzy_set_type == "HS" && tnorm == "HTSK2")

    h = sigmoid(learnable_parameters.input_h);

    output_upper = custom_gaussmf(input_matrix, (sqrt(number_inputs) * (abs(learnable_parameters.input_sigmas) + abs(learnable_parameters.delta_sigmas))), learnable_parameters.input_centers);
    output_lower = custom_gaussmf(input_matrix, (sqrt(number_inputs) * (abs(learnable_parameters.input_sigmas) - abs(learnable_parameters.delta_sigmas))), learnable_parameters.input_centers);
     output_lower = output_lower .* h;

elseif(membership_type ~= "gaussmf") %for future expansion
else %for future expansion
end

end
%%

% Custom Gaussian function
function output = custom_gaussmf(x, s, c)
    exponent = -0.5 * ((x - c).^2 ./ s.^2);
    output = exp(exponent);
end