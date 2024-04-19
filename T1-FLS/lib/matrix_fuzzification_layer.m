% Created on 20 JULY 2023
% Last shanged and commented on 26 JAN 2024 by Ata Koklu
function output = matrix_fuzzification_layer(x, membership_type, learnable_parameters, number_of_rule, number_inputs, mbs)
% calculating fuzzified values
%
% @param output -> output
%
%       (mfc,ic,mbs) tensor
%       mfc = number of input membership functions (number of rules)
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> fuzzified value of the first input with first Membership
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
%       type of the membership function
%       it is gaussmf for now but gauss2mf will be added
%
% @param input 3 -> learnable_parameters
%
%       struct
%       consist of sigma and center values of each mf
%
% @param input 4 -> number_of_rule
%
%       constant
%       number of membership function for inputs
%       In other words number of rules of FLS 
%
% @param input 5 -> number_inputs
%
%       constant
%       number of inputs
%
% @param input 6 -> mbs
%
%       constant
%       number of mini-batch size
%

output = zeros(number_of_rule, number_inputs, mbs,"gpuArray");


if(membership_type == "gaussmf")

    input_matrix = repmat(x, number_of_rule, 1, 1); % expanding the input matrix by the number of rules
    output = custom_gaussmf(input_matrix, learnable_parameters.input_sigmas, learnable_parameters.input_centers); % calculating fuzzified values


elseif(membership_type ~= "gaussmf") %for future expansion
else %for future expansion
end

output = dlarray(output);


end


%%

% Custom Gaussian function
function output = custom_gaussmf(x, s, c) % s -> sigma of the Gauss MF / c -> center of the Gauss MF
    exponent = -0.5 * ((x - c).^2 ./ s.^2); % calculating exponent of the Gauss MF
    output = exp(exponent);
end
