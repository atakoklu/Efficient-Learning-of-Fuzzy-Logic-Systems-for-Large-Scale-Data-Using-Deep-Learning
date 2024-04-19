% Last shanged and commented on 26 JAN 2024 by Ata Koklu
function output = HTSK(x, learnable_parameters, number_of_rule, number_inputs, mbs)

% Calculating normalized firing strength values with the HTSK method
% See "Curse of Dimensionality for TSK Fuzzy Neural Networks: Explanation
% and Solutions" by Yuqi Cui, Dongrui Wu and Yifan Xu
%
% Compatible with mini-batch
%
% @param output -> output
%
%       (rc,1,mbs) tensor
%       rc = number of rules  
%       mbs = mini-batch size
%       (1,1,1) -> normalized firing strength of the first rule of the
%       first element of the batch
%
% @param input 1 -> x
%
%       (1,ic,mbs) tensor
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> firts input of the first element of the batch
%
% @param input 2 -> learnable_parameters
%
%       struct
%       consist of sigma and center values of each mf
%
% @param input 3 -> number_of_rule
%
%       constant
%       number of Membership Function (MF) for inputs
%       In other words number of rules of the FLS
%
% @param input 4 -> number_inputs
%
%       constant
%       number of inputs
%
% @param input 5 -> mbs
%
%       constant
%       number of mini-batch size
%

input_matrix = repmat(x, number_of_rule, 1, 1); % expanding the input matrix by the number of rules
output = zeros(number_of_rule, number_inputs, mbs, "gpuArray"); % alocating the empty output matrix in advance

exponent = 0.5 * ((input_matrix - learnable_parameters.input_centers).^2 ./ (learnable_parameters.input_sigmas.^2)); % calculating the exponent

Z = mean(exponent,2); % calculating the Z, see the corresponding paper if you want to learm about the mathematics of it
frs = -Z;

output = softmax(frs,"DataFormat","CSB"); % calculating normalized firing strength

end

