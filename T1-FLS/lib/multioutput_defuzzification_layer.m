% Created on 15 JULY 2023
% Commented on 16 JAN 2024 by Ata Koklu
function output = multioutput_defuzzification_layer(x,normalized_firing_strength, learnable_parameters,number_outputs, output_type)
%
% Calculating the crisp output of the FLS from the normalized firing
% strength
%
% Compatible with multiple inputs, make sure you check the loss is compatible 
% before using with multi outputs!
%
% @param output -> output
%
%       (1,oc,mbs) tensor
%       oc = number of outputs
%       mbs = mini batch size
%       (1,1,1) -> firts crisp output of the first element of the batch
%
% @param input 1 -> x
%
%       (1,ic,mbs) tensor
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> firts input of the first element of the batch
%
% @param input 2 -> normalized_firing_strength 
%
%       (rc,1,mbs) tensor
%       rc = number of rules  
%       mbs = mini-batch size
%       (1,1,1) -> normalized firing strength of the first rule of the
%       first element of the batch
%
% @param input 3 -> learnable_parameters
%
%       struct
%       consist of required learnable parameters of singleton or linear
%       consequent MF
%
% @param input 4 -> number_outputs
%
%       constant
%       number of outputs
%
% @param input 5 -> output_type
%
%       a string
%       consequent MF type
%       2 options are available for now: 
%       "singleton" , "linear" 
%       other types are excluded since not published yet!
%

if output_type == "singleton"

    normalized_firing_strength = repmat(normalized_firing_strength,1,number_outputs);
    output = normalized_firing_strength.* learnable_parameters.singleton.c; % Multiply firing strengths elementwise with output membership values.
    output = sum(output, 1); % Sum the values across the first dimension to get the final output.

elseif output_type == "linear"

    %preparing the input and slopes and biases of linear consequent to
    %multipication
    temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b]; % adding the bias to the end of slope matrix
    x = permute(x,[2 1 3]); 
    temp_input = [x; ones(1, size(x, 2), size(x, 3))]; % Append a row of ones to the input for bias multiplication.
    temp_input = permute(temp_input, [1 3 2]); % Permute input to align dimensions for further operations.

    % Multiply the membership function parameters with the input.
    c = temp_mf*temp_input;
    % Reshape the result to align with the normalized firing strengths.
    c = reshape(c, [size(normalized_firing_strength, 1), number_outputs, size(normalized_firing_strength, 3)]);

    % Replicate normalized firing strengths for each output.
    normalized_firing_strength = repmat(normalized_firing_strength,1,number_outputs);

    % Multiply normalized firing strengths elementwise with the result of linear operation.
    output = normalized_firing_strength.* c;
    output = sum(output, 1); % Sum the values across the first dimension to get the final output.
    output = dlarray(output);

end