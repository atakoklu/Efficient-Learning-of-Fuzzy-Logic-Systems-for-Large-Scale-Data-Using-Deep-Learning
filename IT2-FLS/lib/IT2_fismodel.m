% commented on 23 JAN 2024 by Ata Koklu
function [output_lower, output_upper, output_mean] = IT2_fismodel(mini_batch_inputs, number_of_rule, number_inputs, number_outputs, mbs, learnable_parameters, output_membership_type, input_membership_type, fuzzy_set_type,CSCM,u,tnorm)
% Interval Type 2 Fuzzy Inferance System Model Structure - Fuzzy Logic System Model Structure
%
% @param output 1 -> output_lower
%
%       (1,oc,mbs) tensor
%       lower bound of the type reduced set of the FLS
%       oc = number of outputs
%       mbs = mini batch size
%       (1,1,1) -> firts lower bound of the type reduced set of the first element of the batch
%
% @param output 2 -> output_upper
%
%       (1,oc,mbs) tensor
%       upper bound of the type reduced set of the FLS
%       oc = number of outputs
%       mbs = mini batch size
%       (1,1,1) -> firts upper bound of the type reduced set of the first element of the batch
%
% @param output 3 -> output_mean
%
%       (1,oc,mbs) tensor
%       Crisp output of the FLS (defuzzified value)
%       oc = number of outputs
%       mbs = mini batch size
%       (1,1,1) -> firts crisp output of the first element of the batch
%
% @param input 1 -> mini_batch_inputs
%
%       (1,ic,mbs) tensor
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> firts input of the first element of the batch
%
% @param input 2 -> number_of_rule
%
%       constant
%       number of Membership Function (MF) for inputs
%       In other words number of rules of the FLS
%
% @param input 3 -> number_inputs
%
%       constant
%       number of inputs
%
% @param input 4 -> number_outputs
%
%       constant
%       number of outputs
%
% @param input 5 -> mbs
%
%       constant
%       mini-batch size
%
% @param input 6 -> learnable_parameters
%
%       struct
%       consist parameters of antecedent and consequent MFs
%
% @param input 7 -> output_membership_type
%
%       a string
%       consequent MF type
%       2 options are available for now: 
%       "singleton" , "linear" 
%       other types are excluded since not published yet!
%
% @param input 8 -> input_membership_type
%
%       a string
%       anticident MF type
%       1 options are available for now: 
%       "gaussmf"
%
% @param input 9 -> fuzzy_set_type
%
%       a string
%       anticident fuzzy set type
%       3 options are available for now: 
%       "H" , "S" , "HS" 
%
% @param input 10 -> CSCM
%
%       a string
%       Center of Set Calculation Method
%       Used for type reduction and defuzzification
%       4 options are available for now: 
%       "SM" , "BMM" , "NT" , "KM" 
%       4 more will be published
%
% @param input 11 -> u
%
%       (R,2^R) binary array
%       contains every possible combination of swithing points of KM for
%       every rule
%       Used for KM and WKM, it is 0 for others
%       Helps to calculate every possible swithcing points in parallel
%
% @param input 12 -> tnorm
%
%       a string
%       T-norm operator type
%       3 options are available for now: 
%       "product" , "HTSK2"     
%       you can use other tnorm operators such as min check
%       "firing_strength_calculation_layer" function
%
%       Note that "HTSK2" uses product but changes fuzzification
%       1 more will be published
%

%fuzzification
[fuzzifed_lower, fuzzifed_upper] = T2_matrix_fuzzification_layer(mini_batch_inputs, input_membership_type,fuzzy_set_type, learnable_parameters, number_of_rule, number_inputs, mbs, tnorm);

%tnorm
if tnorm == "HTSK2"
    [firestrength_lower, firestrength_upper] = T2_firing_strength_calculation_layer(fuzzifed_lower, fuzzifed_upper, "product");
else
    [firestrength_lower, firestrength_upper] = T2_firing_strength_calculation_layer(fuzzifed_lower, fuzzifed_upper, tnorm);
end

%type reduction and defuzzification
[output_lower, output_upper, output_mean] = T2_multioutput_defuzzification_layer(mini_batch_inputs, firestrength_lower, firestrength_upper, learnable_parameters,number_outputs, output_membership_type,CSCM,mbs,number_of_rule,u);


end