% 15 JULY 2023
% Last changed and commented on 23 JAN 2024 by Ata Koklu
function [output_lower, output_upper] = T2_firing_strength_calculation_layer(lower_membership_values,upper_membership_values,operator_type)
%
%       Calculates firing strengths
%
% @param output 1 -> output_lower
%
%       (n,1,mbs) vector 
%       n = number of rows in input
%       (:,1,1) -> lower firing strength of each rule
%
% @param output 2 -> output_upper
%
%       (n,1,mbs) vector 
%       n = number of rows in input
%       (:,1,1) -> upper firing strength of each rule
%
% @param input 1 -> lower_membership_values
%
%       (mfc,ic,mbs) tensor
%       lower fuzzified value
%       mfc = number of input membership functions (number of rules)
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> lower fuzzified value of the first input with first Membership
%       Function (MF) of that input of first element of the batch
%
% @param input 2 -> upper_membership_values
%
%       (mfc,ic,mbs) tensor
%       upper fuzzified value
%       mfc = number of input membership functions (number of rules)
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> upper fuzzified value of the first input with first Membership
%       Function (MF) of that input of first element of the batch
%
% @param input 3 -> operator_type
%
%       a string
%       operator type that will be applied in rules
%       
if operator_type == "product"

    lower_firing_strength = prod(lower_membership_values,2);
    upper_firing_strength = prod(upper_membership_values,2);

elseif operator_type == "sum"

    lower_firing_strength = sum(lower_membership_values,2);
    upper_firing_strength = sum(upper_membership_values,2);

elseif operator_type == "max"

    lower_firing_strength = max(lower_membership_values,[],2);
    upper_firing_strength = max(upper_membership_values,[],2);

elseif operator_type == "min"

    lower_firing_strength = min(lower_membership_values,[],2);
    upper_firing_strength = min(upper_membership_values,[],2);

elseif operator_type == "mean"

    lower_firing_strength = mean(lower_membership_values,2);
    upper_firing_strength = mean(upper_membership_values,2);
end

output_lower = lower_firing_strength;
output_upper = upper_firing_strength;

end