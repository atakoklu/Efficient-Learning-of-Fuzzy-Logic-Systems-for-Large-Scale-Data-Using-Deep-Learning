% 20 OCT 2023
function [output_lower,output_upper] = IT2_firing_strength_normalization_layer(fire_strength_lower,fire_strength_upper)
% v0.1 compatibel with mini-batch
% 
% fire strength are normalized to have their sum equal to 1
%
% @param output -> output
%
%       (rc,1,mbs) tensor
%       rc = number of rules  
%       mbs = mini-batch size
%       (1,1,1) -> normalized firing strength of the first rule of the
%       first element of the batch
%
% @param input -> fire_strength
%
%       (rc,1,mbs) tensor
%       rc = number of rules  
%       mbs = mini-batch size
%       (1,1,1) -> firing strength of the first rule of the
%       first element of the batch

    output_lower = fire_strength_lower./sum(fire_strength_lower, 1);
    output_upper = fire_strength_upper./sum(fire_strength_upper, 1);

end