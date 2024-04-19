% 22 JUN 2023
function output = firing_strength_normalization_layer(fire_strength)
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

    output = fire_strength./sum(fire_strength, 1);

end