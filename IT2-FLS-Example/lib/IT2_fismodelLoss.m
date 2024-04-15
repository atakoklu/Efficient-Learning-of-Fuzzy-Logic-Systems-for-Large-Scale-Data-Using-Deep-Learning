% commented on 23 JAN 2024
function [loss, gradients, yPred_lower, yPred_upper, yPred] = IT2_fismodelLoss(x, number_inputs, targets,number_outputs, number_of_rules, mbs, learnable_parameters, output_type, input_mf_type, fuzzy_set_type,CSCM,u,tnorm)
% IT2 FLS Model
[yPred_lower, yPred_upper, yPred] = IT2_fismodel(x, number_of_rules, number_inputs,number_outputs, mbs, learnable_parameters, output_type, input_mf_type, fuzzy_set_type,CSCM,u,tnorm);

% loss for accuracy 
loss = l2loss(yPred, targets, DataFormat="SCB",NormalizationFactor="batch-size");

gradients = dlgradient(loss, learnable_parameters);

end