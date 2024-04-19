function [loss, gradients, yPred] = fismodelLoss(mini_batch_inputs, number_inputs, targets, number_outputs, number_of_rules, mbs, learnable_parameters, output_membership_type,tnorm)

yPred = fismodel(mini_batch_inputs, number_of_rules, number_inputs,number_outputs, mbs, learnable_parameters, output_membership_type,tnorm);

loss = l2loss(yPred, targets, DataFormat="SCB",NormalizationFactor="batch-size");
gradients = dlgradient(loss, learnable_parameters);

end