% 01 AUG 2023
% commented on 23 JAN 2024 by Ata Koklu
function [output_lower, output_upper, output_mean] = T2_multioutput_defuzzification_layer(x,lower_firing_strength,upper_firing_strength, learnable_parameters, number_outputs, output_membership_type,CSCM,mbs,number_of_rules,u)
%       Calculates the type reduced sets and defuzzified crisp output
%
%       Mathematical manupulations are too complicated to be explained via
%       comments, please read the paper. Also an aditional document on the
%       implementation of KM will be published.
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
% @param input 1 -> x
%
%       (1,ic,mbs) tensor
%       ic = number of inputs
%       mbs = mini batch size
%       (1,1,1) -> firts input of the first element of the batch
%
% @param input 2 -> lower_firing_strength
%
%       (n,1,mbs) vector 
%       n = number of rows in input
%       (:,1,1) -> lower firing strength of each rule
%
% @param input 3 -> upper_firing_strength
%
%       (n,1,mbs) vector 
%       n = number of rows in input
%       (:,1,1) -> upper firing strength of each rule
%
% @param input 4 -> learnable_parameters
%
%       struct
%       consist parameters of antecedent and consequent MFs
%
% @param input 5 -> number_outputs
%
%       constant
%       number of outputs
%
% @param input 6 -> output_membership_type
%
%       a string
%       consequent MF type
%       2 options are available for now: 
%       "singleton" , "linear" 
%       other types are excluded since not published yet!
%
% @param input 7 -> CSCM
%
%       a string
%       Center of Set Calculation Method
%       Used for type reduction and defuzzification
%       8 options are available for now: 
%       "SM" , "BMM" , "NT" , "KM" 
%       4 more will be published
%
% @param input 8 -> mbs
%
%       constant
%       mini-batch size
%
% @param input 9 -> number_of_rule
%
%       constant
%       number of Membership Function (MF) for inputs
%       In other words number of rules of the FLS
%
% @param input 10 -> u
%
%       (R,2^R) binary array
%       contains every possible combination of swithing points of KM for
%       every rule
%       Used for KM and WKM, it is 0 for others
%       Helps to calculate every possible swithcing points in parallel
%
if output_membership_type == "singleton"

    if CSCM == "SM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);

        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);

        output_lower = normalized_lower_firing_strength.* learnable_parameters.singleton.c;% first we multiply elementwise our firing strengths with output memberships
        output_upper = normalized_upper_firing_strength.* learnable_parameters.singleton.c;

        output_lower = sum(output_lower, 1); %then we sum with respect to dimension 1
        output_upper = sum(output_upper, 1); %then we sum with respect to dimension 1

        output_mean = (output_lower + output_upper)./2;

    elseif CSCM == "BMM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);

        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);

        output_lower = normalized_lower_firing_strength.* learnable_parameters.singleton.c;% first we multiply elementwise our firing strengths with output memberships
        output_upper = normalized_upper_firing_strength.* learnable_parameters.singleton.c;

        output_lower = sum(output_lower, 1); %then we sum with respect to dimension 1
        output_upper = sum(output_upper, 1); %then we sum with respect to dimension 1

        alpha = sigmoid(learnable_parameters.singleton.alpha);

        output_mean = (output_lower .* alpha) + (output_upper .* (1 - alpha));

    elseif CSCM == "NT"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        numerator_lower = lower_firing_strength_temp.* learnable_parameters.singleton.c;
        numerator_upper = upper_firing_strength_temp.* learnable_parameters.singleton.c;

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength, 1) + sum(upper_firing_strength, 1);

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;


    elseif CSCM == "KM"


        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

%         pay2 = reshape((permute(learnable_parameters.singleton.c,[3 2 1]).*permute(delta_f,[1 3 2])),[mbs*number_outputs,number_of_rules])*u;
        pay2 = pagemtimes((permute(learnable_parameters.singleton.c,[3 1 2]).*delta_f),u);

%         pay2 = reshape(pay2,mbs, number_outputs,[]);
        pay2 = permute(pay2,[2,3,1]);
        pay1 = sum(learnable_parameters.singleton.c.* lower_firing_strength,1);

        pay = pay1 + pay2;

        %         clear pay1_upper pay2_upper
        %         clear delta_f u

        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;

        %         clear payda1 payda2

        pay = permute(pay,[2 1 3]);

        output = pay./payda;

        %         clear pay_lower pay_upper payda

        output_lower = permute(min(output,[],2),[2 1 3]);
        output_upper = permute(max(output,[],2),[2 1 3]);

        %         clear output_lower_temp output_upper_temp

        output_mean = (output_lower + output_upper)./2;

    
    end


elseif output_membership_type == "linear"


    if CSCM == "SM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        c = temp_mf*temp_input;
        c= reshape(c, [size(normalized_lower_firing_strength, 1), number_outputs, size(normalized_lower_firing_strength, 3)]);


        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);


        output_lower = normalized_lower_firing_strength.* c;
        output_upper = normalized_upper_firing_strength.* c;

        output_lower = sum(output_lower, 1); %then we sum with respect to dimension 1
        output_upper = sum(output_upper, 1); %then we sum with respect to dimension 1

        output_mean = (output_lower + output_upper)./2;

        

    elseif CSCM == "BMM"

        normalized_lower_firing_strength = lower_firing_strength./sum(lower_firing_strength, 1);
        normalized_upper_firing_strength = upper_firing_strength./sum(upper_firing_strength, 1);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation


        c = temp_mf*temp_input;
        c= reshape(c, [size(normalized_lower_firing_strength, 1), number_outputs, size(normalized_lower_firing_strength, 3)]);


        normalized_lower_firing_strength = repmat(normalized_lower_firing_strength,1,number_outputs);
        normalized_upper_firing_strength = repmat(normalized_upper_firing_strength,1,number_outputs);


        output_lower = normalized_lower_firing_strength.* c;
        output_upper = normalized_upper_firing_strength.* c;

        output_lower = sum(output_lower, 1); %then we sum with respect to dimension 1
        output_upper = sum(output_upper, 1); %then we sum with respect to dimension 1


        alpha = sigmoid(learnable_parameters.linear.alpha);

        output_mean = (output_lower .* alpha) + (output_upper .* (1 - alpha));


    elseif CSCM == "NT"

        lower_firing_strength_temp = repmat(lower_firing_strength,1,number_outputs);
        upper_firing_strength_temp = repmat(upper_firing_strength,1,number_outputs);

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

        numerator_lower = lower_firing_strength_temp.* c;
        numerator_upper = upper_firing_strength_temp.* c;

        numerator_lower = 2*sum(numerator_lower, 1);
        numerator_upper = 2*sum(numerator_upper, 1);

        denominator = sum(lower_firing_strength, 1) + sum(upper_firing_strength, 1);

        output_lower = numerator_lower./denominator;
        output_upper = numerator_upper./denominator;

        output_mean = (output_lower + output_upper)./2;


    elseif CSCM == "KM"

        temp_mf = [learnable_parameters.linear.a,learnable_parameters.linear.b];
        x = permute(x,[2 1 3]);
        temp_input = [x; ones(1, size(x, 2), size(x, 3))];
        temp_input = permute(temp_input, [1 3 2]); %for dlarray implementation

        c = temp_mf*temp_input;
        c = reshape(c, [size(lower_firing_strength, 1), number_outputs, size(lower_firing_strength, 3)]);

        delta_f = upper_firing_strength - lower_firing_strength;
        delta_f = permute(delta_f,[3 1 2]);

        payda2 = delta_f*u;

        %         pay2 = pagemtimes((permute(c,[3 1 2]).*repmat(delta_f,1,1,number_outputs)),u);
        pay2 = pagemtimes((permute(c,[3 1 2]).*delta_f),u);

        pay2 = permute(pay2,[3,2,1]);
        pay1 = permute(sum(c .* lower_firing_strength,1),[2 1 3]);

        pay = pay1 + pay2;

        %         clear pay1_upper pay2_upper
        %         clear delta_f u

        payda2 = permute(payda2,[3,2,1]);
        payda1 = sum(lower_firing_strength,1);

        payda = payda1 + payda2;

        %         clear payda1 payda2

        output = pay./payda;

        %         clear pay_lower pay_upper payda

        output_lower = permute(min(output,[],2),[2 1 3]);
        output_upper = permute(max(output,[],2),[2 1 3]);

        %         clear output_lower_temp output_upper_temp

        output_mean = (output_lower + output_upper)./2;

   
    end

end

output_lower = dlarray(output_lower);
output_upper = dlarray(output_upper);
output_mean = dlarray(output_mean);

end