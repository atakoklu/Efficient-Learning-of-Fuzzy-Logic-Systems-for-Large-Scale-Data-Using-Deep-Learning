function tilted_loss = tilted_loss(y, y_lower, y_upper, q1, q2, mbs)

lower_loss = 1/mbs*(sum(max(q1*(y-y_lower), (q1-1)*(y-y_lower)),3));

% temp_lower_1 = q1*(y-y_lower);
% temp_lower_2 = (q1-1)*(y-y_lower);
% temp_lower = max(temp_lower_1,temp_lower_2);
% temp_lower = sum(temp_lower,3); %summing in the direction of minibatch
% 
% 
% 
% lower_loss = 1/mbs*temp_lower;



upper_loss = 1/mbs*(sum(max(q2*(y-y_upper), (q2-1)*(y-y_upper)),3));

% temp_upper_1 = q2*(y-y_upper);
% temp_upper_2 = (q2-1)*(y-y_upper);
% temp_upper = max(temp_upper_1,temp_upper_2);
% temp_upper = sum(temp_upper,3); %summing in the direction of minibatch
% 
% 
% 
% upper_loss = 1/mbs*temp_upper;




tilted_loss = lower_loss + upper_loss;

end

