function [x_norm,mean,std] = zscore_norm(x,mean,std)
arguments
    x = [];
    mean = [];
    std = [];
end

if isempty(mean) || isempty(std)
    [x_norm,mean,std] = zscore(x);
else
    x_norm = (x - mean)./std;
end

end