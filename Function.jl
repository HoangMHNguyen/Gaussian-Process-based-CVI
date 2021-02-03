
function SE(x1, x2, σ_l, l)
    """
    x1 and x2 are 2 input vectors 
    σ_l: factor that governs the width of the uncertainty of factor
    l: length-scale
    """
    K = zeros(length(x1),length(x2)) #create Kernel matrix
    for i=1:size(K,1)
        for j=1:size(K,2)
            K[i,j]=σ_l^2 * exp(-((x1[i]-x2[j])^2)/(2*(l^2)))
        end
    end
    return K

end