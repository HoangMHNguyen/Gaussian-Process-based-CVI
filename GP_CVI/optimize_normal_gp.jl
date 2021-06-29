include("kernel function.jl")
"""
This file stores optimization function for normal GP.
"""
function neg_llh(X_obser::Matrix, y_obser::Vector, σ::Real, l::Real)
    Kff = autocov_mat(X_obser,σ,l);

    eye_ff = 1.0* Matrix(I, size(Kff,1),size(Kff,1));
    Kff = Kff + 1e-5*eye_ff; #add noise for stable computation

    negllh = 1/2*log(det(Kff) + 1e-6) + 1/2*y_obser'*Kff^(-1)*y_obser;
    return negllh

end

function neg_llh(X_obser::Matrix, y_obser::Vector, σ::Real, l::Vector)
    Kff = autocov_mat(X_obser,σ,l);

    eye_ff = 1.0* Matrix(I, size(Kff,1),size(Kff,1));
    Kff = Kff + 1e-5*eye_ff; #add noise for stable computation

    negllh = 1/2*log(det(Kff) + 1e-6) + 1/2*y_obser'*Kff^(-1)*y_obser;
    return negllh

end

##### Optimization
function optim_gp(X_obser::Matrix,y_obser::Vector,σ_val::Real,l_val::Real,η::Real,num_itr::Int)
    for iter = 1:num_itr
        old_σ_val = σ_val; #store old values of σ
        old_l_val = l_val; #store old values of l
        
        # get gradient 
        grads = gradient(σ_val,l_val) do σ, l
            neg_llh(X_obser,y_obser,σ,l)
        end
        
        #Update inducing points and parameters
        σ_val = σ_val - η*grads[1];
        l_val = l_val - η*grads[2];
        if isnan(neg_llh(X_obser,y_obser,σ_val,l_val))
            σ_val = old_σ_val;
            l_val = old_l_val;
            println("The loglikelihood = NaN");
            break
        end
    end

    return σ_val, l_val, neg_llh(X_obser,y_obser,σ_val,l_val)
end

function optim_gp(X_obser::Matrix,y_obser::Vector,σ_val::Real,l_val::Vector,η::Real,num_itr::Int)
    for iter = 1:num_itr
        old_σ_val = σ_val; #store old values of σ
        old_l_val = l_val; #store old values of l
        
        # get gradient 
        grads = gradient(σ_val,l_val) do σ, l
            neg_llh(X_obser,y_obser,σ,l)
        end
        
        #Update inducing points and parameters
        σ_val = σ_val - η*grads[1];
        l_val = l_val - η*grads[2];
        if isnan(neg_llh(X_obser,y_obser,σ_val,l_val))
            σ_val = old_σ_val;
            l_val = old_l_val;
            println("The loglikelihood = NaN");
            break
        end
    end

    return σ_val, l_val, neg_llh(X_obser,y_obser,σ_val,l_val)
end

function optim_gp_adam(X_obser::Matrix,y_obser::Vector,σ_val::Real,l_val::Real,η::Real,batch_size::Int, epoch::Int)
    opt = ADAM(η); #adam optimizer
    N = size(X_obser,2)
    for i=1:epoch
        for batch = 1:batch_size:N
            grads = gradient(σ_val, l_val) do σ, l
                neg_llh(X_obser[:,batch:(batch+batch_size-1)],y_obser[batch:(batch+batch_size-1)],σ,l)
            end
            g = [grads[1];grads[2]]
            ps = [σ_val;l_val]
            update!(opt,ps,g);
            σ_val, l_val = ps[1], ps[2]
        end
    end
    return σ_val, l_val, neg_llh(X_obser,y_obser,σ_val,l_val)
end

function optim_gp_adam(X_obser::Matrix,y_obser::Vector,σ_val::Real,l_val::Vector,η::Real,batch_size::Int, epoch::Int)
    opt = ADAM(η); #adam optimizer
    N = size(X_obser,2)
    for i=1:epoch
        for batch = 1:batch_size:N
            grads = gradient(σ_val, l_val) do σ, l
                neg_llh(X_obser[:,batch:(batch+batch_size-1)],y_obser[batch:(batch+batch_size-1)],σ,l)
            end
            g = [grads[1];grads[2]]
            ps = [σ_val;l_val]
            update!(opt,ps,g);
            σ_val, l_val = ps[1], ps[2:end]
        end
    end

    return σ_val, l_val, neg_llh(X_obser,y_obser,σ_val,l_val)
end

