include("kernel_ICM.jl")

### Negative log-likelihood ###
function neg_llh(X_obser::Matrix, y_obser::Vector, A::Matrix, σ::Real, l::Real)
    Kff = autocov_mat(X_obser,A, σ, l);

    eye_ff = 1.0* Matrix(I, size(Kff,1),size(Kff,1));
    Kff = Kff + 1e-5*eye_ff; #add noise for stable computation

    negllh = 1/2*log(det(Kff) + 1e-6) + 1/2*y_obser'*Kff^(-1)*y_obser;
    return negllh

end

function neg_llh(X_obser::Matrix, y_obser::Vector, A::Matrix, σ::Real, l::Vector)
    Kff = autocov_mat(X_obser,A, σ, l);

    eye_ff = 1.0* Matrix(I, size(Kff,1),size(Kff,1));
    Kff = Kff + 1e-5*eye_ff; #add noise for stable computation

    negllh = 1/2*log(det(Kff) + 1e-6) + 1/2*y_obser'*Kff^(-1)*y_obser;
    return negllh

end

### Optimization
# Gradient descent
function optim_gp(X_obser::Matrix,y_obser::Matrix, A::Matrix, σ_val::Real, l_val::Real, η::Real,num_itr::Int)
    y_obser = vcat(y_obser'...);
    for iter = 1:num_itr
        old_σ_val = σ_val; #store old values of σ
        old_l_val = l_val; #store old values of l
        
        # get gradient 
        grads = gradient(σ_val,l_val) do σ, l 
            neg_llh(X_obser,y_obser,A,σ,l)
        end
        
        #Update inducing points and parameters
        σ_val = σ_val - η*grads[1];
        l_val = l_val - η*grads[2];
        if isnan(neg_llh(X_obser,y_obser,A,σ_val,l_val))
            σ_val = old_σ_val;
            l_val = old_l_val;
            println("The loglikelihood = NaN");
            break
        end
    end

    return σ_val, l_val
end

function optim_gp(X_obser::Matrix,y_obser::Matrix, A::Matrix, σ_val::Real, l_val::Vector, η::Real,num_itr::Int)
    y_obser = vcat(y_obser'...);
    for iter = 1:num_itr
        old_σ_val = σ_val; #store old values of σ
        old_l_val = l_val; #store old values of l
        # get gradient 
        grads = gradient(σ_val,l_val) do σ, l 
            neg_llh(X_obser,y_obser,A,σ,l)
        end
        
        #Update inducing points and parameters
        σ_val = σ_val - η*grads[1];
        l_val = l_val - η*grads[2];

        if isnan(neg_llh(X_obser,y_obser,A,σ_val,l_val))
            σ_val = old_σ_val;
            l_val = old_l_val;
            println("The loglikelihood = NaN");
            break
        end
    end

    return σ_val, l_val
end

# Adam
function optim_gp_adam(X_obser::Matrix,y_obser::Matrix, A::Matrix, σ_val::Real, l_val::Real, η::Real, batch_size::Int, epoch::Int)
    opt = ADAM(η); #adam optimizer
    N = size(X_obser,2)
    for i=1:epoch
        for batch = 1:batch_size:N
            y_batch = y_obser[:,batch:(batch+batch_size-1)]
            y_batch = vcat(y_batch'...);
            grads = gradient(σ_val,l_val) do σ, l 
                neg_llh(X_obser[:,batch:(batch+batch_size-1)],y_batch, A, σ, l)
            end
            g = [grads[1];grads[2]]
            ps = [σ_val;l_val]
            update!(opt,ps,g);
            σ_val, l_val = ps[1], ps[2]

        end
    end

    return σ_val, l_val
end


function optim_gp_adam(X_obser::Matrix,y_obser::Matrix, A::Matrix, σ_val::Real, l_val::Vector, η::Real, batch_size::Int, epoch::Int)
    opt = ADAM(η); #adam optimizer
    N = size(X_obser,2)
    for i=1:epoch
        for batch = 1:batch_size:N
            y_batch = y_obser[:,batch:(batch+batch_size-1)]
            y_batch = vcat(y_batch'...);
            grads = gradient(σ_val,l_val) do σ, l 
                neg_llh(X_obser[:,batch:(batch+batch_size-1)],y_batch, A, σ, l)
            end
            g = [grads[1];grads[2]]
            ps = [σ_val;l_val]
            update!(opt,ps,g);
            σ_val, l_val = ps[1:length(σ_val)], ps[length(σ_val)+1:length(σ_val)+length(l_val)]
            σ_val= σ_val[1];
        end
    end

    return σ_val, l_val
end

#AdaMax
function optim_gp_adamax(X_obser::Matrix,y_obser::Matrix, A::Matrix, σ_val::Real, l_val::Real, η::Real, batch_size::Int, epoch::Int)
    opt = AdaMax(η); #adam optimizer
    N = size(X_obser,2)
    for i=1:epoch
        for batch = 1:batch_size:N
            y_batch = y_obser[:,batch:(batch+batch_size-1)]
            y_batch = vcat(y_batch'...);
            grads = gradient(σ_val,l_val) do σ, l 
                neg_llh(X_obser[:,batch:(batch+batch_size-1)],y_batch, A, σ, l)
            end
            g = [grads[1];grads[2]]
            ps = [σ_val;l_val]
            update!(opt,ps,g);
            σ_val, l_val = ps[1], ps[2]

        end
    end

    return σ_val, l_val
end


function optim_gp_adamax(X_obser::Matrix,y_obser::Matrix, A::Matrix, σ_val::Real, l_val::Vector, η::Real, batch_size::Int, epoch::Int)
    opt = AdaMax(η); #adam optimizer
    N = size(X_obser,2)
    for i=1:epoch
        for batch = 1:batch_size:N
            y_batch = y_obser[:,batch:(batch+batch_size-1)]
            y_batch = vcat(y_batch'...);
            grads = gradient(σ_val,l_val) do σ, l 
                neg_llh(X_obser[:,batch:(batch+batch_size-1)],y_batch, A, σ, l)
            end
            g = [grads[1];grads[2]]
            ps = [σ_val;l_val]
            update!(opt,ps,g);
            σ_val, l_val = ps[1:length(σ_val)], ps[length(σ_val)+1:length(σ_val)+length(l_val)]
            σ_val = σ_val[1];
        end
    end

    return σ_val, l_val
end