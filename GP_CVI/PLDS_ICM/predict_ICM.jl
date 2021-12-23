include("kernel_ICM.jl")

# This file stores the multiGP prediction function for the future value
function icm_predict(x_obser::Matrix, y_obser::Matrix, x_new::Vector, A::Matrix, σ::Real, l::Real)
    Kff = autocov_mat(x_obser,A,σ,l); #auto-covariance of training points
    K_new = autocov_mat(x_new,A,σ,l); #auto-covariance of test points

    eye_ff = 1.0* Matrix(I, size(Kff,1),size(Kff,1));
    eye_new = 1.0* Matrix(I, size(K_new,1),size(K_new,1));
    Kff = Kff + 1e-5*eye_ff; #add noise for stable computation
    K_new = K_new .+ 1e-5*eye_new; #add noise for stable computation

    K_cross_f_new = cross_cov_mat(x_obser,x_new,A,σ,l); #cross-covariance

    y_obser = vcat(y_obser'...);

    predictive_mean = K_cross_f_new' * Kff^(-1) * y_obser;
    predictive_var  = K_new - K_cross_f_new' * Kff^(-1) * K_cross_f_new;

    return predictive_mean, predictive_var
    
end

function icm_predict(x_obser::Matrix, y_obser::Matrix, x_new::Vector, A::Matrix, σ::Real, l::Vector)
    y_obser = vcat(y_obser'...);
    Kff = autocov_mat(x_obser,A,σ,l); #auto-covariance of training points
    K_new = autocov_mat(x_new,A,σ,l); #auto-covariance of test points

    eye_ff = 1.0* Matrix(I, size(Kff,1),size(Kff,1));
    eye_new = 1.0* Matrix(I, size(K_new,1),size(K_new,1));
    Kff = Kff + 1e-5*eye_ff; #add noise for stable computation
    K_new = K_new .+ 1e-5*eye_new; #add noise for stable computation

    K_cross_f_new = cross_cov_mat(x_obser,x_new,A,σ,l); #cross-covariance

    predictive_mean = K_cross_f_new' * Kff^(-1) * y_obser;
    predictive_var  = K_new - K_cross_f_new' * Kff^(-1) * K_cross_f_new;

    return predictive_mean, predictive_var

end