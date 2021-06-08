include("kernel function.jl")
"""
This file contains functions for:
   - Finding the posterior p(f_u|f), i.e. the info of inducing points given observations
   - Predicting new values
   - Online updating 
Remember: X always has the shape DxN. If X is 1-D, then X will be a row vector 1xN.
If we let X be Nx1, then our computation will be wrong.
"""
######### Find the posterior p(f_u|f) of the inducing points ###########
function FITC_post_indu(X::Matrix, Xu::Matrix, y::Vector, σ::Real, l::Real)
   """
   X: matrix of training points (DxN), N is observation number
   Xu: matrix of inducing points (DxU), U is number of inducing points
   y: vector of observed output function values (Nx1)
   """
   Kff = autocov_mat(X,σ,l); #auto-covariance matrix of training set
   Kuu = autocov_mat(Xu,σ,l); #auto-covariance matrix of inducing set

   eye_ff = 1.0* Matrix(I, size(Kff,1),size(Kff,1));
   eye_uu = 1.0* Matrix(I, size(Kuu,1),size(Kuu,1));
   Kff = Kff + 1e-5*eye_ff; #add noise for stable computation
   Kuu = Kuu + 1e-5*eye_uu; #add noise for stable computation

   Kuf = cross_cov_mat(Xu,X,σ,l); #cross-covariance matrix

   Λff = Diagonal(Kff - Kuf'*Kuu^(-1)*Kuf); # diagonal matrix 

   #calculate the posterior mean and variance
   Σu = Kuu*(Kuu + Kuf*Λff^(-1)*Kuf')^(-1)*Kuu; #variance of inducing points (MxM)
   μu = Σu*Kuu^(-1)*Kuf*Λff^(-1)*y; # mean of inducing points (Mx1)

   return μu, Σu
end

function FITC_post_indu(X::Matrix, Xu::Matrix, y::Vector, σ::Real, l::Vector)
   """
   X: matrix of training points (DxN), N is observation number
   Xu: matrix of inducing points (DxU), U is number of inducing points
   y: vector of observed function values (Nx1)
   """
   Kff = autocov_mat(X,σ,l); #auto-covariance matrix of training set
   Kuu = autocov_mat(Xu,σ,l); #auto-covariance matrix of inducing set

   eye_ff = 1.0* Matrix(I, size(Kff,1),size(Kff,1));
   eye_uu = 1.0* Matrix(I, size(Kuu,1),size(Kuu,1));
   Kff = Kff + 1e-5*eye_ff; #add noise for stable computation
   Kuu = Kuu + 1e-5*eye_uu; #add noise for stable computation

   Kuf = cross_cov_mat(Xu,X,σ,l); #cross-covariance matrix

   Λff = Diagonal(Kff - Kuf'*Kuu^(-1)*Kuf); # diagonal matrix 

   #calculate the posterior mean and variance
   Σu = Kuu*(Kuu + Kuf*Λff^(-1)*Kuf')^(-1)*Kuu; #variance of inducing points (MxM)
   μu = Σu*Kuu^(-1)*Kuf*Λff^(-1)*y; #mean of inducing points (Mx1)

   return μu, Σu
end

############# Predicting ################
function FITC_predict(Xu::Matrix, x_new::Vector, μu::Vector, Σu::Matrix, σ::Real, l::Real)
   """
   Xu: matrix of inducing points (DxU), U is number of inducing points
   x_new: the point at which we need to predict (Dx1 vector)
   μu: mean of inducing points (Ux1 vector)
   Σu: variance of inducing points (UxU matrix)
   """
   Kuu = autocov_mat(Xu,σ,l); #auto-covariance matrix of inducing set
   K_new = autocov_mat(x_new,σ,l); #auto-covariance of new point (scalar)

   eye_uu = 1.0* Matrix(I, size(Kuu,1),size(Kuu,1));
   Kuu = Kuu + 1e-5*eye_uu; #add noise for stable computation

   K_new_u = cross_cov_mat(x_new,Xu,σ,l); #cross-covariance matrix

   #Prediction
   μ_predict = K_new_u*Kuu^(-1)*μu; #predictive mean (scalar)
   σ_predict = K_new .- (K_new_u*Kuu^(-1)*(Kuu - Σu)*Kuu^(-1)*K_new_u'); #predictive variance (scalar), change here


   return μ_predict, σ_predict
end

function FITC_predict(Xu::Matrix, x_new::Vector,μu::Vector,Σu::Matrix,σ::Real,l::Vector)
   """
   Xu: matrix of inducing points (DxU), U is number of inducing points
   x_new: the point at which we need to predict (Dx1 vector)
   μu: mean of inducing points (Ux1 vector)
   Σu: variance of inducing points (UxU matrix)
   """
   Kuu = autocov_mat(Xu,σ,l); #auto-covariance matrix of inducing set
   K_new = autocov_mat(x_new,σ,l); #auto-covariance of new point (scalar) 

   eye_uu = 1.0* Matrix(I, size(Kuu,1),size(Kuu,1));
   Kuu = Kuu + 1e-5*eye_uu; #add noise for stable computation

   K_new_u = cross_cov_mat(x_new,Xu,σ,l); #cross-covariance matrix

   #Prediction
   μ_predict = K_new_u*Kuu^(-1)*μu; #predictive mean
   σ_predict = K_new .- K_new_u*Kuu^(-1)*(Kuu - Σu)*Kuu^(-1)*K_new_u'; #predictive variance, change here

   return μ_predict, σ_predict
end

function FITC_predict(Xu::Matrix, x_new::Matrix, μu::Vector, Σu::Matrix, σ::Real, l::Real)
   """
   Xu: matrix of inducing points (DxU), U is number of inducing points
   x_new: Matrix of points at which we need to predict (DxM)
   μu: mean of inducing points (Ux1 vector)
   Σu: variance of inducing points (UxU matrix)
   """
   Kuu = autocov_mat(Xu,σ,l); #auto-covariance matrix of inducing set
   K_new = autocov_mat(x_new,σ,l); #auto-covariance matrix of new point

   eye_uu = 1.0* Matrix(I, size(Kuu,1),size(Kuu,1));
   eye_new = 1.0* Matrix(I, size(K_new,1),size(K_new,1));
   Kuu = Kuu + 1e-5*eye_uu; #add noise for stable computation
   K_new = K_new + 1e-5*eye_new; #add noise for stable computation

   K_new_u = cross_cov_mat(x_new,Xu,σ,l); #cross-covariance matrix

   #Prediction
   μ_predict = K_new_u*Kuu^(-1)*μu; #predictive mean
   σ_predict = K_new - K_new_u*Kuu^(-1)*(Kuu - Σu)*Kuu^(-1)*K_new_u'; #predictive variance

   return μ_predict, σ_predict
end

function FITC_predict(Xu::Matrix, x_new::Matrix,μu::Vector,Σu::Matrix,σ::Real,l::Vector)
   """
   Xu: matrix of inducing points (DxU), U is number of inducing points
   X_new: matrix of points at which we need to predict (DxM)
   μu: mean of inducing points (Ux1 vector)
   Σu: variance of inducing points (UxU matrix)
   """
   Kuu = autocov_mat(Xu,σ,l); #auto-covariance matrix of inducing set
   K_new = autocov_mat(x_new,σ,l); #auto-covariance matrix of new point

   eye_uu = 1.0* Matrix(I, size(Kuu,1),size(Kuu,1));
   eye_new = 1.0* Matrix(I, size(K_new,1),size(K_new,1));
   Kuu = Kuu + 1e-5*eye_uu; #add noise for stable computation
   K_new = K_new + 1e-5*eye_new; #add noise for stable computation

   K_new_u = cross_cov_mat(x_new,Xu,σ,l); #cross-covariance matrix

   #Prediction
   μ_predict = K_new_u*Kuu^(-1)*μu; #predictive mean
   σ_predict = K_new - K_new_u*Kuu^(-1)*(Kuu - Σu)*Kuu^(-1)*K_new_u'; #predictive variance

   return μ_predict, σ_predict
end

################ online updating  ##################
function FITC_online_update(Xu::Matrix, x_new::Vector, y_new::Real, μu::Vector, Σu::Matrix, σ::Real, l::Real)
   """
   Xu: inducing points
   x_new: new observation (Dx1 vector)
   μ and Σ: old info of inducing points
   y_new: new observed value (scalar)
   """
   Kuu = autocov_mat(Xu,σ,l); 
   K_new = autocov_mat(x_new,σ,l); #auto-covariance of new observation Kf+f+ (scalar)

   eye_uu = 1.0* Matrix(I, size(Kuu,1),size(Kuu,1));
   Kuu = Kuu + 1e-5*eye_uu; #add noise for stable computation

   K_new_u = cross_cov_mat(x_new,Xu,σ,l); #cross-covariance matrix

   Λ = Diagonal(K_new .- K_new_u*Kuu^(-1)*K_new_u');
   P = Kuu^(-1) * K_new_u' * Λ^(-1) * K_new_u * Kuu^(-1);

   eye = 1.0* Matrix(I, size(Kuu,1),size(Kuu,1));
   Σu_new = Σu - (Σu*P*Σu)/(1+tr(Σu*P)); #update variance
   μu_new = (eye - (Σu*P)/(1+tr(Σu*P)))*μu + Σu_new*Kuu^(-1)*K_new_u'*Λ^(-1)*y_new; #update mean

   return μu_new[:], Σu_new 

end

function FITC_online_update(Xu::Matrix, x_new::Vector, y_new::Real, μu::Vector, Σu::Matrix, σ::Real, l::Vector)
   """
   Xu: inducing points
   x_new: new observation (Dx1 vector)
   μ and Σ: old info of inducing points
   y_new: new observed value (scalar)
   """
   Kuu = autocov_mat(Xu,σ,l); 
   K_new = autocov_mat(x_new,σ,l); #auto-covariance of new observation Kf+f+ (scalar)

   eye_uu = 1.0* Matrix(I, size(Kuu,1),size(Kuu,1));
   Kuu = Kuu + 1e-5*eye_uu; #add noise for stable computation

   K_new_u = cross_cov_mat(x_new,Xu,σ,l); #cross-covariance matrix

   Λ = Diagonal(K_new .- K_new_u*Kuu^(-1)*K_new_u');
   P = Kuu^(-1) * K_new_u' * Λ^(-1) * K_new_u * Kuu^(-1);

   eye = 1.0* Matrix(I, size(Kuu,1),size(Kuu,1));
   Σu_new = Σu - (Σu*P*Σu)/(1+tr(Σu*P)); #update variance
   μu_new = (eye - (Σu*P)/(1+tr(Σu*P)))*μu + Σu_new*Kuu^(-1)*K_new_u'*Λ^(-1)*y_new; #update mean

   return μu_new[:], Σu_new 

end
