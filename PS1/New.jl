using Pkg
using Random
using Distributions
using LinearAlgebra 
using Optim
using PlotlyJS
using DataFrames
using CSV
using DataFrames
using StatsBase
using Optim
#using Symbolics

#********************
#Step0 Load Data
#********************
simulated_Rodrigo = DataFrame(CSV.read("D:\\BC PhD\\Sem 3\\IO\\PS1\\pset1_data_new.csv",DataFrame))

X = zeros(100,3,3)
for j=1:3
    X[:,:,j] = Matrix(groupby(simulated_Rodrigo,"Product")[j][:,["X_1","X_2","X_3"]])
end

p=zeros(100,3)
for j=1:3
    p[:,j] = Matrix(groupby(simulated_Rodrigo,"Product")[j][:,["prices"]])
end

s=zeros(100,3)
for j=1:3
    s[:,j] = Matrix(groupby(simulated_Rodrigo,"Product")[j][:,["shares"]])
end

W=zeros(100,3)
Z=zeros(100,3)
for j=1:3
    W[:,j] = Matrix(groupby(simulated_Rodrigo,"Product")[j][:,["W"]])
    Z[:,j] = Matrix(groupby(simulated_Rodrigo,"Product")[j][:,["Z"]])
end
    

#********************
#Step1 Demand Side
#********************

#Guess Parameter Estimates (β_1,β_2,β_3,α,σ_α):
actual = [5,1,1,1,1]
parameter_guess = [6,0.5,2,1.2,2]


d_3 = LogNormal(0,1)
ν_sim = rand(d_3,100000)

#Step 1.2: Inversion to find δ from shares
function share_prediction(δ, p, θ, ν)
  
    s = zeros(100,3)
    s_i = zeros(100,1000,3)
    l=0
    for m=1:100
        for i=1:1000
            μ_i = θ[5]*p[m,1]*ν[i+l]
            #μ_i = 0

            s_i[m,i,1] = exp(δ[m,1]+μ_i)/(1+exp(δ[m,1]+μ_i)+exp(δ[m,2]+μ_i)+exp(δ[m,3]+μ_i))     
            s_i[m,i,2] = exp(δ[m,2]+μ_i)/(1+exp(δ[m,1]+μ_i)+exp(δ[m,2]+μ_i)+exp(δ[m,3]+μ_i))     
            s_i[m,i,3] = exp(δ[m,3]+μ_i)/(1+exp(δ[m,1]+μ_i)+exp(δ[m,2]+μ_i)+exp(δ[m,3]+μ_i))           
        end
        s[m,1] = sum(s_i[m,:,1])/1000
        s[m,2] = sum(s_i[m,:,2])/1000
        s[m,3] = sum(s_i[m,:,3])/1000  
        l=1000*m
    end
    return s
end

function contraction_map(s, p, δ_new, θ, ν)
    δ_guess = zeros(100,3)
    while norm(δ_new - δ_guess) > .1
        δ_guess = δ_new
        s_pred = share_prediction(δ_guess,p, θ, ν)
        δ_new = δ_guess + log.(s) - log.(s_pred)
    end
    return δ_new
end


function moment_objective_fn(θ, s, p, X, W, Z, ν_sim)
    δ_guess = rand(100,3)
    δ_cm = contraction_map(s, p, δ_guess, θ, ν_sim)

    # IV-GMM (Only Demand Size)
    x = zeros(100,4,3)
    x = cat([X[:,:,1] p[:,1] ], [X[:,:,2] p[:,2] ], [ X[:,:,3] p[:,3] ], dims=1) #100*4

    #100x3x3 (Summed over each characteristic)
    demand_instruments = cat(sum(X,dims=3)-X[:,:,1], sum(X,dims=3)-X[:,:,2],sum(X,dims=3)-X[:,:,3], dims=1)[:,:,1]

    #100x2x3 (2 Level Shocks for Supply)
    supply_instruments = cat([W[:,1] Z[:,1]], [W[:,2] Z[:,2]], [W[:,3] Z[:,3]], dims=1)

    #5 Instruments (3 Demands Characteristics and 2 supply shocks)
    z= cat(demand_instruments, supply_instruments, dims=2)

    wt = Matrix(I,5,5)
    
    #β_hat = zeros(4,3)
    β_hat = inv(transpose(x)*z*wt*transpose(z)*x)*transpose(x)*z*wt*transpose(z)*reshape(δ_cm,300,1)
    #β_hat[:,2] = inv(transpose(x[:,:,2])*z[:,:,2]*wt*transpose(z[:,:,2])*x[:,:,2])*transpose(x[:,:,2])*z[:,:,2]*wt*transpose(z[:,:,2])*δ_cm[:,2]
    #β_hat[:,3] = inv(transpose(x[:,:,3])*z[:,:,3]*wt*transpose(z[:,:,3])*x[:,:,3])*transpose(x[:,:,3])*z[:,:,3]*wt*transpose(z[:,:,3])*δ_cm[:,3]


    #ξ = δ_cm - cat(x[:,:,1]*β_hat[:,1],x[:,:,2]*β_hat[:,2],x[:,:,3]*β_hat[:,3],dims=2)  #+ guess[5]*diagm(vec(agg_ν))*p

    ξ = δ_cm - cat([X[:,:,1] p[:,1]]*β_hat,[X[:,:,2] p[:,2]]*β_hat,[X[:,:,3] p[:,3]]*β_hat,dims=2)  

    #Moments
    demand_moments = transpose(z[:,1:3])*reshape(ξ,300,1)
    supply_moments = transpose(z[:,4:5])*reshape(ξ,300,1)

    moments = cat(demand_moments, supply_moments,dims=1)
    
    return vec(moments)
end    

#Method of Moments Estimation
Objective(θ) = transpose(moment_objective_fn(θ, s, p, X, W, Z, ν_sim))*moment_objective_fn(θ, s, p, X, W, Z, ν_sim) 

#Checking If Function Works
#Objective([1.0,2.2,3.3,4.1,-2.9])
Objective(a)

#Optimization
gmm_id = optimize(Objective, parameter_guess,
                    Optim.Options(g_tol = 1e-12,
                                    iterations = 10,
                                    store_trace = true,
                                    show_trace = false,
                                    time_limit=1000))

θ_id = Optim.minimizer(gmm_id)
