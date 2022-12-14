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
simulated_Rodrigo = DataFrame(CSV.read("C:\\Users\\kevin\\Documents\\GitHub\\IO1-PS-CPT\\PS1\\pset1_data_new.csv",DataFrame))

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
    s_i = zeros(100,100,3)
    l=0
    for m=1:100
        for i=1:100
            μ_i1 = θ[5]*p[m,1]*ν[i+l]
            μ_i2 = θ[5]*p[m,2]*ν[i+l]
            μ_i3 = θ[5]*p[m,3]*ν[i+l]
            #μ_i = 0

            s_i[m,i,1] = exp(δ[m,1]+μ_i1)/(1+exp(δ[m,1]+μ_i1)+exp(δ[m,2]+μ_i2)+exp(δ[m,3]+μ_i3))     
            s_i[m,i,2] = exp(δ[m,2]+μ_i2)/(1+exp(δ[m,1]+μ_i1)+exp(δ[m,2]+μ_i2)+exp(δ[m,3]+μ_i3))     
            s_i[m,i,3] = exp(δ[m,3]+μ_i3)/(1+exp(δ[m,1]+μ_i1)+exp(δ[m,2]+μ_i2)+exp(δ[m,3]+μ_i3))           
        end
        s[m,1] = sum(s_i[m,:,1])/100
        s[m,2] = sum(s_i[m,:,2])/100
        s[m,3] = sum(s_i[m,:,3])/100  
        l=100*m
    end
    return s
end

function contraction_map(s, p, δ_new, θ, ν)
    δ_guess = rand(100,3)
    while norm(δ_new - δ_guess) > .1
        δ_guess = δ_new
        s_pred = share_prediction(δ_guess,p, θ, ν)
        δ_new = δ_guess + log.(s) - log.(s_pred)
    end
    return δ_new
end

δ_rand = rand(100,3)*25

contraction_map(s,p,δ_rand,actual,ν_sim)

function moment_objective_fn(θ, s, p, X, W, Z, ν_sim)
    δ_guess = rand(100,3)*25
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
Objective(parameter_guess)

#Optimization
gmm_id = optimize(Objective, parameter_guess,
                    Optim.Options(g_tol = 1e-2,
                                    iterations = 2500,
                                    time_limit=14400))

θ_id = Optim.minimizer(gmm_id)

##P2
#1a
θ_id = [6.948,0.4632,2.734,1.642,0.1643]

ξ_estimate = back_ξ(X, s, p, θ_id, ν_sim)

function model_elasticity(p, X, β, α, σ_α, ξ, ν)

    s_1 = zeros(100)
    s_2 = zeros(100)
    s_3 = zeros(100)

    prob = zeros(100,1000,3)
    l=0
    for m=1:100
        for i=1:1000
            δ_1 = transpose(X[m,:,1])*β +ξ[m,1] + α*p[m,1]
            δ_2 = transpose(X[m,:,2])*β +ξ[m,2] + α*p[m,2]
            δ_3 = transpose(X[m,:,3])*β +ξ[m,3] + α*p[m,3]

            μ_i = σ_α*p[m,1]*ν[i+l]

            prob[m,i,1] = exp(δ_1+μ_i)/(1+exp(δ_1+μ_i)+exp(δ_2+μ_i)+exp(δ_3+μ_i))     
            prob[m,i,2] = exp(δ_2+μ_i)/(1+exp(δ_1+μ_i)+exp(δ_2+μ_i)+exp(δ_3+μ_i))     
            prob[m,i,3] = exp(δ_3+μ_i)/(1+exp(δ_1+μ_i)+exp(δ_2+μ_i)+exp(δ_3+μ_i))           
        end
        l=1000*m
        s_1[m] = sum(prob[m,:,1])/1000
        s_2[m] = sum(prob[m,:,2])/1000
        s_3[m] = sum(prob[m,:,3])/1000   
    end

    ϵ = ones(100,3)

    q=0
    for m=1:100
        α_i = α.+σ_α*ν[q+1:q+1000]
        ϵ[m,1] = (sum(α_i.*prob[m,:,1].*(ones(1) .- prob[m,:,1]))/(-1000))
        ϵ[m,2] = (sum(α_i.*prob[m,:,2].*(ones(1) .- prob[m,:,2]))/(-1000))
        ϵ[m,3] = (sum(α_i.*prob[m,:,3].*(ones(1) .- prob[m,:,3]))/(-1000))
        q=1000*m
    end

    s = hcat(s_1, s_2, s_3)
    return s,ϵ
end

s_oli, ϵ_oli = model_elasticity(p, X, θ_id[1:3], θ_id[4], θ_id[5], ξ_estimate, ν_sim) 
mc_oli = zeros(100,3)
for m = 1:100
    mc_oli[m, :] = p[m,:] + inv(diagm(ϵ_oli[m,:]))*s[m,:]
end

function model_crosselasticity(p, X, θ, ξ, ν)
    s_1 = zeros(100)
    s_2 = zeros(100)
    s_3 = zeros(100)

    prob = zeros(100,1000,3)
    l=0
    for m=1:100
        for i=1:1000
            δ_1 = transpose(X[m,:,1])*θ[1:3] +ξ[m,1] + θ[4]*p[m,1]
            δ_2 = transpose(X[m,:,2])*θ[1:3] +ξ[m,2] + θ[4]*p[m,2]
            δ_3 = transpose(X[m,:,3])*θ[1:3] +ξ[m,3] + θ[4]*p[m,3]

            μ_i = θ[5]*p[m,1]*ν[i+l]

            prob[m,i,1] = exp(δ_1+μ_i)/(1+exp(δ_1+μ_i)+exp(δ_2+μ_i)+exp(δ_3+μ_i))     
            prob[m,i,2] = exp(δ_2+μ_i)/(1+exp(δ_1+μ_i)+exp(δ_2+μ_i)+exp(δ_3+μ_i))     
            prob[m,i,3] = exp(δ_3+μ_i)/(1+exp(δ_1+μ_i)+exp(δ_2+μ_i)+exp(δ_3+μ_i))           
        end
        l=1000*m
        s_1[m] = sum(prob[m,:,1])/1000
        s_2[m] = sum(prob[m,:,2])/1000
        s_3[m] = sum(prob[m,:,3])/1000   
    end

    ϵ = ones(300,3)

    q=0
    for m=1:100
        α_i = θ[4].+θ[5]*ν[q+1:q+1000]
        ϵ[3*m-2,1] = sum(α_i.*prob[m,:,1].*(ones(1) .- prob[m,:,1]))/(-1000)
        ϵ[3*m-1,2] = sum(α_i.*prob[m,:,2].*(ones(1) .- prob[m,:,2]))/(-1000)
        ϵ[3*m,3] = sum(α_i.*prob[m,:,3].*(ones(1) .- prob[m,:,3]))/(-1000)
        ϵ[3*m-2,2] = sum(α_i.*prob[m,:,2])/(1000)
        ϵ[3*m-2,3] = sum(α_i.*prob[m,:,3])/(1000)
        ϵ[3*m-1,1] = sum(α_i.*prob[m,:,1])/(1000)
        ϵ[3*m-1,3] = sum(α_i.*prob[m,:,3])/(1000)
        ϵ[3*m,1] = sum(α_i.*prob[m,:,1])/(1000)
        ϵ[3*m,2] = sum(α_i.*prob[m,:,2])/(1000)
        q=1000*m
    end

    s = hcat(s_1, s_2, s_3)
    return s,ϵ
end

s_coll,ϵ_coll = model_crosselasticity(p, X, θ_id, ξ_estimate, ν_sim)
mc_coll = zeros(100,3)
for m = 1:100
    mc_coll[m, :] = p[m,:] + (1 ./ϵ_coll[3*m-2:3*m,:])*s_coll[m,:]
end

mc_pc = p
#1b
comp = DataFrame(firm1 = mc_pc[:,1], firm2 = mc_pc[:,2], firm3 = mc_pc[:,3])
stackcomp = stack(comp, 1:3)
plot(stackcomp, y =:value, x =:variable,  kind = "box")

oli = DataFrame(firm1 = mc_oli[:,1], firm2 = mc_oli[:,2], firm3 = mc_oli[:,3])
stackoli = stack(oli, 1:3)
plot(stackoli, y =:value, x =:variable,  kind = "box")

cost_data = DataFrame(Competition = mc_pc[:], Oligopoly = mc_oli[:], Collusion = mc_coll[:], Actual = MC[:])
cost_data = stack(cost_data, 1:4)

plot(cost_data, y =:value, x =:variable, kind = "box")