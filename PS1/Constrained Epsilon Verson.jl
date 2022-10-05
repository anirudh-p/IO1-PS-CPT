using Pkg
using Random
using Distributions
using LinearAlgebra 
using Optim
using PlotlyJS
using DataFrames
#using Symbolics

#********************
#Step0 Simulate Data
#********************
Random.seed!(123)
d_1 = Uniform(0,1)
d_2 = Normal(0,1)

#0.1: Product Characteristics (X is a tensor of product Characteristics across Market * Characteristic * Firm)
X = zeros(100,3,3)

x_2 = rand(d_1,300) 
x_3 = rand(d_2,300) 
for j = 1:3
    k=(j-1)*100
    for i=1:100
        X[i,1,j] = 1
        X[i,2,j] = x_2[k+i]
        X[i,3,j] = x_3[k+i]
    end
end

#0.2: Demand Shock
ξ = hcat(last(rand(d_2,400),100),last(rand(d_2,500),100),last(rand(d_2,600),100)) #First 300 Draws used up for Product Characteristics (Seed Set)

#0.3: Cost Shifters
W = reshape(repeat(vcat(last(rand(d_2,700),3)),100),3,100)'
Z = hcat(last(rand(d_2,1000),100),last(rand(d_2,1100),100),last(rand(d_2,1200),100)) 
η = hcat(last(rand(d_2,1300),100),last(rand(d_2,1400),100),last(rand(d_2,1500),100)) 

#4.Fix parameter values
β = [5,1,1]
α = 1
σ_α = 1
γ = [2,1,1]

#5.Derivation of prices and market share

d_3 = LogNormal(0,1)
ν = rand(d_3,100000)

function model_elasticity(p, X, β, α, σ_α, ξ, ν)

    s = zeros(100,3)
    ϵ = ones(100,3)

    prob = zeros(100,1000,3)
    l=0
    for m=1:100
        δ_1 = transpose(X[m,:,1])*β +ξ[m,1] - α*p[m,1]
        δ_2 = transpose(X[m,:,2])*β +ξ[m,2] - α*p[m,2]
        δ_3 = transpose(X[m,:,3])*β +ξ[m,3] - α*p[m,3]
        for i=1:1000
            μ_i = σ_α*p[m,1]*ν[i+l]

            prob[m,i,1] = exp(δ_1-μ_i)/(1+exp(δ_1-μ_i)+exp(δ_2-μ_i)+exp(δ_3-μ_i))     
            prob[m,i,2] = exp(δ_2-μ_i)/(1+exp(δ_1-μ_i)+exp(δ_2-μ_i)+exp(δ_3-μ_i))     
            prob[m,i,3] = exp(δ_3-μ_i)/(1+exp(δ_1-μ_i)+exp(δ_2-μ_i)+exp(δ_3-μ_i))           
        end
        l=1000*m
        s[m,1] = sum(prob[m,:,1])/1000
        s[m,2] = sum(prob[m,:,2])/1000
        s[m,3] = sum(prob[m,:,3])/1000   
    end
    
    q=0
    for m=1:100
        α_i = α.+σ_α*ν[q+1:q+1000]
        ϵ[m,1] = (sum(α_i.*prob[m,:,1].*(ones(1) .- prob[m,:,1]))/(-1000))*(p[m,1]/s[m,1])
        ϵ[m,2] = (sum(α_i.*prob[m,:,2].*(ones(1) .- prob[m,:,2]))/(-1000))*(p[m,2]/s[m,2])
        ϵ[m,3] = (sum(α_i.*prob[m,:,3].*(ones(1) .- prob[m,:,3]))/(-1000))*(p[m,3]/s[m,3])
        q=1000*m
    end
    
    return s,ϵ
end

# MC from the supply side 
MC = zeros(100,3)
for j=1:3
    MC[:,j] = hcat(ones(100),W[:,j],Z[:,j],η[:,j])*vcat(γ,1)
end

#set negative values to 0
zero = zeros(100,3)
MC = max.(MC, zero)

#Equilibrium Prices 
p_guess=rand(Uniform(14,15),100,3)
p = ones(100,3)
s = zeros(100,3)
ϵ = zeros(100,3)

while (norm(p - p_guess) > 0.01) 
    while maximum(ϵ) == -1   
        p_guess = p
        s, ϵ = model_elasticity(p, X, β, α, σ_α, ξ, ν)
        ϵ = min.(ϵ,-1)
        for m = 1:100
            p[m,1] = (ϵ[m,1]*MC[m,1]) / (1 + ϵ[m,1])
            p[m,2] = (ϵ[m,2]*MC[m,2]) / (1 + ϵ[m,2])
            p[m,3] = (ϵ[m,3]*MC[m,3]) / (1 + ϵ[m,3])
        end
    end
end

#p

#********************
#Step1 Demand Side
#********************

#Guess Parameter Estimates (β_1,β_2,β_3,α,σ_α):
guess = [6,.5,2,1.2,2]
d_3 = LogNormal(0,1)
ν_sim = rand(d_3,100000)


#Step 1.2: Inversion to find δ from shares
function share_prediction(δ, p, θ, ν)

    s = zeros(100,3)

    prob = zeros(100,1000,3)
    l=0
    for m=1:100
        for i=1:1000
            μ_i = θ[5]*p[m,1]*ν[i+l]

            prob[m,i,1] = exp(δ[m,1]+μ_i)/(1+exp(δ[m,1]+μ_i)+exp(δ[m,2]+μ_i)+exp(δ[m,3]+μ_i))     
            prob[m,i,2] = exp(δ[m,2]+μ_i)/(1+exp(δ[m,1]+μ_i)+exp(δ[m,2]+μ_i)+exp(δ[m,3]+μ_i))     
            prob[m,i,3] = exp(δ[m,3]+μ_i)/(1+exp(δ[m,1]+μ_i)+exp(δ[m,2]+μ_i)+exp(δ[m,3]+μ_i))           
        end
        s[m,1] = sum(prob[m,:,1])/1000
        s[m,2] = sum(prob[m,:,2])/1000
        s[m,3] = sum(prob[m,:,3])/1000  
        l=1000*m
    end
    return s
end

act = [5,1,1,1,1]
δ_guess = X[:,1,:]*5 + X[:,2,:] + X[:,3,:] - p
s_guess = share_prediction(δ_guess, p, act, ν_sim)

function contraction_map(s, p, δ_new, θ, ν)
    δ_guess = zeros(100,3)
    while norm(δ_new - δ_guess) > .01
        δ_guess = δ_new
        s_pred = share_prediction(δ_guess,p, θ, ν)
        δ_new = δ_guess + log.(s) - log.(s_pred)
    end
    return δ_new
end

δ_test = contraction_map(s, p, rand(100,3), guess, ν_sim)

function back_ξ(X, s, p, guess, ν)
    δ_guess = X[:,1,:]*guess[1] + X[:,2,:]*guess[2] + X[:,3,:]*guess[3] - guess[4]*p
    #δ_guess = rand(100,3)
    δ = contraction_map(s, p, δ_guess, guess, ν)

    #Step 1.3: Estimate the ξ from δ 
    #agg_ν = zeros(100,1)
    #q = 0
    #for m = 1:100
    #    agg_ν[m] = sum(ν[q+1:q+1000])/1000
    #    q = m*1000
    #end
    #agg_ν
    #diagm(vec(agg_ν))

    ξ = δ - X[:,1,:].*guess[1] - X[:,2,:].*guess[2] - X[:,3,:].*guess[3] + guess[4]*p #+ guess[5]*diagm(vec(agg_ν))*p
    return ξ
end

#ξ_guess = back_ξ(s, p, guess, ν_sim)
##P1

#2
#(a) The moment condition and GMM

function g_demand_id(X, W, Z, s, p, θ, ν)
    ξ = back_ξ(X, s, p, θ, ν)
    ## 3 moment conditions of characteristics
    g121 = (transpose(X[:,1,2])*reshape(ξ[:,1], 100, 1))[1] /100
    g122 = (transpose(X[:,2,2])*reshape(ξ[:,1], 100, 1))[1] /100
    g123 = (transpose(X[:,3,2])*reshape(ξ[:,1], 100, 1))[1] /100
    g131 = (transpose(X[:,1,3])*reshape(ξ[:,1], 100, 1))[1] /100
    g132 = (transpose(X[:,2,3])*reshape(ξ[:,1], 100, 1))[1] /100
    g133 = (transpose(X[:,3,3])*reshape(ξ[:,1], 100, 1))[1] /100
    g211 = (transpose(X[:,1,1])*reshape(ξ[:,2], 100, 1))[1] /100
    g212 = (transpose(X[:,2,1])*reshape(ξ[:,2], 100, 1))[1] /100
    g213 = (transpose(X[:,3,1])*reshape(ξ[:,2], 100, 1))[1] /100
    g231 = (transpose(X[:,1,3])*reshape(ξ[:,2], 100, 1))[1] /100
    g232 = (transpose(X[:,2,3])*reshape(ξ[:,2], 100, 1))[1] /100
    g233 = (transpose(X[:,3,3])*reshape(ξ[:,2], 100, 1))[1] /100
    g311 = (transpose(X[:,1,1])*reshape(ξ[:,3], 100, 1))[1] /100
    g312 = (transpose(X[:,2,1])*reshape(ξ[:,3], 100, 1))[1] /100
    g313 = (transpose(X[:,3,1])*reshape(ξ[:,3], 100, 1))[1] /100
    g321 = (transpose(X[:,1,2])*reshape(ξ[:,3], 100, 1))[1] /100
    g322 = (transpose(X[:,2,2])*reshape(ξ[:,3], 100, 1))[1] /100
    g323 = (transpose(X[:,3,2])*reshape(ξ[:,3], 100, 1))[1] /100

    ## 6 moment conditions of common and market specific cost Shifters
    g1w = transpose(W[:,1])*ξ[:,1] /100
    g1z = transpose(Z[:,1])*ξ[:,1] /100
    g2w = transpose(W[:,2])*ξ[:,2] /100
    g2z = transpose(Z[:,2])*ξ[:,2] /100
    g3w = transpose(W[:,3])*ξ[:,3] /100
    g3z = transpose(Z[:,3])*ξ[:,3] /100

    g1 = (1/6)*(g121 + g122 + g123 + g131 + g132 + g133)
    g2 = (1/6)*(g211 + g212 + g213 + g231 + g232 + g233)
    g3 = (1/6)*(g311 + g312 + g313 + g321 + g322 + g323)
    gw = (1/3)*(g1w + g2w + g3w)
    gz = (1/3)*(g1z + g2z + g3z)

    g = (1/3).*[g1, g2, g3, gw, gz]

    return g
end

g_id(θ) = transpose(g_demand_id(X, W, Z, s, p, θ, ν_sim))*g_demand_id(X, W, Z, s, p, θ, ν_sim)
g_id(guess)

gmm_id = optimize(θ->g_id(θ), guess)
θ_id = Optim.minimizer(gmm_id)


##P2
#1a
θ_id = [5,1,1,1,1]

ξ_estimate = back_ξ(X, s, p, θ_id, ν_sim)

s_oli, ϵ_oli = model_elasticity(p, X, θ_id[1:3], θ_id[4], θ_id[5], ξ_estimate, ν_sim) 
mc_oli = zeros(100,3)
for m = 1:100
    mc_oli[m, :] = p[m,:] - diagm(ϵ_oli[m,:])*s_oli[m,:]
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
        ϵ[3*m-2,1] = sum(α_i.*prob[m,:,1].*(ones(1) .- prob[m,:,1]))/(1000)
        ϵ[3*m-1,2] = sum(α_i.*prob[m,:,2].*(ones(1) .- prob[m,:,2]))/(1000)
        ϵ[3*m,3] = sum(α_i.*prob[m,:,3].*(ones(1) .- prob[m,:,3]))/(1000)
        ϵ[3*m-2,2] = sum(α_i.*prob[m,:,2])/1000
        ϵ[3*m-2,3] = sum(α_i.*prob[m,:,3])/1000
        ϵ[3*m-1,1] = sum(α_i.*prob[m,:,1])/1000
        ϵ[3*m-1,3] = sum(α_i.*prob[m,:,3])/1000
        ϵ[3*m,1] = sum(α_i.*prob[m,:,1])/1000
        ϵ[3*m,2] = sum(α_i.*prob[m,:,2])/1000
        q=1000*m
    end

    s = hcat(s_1, s_2, s_3)
    return s,ϵ
end

s_coll,ϵ_coll = model_crosselasticity(p, X, θ_id, ξ_estimate, ν_sim)
mc_coll = zeros(100,3)
for m = 1:100
    mc_coll[m, :] = p[m,:] - ϵ_coll[3*m-2:3*m,:]*s_coll[m,:]
end

mc_pc = p
#1b

cost_data = DataFrame(Competition = mc_pc[:], Oligopoly = mc_oli[:], Collusion = mc_coll[:], Actual = MC[:])
cost_data = stack(cost_data, 1:4)

plot(cost_data, y =:value, x =:variable, kind = "box")



##P3 Merger
function model_mergeelasticity(p, X, θ, ξ, ν)
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
        ϵ[3*m-2,1] = sum(α_i.*prob[m,:,1].*(ones(1) .- prob[m,:,1]))/(1000)
        ϵ[3*m-1,2] = sum(α_i.*prob[m,:,2].*(ones(1) .- prob[m,:,2]))/(1000)
        ϵ[3*m,3] = sum(α_i.*prob[m,:,3].*(ones(1) .- prob[m,:,3]))/(1000)
        ϵ[3*m-2,2] = sum(α_i.*prob[m,:,2])/1000
        ϵ[3*m-2,3] = 0
        ϵ[3*m-1,1] = sum(α_i.*prob[m,:,1])/1000
        ϵ[3*m-1,3] = 0
        ϵ[3*m,1] = 0
        ϵ[3*m,2] = 0
        q=1000*m
    end

    s = hcat(s_1, s_2, s_3)
    return s,ϵ
end
p_guess=rand(Uniform(10,15),100,3)
p_merge = ones(100,3)
s_merge = zeros(100,3)
ϵ_merge = zeros(100,3)

while norm(p - p_guess) > 0.01 
    p_guess = p_merge
    s_merge, ϵ_merge = model_mergeelasticity(p, X, θ_id, ξ, ν_sim)
    for m = 1:100
        p_merge[m,:] =  mc_oli[m, :]  + ϵ_merge[3*m-2:3*m,:]*s_merge[m,:]
    end
end


