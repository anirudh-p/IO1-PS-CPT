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

#############################
#Simulate Data for 1 market
#############################

Random.seed!(321)
d1 = Normal(0,1)
d2 = Uniform(0,1)
d3 = LogNormal(0,1)

θ_true = [5,1,1,1,1,2,1,1]

#Demand
X = cat(ones(3), rand(d2,3), rand(d1,3),  dims = 2)
ξ = rand(d1,3)
ν_true = rand(d3,100)


β = θ_true[1:3]
α = θ_true[4]
σ_α = θ_true[5]

function model_ind_shares(β,α,σ_α,ξ,p,ν)  

    δ = X*β - α*p + ξ
    s_i = zeros(100,3)

    for i=1:100
        μ_i1 = σ_α*p[1]*ν[i]
        μ_i2 = σ_α*p[2]*ν[i]
        μ_i3 = σ_α*p[3]*ν[i]
        
        s_i[i,1] = exp(δ[1] + μ_i1)/(1+exp(δ[1] + μ_i1)+exp(δ[2] + μ_i2)+exp(δ[3] + μ_i3)) 
        s_i[i,2] = exp(δ[2] + μ_i2)/(1+exp(δ[1] + μ_i1)+exp(δ[2] + μ_i2)+exp(δ[3] + μ_i3)) 
        s_i[i,3] = exp(δ[3] + μ_i3)/(1+exp(δ[1] + μ_i1)+exp(δ[2] + μ_i2)+exp(δ[3] + μ_i3))
    end
    return s_i
end

function model_elasticity(s_i,α,σ_α,p)
    s = zeros(1,3)
    s[1,1] = mean(s_i[:,1])
    s[1,2] = mean(s_i[:,2])
    s[1,3] = mean(s_i[:,3])

    α_i = α*ones(100) + σ_α*ν_true

    ϵ = zeros(1,3)
    ϵ[1,1] = -1*mean(α_i.*s_i[:,1].*(1 .-s_i[:,1]))*(p[1,1]/s[1,1]) 
    ϵ[1,2] = -1*mean(α_i.*s_i[:,2].*(1 .-s_i[:,2]))*(p[2,1]/s[1,2])  
    ϵ[1,3] = -1*mean(α_i.*s_i[:,3].*(1 .-s_i[:,3]))*(p[3,1]/s[1,3])  

    return s,ϵ
end

#Supply
γ_0 = θ_true[6]
γ_1 = θ_true[7]
γ_2 = θ_true[8]

W = max.(rand(d1,3),0)
Z = max.(rand(d1,3),0)
η = max.(rand(d1,3),0)

MC = γ_0*ones(3) + γ_1*W + γ_2*Z + η

#Equilibrium
p_guess=rand(Uniform(0,15),3,1)
p = rand(Uniform(15,30),3,1)
s = zeros(1,3)
ϵ = zeros(1,3)

while norm(p - p_guess) > 0.01
    p_guess = p
    s_i = model_ind_shares(β,α,σ_α,ξ,p,ν_true) 
    s,ϵ = model_elasticity(s_i,α,σ_α,p)  
    p[1,1] = (ϵ[1,1]/(1+ϵ[1,1])) * MC[1]
    p[2,1] = (ϵ[1,2]/(1+ϵ[1,2])) * MC[2]
    p[3,1] = (ϵ[1,3]/(1+ϵ[1,3])) * MC[3]
end
