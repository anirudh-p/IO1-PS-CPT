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
#Simulate Data for 100 market
#############################

Random.seed!(321)
d1 = Normal(0,1)
d2 = Uniform(0,1)
d3 = LogNormal(0,1)

θ_true = [5,1,1,1,1,2,1,1]

#Demand
x1 = ones(100,3)
x2 = rand(d2,100,3)
x3 = rand(d1,100,3)

ξ = rand(d1,100,3)
ν_true = rand(d3,100,100)

β = θ_true[1:3]
α = θ_true[4]
σ_α = θ_true[5]

function model_ind_shares(β,α,σ_α,ξ,p,ν)  

    δ = x1*β[1] + x2*β[2] + x3*β[3] - α*p + ξ
    s_i = zeros(100,3,100)

    for m=1:100
        for i=1:100
            μ_i1 = σ_α*p[1]*ν[i,m]
            μ_i2 = σ_α*p[2]*ν[i,m]
            μ_i3 = σ_α*p[3]*ν[i,m]
            
            s_i[i,1,m] = exp(δ[1] + μ_i1)/(1+exp(δ[1] + μ_i1)+exp(δ[2] + μ_i2)+exp(δ[3] + μ_i3)) 
            s_i[i,2,m] = exp(δ[2] + μ_i2)/(1+exp(δ[1] + μ_i1)+exp(δ[2] + μ_i2)+exp(δ[3] + μ_i3)) 
            s_i[i,3,m] = exp(δ[3] + μ_i3)/(1+exp(δ[1] + μ_i1)+exp(δ[2] + μ_i2)+exp(δ[3] + μ_i3))
        end
    end
    return s_i
end

function model_elasticity(s_i,α,σ_α,p)
    s = zeros(100,3)
    ϵ = zeros(100,3)

    for m = 1:100
        s[m,1] = mean(s_i[:,1,m])
        s[m,2] = mean(s_i[:,2,m])
        s[m,3] = mean(s_i[:,3,m])

        α_i = α*ones(100) + σ_α*ν_true[:,m]


        ϵ[m,1] = -1*mean(α_i.*s_i[:,1,m].*(1 .-s_i[:,1,m]))*(p[m,1]./s[m,1]) 
        ϵ[m,2] = -1*mean(α_i.*s_i[:,2,m].*(1 .-s_i[:,2,m]))*(p[m,2]./s[m,2])  
        ϵ[m,3] = -1*mean(α_i.*s_i[:,3,m].*(1 .-s_i[:,3,m]))*(p[m,3]./s[m,3])  
    end

    return s,ϵ
end

#Supply
γ_0 = θ_true[6]
γ_1 = θ_true[7]
γ_2 = θ_true[8]

W = repeat(max.(rand(d1,1,3),0),100)
Z = max.(rand(d1,100,3),0)
η = max.(rand(d1,100,3),0)

MC = γ_0*ones(100,3) + γ_1*W + γ_2*Z + η

#Equilibrium
p_guess=rand(Uniform(0,5),100,3)
p = rand(Uniform(5,10),100,3)
s = zeros(100,3)
ϵ = zeros(100,3)

while norm(p - p_guess) > 0.0001
    p_guess = p
    s_i = model_ind_shares(β,α,σ_α,ξ,p,ν_true) 
    s,ϵ = model_elasticity(s_i,α,σ_α,p)  
    for m=1:100
        p[m,1] = (ϵ[m,1]/(1+ϵ[m,1])) * MC[m,1]
        p[m,2] = (ϵ[m,2]/(1+ϵ[m,2])) * MC[m,2]
        p[m,3] = (ϵ[m,3]/(1+ϵ[m,3])) * MC[m,3]
    end
end

