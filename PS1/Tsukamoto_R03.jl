using Pkg
using Random
using Distributions
using LinearAlgebra 
#using Symbolics

#********************
#Step0 Simulate Data
#********************
Random.seed!(123)
d_1 = Uniform(0,1)
d_2 = Normal(0,1)

#0.1: Product Characteristics (X is a tensor of product Characteristics across Market * Characteristic * Firm)
x_2 = rand(d_1,300) #Uniform [0,1]
x_3 = rand(d_2,300) #Normal(0,1)
X = zeros(100,3,3)

k=0
for j = 1:3
    for i=1:100
        X[i,1,j] = 1
        X[i,2,j] = x_2[k+i]
        X[i,3,j] = x_3[k+i]
    end
    k=j*100
end

#0.2: Demand Shock
ξ = hcat(last(rand(d_2,400),100),last(rand(d_2,500),100),last(rand(d_2,600),100)) #First 300 Draws used up for Product Characteristics (Seed Set)

#0.3: Cost Shifters
W = hcat(last(rand(d_2,700),100),last(rand(d_2,800),100),last(rand(d_2,900),100)) #First 600 Draws used up (Seed Set)
Z = hcat(last(rand(d_2,1000),100),last(rand(d_2,1100),100),last(rand(d_2,1200),100)) 
η = hcat(last(rand(d_2,1300),100),last(rand(d_2,1400),100),last(rand(d_2,1500),100)) 

#4.Fix parameter values
β = [5,1,1]
α = 1
σ_α = 1
γ = [2,1,1]

#5.Derivation of prices and market share

function model_elasticity(p, X, β, α, σ_α, ξ, ν)

    d_3 = LogNormal(0,1)
    ν = rand(d_3,100000)

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
        α_i = σ_α*ν[q+1:q+1000]
        ϵ[m,1] = sum(α_i.*prob[m,:,1].*(ones(1) .- prob[m,:,1]))/1000
        ϵ[m,2] = sum(α_i.*prob[m,:,2].*(ones(1) .- prob[m,:,2]))/1000
        ϵ[m,3] = sum(α_i.*prob[m,:,3].*(ones(1) .- prob[m,:,3]))/1000
        q=1000*m
    end

    s = hcat(s_1, s_2, s_3)
    return s,ϵ
end

# MC from the supply side 
MC = zeros(100,3)
for j=1:3
    MC[:,j] = hcat(ones(100),W[:,j],Z[:,j],η[:,j])*vcat(γ,1)
end

#Equilibrium Prices 
p_guess=rand(Uniform(10,15),100,3)
p = ones(100,3)

while norm(p - p_guess) > 0.00001 
    p_guess = p
    s,ϵ = model_elasticity(p, X, β, α, σ_α, ξ, ν)
    p = MC./(ones(100,3) .+ 1 ./ϵ)
end
p

#********************
#Step1 Demand Side
#********************

#Guess Parameter Estimates (β_1,β_2,β_3,α,σ_α):
guess = [4,1.5,2,1,0.8]

#Step 1.2: Inversion to find δ from shares
function share_prediction(δ, p)

    d_3 = LogNormal(0,1)
    ν = rand(d_3,100000)

    s = zeros(100,3)

    prob = zeros(100,1000,3)
    l=0
    for m=1:100
        for i=1:1000
            μ_i = σ_α*p[m,1]*ν[i+l]

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
    
function contraction_map(s,p)
        while norm(δ_new - δ_guess) < 100
            δ_guess = δ_new
            s_pred = share_prediction(δ_guess,p)
            δ_new = δ_guess + ln.(s) - ln.(s_pred)
        end
    end

#Step 1.3: Estimate the ξ from δ 
ξ = X*β


