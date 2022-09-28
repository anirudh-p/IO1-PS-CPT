using Pkg
using Random
using Distributions
using LinearAlgebra 
using Symbolics

#********************
#Step0 Simulate Data
#********************
Random.seed!(123)
d_1 = Uniform(0,1)
d_2 = Normal(0,1)

#0.1: Product Characteristics
#X is a tensor of product Characteristics across Market*Characteristic*Firm

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
#Simulate α_i's
d_3 = LogNormal(0,1)
temp = rand(d_3,100000)
α_i = α .+ temp.*σ_α

function elasticity(p, X, β, α_i, ξ)

    s_1 = zeros(100)
    s_2 = zeros(100)
    s_3 = zeros(100)

    prob = zeros(100,1000,3)
    l=0
    for m=1:100
        for i=1:1000
            delta1 = transpose(X[m,:,1])*β +ξ[m,1] + α_i[l+i]*p[m,1]
            delta2 = transpose(X[m,:,2])*β +ξ[m,2] + α_i[l+i]*p[m,2]
            delta3 = transpose(X[m,:,3])*β +ξ[m,3] + α_i[l+i]*p[m,3]

            prob[m,i,1] = exp(delta1)/(1+exp(delta1)+exp(delta2)+exp(delta3))
            prob[m,i,2] = exp(delta2)/(1+exp(delta1)+exp(delta2)+exp(delta3))
            prob[m,i,3] = exp(delta3)/(1+exp(delta1)+exp(delta2)+exp(delta3))           
        end
        l=1000*m
        s_1[m] = sum(prob[m,:,1])/1000
        s_2[m] = sum(prob[m,:,2])/1000
        s_3[m] = sum(prob[m,:,3])/1000   
    end

    ϵ = zeros(100,3)

    for j= 1:3
        l=0
        for m=1:100
            for i = 1:1000
                ϵ[m,j] += α_i[l+i]*prob[m,i,j]*(1-prob[m,i,j])/1000
            end
            l = m*1000
        end
    end

    s = hcat(s_1, s_2, s_3)
    return s,ϵ
end

MC = zeros(100,3)
for j=1:3
    MC[:,j] = hcat(ones(100),W[:,j],Z[:,j],η[:,j])*vcat(γ,1)
end

p_guess=rand(Uniform(10,15),100,3)
p = ones(100,3)
count = 0

while norm(p .- p_guess) > 0.00001 
    p_guess = p
    s,ϵ = elasticity(p_guess, X, β, α_i, ξ)
    p = MC./(ones(100,3) .+ 1 ./ϵ)
    count +=1
    if count > 1000
        break
    end
end
p

#********************
#Step1 Demand Side
#********************

#Guess Parameter Estimates (β_1,β_2,β_3,α,σ_α):
guess = [4,1.5,2,1,0.8]

#Step 1.2: Inversion to find δ from shares
function contraction_map(s,δ)
        while norm(δ_new - δ_guess) < 0.001
            s_pred = exp()
            δ_new = δ_guess + ln.(s) - ln.(s_pred)
            
            return δ_new
        end

#Step 1.3: Estimate the ξ from δ 
ξ = X*β

