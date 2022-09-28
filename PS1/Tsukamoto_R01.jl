using Random
using Distributions
using LinearAlgebra 

#********************
#Step0 Simulate Data
#********************
Random.seed!(123)
d_1 = Uniform(0,1)
d_2 = Normal(0,1)

#0.1: Product Characteristics

#X is a tensor of product Characteristics across market*Characteristic*firm

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

p = ones(100,3)

δ_0 = zeros(100)
δ_1 = zeros(100)
δ_2 = zeros(100)
δ_3 = zeros(100)

l=1
for m=1:100
    δ_1[m](p) = transpose(X[m,:,1])*β -0.001*transpose(α_i[l:l+999])*ones(1000)*p[m,1] + ξ[m,1]
    δ_2[m](p) = transpose(X[m,:,2])*β -0.001*transpose(α_i[l:l+999])*ones(1000)*p[m,2] + ξ[m,2]
    δ_3[m](p) = transpose(X[m,:,3])*β -0.001*transpose(α_i[l:l+999])*ones(1000)*p[m,3] + ξ[m,3]
    l=l+1000
end

#s_0 = 1-s_1-s_2-s_3
s_1 = (1-s_1-s_2-s_3)*exp(δ_1 - δ_0)
s_2 = (1-s_1-s_2-s_3)*exp(δ_2 - δ_0)
s_3 = (1-s_1-s_2-s_3)*exp(δ_3 - δ_0)

s = hcat(s_1, s_2, s_3)

MC_1 = hcat(ones(100),W[:,1],Z[:,1],η[:,1])*vcat(γ,1) 
MC_2 = hcat(ones(100),W[:,2],Z[:,2],η[:,2])*vcat(γ,1)
MC_3 = hcat(ones(100),W[:,3],Z[:,3],η[:,3])*vcat(γ,1)

MC = hcat(MC_1, MC_2, MC_3)

ϵ = 
(p-MC)./p = -1/