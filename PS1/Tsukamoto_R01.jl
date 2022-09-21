#Step0 Simulate Data
using Random, Distributions, LinearAlgebra 
Random.seed!(123)
d_1 = Uniform(0,1)
d_2 = Normal(0,1)

#1.Product Characteristics
X_21 = rand(d_2,100)
X_31 = rand(d_1,100)
X_1 = zeros(100,3)
for i in 1:100
    X_1[i,1] = 1
    X_1[i,2] = X_21[i]
    X_1[i,3] = X_31[i]
end

X_22 = rand(d_2,100)
X_32 = rand(d_1,100)
X_2 = zeros(100,3)
for i in 1:100
    X_1[i,1] = 1
    X_1[i,2] = X_21[i]
    X_1[i,3] = X_31[i]
end

X_23 = rand(d_2,100)
X_33 = rand(d_1,100)
X_3 = zeros(100,3)
for i in 1:100
    X_1[i,1] = 1
    X_1[i,2] = X_21[i]
    X_1[i,3] = X_31[i]
end

#2.Demand Shock
ϵ_1 = rand(d_2,100)
ϵ_2 = rand(d_2,100)
ϵ_3 = rand(d_2,100)

#3.Cost Shifters
W_1 = rand(d_1,1)
W_2 = rand(d_1,1)
W_3 = rand(d_1,1)
Z_1 = rand(d_2,100)
Z_2 = rand(d_2,100)
Z_3 = rand(d_2,100)

#4.Fix parameter values
β = [5,1,1]
α = 1
σ_α = 1
γ = [2,1,1]

#5.Derivation of prices and market shares
