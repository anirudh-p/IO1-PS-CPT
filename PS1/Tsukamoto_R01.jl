using Random, Distributions, LinearAlgebra 

#********************
#Step0 Simulate Data
#********************
Random.seed!(123)
d_1 = Uniform(0,1)
d_2 = Normal(0,1)

#0.1: Product Characteristics

#Firm j =1
X_21 = rand(d_1,100) #Uniform [0,1]
X_31 = rand(d_2,100) #Normal [0,1]
X_1 = zeros(100,3)
for i in 1:100
    X_1[i,1] = 1
    X_1[i,2] = X_21[i]
    X_1[i,3] = X_31[i]
end

#Firm j =2
X_22 = rand(d_1,100)
X_32 = rand(d_2,100)
X_2 = zeros(100,3)
for i in 1:100
    X_1[i,1] = 1
    X_1[i,2] = X_21[i]
    X_1[i,3] = X_31[i]
end

#Firm j =3
X_23 = rand(d_1,100)
X_33 = rand(d_2,100)
X_3 = zeros(100,3)
for i in 1:100
    X_1[i,1] = 1
    X_1[i,2] = X_21[i]
    X_1[i,3] = X_31[i]
end

#0.2: Demand Shock
ξ_1 = rand(d_2,100)
ξ_2 = rand(d_2,100) 
ξ_3 = rand(d_2,100)

#0.3: Cost Shifters
W_1 = rand(d_1,1)
W_2 = rand(d_1,1)
W_3 = rand(d_1,1)
Z_1 = rand(d_2,100)
Z_2 = rand(d_2,100)
Z_3 = rand(d_2,100)
η_1 = rand(d_1,100)
η_2 = rand(d_2,100)
η_3 = rand(d_3,100)

#4.Fix parameter values
β = [5,1,1]
α = 1
σ_α = 1
γ = [2,1,1]

#5.Derivation of prices and market shares
δ_1 = X_1*β - α* #Mean Utilities

MC_1 = hcat(ones(100), W_1, Z_1)*γ + η_1
MC_2 = hcat(ones(100), W_2, Z_2)*γ + η_2
MC_3 = hcat(ones(100), W_3, Z_3)*γ + η_3

ln(s_1) - ln(s_0) = δ_1
(p_1 - MC_1)/p_1 = 