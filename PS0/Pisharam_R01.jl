#Step 0: Import Packages
    #import Pkg;
    #using Pkg;
    #Pkg.add("Distributions") // Importing the Distributions Package
    #Pkg.add("Plots")         // Importing the Plot Package
    #Pkg.add("DataFrame")     // Importing the Dataframe Package
    #Pkg.add("LinearAlgebra") // Importing the Linear LinearAlgebra Package
    #Pkg.add("Optim")         // Importing the Optimization Package
#Step 1.1.1: Fix Parameter Values
β_1 = 0
β_2 = 2
δ = 0.2
μ = 1 
s_D = 1 
s_S = 1

#Step 1.1.2: Generate Realizations of Shocks
using Random, Distributions
Random.seed!(123)   

ϵ_D = rand(Normal(0,s_D),50)
ϵ_a = rand(Normal(0,s_D),50)

#Step 1.1.3: Generate Price Data
ln_P = (1/(1+β_2*δ))*(δ*β_1 .+ δ*ϵ_D .+ log(μ) .- ϵ_a)
P = exp.(ln_P)

#Step 1.1.4: Use Price and Equilibrium Conditions to generate (P,Q)
ln_Q = β_1 .- β_2*ln_P .+ ϵ_D
Q = exp.(ln_Q)

#Step 1.1.5: Plot Empirical Distributions of P and Q
using Plots
plot(
ln_Q, ln_P, 
seriestype = :scatter, title = "Log Price and Quantity", xlabel = "Log Quantity", ylabel = "Log Price")

plot(
histogram(ln_Q; bins=:sqrt, title = "Distribution of ln Q", ylabel = "Log Quantity"), 
histogram(ln_P, bins=:sqrt, title = "Distribution of ln P", ylabel = "Log Price");
layout=(2,1))


#Step 1.2.1.1: Report OLS Estimate of β_2
using DataFrames
simulated = DataFrame(ln_Q = ln_Q, ln_P = ln_P)
using GLM
ols = lm(@formula(ln_Q ~ ln_P), simulated)

#*****************************
#Step 1.2.2.1: MoM Population
#Set Parameter
γ = .6 
s_z = 1

g1 = (β_2*γ^2*s_z)/(1+δ*β_2); #Based on our calculations
g2 = (-1*γ*s_z)/(1+δ*β_2); #Based on our calculations

#Step 1.2.2.2: MoM Simulation
using LinearAlgebra 

function sim_moments(x, γ, s_z,num_s,seed)
    Random.seed!(seed)   #To ensure stability of values each time the function is called
    β_1 = 0
    δ = 0.2
    μ = 1 
    s_D = 1 
    s_S = 1
    g = zeros(100,2)
    for i in 1:num_s
        ln_Z = rand(Normal(0,s_z), 50)
        s_a = s_S - γ^2*s_z
        ϵ_D = rand(Normal(0,s_D), 50)
        ln_a = rand(Normal(0,s_a), 50)  
        ϵ_a = γ*ln_Z + ln_a
        ln_P = (1/(1+x*δ))*(δ*β_1 .+ δ*ϵ_D .+ log(μ) .- ϵ_a)
        ln_Q = β_1 .- x*ln_P .+ ϵ_D
        g[i,1] = dot(ln_Z, ln_Q) 
        g[i,2] = dot(ln_Z, ln_P)
    end
    sim_g = (1/num_s)*sum(g',dims=2)
    pop_g = [(x*γ^2*s_z)/(1+δ*x);(-1*γ*s_z)/(1+δ*x)]
    err_g = pop_g - sim_g
    return sim_g, err_g
end

#Compute Empirical Moment
ln_Z = rand(Normal(0,s_z), 50)
s_a = s_S - γ^2*s_z
ϵ_D = rand(Normal(0,s_D), 50)
ln_a = rand(Normal(0,s_a), 50)  
ϵ_a = γ*ln_Z + ln_a
ln_P = (1/(1+β_2*δ))*(δ*β_1 .+ δ*ϵ_D .+ log(μ) .- ϵ_a)
ln_Q = β_1 .- β_2*ln_P .+ ϵ_D
g = zeros(1,2)
g[1,1] = dot(ln_Z, ln_Q) 
g[1,2] = dot(ln_Z, ln_P)
emp_g = sum(g',dims=2)

#Step 1.2.2.4 Opitmize
using Optim
A(x) = [(x*γ^2*s_z)/(1+δ*x);(-1*γ*s_z)/(1+δ*x)]
B = emp_g
C(x) = sim_moments(x,0.8,1,100,123) 

obj_pop(x) = norm(A(x)-B)
obj_sim(x) = norm(C(x)-B)

optimize(obj_pop,[0.0])

