import Pkg; Pkg.add("Distributions")
import Pkg; Pkg.add("Plots")
import Pkg; Pkg.add("GLM")
using Distributions
using Plots
using GLM
using DataFrames

#Step 1.1.1: Fix Parameter Values
β_1 = 0
β_2 = 2
δ = 0.2
μ = 0
σ_D = 1 
σ_S = 1

#Step 1.1.2: Generate Realizations of Shocks
ϵ_D = rand(Normal(0,σ_D),50)
a = rand(Normal(0,σ_S), 50)

#Step 1.1.3: Generate Price Data
logP = (1/(1+β_2*δ))*(δ*ϵ_D - a)

#Step 1.1.4: Use Price and Equilibrium Conditions to generate (P,Q)
logQ = β_2*logP + ϵ_D

#Step 1.1.5: Plot Empirical Distributions of P and Q

plot(logQ, logP, seriestype = :scatter, title = "Log Price and Quantity", xlabel = "Log Quantity", ylabel = "Log Price")



#Step 1.2.1.1: Report OLS Estimate of β_2
simulated = DataFrame(logQ = logQ, logP = logP)
ols = lm(@formula(logP ~ logQ), simulated)

#Step 1.2.2.1: MoM
