#Step 1.1.1: Fix Parameter Values
β_1 = 0
β_2 = 2
δ = 0.2
μ = 0
σ^D = 1 
σ^S = 1

#Step 1.1.2: Generate Realizations of Shocks
ϵ_D = Array
a = Array

#Step 1.1.3: Generate Price Data
ln(P) = (1/(1+β_2*δ))*(δ*β_1 + δ*ϵ_D + ln(μ) - ln(a))

#Step 1.1.4: Use Price and Equilibrium Conditions to generate (P,Q)

#Step 1.1.5: Plot Empirical Distributions of P and Q

