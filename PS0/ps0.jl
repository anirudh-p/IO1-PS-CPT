#############

using DataFrames, Statistics, LinearAlgebra, Roots, Optim, ForwardDiff
using Random, Plots
using Optim

# Define parameters
β1  = 0 
β2  = 2
δ   = 0.2
μ   = 1
σd  = 1
σs  = 1
T   = 50

# Pick seed
Random.seed!(1);

# Generate shocks
ϵd = randn(T).*σd
lna = randn(T).*σs
# Calculate equilibrium elements based on shocks and parameters
qt = (β1 .- β2.*(log(μ) .- lna) .+ ϵd).*((1+δ*β2).^-1)
pt = δ.*qt .+ log(μ) .- lna 

plot(qt, pt, seriestype = :scatter, title = "50 Market Equilibriums", xlabel = "qt", ylabel = "pt")

# Estimate β OLS qt = -β2*pt + νt
β2ols  = (pt'*pt)\*(pt'*qt)

# Pick additional parameters
# "However, finding instruments which are strong and valid in the presence of price 
# endogeneity remains a challenge without general solutions." (De Loecker and Scott, 2016)
# High information instrument (subject to 1 > (γ^2 * σz) > 0 as σs = 1)
γ = 2
σz = 0.2        # Variance of lnzt

# Low information instrument
# γ = 0.1
# σz = 0.9

# Generate additional random shocks
lnau = randn(T).*(σs - γ^2*σz)
# Calculate equilibrium supply related component
zt = (lna .- lnau)./γ

## Estimate First Stage, effect of zt on pt
αols  = (zt'*zt)\*(zt'*pt)
pt_hat = αols.*zt 
err = pt .- pt_hat

# Estimate β IV
β2iv = (zt'*pt)\*(zt'*qt)
# Estimate in the 2nd stage
β2iv = (pt_hat'*pt_hat)\*(pt_hat'*qt)

# Generate two vectors with the realizations of the random variables g1(W,θ) and g2(W,θ)
g1 = zt.*qt
g2 = zt.*pt

E_emp1 = mean(g1)
E_emp2 = mean(g2)

## Empirical vs theoretical moments
function f(β2::Float64)
    E_pop1 = (γ*β2[1]*σz)/(1+δ*β2[1])
    E_pop2 = -γ*σz
    
    ν1 = E_pop1 - E_emp1
    ν2 = E_pop2 - E_emp2

    return sqrt(ν1^2 + ν2^2)
end
# Pick an initial value
β2_init = 1.0

#param_gmm = optimize(f, x_init)
β2_gmm = optimize(β2->f(first(β2)), [β2_init])

## Empirical vs Simulated moments
S = 100
function g(β2::Float64)
    
    G1s = zeros(S,1); G2s = zeros(S,1)

    Random.seed!(1234);
    for s in 1:S
    
    # Generate shocks
    ϵds = randn(T).*σd
    lnas = randn(T).*σs
    lnaus = randn(T).*(σs - γ^2*σz)
    
    # Generate simulated realizations of random variables
    qs = (β1 .- β2[1].*(log(μ) .- lnas) .+ ϵds).*((1+δ*β2[1]).^-1)
    ps = δ.*qt .+ log(μ) .- lnas 
    zs = (lnas .- lnaus)./γ
    
    G1s[s] = mean(qs.*zs)
    G2s[s] = mean(ps.*zs)
    end

    E_sim1 = mean(G1s)
    E_sim2 = mean(G2s)

    ν1 = E_sim1 - E_emp1
    ν2 = E_sim2 - E_emp2

    return sqrt(ν1^2 + ν2^2)

end

#param_gmm_sim = optimize(g, x_init)
β2_gmm_sim = optimize(β2->g(first(β2)), [β2_init])

# Check your results
println(abs(β2ols))
println(abs(β2iv))
println(Optim.minimizer(β2_gmm))
println(Optim.minimizer(β2_gmm_sim))

function f1(β2::Float64)
    E_pop1 = (γ*β2[1]*σz)/(1+δ*β2[1])
    E_pop2 = -γ*σz
    
    ν1 = E_pop1 - E_emp1
    ν2 = E_pop2 - E_emp2

    return abs(ν1) + abs(ν2)
end

# Plot functions g and f 
plot([-1:0.1:5],g, title = "g(β2) - Obj fun - sim moments", label = "L2" )
plot([-1:0.1:8],f, title = "f(β2) - Obj fun - theor moments", label = "L2")
plot!([-1:0.1:8],f1, title = "f(β2) - Obj fun - theor moments", label = "L1")





