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
using Expectations
using Parameters

Random.seed!(612)
#Set up the Parameters
β = 0.9;
μ = -1;
R = -3;

###################################
#STEP 1: VALUE FUNCTION ITERATION
###################################

#Step 1.1: Set up the grid of possible state variables 
a = [1,2,3,4,5] #Note: We have a small, finite set of possible states: a_t = 1,2,3,4,5


function VFI2(μ,R, max_iter = 500, tol = 1e-6)

    #@unpack μ, R = θ
    v = rand(5,2)
    vprev = zeros(5,2)
    err = 1
    i = 1
    while err > tol && i < max_iter
        for a = 1:5
            j = min(a+1,5)
            v[a,1] = R + β*log(exp(vprev[1,1]) + exp(vprev[1,2]) ) 
            v[a,2] = μ*j + β*log( exp(vprev[j,2]) + exp(vprev[j,1] ) ) 
        end

        err = norm(v-vprev)
        vprev = copy(v)
        i = i+1
    end
    return vprev
end

vf2 = VFI2(-1,-3)

θ_init = [μ, R]
#vf2 = VFI2(θ_init)

#The probabilities of observing each choice
prob2 = zeros(5,2)

for a in 1:5
    prob2[a,1] = exp(vf2[a,1])/(exp(vf2[a,1]) + exp(vf2[a,2]))
    prob2[a,2] = exp(vf2[a,2])/(exp(vf2[a,1]) + exp(vf2[a,2]))
end

###################################
#STEP 2: DATA SIMULATION 
###################################

d1 = Gumbel()
draw = rand(d1, 40000)
ϵ1 = draw[1:20000]
ϵ0 = draw[20001:40000]
a_obs = sample(a, 20000, replace = true)
#Assuming (μ, R) = (-1, -3)
i_obs = zeros(20000)
for i in 1:20000
    if β*vf2[a_obs[i],1]+ R + ϵ1[i] > β*vf2[a_obs[i],2]+ μ*a_obs[i]+ϵ0[i]
        i_obs[i] = 0
    else
        i_obs[i] = 1
    end
end

###################################
#QUESTION 5.1: INNER NXFP LOOP
###################################

#Step 5.a: Guess Parameter (μ,R)
init_θ = (0,0)

#Step 5.b: Estimate Dynamic Logit Probability using V(1), V(0) and VFI
VFI2(θ)

#Step 5.c: Estime the LL using the EV distribution of the Errors and the formula
P(μ,R) = sum( u() +  )

###################################
#QUESTION 5.4: OUTER NXFP LOOP
###################################

#Step 5.d: Run the Inner loop over a set of parameter to maximize LL
optimize(P,init_θ)
