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
#QUESTION 3: VALUE FUNCTION ITERATION
###################################

#Set up the grid of possible state variables 
a = [1,2,3,4,5] #Note: We have a small, finite set of possible states: a_t = 1,2,3,4,5

#set up the parameter input structure
@with_kw struct Inputs
    μ::Float64 = -1 #Mean Transition Cost
    R::Float64 = -3 # Replacement Cost
end

θ = Inputs(-1,-3) #Test

# Main Value Function Iteration (VFI)
function VFI2(θ, max_iter = 500, tol = 1e-6)

    @unpack μ, R = θ
    v = rand(5,2)
    vprev = zeros(5,2)
    err = 1
    i = 1
    while err > tol && i < max_iter
        for a = 1:5
            j = min(a+1,5)
            v[a,1] = R + β*log(exp(vprev[1,1]) + exp(vprev[1,2]) )  # Left Column (Indexed 1) is Renewal
            v[a,2] = μ*j + β*log( exp(vprev[j,2]) + exp(vprev[j,1] ) ) #Right Column (Indexed 2) is Non-Renewal
        end

        err = norm(v-vprev)
        vprev = copy(v)
        i = i+1
    end
    return vprev
end

vf2 = VFI2(θ) 

###################################
#QUESTION 4: DATA SIMULATION 
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

######################
# Deterministic states
a_obs2 = zeros(20001)
a_obs2[1] = 1
i_obs2 = zeros(20000)

for i in 1:20000
    #Current Decision based on Current State VF (LHS is V1, RHS is V0)
    if R + ϵ1[i] + β*vf2[Int(a_obs2[i]),1] > μ*a_obs2[i] + ϵ0[i] + β*vf2[Int(a_obs2[i]),2] 
        i_obs2[i] = 1
    else
        i_obs2[i] = 0
    end
    #Next State based on Current Decision
    if i_obs2[i] == 1
        a_obs2[i+1] = 1
    else 
        a_obs2[i+1] = min(5,a_obs2[i]+1)
    end
end

count(i -> (i==5), a_obs2) #Check there is some variation in the data

###################################
#QUESTION 5.1: INNER NXFP LOOP
###################################

#Question 5.a: Guess Initial Parameter (μ,R)
init_θ = Inputs(-0.5,-3.5)

#Step 5.b: Run the VFI
VFI2(init_θ)

#Step 5.c: Estime the LL using the EV distribution of the Errors and the formula
#The probabilities of observing each choice
prob2 = zeros(5,2)

for a in 1:5
    prob2[a,1] = exp(vf2[a,1])/(exp(vf2[a,1]) + exp(vf2[a,2]))
    prob2[a,2] = exp(vf2[a,2])/(exp(vf2[a,1]) + exp(vf2[a,2]))
end

function LL(θ, i_obs, a_obs)
    vf = VFI2(θ)                #Combined Steps a, b and c by including the VFI(guess_θ) inside this inner loop
    num = zeros(20000)
    den = zeros(20000)

    for i in 1:20000
        if i_obs2[i] == 0.0
            d = 2
        else
            d = 1
        end
        Π = i_obs[i] * (R + ϵ1[i]) + (1-i_obs[i])*(μ*a_obs[i] + ϵ0[i])

        num[i] = β*vf[Int(a_obs[i]),d]  + Π     #Numerator of obs i is a function of the state at i and decision at i
        den[i] = (β*vf[Int(a_obs[i]),1] + R + ϵ1[i]) + (β*vf[Int(a_obs[i]),2] + μ*a_obs[i] + ϵ0[i]) 
    end

    return sum(num)  - sum(log.(den))
end

θ_prime = Inputs(-1,-5)
P = LL(θ_prime, i_obs2, a_obs2)

###################################
#QUESTION 5.4: OUTER NXFP LOOP
###################################

#Step 5.d: Run the Inner loop over a set of parameter to maximize LL
optimize(P,init_θ)
