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

#Step 1.2: Initial Guess of the Value function
init_v = zeros(5,2) # 5 states x 2 Actions
vprev = copy(init_v)

v = zeros(5,2)

#Step 1.3: Use the log-sum closed form formula to calculate value for each state action pair (hence function)

err = 1000
while err >= 1e-12

    v[1,1] = log( exp((R) + β*vprev[1,2]) ) 
    v[1,2] = log( exp(μ*1 + β*vprev[1,1]) )  
    v[2,1] = log( exp((R) + β*vprev[2,2]) ) 
    v[2,2] = log( exp(μ*2 + β*vprev[2,1]) )  
    v[3,1] = log( exp((R) + β*vprev[3,2]) ) 
    v[3,2] = log( exp(μ*3 + β*vprev[3,1]) )  
    v[4,1] = log( exp((R) + β*vprev[4,2]) ) 
    v[4,2] = log( exp(μ*4 + β*vprev[4,1]) )  
    v[5,1] = log( exp((R) + β*vprev[5,2]) ) 
    v[5,2] = log( exp(μ*5 + β*vprev[5,1]) )  

    #Step 1.4 Check distance to see convergence
    err = norm(v-vprev)

    #Step 1.5: Continue iteration by updating previous value
    vprev = copy(v)

end

function VFI(μ, R, max_iter = 500, tol = 1e-6)

    v = rand(5,2)
    vprev = zeros(5,2)
    err = 1
    i = 1
    while err > tol && i < max_iter
        for a = 1:5
            v[a,1] = log( exp((R) + β*vprev[a,1]) ) 
            v[a,2] = log( exp(μ*a + β*vprev[a,2]) )  
        end

        err = norm(v-vprev)
        vprev = copy(v)
        i = i+1
    end
    return vprev
end


function VFI2(μ, R, max_iter = 500, tol = 1e-6)

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

vf1 = VFI(-1,-3)
vf2 = VFI2(-1,-3)


#The probabilities of observing each choice
prob1 = zeros(5,2)
prob2 = zeros(5,2)
for a in 1:5
    prob1[a,1] = exp(vf1[a,1])/(exp(vf1[a,1]) + exp(vf1[a,2]))
    prob1[a,2] = exp(vf1[a,2])/(exp(vf1[a,1]) + exp(vf1[a,2]))
end

for a in 1:5
    prob2[a,1] = exp(vf2[a,1])/(exp(vf2[a,1]) + exp(vf2[a,2]))
    prob2[a,2] = exp(vf2[a,2])/(exp(vf2[a,1]) + exp(vf2[a,2]))
end


#Data generation
d1 = Gumbel()
draw = rand(d1, 4000)
ϵ1 = draw[1:2000]
ϵ0 = draw[2001:4000]
a_obs = sample(a, 2000, replace = true)
#Assuming (μ, R) = (-1, -3)
i_obs = zeros(2000)
for i in 1:2000
    if β*vf2[a_obs[i],1]+ R + ϵ1[i] > β*vf2[a_obs[i],2]+ μ*a_obs[i]+ϵ0[i]
        i_obs[i] = 0
    else
        i_obs[i] = 1
    end
end

###################################
#STEP 2: INNER NXFP LOOP
###################################

#Step 2.1: Guess Parameter (μ,R)

#Step 2.2: Estimate Dynamic Logit Probability using V(1), V(0) and VFI

#Step 2.3: Estime the LL using the EV distribution of the Errors and the formula

###################################
#STEP 3: OUTER NXFP LOOP
###################################

#Step 3.1: Run the Inner loop over a set of parameter to maximize LL

#This is STEP 5.d