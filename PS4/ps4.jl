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

#Set up inner loop contraction mapping
μ, R = -1, -3
β = .9

function in_loop(μ, R)
    #Set initial guess for EV to 0
    #EV(a,1) should be the same for all a
    #EV(a,0) should be the same for a >= 4
    #evguess and evupdate will store EV values starting with EV(a,1) and then EV(1-4,0)
    evguess = zeros(5)
    evupdate = ones(5)
    u(a, i) = μ*a*(1-i) + R*(i)
    econst = .5775
    yt = [1,2,3,4,5]
    nor = 100
    tol = .1

    #Loop while norm of difference is greater than set tolerance
    while nor > tol
        evupdate = log.(exp.(u.(yt,0)+β*evguess) + exp.(u.(yt,1)+β*fill(evguess[1],5)))
        nor = norm(evupdate-evguess)
        evguess = evupdate
    end

    #return vector of expected values
    return evupdate
end

evupdate = in_loop(μ,R)
p0(a) = exp(evupdate[1])/(exp(evupdate[1])+exp(evupdate[a+1]))

p0(1)