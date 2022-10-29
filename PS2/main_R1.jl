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

#Import entryData.csv and rename columns
entryData = DataFrame(CSV.read("PS2/entryData.csv",DataFrame, header=false))
column_names = ["X", "Z_1", "Z_2", "Z_3", "E1", "E2", "E3"]
rename!(entryData,Symbol.(column_names))

####################
# 1.3.1: Berry 1992
####################

#Equilibrium Number of Firms
entryData[:,"N_star"] = entryData.E1 + entryData.E2 + entryData.E3

#True Parameters
(α,β,δ) = (1,1,1)

#Fixed Costs for Firm-Market
μ = 2
σ = 1
d1 = Normal(μ,σ)
n_hat = zeros(100,4,100)

for t in 1:100
    u = rand(d1,100,3)

    Φ = zeros(100,3)

    Φ = α*Matrix(entryData[:,2:4]) + u 


    #Order firms by fixed costs
    q = Matrix{Int64}(undef, 100,3)
    for m in 1:100
        q[m,:] = sortperm(Φ[m,:])
    end

    π = zeros(100,3)

    for m in 1:100
        π[m,q[m,1]] = β*entryData.X[m] - δ*log(1) - Φ[m,q[m,1]]
        π[m,q[m,2]] = β*entryData.X[m] - δ*log(2) - Φ[m,q[m,2]]
        π[m,q[m,3]] = β*entryData.X[m] - δ*log(3) - Φ[m,q[m,3]]
    end

    predentry = (π .>0)
    n = predentry[:,1]+predentry[:,2] + predentry[:,3]
    n_hat[:,:,t] = cat(predentry, n, dims = 2)
end

mean(n_hat, dims=3)
