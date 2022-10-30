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

function sim_n(μ, σ, T)
    d1 = Normal(μ,σ)
    n_hat = zeros(100,4,100)

    for t in 1:T
        u = rand(d1,100,3)

        Φ = zeros(100,3)

        Φ = α*Matrix(entryData[:,2:4]) + u 


        #Order firms by fixed costs
        q = Matrix{Int64}(undef, 100,3)
        for m in 1:100
            q[m,:] = sortperm(Φ[m,:])
        end

        #Calculate profits
        π = zeros(100,3)

        for m in 1:100
            π[m,q[m,1]] = β*entryData.X[m] - δ*log(1) - Φ[m,q[m,1]]
            π[m,q[m,2]] = β*entryData.X[m] - δ*log(2) - Φ[m,q[m,2]]
            π[m,q[m,3]] = β*entryData.X[m] - δ*log(3) - Φ[m,q[m,3]]
        end

        #add column for total entrants and save
        predentry = (π .>0)
        n = predentry[:,1]+predentry[:,2] + predentry[:,3]
        n_hat[:,:,t] = cat(predentry, n, dims = 2)
    end

    return mean(n_hat, dims=3)
end

function moment_berry(entryData, μ, σ, T)
    N_hat = sim_n(μ, σ, T)
    N_star = Matrix(entryData[:,5:8])
    ν = N_star-N_hat

    m1 = transpose(Matrix(entryData[:,1:2]))*ν[:,1]
    m2 = transpose(cat(entryData.X, entryData.Z_2, dims=2))*ν[:,2]
    m3 = transpose(cat(entryData.X, entryData.Z_3, dims=2))*ν[:,3]
    m4 = transpose(entryData.X)*ν[:,4]

    moments = cat(m1, m2, m3, m4, dims=1)
    return vec(moments)
end

obj_berry(θ) = transpose(moment_berry(entryData, θ[1], θ[2], 100))*moment_berry(entryData, θ[1], 
θ[2], 100)

obj_berry([.5,.5])
initial = [.5,.5]
lower = [-3,.1]
upper = [Inf,Inf]

gmm_berry = optimize(obj_berry, lower, upper, initial)
θ_berry = Optim.minimizer(gmm_berry)