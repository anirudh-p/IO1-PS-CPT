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

function sim_berry(μ, σ, T)
    (α,β,δ) = (1,1,1)
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
        Π = zeros(100,3)

        for m in 1:100
            Π[m,q[m,1]] = β*entryData.X[m] - δ*log(1) - Φ[m,q[m,1]]
            Π[m,q[m,2]] = β*entryData.X[m] - δ*log(2) - Φ[m,q[m,2]]
            Π[m,q[m,3]] = β*entryData.X[m] - δ*log(3) - Φ[m,q[m,3]]
        end

        #add column for total entrants and save
        predentry = (Π .>0)
        n = predentry[:,1]+predentry[:,2] + predentry[:,3]
        n_hat[:,:,t] = cat(predentry, n, dims = 2)
    end

    return mean(n_hat, dims=3)
end

function moment_berry(entryData, μ, σ, T)
    N_hat = sim_berry(μ, σ, T)
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


function sim_tamer(μ, σ, T)
    (α,β,δ) = (1,1,1)
    d1 = Normal(μ,σ)
    h1_hat = zeros(100,3,100)
    h2_hat = zeros(100,3,100)
    for t in 1:T
        u = rand(d1,100,3)

        Φ = zeros(100,3)

        Φ = α*Matrix(entryData[:,2:4]) + u 


        #Calculate profits
        Π = zeros(8,3)
        high = zeros(100, 3)
        low = zeros(100, 3)
        for m in 1:100
            Π[1,1] = β*entryData.X[m] - δ*log(3) - Φ[m,1]
            Π[1,2] = β*entryData.X[m] - δ*log(3) - Φ[m,2]
            Π[1,3] = β*entryData.X[m] - δ*log(3) - Φ[m,3]
            if all(>=(0),Π[1,:])
                high[m,:] = ones(1,3)
                low[m,:] = ones(1,3)
                continue
            end
            #firms 2&3 enter
            Π[2,1] = 0
            Π[2,2] = β*entryData.X[m] - δ*log(2) - Φ[m,2]
            Π[2,3] = β*entryData.X[m] - δ*log(2) - Φ[m,3]
            #firms 1&3 enter
            Π[3,1] = β*entryData.X[m] - δ*log(2) - Φ[m,1]
            Π[3,2] = 0
            Π[3,3] = β*entryData.X[m] - δ*log(2) - Φ[m,3]
            #firms 1&2 enter
            Π[4,1] = β*entryData.X[m] - δ*log(2) - Φ[m,1]
            Π[4,2] = β*entryData.X[m] - δ*log(2) - Φ[m,2]
            Π[4,3] = 0
            for f in 1:3 
                if all(>=(0), Π[f+1,:]) && Π[1,f] >= 0
                    high[m,1:end .!=f] = ones(1,2)
                    low[m,1:end .!=f] .+= ones(2,1)
                end
            end
            if any(>(0), high[m,:])
                equil = sum(high[m,:])
                for f in 1:3
                    if low[m,f] == equil
                        low[m,f] = 1
                    elseif low[m,f] < equil
                        low[m,f] = 0
                    end 
                end
                continue
            end
            #firm 1 enters
            Π[5,1] = β*entryData.X[m] - δ*log(1) - Φ[m,1]
            #firm 2 enters
            Π[6,2] = β*entryData.X[m] - δ*log(1) - Φ[m,2]
            #firm 3 enters
            Π[7,3] = β*entryData.X[m] - δ*log(1) - Φ[m,3]

            for f in 1:3
                if Π[f+4,f] >= 0 && all(<=(0),Π[2:4,f])
                    high[m,f] = 1
                    low[m,f] += 1
                end
            end
            if any(>(0), high[m,:])
                equil = sum(high[m,:])
                for f in 1:3
                    if low[m,f] == equil
                        low[m,f] = 1
                    elseif low[m,f] < equil
                        low[m,f] = 0
                    end 
                end
                continue
            end
        end
        
        h1_hat[:,:,t] = low
        h2_hat[:,:,t] = high
    end

    return mean(h1_hat, dims=3),mean(h2_hat,dims=3)
end

sim_tamer(2,1,100)