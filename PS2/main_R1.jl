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
(α,β,δ) = 1,1,1

#Fixed Costs for Firm-Market
μ = 2
σ = 1

function sim_berry(μ, σ, T)
    d1 = Normal(μ,σ)
    n_hat = zeros(100,4,100)

    for t in 1:T
        u = rand(MersenneTwister(t),d1,100,3)

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

obj_berry(θ) = transpose(moment_berry(entryData, θ[1], θ[2], 100))*
                         moment_berry(entryData, θ[1],θ[2], 100)

initial = [0.5,0.5]
lower = [-3,.1]
upper = [Inf,Inf]

gmm_berry = optimize(obj_berry, lower, upper, initial)
θ_berry = Optim.minimizer(gmm_berry)

####################
# 1.3.2: CT 2009
####################

# Simulation for H1 and H2 
function sim_tamer(μ, σ, T, data)
    entryData = data
    s = size(entryData)[1]
    (α,β,δ) = (1,1,1)
    d1 = Normal(μ,σ)
    h1_hat = zeros(s,3,T)
    h2_hat = zeros(s,3,T)
    for t in 1:T
        u = rand(d1,s,3)

        Φ = zeros(s,3)

        Φ = α*Matrix(entryData[:,2:4]) + u 


        #Calculate profits
        Π = zeros(8,3)
        high = zeros(s, 3)
        low = zeros(s, 3)
        for m in 1:s
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
                for f in 1:3
                    if low[m,f] == 1
                        low[m,f] = 1
                    else
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
                for f in 1:3
                    if low[m,f] == 1
                        low[m,f] = 1
                    else
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

# Generating the Objective Function 
function calc_mi(μ, data,T)
    σ = 1
    h1, h2 = sim_tamer(μ, σ, T, data)
    Q = (1/T)*sum(norm(Matrix(data[:,5:7])-h1[:,:,1])+norm(Matrix(data[:,5:7])-h2[:,:,1]))
    return Q
end

# Let min_mi be the minimized value of the objective calc_mi(μ; data) = a_n Q_n(μ, data)
res = optimize(μ -> calc_mi(μ, entryData,100), .5, 4)
min_mi = Optim.minimum(res)

# Define c0 as the 1.25*min_mi following Ciliberto-Tamer
c0 = 1.25 * min_mi

# Find initial confidence region by evaluating the obj in a grid of 51 points (-1 to 4)
function calc_mi2(μ)
    calc_mi(μ,entryData, 100)
end
MU = -1:0.1:4 
MU_I = MU[calc_mi2.(MU) .<= c0] 
μ0_lb = minimum(MU_I)
μ0_ub = maximum(MU_I)

# Generate subsamples with a subsample size of M/4 following Ciliberto-Tamer
# Compute max value of the obj function of each subsample over initial 
# confidence region by finding the min of the negative obj function, and
# subtract from that the min of the obj function in the same region
# following Ciliberto-Tamer to correct for potential misspecification.

M = 100
B = 100

# Write a function subsample(data, size) to generate subsamples from data of a particular size
function subsample(data, m, b)
    selec = rand(MersenneTwister(b),1:100,m)
    sub_data = data[selec, :]
    return sub_data
end
function calc_mi_subsample(MU_I, data, M, b)
    m = Int64(M/4)
    sub_data = subsample(data, m, b)
    obj_values = map(μ -> calc_mi(μ, sub_data,m), MU_I) #Calculate a_n Q_n(μ, sub_data) for all μ in μ_I
    max_mi_sub = maximum(obj_values) # C_n = sup_{μ ∈ μ_I} a_n Q_n(μ, sub_data)
    min_mi_sub = minimum(obj_values) # to correct for misspecification
    return (max_mi_sub - min_mi_sub) 
end

# Take 1/4 the 95th percentile and set equal to c1 to compute 95% CI (1/4
# because #subsample=M/4)
c1_subsamples = zeros(100)
for b=1:B
    c1_subsamples[b] = calc_mi_subsample(MU_I, entryData, M, b)
end
c1 = (1/4)*quantile(c1_subsamples, 0.95)

# compute ci1 using Ciliberto and Tamer's estimator modified for
# misspecification
MU_I1 = MU[calc_mi2.(MU) .- min_mi .<= c1] 
μ1_lb = minimum(MU_I1)
μ1_ub = maximum(MU_I1)

#Last, repeat the subsampling procedure a few times to update the bounds
#further. In doing so, use a finer grid to obtain more accurate bounds (e.g.,
#MU = -1:0.025:4).

c2_subsamples = zeros(100)
for b=1:B
    c2_subsamples[b] = calc_mi_subsample(MU_I1, entryData, M, b)
end
c2 = (1/4)*quantile(c2_subsamples, 0.95)

# compute ci1 using Ciliberto and Tamer's estimator modified for
# misspecification
MU_I2 = MU[calc_mi2.(MU) .- min_mi .<= c2] 
μ2_lb = minimum(MU_I2)
μ2_ub = maximum(MU_I2)