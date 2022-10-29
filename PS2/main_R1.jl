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
μ = 0.5
σ = 0.5
d1 = Normal(μ,σ)
u = rand(d1,100,3)

Φ = zeros(100,3) 
Φ[:,1] = α*entryData[:,Z_1] .+ u[:,1] 

