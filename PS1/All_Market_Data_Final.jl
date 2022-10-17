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
#using Symbolics

##############################
#Simulate Data for 100 markets
##############################

Random.seed!(321)
d1 = Normal(0,1)
d2 = Uniform(0,1)
d3 = LogNormal(0,1)

θ_true = [5,1,1,1,1,2,1,1]

#Demand
x1 = ones(100,3)
x2 = rand(d2,100,3)
x3 = rand(d1,100,3)

ξ = rand(d1,100,3)
ν_true = rand(d3,100,100)

β = θ_true[1:3]
α = θ_true[4]
σ_α = θ_true[5]

function model_ind_shares(β,α,σ_α,ξ,p,ν)  

    δ = x1*β[1] + x2*β[2] + x3*β[3] - α*p + ξ
    s_i = zeros(100,3,100)

    for m=1:100
        for i=1:100
            μ_i1 = σ_α*p[1]*ν[i,m]
            μ_i2 = σ_α*p[2]*ν[i,m]
            μ_i3 = σ_α*p[3]*ν[i,m]
            
            s_i[i,1,m] = exp(δ[1] + μ_i1)/(1+exp(δ[1] + μ_i1)+exp(δ[2] + μ_i2)+exp(δ[3] + μ_i3)) 
            s_i[i,2,m] = exp(δ[2] + μ_i2)/(1+exp(δ[1] + μ_i1)+exp(δ[2] + μ_i2)+exp(δ[3] + μ_i3)) 
            s_i[i,3,m] = exp(δ[3] + μ_i3)/(1+exp(δ[1] + μ_i1)+exp(δ[2] + μ_i2)+exp(δ[3] + μ_i3))
        end
    end
    return s_i
end

function model_elasticity(s_i,α,σ_α,p,ν)
    s = zeros(100,3)
    ϵ = zeros(100,3)

    for m = 1:100
        α_i = α*ones(100) + σ_α*ν[:,m]

        s[m,1] = mean(s_i[:,1,m])
        s[m,2] = mean(s_i[:,2,m])
        s[m,3] = mean(s_i[:,3,m])

        ϵ[m,1] = -1*mean(α_i.*s_i[:,1,m].*(1 .-s_i[:,1,m]))*(p[m,1]./s[m,1]) 
        ϵ[m,2] = -1*mean(α_i.*s_i[:,2,m].*(1 .-s_i[:,2,m]))*(p[m,2]./s[m,2])  
        ϵ[m,3] = -1*mean(α_i.*s_i[:,3,m].*(1 .-s_i[:,3,m]))*(p[m,3]./s[m,3])  
    end

    return s,ϵ
end

# function model_allelasticity(s_i,α,σ_α,p,ν)
#     s = zeros(100,3)
#     ϵ = zeros(100,3)

#     for m = 1:100
#         α_i = α*ones(100) + σ_α*ν[:,m]

#         s[m,1] = mean(s_i[:,1,m])
#         s[m,2] = mean(s_i[:,2,m])
#         s[m,3] = mean(s_i[:,3,m])

#         ϵ[1,2,m] = -1*mean(α_i.*s_i[:,1,m].*s_i[:,2,m])*(p[m,1]./s[m,1]) 
#         ϵ[1,3,m] = -1*mean(α_i.*s_i[:,1,m].*s_i[:,3,m])*(p[m,2]./s[m,2])  
#         ϵ[2,3,m] = -1*mean(α_i.*s_i[:,2,m].*s_i[:,3,m])*(p[m,3]./s[m,3])  

#         ϵ[2,1,m] = -1*mean(α_i.*s_i[:,2,m].*s_i[:,1,m])*(p[m,1]./s[m,1]) 
#         ϵ[3,1,m] = -1*mean(α_i.*s_i[:,3,m].*s_i[:,1,m])*(p[m,2]./s[m,2])  
#         ϵ[3,2,m] = -1*mean(α_i.*s_i[:,3,m].*s_i[:,2,m])*(p[m,3]./s[m,3])  

#         ϵ[1,1,m] = -1*mean(α_i.*s_i[:,1,m].*(1 .-s_i[:,1,m]))*(p[m,1]./s[m,1]) 
#         ϵ[2,2,m] = -1*mean(α_i.*s_i[:,2,m].*(1 .-s_i[:,2,m]))*(p[m,2]./s[m,2])  
#         ϵ[3,3,m] = -1*mean(α_i.*s_i[:,3,m].*(1 .-s_i[:,3,m]))*(p[m,3]./s[m,3])  

#     end
#     return s,ϵ
# end

#Supply
γ_0 = θ_true[6]
γ_1 = θ_true[7]
γ_2 = θ_true[8]

W = repeat(max.(rand(d1,1,3),0),100)
Z = max.(rand(d1,100,3),0)
η = max.(rand(d1,100,3),0)

MC = γ_0*ones(100,3) + γ_1*W + γ_2*Z + η

#Equilibrium
p_guess=rand(Uniform(0,5),100,3)
p = rand(Uniform(5,10),100,3)
s = zeros(100,3)
ϵ = zeros(100,3)

while norm(p - p_guess) > 0.0001
    p_guess = p
    s_i = model_ind_shares(β,α,σ_α,ξ,p,ν_true) 
    s,ϵ = model_elasticity(s_i,α,σ_α,p,ν_true) 
    for m=1:100
        p[m,1] = (ϵ[m,1]/(1+ϵ[m,1])) * MC[m,1]
        p[m,2] = (ϵ[m,2]/(1+ϵ[m,2])) * MC[m,2]
        p[m,3] = (ϵ[m,3]/(1+ϵ[m,3])) * MC[m,3]
    end
end

#######################
# P1
#######################
ν_sim = rand(d3,100,100)


#Step 1.2: Inversion to find δ from shares
function share_prediction(δ, p, θ, ν)
  
    s = zeros(100,3)
    s_i = zeros(100,100,3)
    for m=1:100
        for i=1:100
            μ_i1 = θ[5]*p[m,1]*ν[i,m]
            μ_i2 = θ[5]*p[m,2]*ν[i,m]
            μ_i3 = θ[5]*p[m,3]*ν[i,m]
            #μ_i = 0

            s_i[m,i,1] = exp(δ[m,1]+μ_i1)/(1+exp(δ[m,1]+μ_i1)+exp(δ[m,2]+μ_i2)+exp(δ[m,3]+μ_i3))     
            s_i[m,i,2] = exp(δ[m,2]+μ_i2)/(1+exp(δ[m,1]+μ_i1)+exp(δ[m,2]+μ_i2)+exp(δ[m,3]+μ_i3))     
            s_i[m,i,3] = exp(δ[m,3]+μ_i3)/(1+exp(δ[m,1]+μ_i1)+exp(δ[m,2]+μ_i2)+exp(δ[m,3]+μ_i3))           
        end
        s[m,1] = sum(s_i[m,:,1])/100
        s[m,2] = sum(s_i[m,:,2])/100
        s[m,3] = sum(s_i[m,:,3])/100  
    end
    return s
end

function contraction_map(s, p, δ_new, θ, ν)
    δ_guess = rand(100,3)
    while norm(δ_new - δ_guess) > .01
        δ_guess = δ_new
        s_pred = share_prediction(δ_guess,p, θ, ν)
        δ_new = δ_guess + log.(s) - log.(s_pred)
    end
    return δ_new
end


#######################
# P2
#######################
#1a
θ_id = [6.948,0.4632,2.734,1.642,0.1643]
#θ_id = [5,1,1,1,1]

X = zeros(100,3,3)
for j=1:3
    X[:,:,j] = cat(x1[:,j], x2[:,j], x3[:,j], dims=2)
end

function back_ξ(θ, s, p, X, W, Z, ν_sim)
    δ_guess = rand(100,3)*25
    δ_cm = contraction_map(s, p, δ_guess, θ, ν_sim)

    # IV-GMM (Only Demand Size)
    x = zeros(100,4,3)
    x = cat([X[:,:,1] p[:,1] ], [X[:,:,2] p[:,2] ], [ X[:,:,3] p[:,3] ], dims=1) #100*4

    #100x3x3 (Summed over each characteristic)
    demand_instruments = cat(sum(X,dims=3)-X[:,:,1], sum(X,dims=3)-X[:,:,2],sum(X,dims=3)-X[:,:,3], dims=1)[:,:,1]

    #100x2x3 (2 Level Shocks for Supply)
    supply_instruments = cat([W[:,1] Z[:,1]], [W[:,2] Z[:,2]], [W[:,3] Z[:,3]], dims=1)

    #5 Instruments (3 Demands Characteristics and 2 supply shocks)
    z= cat(demand_instruments, supply_instruments, dims=2)

    wt = Matrix(I,5,5)
    
    #β_hat = zeros(4,3)
    β_hat = inv(transpose(x)*z*wt*transpose(z)*x)*transpose(x)*z*wt*transpose(z)*reshape(δ_cm,300,1)

    ξ = δ_cm - cat([X[:,:,1] p[:,1]]*β_hat,[X[:,:,2] p[:,2]]*β_hat,[X[:,:,3] p[:,3]]*β_hat,dims=2)  

    return ξ
end    

ξ_hat = back_ξ(θ_id,s,p,X,W,Z,ν_sim)
#ξ_hat = rand(Normal(0,1),100,3)

s_isim = model_ind_shares(θ_id[1:3],θ_id[4],θ_id[5],ξ_hat,p,ν_sim)
s_oli_hat,ϵ_own_hat = model_elasticity(s_isim,θ_id[4],θ_id[5],p,ν_sim)
s_coll_hat, ϵ_coll_hat = model_crosselasticity(s_isism,θ_id[4],θ_id[5],p,ν_sim)

function model_crosselasticity(p, X, θ, ξ, ν)
    s_1 = zeros(100)
    s_2 = zeros(100)
    s_3 = zeros(100)

    prob = zeros(100,1000,3)
    l=0
    for m=1:100
        for i=1:1000
            δ_1 = transpose(X[m,:,1])*θ[1:3] +ξ[m,1] + θ[4]*p[m,1]
            δ_2 = transpose(X[m,:,2])*θ[1:3] +ξ[m,2] + θ[4]*p[m,2]
            δ_3 = transpose(X[m,:,3])*θ[1:3] +ξ[m,3] + θ[4]*p[m,3]

            μ_i = θ[5]*p[m,1]*ν[i+l]

            prob[m,i,1] = exp(δ_1+μ_i)/(1+exp(δ_1+μ_i)+exp(δ_2+μ_i)+exp(δ_3+μ_i))     
            prob[m,i,2] = exp(δ_2+μ_i)/(1+exp(δ_1+μ_i)+exp(δ_2+μ_i)+exp(δ_3+μ_i))     
            prob[m,i,3] = exp(δ_3+μ_i)/(1+exp(δ_1+μ_i)+exp(δ_2+μ_i)+exp(δ_3+μ_i))           
        end
        l=1000*m
        s_1[m] = sum(prob[m,:,1])/1000
        s_2[m] = sum(prob[m,:,2])/1000
        s_3[m] = sum(prob[m,:,3])/1000   
    end

    ϵ = ones(300,3)

    q=0
    for m=1:100
        α_i = θ[4].+θ[5]*ν[q+1:q+1000]
        ϵ[3*m-2,1] = sum(α_i.*prob[m,:,1].*(ones(1) .- prob[m,:,1]))/(-1000)
        ϵ[3*m-1,2] = sum(α_i.*prob[m,:,2].*(ones(1) .- prob[m,:,2]))/(-1000)
        ϵ[3*m,3] = sum(α_i.*prob[m,:,3].*(ones(1) .- prob[m,:,3]))/(-1000)
        ϵ[3*m-2,2] = sum(α_i.*prob[m,:,2])/(1000)
        ϵ[3*m-2,3] = sum(α_i.*prob[m,:,3])/(1000)
        ϵ[3*m-1,1] = sum(α_i.*prob[m,:,1])/(1000)
        ϵ[3*m-1,3] = sum(α_i.*prob[m,:,3])/(1000)
        ϵ[3*m,1] = sum(α_i.*prob[m,:,1])/(1000)
        ϵ[3*m,2] = sum(α_i.*prob[m,:,2])/(1000)
        q=1000*m
    end

    s = hcat(s_1, s_2, s_3)
    return ϵ
end

#Marginal Cost (Perfect Competition)
mc_pc = p

#Marginal Cost (Oligopoly)
mc_oli = zeros(100,3)
for m = 1:100
    mc_oli[m, :] = p[m,:] + inv(diagm(ϵ_own_hat[m,:]))*s[m,:]
end

#Marginal Cost (Perfect Collusion)
ϵ_coll = model_crosselasticity(p, X, θ_id, ξ_hat, ν_sim)
mc_coll = zeros(100,3)
for m = 1:100
    mc_coll[m, :] = p[m,:] + (1 ./ϵ_coll[3*m-2:3*m,:])*s[m,:]
end

#1b
comp = DataFrame(firm1 = mc_pc[:,1], firm2 = mc_pc[:,2], firm3 = mc_pc[:,3])
stackcomp = stack(comp, 1:3)
plot(stackcomp, y =:value, x =:variable,  kind = "box")

oli = DataFrame(firm1 = mc_oli[:,1], firm2 = mc_oli[:,2], firm3 = mc_oli[:,3])
stackoli = stack(oli, 1:3)
plot(stackoli, y =:value, x =:variable,  kind = "box")

cost_data = DataFrame(Competition = mc_pc[:], Oligopoly = mc_oli[:], Collusion = mc_coll[:])
cost_data = stack(cost_data, 1:3)
plot(cost_data, y =:value, x =:variable, kind = "box")