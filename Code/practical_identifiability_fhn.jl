using DifferentialEquations
using ForwardDiff
using DiffResults
using DiffEqSensitivity
using Statistics
using Random
using LinearAlgebra
using DataFrames
# using CSV
# using Dierckx
# using Plots
# using StatsPlots
using LaTeXStrings



# function practical_identifiability_fhn(p = [-0.3008, 1.104, 19.98, 0.2301], trainable=[false,true,true,false], states=[1], noise = 0.1)
function practical_identifiability_fhn(p = [1.104, 19.98], trainable=[false,true,true,false], states=[1], noise = 0.1)
# function practical_identifiability_fhn(noise, p0 = [-0.3008, 1.104, 19.98, 0.2301], trainable=[false,true,true,false], states=[1])
# function practical_identifiability_fhn(p, trainable=[false,true,true,false], states=[1], noise = 0.1)
    
    p_real = [-0.3, 1.1, 20., 0.23]

    # p = p0[trainable]
    # println(p0[trainable])
    # println(p0[.!trainable], p)

    function f_fhn(dx, x, p, t)
        x1, x2 = x
        #this is kinda sketchy, but it will work since all trainable variables are next to each other in our cases
        if trainable[1]
            pa = [ trainable[x] ? p[x] : p_real[x] for x in 1:4  ] 
        elseif trainable[2]
            pa = [ trainable[x] ? p[x-1] : p_real[x] for x in 1:4  ]
        elseif trainable[3]
            pa = [ trainable[x] ? p[x-2] : p_real[x] for x in 1:4  ]
        else
            pa = [ trainable[x] ? p[x-3] : p_real[x] for x in 1:4  ]
        end
        dx[1] = (x1 - x1^3 - x2 + pa[4])
        dx[2] = (x1 - pa[1] - pa[2] * x2) / pa[3]
    end

    x0 = [0., 0.]
    tspan = (0.0, 999.0)
    prob_fhn = ODEForwardSensitivityProblem(f_fhn, x0, tspan, p)

    sol_fhn = solve(prob_fhn, alg_hints=[:stiff], saveat=0.1)
    x_fhn, dp_fhn = extract_local_sensitivities(sol_fhn)

    lab = [L"a", L"b", L"\tau", L"I_{ext}"][trainable]
    
    __noise__ = (noise>0) ? noise : .01
    # println(__noise__)
    σ = __noise__ * std(x_fhn, dims=2)
    cov_ϵ = Diagonal( σ[ states[1]:states[end] ] )
    dp = dp_fhn
    cols = states[1]:states[end]


    Nt = length(dp[1][1,:])
    Nstate = length(dp[1][:,1])
    Nparam = length(dp[:,1])
    F = zeros(Float64, Nparam, Nparam)
    perm = vcat(1, sort(rand(2:Nt-1, Nt÷5)), Nt)

    for i in perm
        S = reshape(dp[1][:,i], (Nstate,1))
        for j = 2:Nparam
            S = hcat(S, reshape(dp[j][:,i], (Nstate,1)))
        end
        F += S[cols,:]' * inv(cov_ϵ) * S[cols,:]
    end


    # correlation matrix
    C = inv(F)
    R = ones(size(C))
    R = [C[i,j]/sqrt(C[i,i]*C[j,j]) for i = 1:size(C)[1], j = 1:size(C)[1]]
    # heatmap(R, xlims=(0.5,size(R)[1]+0.5), aspect_ratio = 1, color = :inferno, clims = (-1, 1),
    #         xticks = (1:1:size(C)[1], lab), xtickfont = font(14, "Times"),
    #         yticks = (1:1:size(C)[1], lab), ytickfont = font(14, "Times"), fmt = :png, dpi=300)
    # savefig("correlation_matrix")

    abs.(R) .> 0.99

    lowerbound = sqrt.(diag(inv(F)))
    err = [abs(lowerbound[o]/p_real[trainable][o]) for o in 1:length(p)]
    # err = [lowerbound[o]/p[o] for o in 1:length(p)]

    return lowerbound, lab, err
end


# lowerbound, lab, err = practical_identifiability_fhn()

# for i = 1:length(lab)
#     println( lab[i], '\t', lowerbound[i], '\t', err[i] )
# end


