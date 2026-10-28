#Data Generation
using Random, Distributions, LinearAlgebra, Statistics
using DataFrames
using StatsBase
using DelimitedFiles
using FastGaussQuadrature
using Optim
using NLSolversBase: only_fg!
using PrettyTables
using MAT

######################################
# Prepare functions
######################################
## Nonlinear specification
struct NLSpec_IO_HW
    K2::Int          # # product chars
    D::Int           # # demographics
    Knu::Int         # # ν dimensions (usually = K2 if Σ is K2×K2)
        
    maskΣ_lt::BitVector     # length nΣ = K2*(K2+1)÷2, lower-tri part mask (true = estimated)
    maskΠ::BitMatrix        # size K2×D, true = estimated

    fixedΣ::Matrix{Float64} # K2×K2 (lower-tri entries used when maskΣ_lt=false; upper tri ignored)
    fixedΠ::Matrix{Float64} # K2×D   (used when maskΠ=false)
end


# Translation between θ vector and (Σ, Π) matrices
tril_inds(K2) = findall(tril(trues(K2,K2)))  # vector of CartesianIndex in column-major LT order


function build_maps_IO_HW(sp::NLSpec_IO_HW)
    K2, D = sp.K2, sp.D

    param_char_idx  = Int[]
    param_src_kind  = Symbol[]
    param_src_idx   = Int[]
    param_scale     = Float64[]

    # --- Σ block (lower-tri), same order as pack/unpack ---
    LT = tril_inds(K2)
    @inbounds for (idx, isfree) in enumerate(sp.maskΣ_lt)
        if isfree
            ij = LT[idx]
            row = ij[1]
            col = ij[2]

            push!(param_char_idx, row)
            push!(param_src_kind, :nu)
            push!(param_src_idx, col)
            #push!(param_scale, sp.scaleΣ_lt[idx])
        end
    end

    # --- Π block ---
    for r in 1:D, k in 1:K2
        if sp.maskΠ[k,r]
            push!(param_char_idx, k)
            push!(param_src_kind, :d)
            push!(param_src_idx, r)
            #push!(param_scale, sp.scaleΠ[k,r])
        end
    end

    return param_char_idx, param_src_kind, param_src_idx
end


function pack_IO_HW(Σ::Matrix, Π::Matrix, sp::NLSpec_IO_HW)
    LT = tril_inds(sp.K2)
    θΣ_full = Σ[LT]                         # length nΣ
    θΠ_full = vec(Π)                        # length K2*D

    θ = Float64[]
    append!(θ, θΣ_full[sp.maskΣ_lt])
    append!(θ, θΠ_full[vec(sp.maskΠ)])
    return θ
end

function unpack_full_from_IO_HW(theta::AbstractVector, sp::NLSpec_IO_HW)
    K2, D = sp.K2, sp.D
    Σ = copy(sp.fixedΣ)        # start from fixed values
    Π = copy(sp.fixedΠ)

    # Fill in estimated Σ (lower-tri) entries
    LT = tril_inds(K2)
    θΣ_est = theta[1:count(sp.maskΣ_lt)]
    p = 1
    @inbounds for (idx, isfree) in enumerate(sp.maskΣ_lt)
        if isfree
            Σ[LT[idx]] = θΣ_est[p]     # apply chain-rule scale if needed
            p += 1
        end
    end

    # Fill in estimated Π entries
    θΠ_est = theta[p:end]
    q = 1
    @inbounds for r in 1:D, k in 1:K2
        if sp.maskΠ[k,r]
            Π[k,r] = θΠ_est[q]
            q += 1
        end
    end
    return Σ, Π
end


#seed = 1
function transform_data_IO_HW(alphas, eta, P_opt, Z, shares, w, x1, xi_all;
        include_price_moment::Bool)

    M = size(shares, 2)
    Ng = M
    J = repeat([size(shares, 1)], M)

    N = sum(J)  # Total number of products
    
    k = 1
    market_id_data = zeros(Int, N, 4)
    for g in 1:Ng
        for j in 1:J[g]
            market_id_data[k,1] = g  # market id
            market_id_data[k,2] = 1              # time period
            market_id_data[k,3] = g              # geographic market
            market_id_data[k,4] = j              # product id
            k += 1
        end
    end

    market_id = market_id_data[:,1]  # market id for each observation


    D = 0 # number of demographics variables
    
    
    
    X_mean_end = reshape(-P_opt, N)  # price level in mean utility
    X_mean_exo = copy(x1)  # product characteristics in mean utility 
    X_mean = hcat(X_mean_exo, X_mean_end)  # product characteristics in mean utility
    X_micro = reshape(-P_opt, N)  # micro BLP typically uses same X for micro moments, but could be different

    # #Hausman_IV: average of j-prices in other markets
    # Hausman_IV = (sum(P_opt, dims=2) .- P_opt)/(M-1)
    #BLP IV: mean(X_exo_{-j})
    X2_reshape = reshape(X_mean_exo[:,2], J[1], M)
    X3_reshape = reshape(X_mean_exo[:,3], J[1], M)
    BLP_IV2 = (sum(X2_reshape, dims=1) .- X2_reshape)/(J[1]-1)
    BLP_IV3 = (sum(X3_reshape, dims=1) .- X3_reshape)/(J[1]-1)
    # #Diﬀerentiation IV: sum((X_exo_{-j} - X_exo_j)^2)
    # diff_IV2 = zeros(J[1], M)
    # diff_IV3 = zeros(J[1], M)
    # for j in 1:J[1]
    #     diff_IV2[j,:] = sum((X2_reshape[setdiff(1:J[1], j), :] .- X2_reshape[j:j,:]).^2, dims=1)
    #     diff_IV3[j,:] = sum((X3_reshape[setdiff(1:J[1], j), :] .- X3_reshape[j:j,:]).^2, dims=1)
    # end

    #Z_Xexo = hcat(X_mean_exo, reshape(Hausman_IV, N), reshape(BLP_IV2, N), reshape(BLP_IV3, N), reshape(diff_IV2, N), reshape(diff_IV3, N))
    Z_Xexo = hcat(X_mean_exo, reshape(BLP_IV2, N), reshape(BLP_IV3, N)) 

    # Explicit opt-in for Q8(f): add the invalid moment E[p * xi] = 0.
    if include_price_moment
        Z_Xexo = hcat(Z_Xexo, vec(P_opt))
    end

    K2 = size(X_micro, 2)  # number of product characteristics with random coeffs
    
    
    
    # Monte Carlo Integration!
    # unobserved taste shocks
    S = repeat([size(alphas, 2)], M)
    weight = vcat([fill(1.0 / S[g], S[g]) for g in 1:M]...)
    ν = exp.(randn(sum(S), K2))  # unobserved draws for random coeffs (S1 consumers, S2 consumers, ..., each with K2 draws)
    # individual dataset
    indivdata_est = hcat(1:sum(S), zeros(sum(S)), weight, ν)  # id, geographic market, weight, demographics, unobsDraws
    k = 1
    for g in 1:Ng
        indivdata_est[k:k+S[g]-1, 2] .= g
        k += S[g]
    end
    

    # Gauss-Hermite quadrature draws and weights (for normal distribution)
    Q = 10 # number of quadrature points
    function gh_normal_rule(Q::Int)
        x, w = gausshermite(Q)           # ∫ e^{-x^2} f(x) dx
        z = sqrt(2) .* x                 # map to N(0,1)
        ω = w ./ sqrt(pi)                # normalize to probability weights
        return z, ω
    end
    function gh_lognormal_rule(Q::Int)
        z, ω = gh_normal_rule(Q)   # z are N(0,1) nodes, ω are weights
        ν = exp.(z)
        return ν, ω
    end
    ν_gh, w_gh = gh_lognormal_rule(Q)  # ν_gh is Q×K2 matrix of lognormal nodes, w_gh is vector of length Q of weights
    S_gh = Q*M
    indivdata_est_gh = hcat(1:S_gh, zeros(S_gh), repeat(w_gh, M), repeat(ν_gh, M))  # id, geographic market, weight, demographics, unobsDraws
    for g in 1:Ng
        indivdata_est_gh[(g-1)*Q+1:g*Q, 2] .= g
    end

    true_beta = [5.0, 1.0, 1.0, 1.0]  # true mean utility coefficients (price, x1)
    true_Sigma = reshape([1.0], 1, 1)
    true_Pi = reshape([], K2, D)
    sp = NLSpec_IO_HW(
        K2, D, K2,
        trues(K2*(K2+1) ÷ 2),
        trues(K2, D),
        zeros(K2, K2),
        zeros(K2, D)
    )
    true_nonlinear_param = pack_IO_HW(true_Sigma, true_Pi, sp)
    (Σ, Π) = unpack_full_from_IO_HW(true_nonlinear_param, sp)
    K3 = length(true_nonlinear_param)

    



    # ---------------------------------------------------------------
    # Package data
    # ---------------------------------------------------------------
    data = (
        share = reshape(shares, N),  # observed market share
        X_mean = X_mean,
        X_micro = X_micro,
        Z = Z_Xexo
    )

    indiv_datasets = (
        indivdata_est = indivdata_est,
        indivdata_est_gh = indivdata_est_gh
    )
    


    market_info = (
        market_id_data = market_id_data,
        J = J,
        Ng = Ng,
        S = S,
        S_gh = S_gh,
        D = D,
        N = N,
        M = M,
        K2 = K2
    )


    return (
        data = data,
        W_agg = inv(data.Z' * data.Z / N),
        market_info = market_info,
        indiv_datasets = indiv_datasets,
        true_Sigma = true_Sigma,
        true_Pi = true_Pi,
        true_beta = true_beta,
        true_nonlinear_param = true_nonlinear_param,
        sp = sp
    )
end



#
function obj_grad_IO_HW!(F, G, theta::AbstractVector,
                   sp::NLSpec_IO_HW,
                   data::NamedTuple,
                   W_agg::AbstractMatrix,
                   market_info::NamedTuple,
                   indivdata_est::AbstractMatrix;
                   get_linear_param_se = false
                   )


    K3 = length(theta)
    Ng = market_info.Ng
    M  = market_info.M
    N  = market_info.N
    K2 = market_info.K2
    Kx = K2 - 1
    D  = sp.D

    

    # maps: build once
    param_char_idx, param_src_kind, param_src_idx = build_maps_IO_HW(sp)

    Σ, Π = unpack_full_from_IO_HW(theta, sp)

    # -------------------------
    # precompute row indices (avoid boolean masks)
    # -------------------------
    indiv_rows_by_g = Vector{Vector{Int}}(undef, Ng)
    for g in 1:Ng
        indiv_rows_by_g[g] = findall(==(g), @view indivdata_est[:,2])
    end

    prod_rows_by_g = Vector{Vector{Int}}(undef, Ng)
    for g in 1:Ng
        prod_rows_by_g[g] = findall(==(g), @view market_info.market_id_data[:,3])
    end

    mu_by_geo = Vector{Matrix{Float64}}(undef, Ng)
    w_by_geo  = Vector{Vector{Float64}}(undef, Ng)
    for g in 1:Ng
        rows = indiv_rows_by_g[g]
        d_g = @view indivdata_est[rows, 4:(3+D)]
        ν_g = @view indivdata_est[rows, (4+D):end]
        w_gv = @view indivdata_est[rows, 3]   # view

        Sm = size(ν_g, 1)

        # materialize weights once (Vector{Float64} expected elsewhere)
        w_g = Vector{Float64}(w_gv)
        w_by_geo[g] = w_g

        # X rows for this geo across all t
        prows = prod_rows_by_g[g]
        X_g = @view data.X_micro[prows, :]

        # A_g = Σ*ν' + Π*d'
        A_g = Σ * ν_g' + Π * d_g'


        mu_by_geo[g] = X_g * A_g
    end

    delta_list = Vector{Vector{Float64}}(undef, M)
    gradient_results = G !== nothing ? Vector{NamedTuple}(undef, M) : nothing

    # ----------------------------------------
    # Market loop
    # ----------------------------------------
    #Threads.@threads 
    for g in 1:Ng
        @inbounds begin


            # ---- Extract individual data for this geo market ONCE per g ----
            idx_g = (indivdata_est[:,2] .== g)
            d_m  = indivdata_est[idx_g, 4:(3+D)]        # Sm × D   (col3 is w)
            nu_m = indivdata_est[idx_g, (4+D):end]      # Sm × K2
            w_m  = w_by_geo[g]                                # Sm, normalized (recommended)



            mask_m = (market_info.market_id_data[:,2] .== 1) .& (market_info.market_id_data[:,3] .== g)
            m = market_info.market_id_data[mask_m,1][1]   # get market index
            Jm = market_info.J[m]
            
            # Extract market-specific data
            X_m = data.X_micro[mask_m, :]
            share_m = data.share[mask_m]

            # ----------------------------------------
            # 2A: Delta contraction for this market
            # ----------------------------------------
            # Extract mu for this (t,g) from pre-computed mu_by_geo[g]
            mask_gt = (market_info.market_id_data[:,2] .== 1)[(market_info.market_id_data[:,3] .== g)]
            mu_m = mu_by_geo[g][mask_gt, :]
            
            # Compute delta_m and indivP_m via contraction mapping
            delta_m, indivP_m = getDelta_IndivP_m_IO_HW(share_m, mu_m, w_m, Jm;
                                                            t=1, g=g, damp = 1.0, maxit = 100000)


            # Store delta for this market
            delta_list[m] = delta_m

            res_m = gradient_m_IO_HW(param_char_idx, param_src_kind, param_src_idx, 
                            theta,
                            X_m, d_m, nu_m, w_m,
                            delta_m, indivP_m)
            

            # Store gradient results
            gradient_results[m] = res_m
        
        end

    end


    # ============================================================================
    # PHASE 3: GLOBAL AGGREGATION
    # ============================================================================
    
    # 3A: Aggregate moments (with demeaning and concentration)
    # Assemble full delta vector
    delta_vec = vcat(delta_list...)

    
    # Concentration: solve for linear parameters
    linear_param = (data.X_mean' * data.Z * W_agg * data.Z' * data.X_mean) \
                    (data.X_mean' * data.Z * W_agg * data.Z' * delta_vec)
    
    # Structural errors
    xi_hat = delta_vec .- data.X_mean * linear_param
    
    
    g_agg = data.Z' * xi_hat
    obj = (g_agg' * W_agg * g_agg)[]/N
    
    # ============================================================================
    # PHASE 4: GRADIENT ASSEMBLY (fill G_mat only)
    # ========================================================================
    # Stack deltam_theta from all markets
    ∂δ_∂θ = zeros(N, K3)
    for m in 1:M
        g = (m-1) % Ng + 1
        mask = (market_info.market_id_data[:,2] .== 1) .&
            (market_info.market_id_data[:,3] .== g)
        ∂δ_∂θ[mask, :] .= gradient_results[m].deltam_theta
    end

    # Aggregate gradient (kept separate by design)
    ∂g_agg_∂θ = data.Z' * ∂δ_∂θ
    G[:] = (2.0 * g_agg' * W_agg  * ∂g_agg_∂θ)' ./ N   # K3-vector
    
    if get_linear_param_se
        #SE
        g_agg_i = data.Z .* xi_hat
        Sigma_0 = (g_agg_i' * g_agg_i) / N          # Ω̂ = (1/N)∑ g_i g_i'

        Gbar_nl = ∂g_agg_∂θ / N                        
        Gbar_lin = -data.Z' * data.X_mean / N
        Gbar = hcat(Gbar_lin, Gbar_nl)                 # K3

        A = (Gbar' * W_agg * Gbar) \ I           
        Avar_sqrtN = A * (Gbar' * W_agg * Sigma_0 * W_agg * Gbar) * A

        #√n (̂θ - θ) ->_d N(0, Avar) => se = sqrt.(diag(Avar./N))
        se = sqrt.(diag(Avar_sqrtN./N))
        return obj, xi_hat, linear_param, se
    else
        return obj
    end
end



function getDelta_IndivP_m_IO_HW(
    share_m::AbstractVector{<:Real},        # Jm (inside shares, unconditional)
    mu_m::AbstractMatrix{<:Real},           # Jm × Sm
    w_m::AbstractVector{<:Real},            # Sm
    Jm::Int;
    t::Int=0, g::Int=0,
    maxit::Int=100000,
    tol::Float64=1e-8,
    damp::Float64=0.3,                      # update rate in (0,1]
    clamp_lo::Float64=-700.0,
    clamp_hi::Float64=700.0,
    tol_safe::Float64=1e-12)

    Sm = size(mu_m, 2)
    @assert size(mu_m, 1) == Jm "mu_m must be Jm×Sm (got $(size(mu_m))) (t=$t,g=$g)"
    @assert length(share_m) == Jm "length(share_m) must be Jm (t=$t,g=$g)"
    @assert length(w_m) == Sm "length(w_m) must equal Sm (t=$t,g=$g)"
    @assert 0 < damp <= 1 "damp must be in (0,1]"

    # ---- data shares used in contraction ----
    share_data = copy(share_m)

    # ---- buffers (allocated once per call) ----
    delta_m     = zeros(Float64, Jm)
    delta_new   = similar(delta_m)
    share_model = zeros(Float64, Jm)

    EXP_m = Matrix{Float64}(undef, Jm, Sm)   # store exp(V) for final P
    denom = Vector{Float64}(undef, Sm)       # denom per i


    conv = false
    it = 0
    while !conv && it < maxit
        it += 1
        fill!(share_model, 0.0)

        # ---- compute EXP_m and share_model ----
        @inbounds for i in 1:Sm
            # max utility for individual i
            mxi = -Inf
            for j in 1:Jm
                u = delta_m[j] + Float64(mu_m[j, i])
                if !isfinite(u)
                    error("Non-finite utility (t=$t,g=$g,it=$it) at (j=$j,i=$i): $u")
                end
                if u > mxi
                    mxi = u
                end
            end

            # exp(V) and sum
            s = 0.0
            for j in 1:Jm
                v = (delta_m[j] + Float64(mu_m[j, i])) - mxi
                v = ifelse(v < clamp_lo, clamp_lo, ifelse(v > clamp_hi, clamp_hi, v))
                e = exp(v)
                EXP_m[j, i] = e
                s += e
            end

            # denom: with/without outside
            di = exp(-mxi) + s   
            denom[i] = di

            wi_over_di = w_m[i] / di
            for j in 1:Jm
                share_model[j] += EXP_m[j, i] * wi_over_di
            end
        end

        # ---- Berry update ----
        @inbounds for j in 1:Jm
            sj  = share_data[j]
            smj = share_model[j]
            if !(sj > 0.0 && smj > 0.0 && isfinite(sj) && isfinite(smj))
                error("Invalid shares in contraction (t=$t,g=$g,it=$it) at j=$j: share_data=$sj, share_model=$smj")
            end
            delta_new[j] = delta_m[j] + log(sj) - log(smj)
        end

        # damping
        if damp != 1.0
            @inbounds for j in 1:Jm
                delta_new[j] = (1 - damp) * delta_m[j] + damp * delta_new[j]
            end
        end

        # convergence check
        max_change = 0.0
        @inbounds for j in 1:Jm
            dj = abs(delta_new[j] - delta_m[j])
            if dj > max_change
                max_change = dj
            end
        end
        conv = (max_change < tol)

        copyto!(delta_m, delta_new)
    end

    # println("Delta contraction converged in $it iterations (t=$t, g=$g)")
    # #print max_share_error minimum(delta_m) maximum(delta_m)
    # println("Min delta: $(minimum(delta_m))")
    # println("Max delta: $(maximum(delta_m))")

    # build indivP_m (Sm×Jm), inside probabilities (unconditional)
    indivP_m = Matrix{Float64}(undef, Sm, Jm)
    @inbounds for i in 1:Sm
        invdi = 1.0 / denom[i]
        for j in 1:Jm
            p = EXP_m[j, i] * invdi    # if inside_contraction: p(j|inside), else: p(j)
            p = ifelse(p < tol_safe, tol_safe, ifelse(p > 1.0 - tol_safe, 1.0 - tol_safe, p))
            indivP_m[i, j] = p
        end
    end


    if !conv
        max_share_error = 0.0
        @inbounds for j in 1:Jm
            err = abs(share_model[j] - share_data[j])
            if err > max_share_error
                max_share_error = err
            end
        end
        max_share = maximum(share_model)
        min_share = minimum(share_model)
        @warn "Delta contraction did not converge after $it iterations (t=$t, g=$g)" max_share min_share max_share_error maximum(share_data) minimum(delta_m) maximum(delta_m)
    end

    return delta_m, indivP_m
end


function gradient_m_IO_HW(param_char_idx::Vector{Int64}, 
                    param_src_kind::Vector{Symbol}, 
                    param_src_idx::Vector{Int64},
                    theta2::AbstractVector,
                    X_m::AbstractMatrix,
                    d_m::AbstractMatrix,
                    nu_m::AbstractMatrix,
                    w_m::AbstractVector,
                    delta_m::AbstractVector,
                    indivP_m::AbstractMatrix)


    Sm, Jm = size(indivP_m)
    K3 = length(theta2)

    # mean_s, cross_s はあとで
    xbar = indivP_m * X_m

    dSbar_dθ_partial = zeros(eltype(indivP_m), Jm, K3)

    a  = similar(w_m)                 # Sm
    ax = similar(w_m)                 # Sm
    t1 = zeros(eltype(w_m), Jm)       # J
    t2 = zeros(eltype(w_m), Jm)       # J


    # ---- dSbar_dθ_partial ----
    @inbounds for c in 1:K3
        kc = param_char_idx[c]
        vc = (param_src_kind[c] === :nu) ? @view(nu_m[:, param_src_idx[c]]) :
                                           @view(d_m[:,  param_src_idx[c]])

        xbarc = @view xbar[:, kc]     # Sm
        xcol  = @view X_m[:, kc]      # J

        @. a  = w_m * vc

        mul!(t1, transpose(indivP_m), a)      # t1 = P' * a

        @. ax = a * xbarc
        mul!(t2, transpose(indivP_m), ax)     # t2 = P' * (a .* xbarc)

        @. @view(dSbar_dθ_partial[:, c]) = xcol * t1 - t2
    end

    # ---- weighted ∂S̄/∂δ = Σ_i w_i [diag(s_i) - s_i s_i'] ----
    mean_s  = vec(indivP_m' * w_m)                  # (Jm,)
    cross_s = indivP_m' * (indivP_m .* w_m)         # (Jm×Jm)
    dSbar_dδ = Diagonal(mean_s) .- cross_s          # (Jm×Jm)

    F = lu(dSbar_dδ)
    deltam_theta = - (F \ dSbar_dθ_partial)   # (Jm×K3)

    return (deltam_theta = deltam_theta,
    other = nothing)
end

# Question 8(d). Adapted to X_mean = [1, x2, x3, -price],
# X_micro = -price, and alpha_i = alpha_bar + sigma_alpha * nu_i.
# Elasticity rows index demand j; columns index price k.
function compute_elasticities_IO_HW(theta::AbstractVector, sp::NLSpec_IO_HW,
        data::NamedTuple, market_info::NamedTuple,
        indivdata_est::AbstractMatrix, linear_param::AbstractVector;
        delta=nothing)
    @assert sp.K2 == 1 && sp.D == 0 "Q8 assumes one random price coefficient and no demographics"
    Σ, _ = unpack_full_from_IO_HW(theta, sp)
    sigma = Σ[1,1]
    alpha_bar = linear_param[end]
    delta_vec = delta === nothing ? zeros(market_info.N) : copy(vec(delta))
    @assert length(delta_vec) == market_info.N
    elasticity_list = Matrix{Float64}[]
    derivative_list = Matrix{Float64}[]
    share_vec = zeros(market_info.N)
    own_elasticity = zeros(market_info.N)
    consumer_surplus = zeros(market_info.M)
    for m in 1:market_info.M
        rows = findall(==(m), market_info.market_id_data[:,1])
        g = market_info.market_id_data[first(rows),3]
        consumers = findall(==(g), indivdata_est[:,2])
        weights = indivdata_est[consumers,3]
        @assert isapprox(sum(weights), 1.0; atol=1e-12)
        nu = indivdata_est[consumers,4]
        alpha_i = alpha_bar .+ sigma .* nu
        # Monetary logsum surplus requires positive marginal utility of income.
        all(alpha_i .> 0) || error("Nonpositive alpha_i in market $m: monetary consumer surplus is undefined")
        prices = -vec(data.X_micro[rows,:])
        mu = -prices * (sigma .* nu)'
        if delta === nothing
            delta_m, _ = getDelta_IndivP_m_IO_HW(data.share[rows], mu,
                weights, length(rows); t=1, g=g, damp=1.0, maxit=100000)
            delta_vec[rows] = delta_m
        end
        utilities = delta_vec[rows] .+ mu
        shift = max.(0.0, maximum(utilities, dims=1))
        numer = exp.(utilities .- shift)
        denominator = exp.(-shift) .+ sum(numer, dims=1)
        probabilities = numer ./ denominator  # products × consumers
        shares_m = probabilities * weights
        # ds_j/dp_k = E[alpha_i P_ij P_ik] - 1{j=k} E[alpha_i P_ij].
        weighted_alpha = weights .* alpha_i
        jacobian = (probabilities .* weighted_alpha') * probabilities'
        jacobian[diagind(jacobian)] .-= probabilities * weighted_alpha
        elasticity = jacobian .* (prices' ./ shares_m)
        push!(derivative_list, jacobian)
        push!(elasticity_list, elasticity)
        share_vec[rows] = shares_m
        own_elasticity[rows] = diag(elasticity)
        # Expected surplus relative to an outside-only choice set; Euler constant cancels.
        logsums = vec(shift .+ log.(denominator))
        consumer_surplus[m] = sum(weights .* logsums ./ alpha_i)
    end
    return (; elasticity_list, derivative_list, delta=delta_vec,
        shares=share_vec, own_elasticity, consumer_surplus,
        max_observed_share_error=maximum(abs.(share_vec .- data.share)))
end

# True pricing model: single-product Nash-Bertrand firms in each market.
# Use derivatives directly, avoiding division by potentially zero prices.
function marginal_cost_estimation_IO_HW(demand::NamedTuple,
        market_info::NamedTuple, data::NamedTuple)
    prices = -vec(data.X_micro)
    markup = zeros(market_info.N)
    for m in 1:market_info.M
        rows = findall(==(m), market_info.market_id_data[:,1])
        own_derivative = diag(demand.derivative_list[m])
        all(own_derivative .< 0) || error("Nonnegative own-price derivative in market $m")
        markup[rows] = -demand.shares[rows] ./ own_derivative
    end
    marginal_cost = prices .- markup
    # Per potential consumer; multiply by market size for total variable profits.
    profit = markup .* demand.shares
    return (; marginal_cost, markup, profit)
end

function run_blp_bfgs_IO_HW(theta0, dgp_data;
                      show_trace=true,
                      iterations=1000,
                      g_tol=1e-6,
                      f_reltol=1e-8,
                      x_reltol=1e-8,
                      store_trace=true,
                      extended_trace=false,
                      optim_algorithm=LBFGS(),
                      use_log_diag=true,
                      use_gh=true)

    sp, data, W_agg, market_info = dgp_data.sp, dgp_data.data, dgp_data.W_agg, dgp_data.market_info
    if use_gh
        # Use Gauss-Hermite quadrature for moment integration
         println("Using Gauss-Hermite quadrature for integration")
        indivdata_est = copy(dgp_data.indiv_datasets.indivdata_est_gh)
    else
        # Use Monte Carlo integration with random draws
         println("Using Monte Carlo integration with random draws for integration")
        indivdata_est = copy(dgp_data.indiv_datasets.indivdata_est)
    end


    # Identify which parameters are diagonal Σ elements
    diagonal_indices = Int[]
    if use_log_diag
        println("Using LOG parameterization for diagonal Σ elements (ensures positivity)")
        LT = tril_inds(sp.K2)
        p = 1
        for (idx, isfree) in enumerate(sp.maskΣ_lt)
            if isfree
                cart_idx = LT[idx]
                row, col = cart_idx[1], cart_idx[2]
                if row == col  # This is a diagonal element
                    push!(diagonal_indices, p)
                end
                p += 1
            end
        end
        println("  Diagonal Σ parameter indices: ", diagonal_indices)
    else
        println("Using standard parameterization (no log transform)")
    end

    theta0_scaled = copy(theta0)
    theta0_scaled[diagonal_indices] .= max.(min.(log.(theta0[diagonal_indices]), 1e+10), -1e+10)



    function fg!(F, G, theta)
        #failed = Threads.Atomic{Bool}(false)
        # Transform theta: for diagonal elements, use exp(theta) instead of theta
        if use_log_diag
            theta_scale_back = copy(theta)
            theta_scale_back[diagonal_indices] .= max.(min.(exp.(theta[diagonal_indices]), 1e+10), 1e-10)
        else
            theta_scale_back = copy(theta)
        end
        

        obj_val = obj_grad_IO_HW!(F, G, theta_scale_back, sp, data, W_agg, market_info,indivdata_est)

        # Apply chain rule to gradient: ∂f/∂(log σ) = ∂f/∂σ × ∂σ/∂(log σ) = ∂f/∂σ × exp(log σ) = ∂f/∂σ × σ
        if G !== nothing && use_log_diag
            for idx in diagonal_indices
                G[idx] *= theta_scale_back[idx]  # Multiply by σ = exp(log σ)
            end
        end


        return obj_val
    end

    # Run optimization
    println("\nStarting L-BFGS optimization...\n")

    result = optimize(
            only_fg!(fg!),
            theta0_scaled,
            optim_algorithm,
            Optim.Options(
                show_trace = show_trace,
                iterations = iterations,
                g_tol = g_tol,
                f_reltol = f_reltol,
                x_reltol = x_reltol,
                store_trace = store_trace,
                extended_trace = extended_trace
                #f_calls_limit = 200_000,
                #callback=cb
                #allow_f_increases = false # if obj = Inf, reject that step
            )
        )
    
    println("\n" * "="^80)
    println("Optimization Complete")
    println("="^80)
    println("Converged: ", Optim.converged(result))
    println("Minimum objective: ", Optim.minimum(result))
    println("Iterations: ", Optim.iterations(result))
    println("="^80)
    
    return result
end

######################################
# Data
######################################
## DATA TRANSFORM
script_dir = @__DIR__
cd(script_dir)

data_T100_J3 = matread("Simulation Data/100markets3products.mat")

dgp_data = transform_data_IO_HW(
    data_T100_J3["alphas"], data_T100_J3["eta"], data_T100_J3["P_opt"],
    data_T100_J3["Z"], data_T100_J3["shares"], data_T100_J3["w"],
    data_T100_J3["x1"], data_T100_J3["xi_all"]; include_price_moment=false)
theta = dgp_data.true_nonlinear_param
G = zeros(length(theta))  # placeholder for gradient
individual_est = dgp_data.indiv_datasets.indivdata_est
individual_est_gh = dgp_data.indiv_datasets.indivdata_est_gh
#@time obj, xi_hat, linear_param,se = obj_grad_IO_HW!(nothing, G, theta, dgp_data.sp, dgp_data.data, dgp_data.W_agg, dgp_data.market_info, individual_est; get_linear_param_se=true)
@time obj, xi_hat, linear_param,se = obj_grad_IO_HW!(nothing, G, theta, dgp_data.sp, dgp_data.data, dgp_data.W_agg, dgp_data.market_info, individual_est_gh; get_linear_param_se=true)


######################################
# Estimation
######################################
init_theta = [0.1] 
# # Monte Carlo
# est_time_mc = @elapsed est_result_monte_carlo = run_blp_bfgs_IO_HW(init_theta, dgp_data; use_gh=false, use_log_diag=true)
# theta_hat_log_diag = Optim.minimizer(est_result_monte_carlo)
# Σ_hat_log, Π_hat_mc = unpack_full_from_IO_HW(theta_hat_log_diag, sp)
# Σ_hat_mc = copy(Σ_hat_log)
# for i in 1:K2
#     Σ_hat_mc[i,i] = exp(Σ_hat_log[i,i])
# end
# theta_hat_mc = pack_IO_HW(Σ_hat_mc, Π_hat, sp)
# @time obj_mc, _, linear_param_mc,se_mc = obj_grad_IO_HW!(nothing, G, theta_hat_mc, dgp_data.sp, dgp_data.data, dgp_data.W_agg, dgp_data.market_info, individual_est; get_linear_param_se=true)


# Gauss-Hermite
est_time_gh = @elapsed est_result_gh = run_blp_bfgs_IO_HW(init_theta, dgp_data; use_gh=true, use_log_diag=true)
theta_hat_log_diag = Optim.minimizer(est_result_gh)
Σ_hat_log, Π_hat_gh = unpack_full_from_IO_HW(theta_hat_log_diag, dgp_data.sp)
Σ_hat_gh = copy(Σ_hat_log)
for i in 1:dgp_data.market_info.K2
    Σ_hat_gh[i,i] = exp(Σ_hat_log[i,i])
end
theta_hat_gh = pack_IO_HW(Σ_hat_gh, Π_hat_gh, dgp_data.sp)
@time obj_gh, _, linear_param_gh,se_gh = obj_grad_IO_HW!(nothing, G, theta_hat_gh, dgp_data.sp, dgp_data.data, dgp_data.W_agg, dgp_data.market_info, individual_est_gh; get_linear_param_se=true)


# Question 8(c): GH estimates and estimation errors (estimate minus truth)
param_names = ["beta_0", "beta_2", "beta_3", "alpha_bar", "sigma_alpha"]
true_param = vcat(dgp_data.true_beta, dgp_data.true_nonlinear_param)
param_gh = vcat(linear_param_gh, theta_hat_gh)
df_param = DataFrame(parameter=param_names, true_value=true_param,
    estimate=param_gh, bias=param_gh .- true_param, standard_error=se_gh)

# The implemented criterion is N*gbar'W*gbar, with gbar = Z'xi/N.
df_fit = DataFrame(GMM_N_gbar_W_gbar=obj_gh,
    GMM_gbar_W_gbar=obj_gh / dgp_data.market_info.N,
    converged=Optim.converged(est_result_gh),
    iterations=Optim.iterations(est_result_gh), seconds=est_time_gh)

using Printf
const Q8_TABLE_DIR = joinpath(@__DIR__, "output", "tables")
mkpath(Q8_TABLE_DIR)
latex_escape(s) = replace(string(s), '&'=>"\\&", '%'=>"\\%", '_'=>"\\_", '#'=>"\\#")
function latex_cell(x)
    ismissing(x) && return "--"
    x isa Bool && return x ? "Yes" : "No"
    if x isa AbstractFloat
        !isfinite(x) && return string(x)
        value = @sprintf("%.4g", x)
        if occursin('e', value)
            mantissa, exponent = split(value, 'e')
            return "\$" * mantissa * "\\times 10^{" * string(parse(Int, exponent)) * "}\$"
        end
        return value
    end
    parameter_labels = Dict("beta_0"=>raw"$\beta_0$", "beta_2"=>raw"$\beta_2$",
        "beta_3"=>raw"$\beta_3$", "alpha_bar"=>raw"$\bar\alpha$",
        "sigma_alpha"=>raw"$\sigma_\alpha$")
    return get(parameter_labels, string(x), latex_escape(x))
end
function save_q8_table(filename, table)
    path = joinpath(Q8_TABLE_DIR, filename * ".tex")
    open(path, "w") do io
        println(io, "% Generated by blp_IO_HW.jl; requires booktabs.")
        println(io, "\\begin{tabular}{", join([eltype(col) <: Number ? "r" : "l" for col in eachcol(table)]), "}")
        println(io, "\\toprule")
        println(io, join(latex_escape.(names(table)), " & "), " \\\\")
        println(io, "\\midrule")
        for row in eachrow(table)
            println(io, join(latex_cell.(collect(row)), " & "), " \\\\")
        end
        println(io, "\\bottomrule\n\\end{tabular}")
    end
    return path
end
function report_q8_table(title, table)
    println("\n", title)
    show(stdout, MIME"text/plain"(), table)
    filename = strip(replace(lowercase(title), r"[^a-z0-9]+"=>"_"), '_')
    println("\nLaTeX table saved: ", save_q8_table(filename, table))
end
report_q8_table("Parameter estimates (SEs on original scale)", df_param)
report_q8_table("GMM criterion and optimization summary", df_fit)

# Recompute shares at the FINAL delta, rather than using the contraction's
# probability buffer, which precedes its last update. Include the outside good
# in the exponential normalization. Errors are maxima over all markets/products.
function gh_contraction_errors(dgp, individuals, sigma, delta)
    share_error = 0.0
    fixed_point_error = 0.0
    for g in 1:dgp.market_info.Ng
        rows = findall(==(g), dgp.market_info.market_id_data[:,3])
        consumers = findall(==(g), individuals[:,2])
        weights = individuals[consumers,3]
        nu = individuals[consumers,4]  # Q8: no demographics, one random coefficient
        utility = delta[rows] .+ dgp.data.X_micro[rows,:] * (sigma .* nu')
        shift = max.(0.0, maximum(utility, dims=1))
        numer = exp.(utility .- shift)
        probabilities = numer ./ (exp.(-shift) .+ sum(numer, dims=1))
        predicted = vec(probabilities * weights)
        observed = dgp.data.share[rows]
        share_error = max(share_error, maximum(abs.(predicted .- observed)))
        fixed_point_error = max(fixed_point_error,
            maximum(abs.(log.(observed) .- log.(predicted))))
    end
    return (; share_error, fixed_point_error)
end

# Same dataset, GH nodes, weights, instruments, and stopping rules in every run.
# Positive starting values are required by the log parameterization.
using Logging
init_theta_vec = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
function gh_starting_value_comparison(dgp, individuals, starts)
    rows = NamedTuple[]
    parameter_tables = DataFrame[]
    for sigma0 in starts
        warning_buffer = IOBuffer()
        started = time()
        try
            result = with_logger(SimpleLogger(warning_buffer, Logging.Warn)) do
                run_blp_bfgs_IO_HW([sigma0], dgp; use_gh=true,
                    use_log_diag=true, show_trace=false)
            end
            sigma = clamp(exp(only(Optim.minimizer(result))), 1e-10, 1e10)
            gradient = zeros(1)
            objective, xi, linear, se = obj_grad_IO_HW!(nothing, gradient,
                [sigma], dgp.sp, dgp.data, dgp.W_agg, dgp.market_info,
                individuals; get_linear_param_se=true)
            errors = gh_contraction_errors(dgp, individuals, sigma,
                xi + dgp.data.X_mean * linear)
            warnings = String(take!(warning_buffer))
            # Retain optimizer warnings, including failed contractions at trial points.
            isempty(warnings) || println("Warnings for sigma0=", sigma0, "\n", warnings)
            push!(rows, (sigma_start=sigma0, sigma_hat=sigma,
                GMM_N_gbar_W_gbar=objective, converged=Optim.converged(result),
                iterations=Optim.iterations(result), seconds=time()-started,
                max_share_error=errors.share_error,
                max_fixed_point_error=errors.fixed_point_error,
                contraction_ok=errors.fixed_point_error < 1e-8,
                trial_contraction_warnings=count("Delta contraction did not converge", warnings),
                log_sigma_gradient=abs(sigma * gradient[1]), error=""))
            estimates = vcat(linear, sigma)
            push!(parameter_tables, DataFrame(sigma_start=fill(sigma0, length(estimates)),
                parameter=param_names, estimate=estimates,
                bias=estimates .- vcat(dgp.true_beta, dgp.true_nonlinear_param),
                standard_error=se))
        catch err
            err isa InterruptException && rethrow()
            warnings = String(take!(warning_buffer))
            push!(rows, (sigma_start=sigma0, sigma_hat=missing,
                GMM_N_gbar_W_gbar=missing, converged=false, iterations=missing,
                seconds=time()-started, max_share_error=missing,
                max_fixed_point_error=missing, contraction_ok=false,
                trial_contraction_warnings=count("Delta contraction did not converge", warnings),
                log_sigma_gradient=missing, error=sprint(showerror, err)))
        end
    end
    runs = DataFrame(rows)
    parameters = isempty(parameter_tables) ? DataFrame() : vcat(parameter_tables...)
    return runs, parameters
end

df_runs, df_runs_parameters = gh_starting_value_comparison(
    dgp_data, individual_est_gh, init_theta_vec)
report_q8_table("Starting-value sensitivity: final residuals and convergence", df_runs)
report_q8_table("Parameter estimates, bias, and SEs for each starting value", df_runs_parameters)

# Summarize sensitivity only across runs passing BOTH convergence checks.
valid_starts = df_runs.sigma_start[df_runs.converged .& df_runs.contraction_ok]
if !isempty(valid_starts) && !isempty(df_runs_parameters)
    valid_parameters = filter(row -> row.sigma_start in valid_starts, df_runs_parameters)
    df_sensitivity = combine(groupby(valid_parameters, :parameter),
        :estimate => minimum => :minimum_estimate,
        :estimate => maximum => :maximum_estimate,
        :estimate => (x -> maximum(x)-minimum(x)) => :range)
    report_q8_table("Sensitivity across converged runs (original parameter scales)", df_sensitivity)
else
    println("No runs passed both optimization and final contraction checks.")
end

# Question 8(d): evaluate the GH estimate reported above at observed prices.
# Estimated xi is absorbed in the share-inverting delta. True xi is used ONLY
# for the benchmark, never for demand estimation. Common GH nodes isolate
# parameter differences, though true shares can differ due to integration error.
demand_gh = compute_elasticities_IO_HW(theta_hat_gh, dgp_data.sp,
    dgp_data.data, dgp_data.market_info, individual_est_gh, linear_param_gh)
true_delta = dgp_data.data.X_mean * dgp_data.true_beta + vec(data_T100_J3["xi_all"])
demand_true = compute_elasticities_IO_HW(dgp_data.true_nonlinear_param,
    dgp_data.sp, dgp_data.data, dgp_data.market_info, individual_est_gh,
    dgp_data.true_beta; delta=true_delta)
costs_gh = marginal_cost_estimation_IO_HW(demand_gh, dgp_data.market_info, dgp_data.data)
costs_true = marginal_cost_estimation_IO_HW(demand_true, dgp_data.market_info, dgp_data.data)

# Full J×J matrices for each market remain available in these lists.
elasticity_gh = demand_gh.elasticity_list
elasticity_true = demand_true.elasticity_list
marginal_cost_gh = costs_gh.marginal_cost
marginal_cost_true = costs_true.marginal_cost

# Long-form table includes BOTH own- and cross-price elasticities.
elasticity_rows = NamedTuple[]
for m in 1:dgp_data.market_info.M
    for j in 1:dgp_data.market_info.J[m], k in 1:dgp_data.market_info.J[m]
        estimate = elasticity_gh[m][j,k]
        benchmark = elasticity_true[m][j,k]
        push!(elasticity_rows, (market=m, demand_product=j, price_product=k,
            own_price=(j==k), estimate=estimate, true_parameter_value=benchmark,
            difference=estimate-benchmark))
    end
end
df_elasticities = DataFrame(elasticity_rows)
df_products = DataFrame(market=dgp_data.market_info.market_id_data[:,1],
    product=dgp_data.market_info.market_id_data[:,4], price=-vec(dgp_data.data.X_micro))
for (label, estimated, benchmark) in [
        (:own_elasticity, demand_gh.own_elasticity, demand_true.own_elasticity),
        (:marginal_cost, costs_gh.marginal_cost, costs_true.marginal_cost),
        (:markup, costs_gh.markup, costs_true.markup),
        (:profit_per_consumer, costs_gh.profit, costs_true.profit)]
    df_products[!, Symbol(label, :_estimate)] = estimated
    df_products[!, Symbol(label, :_true)] = benchmark
    df_products[!, Symbol(label, :_difference)] = estimated .- benchmark
end
df_consumer_surplus = DataFrame(market=1:dgp_data.market_info.M,
    estimate=demand_gh.consumer_surplus,
    true_parameter_value=demand_true.consumer_surplus,
    difference=demand_gh.consumer_surplus .- demand_true.consumer_surplus)
df_share_check = DataFrame(evaluation=["estimate", "true parameters and true xi"],
    max_observed_share_error=[demand_gh.max_observed_share_error,
        demand_true.max_observed_share_error])
# Equal-weight summaries over market-products (cross elasticities: ordered
# product pairs; consumer surplus: market-level average surplus per consumer).
# The difference row summarizes matched estimate-minus-truth observations,
# rather than subtracting separately computed quantiles.
summary_rows = NamedTuple[]
cross_mask = .!df_elasticities.own_price
for (object, estimated, benchmark) in [
        ("Own-price elasticity", demand_gh.own_elasticity, demand_true.own_elasticity),
        ("Cross-price elasticity", df_elasticities.estimate[cross_mask],
            df_elasticities.true_parameter_value[cross_mask]),
        ("Marginal cost", costs_gh.marginal_cost, costs_true.marginal_cost),
        ("Markup", costs_gh.markup, costs_true.markup),
        ("Profit per consumer", costs_gh.profit, costs_true.profit),
        ("Consumer surplus per consumer", demand_gh.consumer_surplus, demand_true.consumer_surplus)]
    for (evaluation, values) in [("Estimate", estimated), ("True parameters", benchmark),
            ("Difference", estimated .- benchmark)]
        qs = quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
        push!(summary_rows, (object=object, evaluation=evaluation, N=length(values),
            mean=mean(values), q10=qs[1], q25=qs[2], median=qs[3], q75=qs[4], q90=qs[5]))
    end
end
df_objects_summary = DataFrame(summary_rows)
report_q8_table("Estimates versus true parameters: summary statistics", df_objects_summary)
report_q8_table("Share discrepancies: includes integration error for the true benchmark", df_share_check)

# Questions 8(e) and 8(f). Each fit bundles its own data and results to prevent
# mixing estimates/SEs from different datasets in an interactive session.
function select_q8_fit(dgp, runs, parameters)
    valid = findall(runs.converged .& runs.contraction_ok .& (runs.error .== ""))
    isempty(valid) && error("No starting value passed optimization and contraction checks")
    # Treat objective differences below 1e-8 as numerical ties. Prefer the
    # starting value closest to 1, consistently with the main (c)/(d) run.
    # Report all solutions: tied objectives do not imply equal parameters.
    minimum_objective = minimum(runs.GMM_N_gbar_W_gbar[valid])
    tied = filter(i -> runs.GMM_N_gbar_W_gbar[i] <= minimum_objective + 1e-8, valid)
    best = tied[argmin(abs.(log.(runs.sigma_start[tied])))]
    sigma0 = runs.sigma_start[best]
    table = filter(row -> row.sigma_start == sigma0, parameters)
    return (; dgp, runs, parameters, best_start=sigma0,
        linear=copy(table.estimate[1:4]), theta=[table.estimate[5]],
        se=copy(table.standard_error), objective=runs.GMM_N_gbar_W_gbar[best])
end

function q8_stability(label, fit)
    r = fit.runs
    ok = r.converged .& r.contraction_ok .& (r.error .== "")
    sigmas = Float64.(r.sigma_hat[ok])
    return (specification=label, starts=nrow(r), successful=sum(ok),
        best_start=fit.best_start, sigma_min=minimum(sigmas), sigma_max=maximum(sigmas),
        criterion_min=minimum(r.GMM_N_gbar_W_gbar[ok]),
        criterion_max=maximum(r.GMM_N_gbar_W_gbar[ok]),
        max_fixed_point_error=maximum(r.max_fixed_point_error[ok]),
        trial_contraction_warnings=sum(r.trial_contraction_warnings))
end

# (e): T=10, using the same estimation and reporting steps as (c).
data_T10_J3 = matread("Simulation Data/10markets3products.mat")
dgp_data_10 = transform_data_IO_HW(
    data_T10_J3["alphas"], data_T10_J3["eta"], data_T10_J3["P_opt"],
    data_T10_J3["Z"], data_T10_J3["shares"], data_T10_J3["w"],
    data_T10_J3["x1"], data_T10_J3["xi_all"]; include_price_moment=false)
individual_est_10 = dgp_data_10.indiv_datasets.indivdata_est_gh
init_theta_10 = [1.0]
est_time_10 = @elapsed est_result_10 = run_blp_bfgs_IO_HW(
    init_theta_10, dgp_data_10; use_gh=true, use_log_diag=true)
theta_hat_log_10 = Optim.minimizer(est_result_10)
Σ_hat_log_10, Π_hat_10 = unpack_full_from_IO_HW(theta_hat_log_10, dgp_data_10.sp)
Σ_hat_10 = copy(Σ_hat_log_10)
for i in 1:dgp_data_10.market_info.K2
    Σ_hat_10[i,i] = exp(Σ_hat_log_10[i,i])
end
theta_hat_10 = pack_IO_HW(Σ_hat_10, Π_hat_10, dgp_data_10.sp)
obj_10, xi_hat_10, linear_param_10, se_10 = obj_grad_IO_HW!(
    nothing, zeros(length(theta_hat_10)), theta_hat_10, dgp_data_10.sp,
    dgp_data_10.data, dgp_data_10.W_agg, dgp_data_10.market_info,
    individual_est_10; get_linear_param_se=true)
param_10 = vcat(linear_param_10, theta_hat_10)
true_param_10 = vcat(dgp_data_10.true_beta, dgp_data_10.true_nonlinear_param)
df_e_parameters = DataFrame(parameter=param_names, true_value=true_param_10,
    estimate=param_10, bias=param_10 .- true_param_10, standard_error=se_10)
df_e_fit = DataFrame(GMM_N_gbar_W_gbar=obj_10,
    GMM_gbar_W_gbar=obj_10/dgp_data_10.market_info.N,
    converged=Optim.converged(est_result_10),
    iterations=Optim.iterations(est_result_10), seconds=est_time_10)
report_q8_table("8(e): T=10 parameter estimates", df_e_parameters)
report_q8_table("8(e): GMM criterion and optimization summary", df_e_fit)

# (f): return to T=100, retaining baseline IVs and adding price itself.
fit_100 = select_q8_fit(dgp_data, df_runs, df_runs_parameters)
dgp_data_price = transform_data_IO_HW(
    data_T100_J3["alphas"], data_T100_J3["eta"], data_T100_J3["P_opt"],
    data_T100_J3["Z"], data_T100_J3["shares"], data_T100_J3["w"],
    data_T100_J3["x1"], data_T100_J3["xi_all"]; include_price_moment=true)
runs_price, params_price = gh_starting_value_comparison(dgp_data_price,
    dgp_data_price.indiv_datasets.indivdata_est_gh, init_theta_vec)
fit_price = select_q8_fit(dgp_data_price, runs_price, params_price)
df_f_parameters = vcat([
    DataFrame(specification=fill(label,5), parameter=param_names,
        true_value=vcat(fit.dgp.true_beta,fit.dgp.true_nonlinear_param),
        estimate=vcat(fit.linear,fit.theta),
        bias=vcat(fit.linear,fit.theta)-vcat(fit.dgp.true_beta,fit.dgp.true_nonlinear_param),
        standard_error=fit.se)
    for (label,fit) in [("Baseline IV",fit_100),("IV plus price moment",fit_price)]]...)
report_q8_table("8(f): parameter comparison with the truth", df_f_parameters)
report_q8_table("8(f): price-moment runs and contraction diagnostics", runs_price)
report_q8_table("8(f): optimization stability", DataFrame([
    q8_stability("Baseline IV",fit_100),q8_stability("IV plus price moment",fit_price)]))

# Common integration points and observed prices for both estimates and truth.
f_individuals = fit_100.dgp.indiv_datasets.indivdata_est_gh
f_baseline_demand = compute_elasticities_IO_HW(fit_100.theta,fit_100.dgp.sp,
    fit_100.dgp.data,fit_100.dgp.market_info,f_individuals,fit_100.linear)
f_price_demand = compute_elasticities_IO_HW(fit_price.theta,fit_price.dgp.sp,
    fit_price.dgp.data,fit_price.dgp.market_info,f_individuals,fit_price.linear)
f_true_demand = compute_elasticities_IO_HW(fit_100.dgp.true_nonlinear_param,
    fit_100.dgp.sp,fit_100.dgp.data,fit_100.dgp.market_info,f_individuals,
    fit_100.dgp.true_beta; delta=fit_100.dgp.data.X_mean*fit_100.dgp.true_beta+
        vec(data_T100_J3["xi_all"]))
f_summary_rows = NamedTuple[]
for (label,demand) in [("Baseline IV",f_baseline_demand),
        ("IV plus price moment",f_price_demand),("True parameters",f_true_demand)]
    costs = marginal_cost_estimation_IO_HW(demand,fit_100.dgp.market_info,fit_100.dgp.data)
    cross = [E[j,k] for E in demand.elasticity_list for j in axes(E,1)
        for k in axes(E,2) if j != k]
    for (object,values) in [("Own-price elasticity",demand.own_elasticity),
            ("Cross-price elasticity",cross),("Markup",costs.markup)]
        qs=quantile(values,[.1,.25,.5,.75,.9])
        push!(f_summary_rows,(object=object,specification=label,mean=mean(values),
            q10=qs[1],q25=qs[2],median=qs[3],q75=qs[4],q90=qs[5]))
    end
end
df_f_objects = DataFrame(f_summary_rows)
report_q8_table("8(f): elasticities and markups versus baseline IV and truth", df_f_objects)
# alpha_bar is positive price disutility; utility's price coefficient is -alpha_i.
# Because nu is not centered, also compare E[alpha_i] = alpha_bar + sigma*exp(1/2).
df_f_price_sensitivity = DataFrame(specification=["Baseline IV","IV plus price moment","Truth"],
    alpha_bar=[fit_100.linear[end],fit_price.linear[end],1.0],
    mean_alpha=[fit_100.linear[end]+only(fit_100.theta)*exp(.5),
        fit_price.linear[end]+only(fit_price.theta)*exp(.5),1+exp(.5)])
report_q8_table("8(f): price disutility (utility coefficient has the opposite sign)", df_f_price_sensitivity)
println("8(f) sample mean(p*xi_true) = ",mean(vec(data_T100_J3["P_opt"]).*vec(data_T100_J3["xi_all"])))
println("Positive demand shocks can raise equilibrium prices, making price an invalid instrument. The price moment can then attribute high demand to price rather than xi, reducing inferred price sensitivity and raising markups. In mixed logit this direction is not guaranteed: alpha_bar and sigma move jointly, so use the reported mean alpha, elasticities, and markups to assess the realized changes.")
println("A perfect first stage when price instruments itself does not establish validity. GMM criteria with different instrument sets are not directly comparable as goodness-of-fit measures.")

# Compact presentation tables used by question8.tex. Full diagnostics are also
# saved by report_q8_table above; all values remain unrounded in memory.
function answer_parameter_table(df)
    rename(select(df, :parameter, :true_value, :estimate, :bias, :standard_error),
        :parameter=>"Parameter", :true_value=>"Truth", :estimate=>"Estimate",
        :bias=>"Estimate - truth", :standard_error=>"SE")
end
save_q8_table("c_parameters", answer_parameter_table(df_param))
save_q8_table("e_parameters", answer_parameter_table(df_e_parameters))
save_q8_table("c_fit", DataFrame(Criterion=[obj_gh], Converged=[Optim.converged(est_result_gh)],
    Iterations=[Optim.iterations(est_result_gh)]))
save_q8_table("e_fit", DataFrame(Criterion=[obj_10], Converged=[Optim.converged(est_result_10)],
    Iterations=[Optim.iterations(est_result_10)]))
save_q8_table("c_starts", rename(select(df_runs,:sigma_start,:sigma_hat,:GMM_N_gbar_W_gbar,
    :converged,:max_share_error,:max_fixed_point_error),
    :sigma_start=>"Start",:sigma_hat=>"Sigma estimate",:GMM_N_gbar_W_gbar=>"Criterion",
    :converged=>"Converged",:max_share_error=>"Share error",:max_fixed_point_error=>"FP residual"))
answer_d = filter(row->row.evaluation != "Difference",df_objects_summary)
answer_d = rename(select(answer_d,:object,:evaluation,:mean,:q10,:q25,:median,:q75,:q90),
    :object=>"Object",:evaluation=>"Evaluation",:mean=>"Mean",:q10=>"Q10",:q25=>"Q25",
    :median=>"Median",:q75=>"Q75",:q90=>"Q90")
replace!(answer_d.Object,"Consumer surplus per consumer"=>"Consumer surplus",
    "Profit per consumer"=>"Profit")
replace!(answer_d.Evaluation,"True parameters"=>"Truth")
save_q8_table("d_summary",answer_d)
answer_f_params = filter(row->row.parameter in ["alpha_bar","sigma_alpha"],df_f_parameters)
answer_f_params = rename(select(answer_f_params,:specification,:parameter,:true_value,:estimate,:standard_error),
    :specification=>"Specification",:parameter=>"Parameter",:true_value=>"Truth",
    :estimate=>"Estimate",:standard_error=>"SE")
save_q8_table("f_parameters",answer_f_params)
answer_f = rename(copy(df_f_objects),:object=>"Object",:specification=>"Specification",
    :mean=>"Mean",:q10=>"Q10",:q25=>"Q25",:median=>"Median",:q75=>"Q75",:q90=>"Q90")
save_q8_table("f_summary",answer_f)
