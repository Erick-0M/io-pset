module blp

    using Statistics, LinearAlgebra, Optim, NLSolversBase, Random
    

    struct BLPData
        T::Int
        J::Int  
        
        X::Matrix{Float64} 
        p::Matrix{Float64} 
        shares::Matrix{Float64} 
        errors::Matrix{Float64}

    end

    Base.@kwdef mutable struct BLPDemand
        

        BLPData::BLPData

        T::Int = BLPData.T
        J::Int  = BLPData.J 

        X::Matrix{Float64} = BLPData.X
        p::Matrix{Float64} = BLPData.p
        shares::Matrix{Float64} = BLPData.shares
        errors::Matrix{Float64} = BLPData.errors

        shares_0::Matrix{Float64}   = reshape(repeat(1.0 .- sum(reshape(shares, J, T), dims=1), inner=(J, 1)), (T*J, 1))

        Z::Matrix{Float64}          = Matrix{Float64}(undef, 0, 0)
        W::Matrix{Float64}          = Matrix{Float64}(undef, 0, 0)
        M::Matrix{Float64}          = Matrix{Float64}(undef, 0, 0)
        moments::Matrix{Float64}    = Matrix{Float64}(undef, 0, 0)
        k_Z::Union{Missing, Int}    = missing

        R::Int                      = size(errors)[2]
        k::Int                      = size(X)[2] + 1
        X_all::Matrix{Float64}      = hcat(X, p)
        sigma::Union{Missing, Float64}    = missing # This would usually will be a matrix          
        beta::Matrix{Float64}       = Matrix{Float64}(undef, k, 1)
        alphas::Matrix{Float64}     = Matrix{Float64}(undef, T * J, R)
        xi::Matrix{Float64}         = Matrix{Float64}(undef, T * J, 1)
        xi_real::Matrix{Float64}    = Matrix{Float64}(undef, T * J, 1)
        delta::Matrix{Float64}      = Matrix{Float64}(undef, T * J, 1)
        v::Matrix{Float64}          = Matrix{Float64}(undef, T, R)
        shares_sim::Matrix{Float64} = Matrix{Float64}(undef, T * J, 1)
        
        Var::Matrix{Float64}        = Matrix{Float64}(undef, k + 1, k + 1)
        util::Matrix{Float64}       = Matrix{Float64}(undef, T * J, R)
        choice::Matrix{Float64}     = Matrix{Float64}(undef, T * J, R)
        ds_dp::Matrix{Float64}      = Matrix{Float64}(undef, T * J, J)
        elas::Matrix{Float64}       = Matrix{Float64}(undef, T * J, J)
        elas_sim::Matrix{Float64}   = Matrix{Float64}(undef, T * J, J)
        markup::Matrix{Float64}     = Matrix{Float64}(undef, T * J, 1)
        markup_sim::Matrix{Float64} = Matrix{Float64}(undef, T * J, 1)
        cost::Matrix{Float64}       = Matrix{Float64}(undef, T * J, 1)
        cost_sim::Matrix{Float64}   = Matrix{Float64}(undef, T * J, 1)
        profit::Matrix{Float64}     = Matrix{Float64}(undef, T * J, 1)
        profit_sim::Matrix{Float64} = Matrix{Float64}(undef, T * J, 1)
        cs::Matrix{Float64}         = Matrix{Float64}(undef, T , 1)
        cs_sim::Matrix{Float64}     = Matrix{Float64}(undef, T , 1)
        ps_sim::Matrix{Float64}     = Matrix{Float64}(undef, T , 1)
        ts_sim::Matrix{Float64}     = Matrix{Float64}(undef, T , 1)
        grad::Matrix{Float64}       = Matrix{Float64}(undef, T * J , 1)
        Omega::Matrix{Float64}      = I(J) # Assumed perform Bertrand
        opt_res::Union{Vector{Any}, Missing} = missing # Optimized results
        params::Matrix{Float64}      = Matrix{Float64}(undef, 0, 0)
    end


    Base.@kwdef mutable struct BLPSupply
        X::Matrix{Float64}          # Linear cost shifters
        k::Int                      = size(X)[2]
        Z::Matrix{Float64}          = Matrix{Float64}(undef, 0, 0)
        W::Matrix{Float64}          = Matrix{Float64}(undef, 0, 0)
        M::Matrix{Float64}          = (I -  X * inv(X' * Z * W * Z' * X) * X'* Z * W * Z');
        moments::Matrix{Float64}    = Matrix{Float64}(undef, 0, 0)
        k_Z::Union{Missing, Int}    = size(Z)[2]
        beta::Matrix{Float64}       = Matrix{Float64}(undef, k, 1)
    end


    Base.@kwdef mutable struct BLP
        BLPDemand::BLPDemand
        BLPSupply::BLPSupply
        k = BLPDemand.k + BLPSupply.k + 1 # Supply and demand parameters and sigma
        opt_res::Union{Vector{Any}, Missing} = missing
        params::Matrix{Float64}   = Matrix{Float64}(undef, 0, 0)
        Var::Matrix{Float64}          = Matrix{Float64}(undef, BLPDemand.T * BLPDemand.J , 1)
    end

    # Random-coefficient part of the price coefficient: alpha_{tr} = beta_p + v_{tr}
    random_coef(model, sigma) = -exp.(sigma .* model.errors)

    # Choice probabilities (J x R) of one market
    function market_choice(delta_m, p_m, v_m)
        ex = exp.(delta_m .+ p_m * v_m')
        return ex ./ (1 .+ sum(ex, dims = 1))
    end

    # Price Jacobian ds/dp (J x J) of one market, averaged over the R draws
    function market_dsdp(ch, alpha_m)
        return (Diagonal(vec(ch * alpha_m)) - (ch .* alpha_m') * ch') ./ size(ch, 2)
    end

    # GMM estimator (X'ZWZ'X)^-1 X'ZWZ' y
    gmm_beta(X, Z, W, y) = (X' * Z * W * Z' * X) \ (X' * Z * W * Z' * y)

    # Residual maker M = I - X (X'ZWZ'X)^-1 X'ZWZ'
    gmm_M(X, Z, W) = I - X * ((X' * Z * W * Z' * X) \ (X' * Z * W * Z'))

    # Simulated shares only (no allocations): the inner loop of the contraction
    function share_sim!(out::Vector{Float64}, buf::Vector{Float64}, model, delta::Vector{Float64}, v::Matrix{Float64})
        J = model.J
        p = model.p
        fill!(out, 0.0)
        @inbounds for r in 1:model.R, m in 1:model.T
            off = (m - 1) * J
            vmr = v[m, r]
            denom = 1.0
            for j in 1:J
                e = exp(delta[off + j] + p[off + j] * vmr)
                buf[j] = e
                denom += e
            end
            for j in 1:J
                out[off + j] += buf[j] / denom
            end
        end
        out ./= model.R
        return out
    end

    # Simulating the model given delta and sigma
    function simulate_BLP!(model::BLPDemand; delta = nothing, sigma = nothing, all = false, return_sim =true, return_grad= false, supply = false)

        delta = something(delta, model.delta)
        sigma = something(sigma, model.sigma)

        T, J, R = model.T, model.J, model.R
        competitive = iszero(model.Omega) # perfect competition: c = p
        need_dp = all || (supply && !competitive)
        need_grad = return_grad || all

        # For each market we compute the random utility part
        v = random_coef(model, sigma)
        alphas = model.beta[end] .+ v

        grad = zeros(T * J, 1)
        dc_dsigma = zeros(T * J, 1)
        dc_dbetap = zeros(T * J, 1)
        ds_dp = zeros(T * J, J)
        util = zeros(T * J, R)
        choice = zeros(T * J, R)
        cs_sim = zeros(T, 1)
        elas = zeros(T * J, J)
        cost_sim = zeros(T * J, 1)
        markup_sim = zeros(T * J, 1)

        for m_i in 1:T

            idx = (m_i - 1) * J + 1:m_i * J
            p_m = model.p[idx, 1]
            s_m = model.shares[idx, 1]

            util[idx, :] = delta[idx, [1]] .+ p_m * v[m_i, :]'
            ch = market_choice(delta[idx, 1], p_m, v[m_i, :])
            choice[idx, :] = ch

            if need_dp
                dsdp = market_dsdp(ch, alphas[m_i, :])
                ds_dp[idx, :] = dsdp
                A = model.Omega .* dsdp
                if all
                    elas[idx, :] = dsdp .* p_m' ./ s_m
                    cs_sim[m_i, 1] = mean(-log.(1 .+ sum(exp.(util[idx, :]), dims = 1)) ./ alphas[[m_i], :])
                end

                # Recovering costs from ownership matrix
                if !competitive
                    cost_sim[idx, 1] = p_m + A \ s_m
                else
                    cost_sim[idx, 1] = p_m
                end
                markup_sim[idx, 1] = (p_m - cost_sim[idx, 1]) ./ p_m
            end

            if need_grad
                dd = vec(v[m_i, :] .* model.errors[m_i, :])                        # dalpha_r/dsigma
                ds_ddelta = (Diagonal(vec(sum(ch, dims = 2))) - ch * ch') ./ R    # ds/ddelta
                ds_dsigma = ((ch .* p_m) * dd - ch * ((ch' * p_m) .* dd)) ./ R    # ds/dsigma at fixed delta
                g_m = -(ds_ddelta \ ds_dsigma)
                grad[idx, 1] = g_m

                if supply && !competitive
                    A = model.Omega .* ds_dp[idx, :]
                    alpha_m = alphas[m_i, :]
                    U = g_m .+ p_m * dd'                                          # J x R
                    DS = ch .* U - ch .* sum(ch .* U, dims = 1)                    # total ds_r/dsigma
                    dDelta = (Diagonal(vec(ch * dd)) - (ch .* dd') * ch' +
                              Diagonal(vec(DS * alpha_m)) - (DS .* alpha_m') * ch' - (ch .* alpha_m') * DS') ./ R
                    dc_dsigma[idx, 1] = -(A \ ((model.Omega .* dDelta) * (A \ s_m)))
                    dc_dbetap[idx, 1] = -(A \ ((model.Omega .* ds_ddelta) * (A \ s_m)))
                end
            end
        end

        if competitive
            cost_sim = copy(model.p)
            markup_sim = zeros(T * J, 1)
        end

        # Store all the computations in the object

        if all
            model.elas_sim = elas;
            model.cost_sim = cost_sim;
            model.markup_sim = markup_sim;
            model.cs_sim = cs_sim;
            model.profit_sim = (model.p - model.cost_sim) .* model.shares;
            model.ps_sim = sum(reshape(model.profit_sim, (J, T)), dims = 1)'
            model.ts_sim = model.cs_sim + model.ps_sim
            model.shares_sim = mean(choice, dims = 2);
            model.grad = grad;
            model.alphas = alphas;
            model.util = util;
            model.choice = choice;
            model.delta = delta;
            model.ds_dp = ds_dp; # To compute diversion ratios
        end

        # Return the gradient
        if return_grad & !supply
            return grad
        end

        # Return the simulation
        if return_sim & !supply
            return mean(choice, dims = 2);
        end

        if return_grad & supply
            return grad, dc_dsigma, dc_dbetap
        end

        # Return the choice and cost return if supply is active
        if return_sim & supply
            return mean(choice, dims = 2), cost_sim
        end

    end

    # Fixed point algorithm. delta0 warm-starts the iteration (any starting point converges).
    function contraction(model, sigma_alpha; toler = 1e-10, max_iter = 10000, compute_grad = false, delta0 = nothing)
        v = random_coef(model, sigma_alpha)
        log_s = log.(vec(model.shares))
        delta = delta0 === nothing ? log_s .- log.(vec(model.shares_0)) : copy(vec(delta0))
        s_sim = similar(delta)
        buf = zeros(model.J)
        for _ in 1:max_iter
            share_sim!(s_sim, buf, model, delta, v)
            nd = 0.0
            @inbounds for i in eachindex(delta)
                d = log_s[i] - log(s_sim[i])
                delta[i] += d
                nd += d * d
            end
            sqrt(nd) <= toler && return reshape(delta, :, 1)
        end
        println("Reached max number of iterations")
        return reshape(delta, :, 1)
    end



    # Merger simulation

    function counterfactual_simulation(model_system::BLP; toler = 1e-8, max_iter = 10000, lambda = 0.1)

        # lambda: damping weight on the new price (the undamped iteration diverges)

        model = model_system.BLPDemand
        T, J = model.T, model.J
        v = random_coef(model, model.sigma)
        alphas = model.beta[end] .+ v
        cost_sim = model.cost_sim
        p_sim = zeros(T * J, 1)
        diff = Inf
        flag = 0

        for count in 1:max_iter
            delta = model.X_all * model.beta + model.xi

            # Given the shares, compute the new price
            for m_i in 1:T
                idx = (m_i - 1) * J + 1:m_i * J
                ch = market_choice(delta[idx, 1], model.p[idx, 1], v[m_i, :])
                dsdp = market_dsdp(ch, alphas[m_i, :])
                p_new = -((model.Omega .* dsdp) \ model.shares[idx, :]) + cost_sim[idx, :]
                p_sim[idx, :] = lambda .* p_new .+ (1 - lambda) .* model.p[idx, :]
            end
            p_old = model.p

            # Replace the last element of the X_all matrix
            model.p = copy(p_sim)
            model.X_all[:, end] = p_sim
            model.delta = model.X_all * model.beta + model.xi

            # Given the new price, compute the shares
            shares_sim = simulate_BLP!(model)

            # Check convergence in prices
            diff = maximum(abs.(p_sim - p_old))

            # Chaging simulated shares and real shares
            model.shares = shares_sim
            model.shares_sim = shares_sim

            toler < diff || break
            if count == max_iter
                flag = -1
                println("Reached max number of iterations")
            end
        end

        if flag == 0
            simulate_BLP!(model, all = true)
        end

        return model_system

    end



    # Objective function with the gradient. warm: optional Ref reused to warm-start the contraction.
    function gmm_fg!(F, G, model, sigma; toler = 1e-10, max_iter = 10000, warm = nothing)

        delta = contraction(model, sigma, toler = toler, max_iter = max_iter, delta0 = warm_delta(warm))
        store_warm!(warm, delta)
        e_xi = model.M * delta;
        Wg = model.W * (model.Z' * e_xi)

        if G !== nothing
            grad_ = simulate_BLP!(model, delta = delta, sigma = sigma, return_grad = true)
            G .= 2 * grad_' * model.Z * Wg;
        end
        if F !== nothing
            return dot(model.Z' * e_xi, Wg)
        end

    end

    warm_delta(warm) = (warm === nothing || !isassigned(warm)) ? nothing : warm[]
    store_warm!(warm, delta) = (warm !== nothing && all(isfinite, delta)) ? (warm[] = delta; nothing) : nothing


    function compute_hausman(model)

        # Computing Hausman style instrument for price
        price_bar = zeros(model.J*model.T, 1)
        ts = sum(reshape(model.p, (model.J, model.T)), dims = 2)

        for m_i in 1: model.T
            idx = (m_i - 1)* model.J + 1:m_i* model.J
            price_bar[idx, :] = (ts - model.p[idx, :])/(model.T-1)

        end

        return price_bar

    end


    function compute_blp(model)

        X_rivals = zeros(model.J*model.T, model.k - 1)
        for m_i in 1:model.T
            idx = (m_i - 1) * model.J + 1: (m_i) * model.J
            X_rivals[idx, :] = (sum(model.X[idx, :], dims = 1) .- model.X[idx, :])
        end
        X_rivals = X_rivals[:, 2:end];
        return X_rivals

    end

    # Minimise over sigma from Ns random starts, one start per task (Threads). make_fg() must return an
    # fg! closure that owns whatever state it mutates, so starts never share mutable data.
    # Returns the Optim results of the starts that did not error, in start order.
    function multistart(make_fg; sigma_lower, sigma_upper, Ns, iterations, time_limit, failures, outer_iterations = 1000)

        # Seed globally
        Random.seed!(704)

        sigma0_ = clamp.(randn(Ns) * 0.5 .+ 1, sigma_lower, sigma_upper);

        results = Vector{Any}(nothing, Ns)
        lk = ReentrantLock()

        # One BLAS thread per Julia thread: avoids oversubscription when the starts run in parallel
        blas_threads = BLAS.get_num_threads()
        Threads.nthreads() > 1 && BLAS.set_num_threads(1)

        Threads.@threads :dynamic for i in 1:Ns
            sigma0 = sigma0_[i]
            t0 = time()
            try
                res = optimize(NLSolversBase.only_fg!(make_fg()), [sigma_lower], [sigma_upper], [sigma0], Fminbox(LBFGS()), Optim.Options(iterations = iterations, outer_iterations = outer_iterations, time_limit = time_limit))
                lock(lk) do
                    println("  start sigma0 = $(round(sigma0, digits = 3)) -> sigma = $(round(Optim.minimizer(res)[1], digits = 4)), obj = $(Optim.minimum(res)), $(round(time() - t0, digits = 1))s"); flush(stdout)
                end
                results[i] = res
            catch err
                msg = sprint(showerror, err)
                lock(lk) do
                    println("  start sigma0 = $(round(sigma0, digits = 3)) FAILED after $(round(time() - t0, digits = 1))s: $(first(msg, 120))"); flush(stdout)
                    failures === nothing || push!(failures, (sigma0 = sigma0, error = msg))
                end
            end
        end

        BLAS.set_num_threads(blas_threads)

        ress = collect(Any, filter(!isnothing, results));
        isempty(ress) && error("all $Ns starting points failed")
        return ress

    end

    function estimate_blp!(model; Z = nothing, W = nothing, sigma_lower = 0.1, sigma_upper = 10, Ns = 10, toler = 1e-10, max_iter = 10000, iterations = 1000, time_limit = Inf, failures = nothing, outer_iterations = 1000)

        model.Z = something(Z, model.Z);
        model.W = something(W, model.W);

        model.k_Z = size(model.Z, 2);
        model.M = gmm_M(model.X_all, model.Z, model.W);

        # gmm_fg! only reads the model, so starts share it; each start has its own warm-start cache
        ress = multistart(() -> (F, G, s) -> gmm_fg!(F, G, model, s, toler = toler, max_iter = max_iter, warm = Ref{Matrix{Float64}}());
                          sigma_lower, sigma_upper, Ns, iterations, time_limit, failures, outer_iterations)

        optimized = ress[argmin(Optim.minimum.(ress))];

        sigma_est = Optim.minimizer(optimized)[1];
        model.sigma = sigma_est
        # Get the estimates for delta (precise: independent of the warm start)
        delta = contraction(model, model.sigma, toler = toler, max_iter = max_iter);
        # Get the estimates for beta
        beta_est = gmm_beta(model.X_all, model.Z, model.W, delta);

        model.delta = delta;
        model.xi = model.M * model.delta;
        model.beta = beta_est;
        model.opt_res = ress
        model.params = vcat(beta_est, [sigma_est])
        simulate_BLP!(model, all = true);

        N = model.J * model.T
        omega_var = model.Z' * ((model.xi .^ 2) .* model.Z) ./ N

        Gamma = 1/N * model.Z' * hcat(-model.X_all, model.grad);
        GWG = Gamma' * model.W * Gamma
        model.Var = (GWG \ (Gamma' * model.W * omega_var * model.W * Gamma)) / GWG / N

    end


    function gmm_joint_fg!(F, G, model, sigma; toler = 1e-10, max_iter = 10000, warm = nothing)

        d, s = model.BLPDemand, model.BLPSupply

        delta = contraction(d, sigma, toler = toler, max_iter = max_iter, delta0 = warm_delta(warm));
        store_warm!(warm, delta)
        # Update beta before computing cost
        B = (d.X_all' * d.Z * d.W * d.Z' * d.X_all) \ (d.X_all' * d.Z * d.W * d.Z')
        d.beta = B * delta;
        shares, cost = simulate_BLP!(d, delta = delta, sigma = sigma, supply = true);
        s.beta = gmm_beta(s.X, s.Z, s.W, cost);
        e_xi = d.M * delta;
        e_omega = s.M * cost;
        g_d = d.Z' * e_xi
        g_s = s.Z' * e_omega
        Wg_d = d.W * g_d
        Wg_s = s.W * g_s

        if G !== nothing

            grad_, dc_dsigma_, dc_dbetap_ = simulate_BLP!(d, delta = delta, sigma = sigma, return_grad = true, supply = true);
            dc_dsigma_total = dc_dsigma_ .+ dc_dbetap_ .* ((B * grad_)[end]) # We need to add the impact of the change of sigma in the cost through changes in beta
            G .= 2 * grad_' * d.Z * Wg_d + 2 * dc_dsigma_total' * s.Z * Wg_s

        end

        if F !== nothing
            return dot(g_d, Wg_d) + dot(g_s, Wg_s)
        end

    end

    function estimate_blp_joint!(model; sigma_lower = 0.1, sigma_upper = 10, Ns = 10, toler = 1e-10, max_iter = 10000, iterations = 100, time_limit = Inf, failures = nothing, outer_iterations = 1000)

        d, s = model.BLPDemand, model.BLPSupply

        # gmm_joint_fg! mutates beta, so each start works on its own copy of the system
        ress = multistart(() -> (m = deepcopy(model); w = Ref{Matrix{Float64}}(); (F, G, sg) -> gmm_joint_fg!(F, G, m, sg, toler = toler, max_iter = max_iter, warm = w));
                          sigma_lower, sigma_upper, Ns, iterations, time_limit, failures, outer_iterations)

        optimized = ress[argmin(Optim.minimum.(ress))];
        sigma_est = Optim.minimizer(optimized)[1];
        d.sigma = sigma_est

        # Everything below is evaluated at the best start (the last evaluation inside the optimizer is not necessarily it)
        delta = contraction(d, sigma_est, toler = toler, max_iter = max_iter);
        d.delta = delta;
        d.xi = d.M * delta;
        d.beta = gmm_beta(d.X_all, d.Z, d.W, delta);
        d.opt_res = ress

        simulate_BLP!(d, delta = delta, sigma = sigma_est, all = true);
        s.beta = gmm_beta(s.X, s.Z, s.W, d.cost_sim);

        N  = d.T * d.J
        kb = size(d.X_all, 2)       # demand linear params (price is the last one)
        kc = size(s.X, 2)           # cost shifters
        kd = size(d.Z, 2)
        ks = size(s.Z, 2)

        grad, dc_dsigma, dc_dbetap = simulate_BLP!(d, delta = delta, sigma = sigma_est, return_grad = true, supply = true)
        cost = d.cost_sim

        xi    = d.xi                # demand residual
        omega = s.M * cost          # supply residual

        # Jacobians of the residuals wrt θ = [β; γ; σ]
        dxi    = hcat(-d.X_all, zeros(N, kc), grad)
        domega = hcat(zeros(N, kb - 1), dc_dbetap, -s.X, dc_dsigma)

        # Γ = ∂ĝ/∂θ'
        Gamma = vcat(d.Z' * dxi, s.Z' * domega) ./ N

        # Moment covariance (full, keeps the ξ–ω cross block)
        g = hcat(d.Z .* xi, s.Z .* omega)               # N × (kd + ks)

        S = g' * g ./ N

        Wg = [d.W zeros(kd, ks); zeros(ks, kd) s.W]
        GWG = Gamma' * Wg * Gamma
        V = (GWG \ (Gamma' * Wg * S * Wg * Gamma)) / GWG ./ N

        model.opt_res = ress
        model.params = vcat(d.beta, s.beta, [sigma_est])
        model.Var = V

    end

end
