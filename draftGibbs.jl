function gev_bhm(data_layer::Vector{BlockMaxima}; δ::Real=0.5, warmup::Int=0, niter::Int=10000, adapt::Symbol=:all)
    #number of groups
    m = length(data_layer) 
    n = Extremes.nparameter(data_layer[1])
    
    # vector initialization
    params = zeros(m, n, niter)
    ν = zeros(n, niter)
    τ = ones(n, niter)
    
    #acceptance counts 
    acc = 0. 
    
    #initilization
    Σ = Matrix{Float64}[]
    for i in 1:m
        fd = Extremes.fit(data_layer[i]) 
        params[i, :, 1] = fd.θ̂ 
        #params[i, :, 1] = randn(n)
        push!(Σ, inv(Symmetric(Extremes.hessian(fd)))+ 0.01 * Matrix(I, n, n))
        #push!(Σ, Matrix(I, n, n))
    end
    ν[:,1] = mean(params[:,:,1], dims=1)
    τ[:,1] = std(params[:,:,1], dims=1)
    u = rand(m)
    
    #Normal distribution N([μ₀, μ₁, ϕ₀, ϕ₁, ξ], δ * I^-1) is used for Metropolis-Hastings simulation (Random walk M-H)
    @showprogress for iter=2:niter
        #Updating the data layer parameters 
        rand!(u)
        for i = 1:m
            candidates = rand(MvNormal(params[i , : , iter-1], δ*Σ[i]))
            lr = Extremes.loglike(data_layer[i], candidates) -
                    Extremes.loglike(data_layer[i], params[i , : , iter-1]) +
                    logpdf(MvNormal(ν[:, iter-1], diagm(τ[:, iter-1])), candidates) -
                    logpdf(MvNormal(ν[:, iter-1], diagm(τ[:, iter-1])), params[i , : , iter-1])
            if lr > log(u[i])
                params[i , : , iter] = candidates
                acc += 1
            else
                params[i , : , iter] = params[i , : , iter-1]
            end
        end
        
        # Updating the process layer parameters
        for j in 1:n
            ν[j,iter], τ[j,iter] = normal_sampling(params[:, j, iter])
        end
        
        # Updating the stepsize
        if iter % 50 == 0
            if !(adapt == :none)
                if (iter <= warmup) | (adapt==:all)
                    accrate = acc / (m*50)
                    δ = update_stepsize(δ, accrate)
                    acc = 0.
                end
            end
        end
        
        #Updating the covariance proposition matrix according to Rosenthal (2008)
        if iter % 50 == 0 && iter <= warmup
            for i in 1:m
                Σ[i] = StatsBase.cov(params[i, :, 1:iter]') + 0.01 * Matrix(I, n, n)
            end
        end
    end
    
    parmnames = String[]
    res = Array{Float64, 2}
    if n == 3
        parmnames = vcat(["μ[$i]" for i=1:m], ["ϕ[$i]" for i=1:m], ["ξ[$i]" for i=1:m],["ν_μ", "ν_ϕ", "ν_ξ", "τ_μ", "τ_ϕ", "τ_ξ"])
        res = vcat(params[:, 1, :], params[:, 2, :], params[:, 3, :], ν, τ)
    end
    if n == 4
        parmnames = vcat(["μ₀[$i]" for i=1:m], ["μ₁[$i]" for i=1:m], ["ϕ[$i]" for i=1:m], ["ξ[$i]" for i=1:m], ["ν_μ₀", "ν_μ₁", "ν_ϕ", "ν_ξ", "τ_μ₀", "τ_μ₁", "τ_ϕ", "τ_ξ"])
        res = vcat(params[:, 1, :], params[:, 2, :], params[:, 3, :], params[:, 4, :], ν, τ)     
    end
    if n == 5 
        parmnames = vcat(["μ₀[$i]" for i=1:m], ["μ₁[$i]" for i=1:m], ["ϕ₀[$i]" for i=1:m], ["ϕ₁[$i]" for i=1:m], ["ξ[$i]" for i=1:m],["ν_μ₀", "ν_μ₁", "ν_ϕ₀", "ν_ϕ₁", "ν_ξ", "τ_μ₀", "τ_μ₁", "τ_ϕ₀", "τ_ϕ₁", "τ_ξ"])
        res = vcat(params[:, 1, :], params[:, 2, :], params[:, 3, :], params[:, 4, :], params[:, 5, :], ν, τ)
    end
    
    res = Mamba.Chains(collect(res'), names=parmnames)
    res = res[warmup+1:10:niter, :,:]
    
    println("Exploration stepsize after warmup: ", δ)
    println("Acceptance rate: ", acc/((niter-warmup)*m))
    
    return res
end