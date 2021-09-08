function gibbs(stations_df::DataFrame, δ::Real=0.5, niter::Int=5000, warmup::Int=2000)
    
    # ---- valeurs initiales
    N = nrow(stations_df)

    Y = Vector{Float64}[]

    # for i=1:size(station_list,1)
    for stationID in stations_df[:,:ID]
    #     stationID = station_list[i,:ID]
        y = IDF.pcp(stationID, DURATION)
        push!(Y, y)
    end

    μ = zeros(N, niter)
    ϕ = zeros(N, niter)
    ξ = zeros(niter)


    μ[:, 1] = stations_df[:μ₀]
    ϕ[:, 1] = stations_df[:ϕ₀]
    ξ[1] = 0.0

    acc = 0 # nombre d'acceptations
    

    # échantillonage
    
    @showprogress for iter = 2:niter
        
        
        for i in 1:N
            u = rand(N)

            # ----- estimation de mu
            
            
            # on va estimer avec Metropolis Hastings, en utilisant une loi normale N([μ₀, ϕ₁, ξ], δ * I^-1) 
            m = [μ[i, iter - 1], ϕ[i, iter - 1], ξ[iter - 1]]
            cov = δ * Matrix(I, 3, 3)
            candidates = rand(MvNormal(m, cov))
            ll = logL(Y[i], candidates) - logL(Y[i], m)

            if ll > log(u[i])
                μ[i, iter] = candidates[1]
                acc += 1
            else
                μ[i, iter] = μ[i, iter - 1]
            end

            # ----- estimation de phi
            
            m = [μ[i, iter], ϕ[i, iter - 1], ξ[iter - 1]]
            cov = δ * Matrix(I, 3, 3)
            candidates = rand(MvNormal(m, cov))
            ll = logL(Y[i], candidates) - logL(Y[i], m)

            if ll > log(u[i])
                ϕ[i, iter] = candidates[1]
                acc += 1
            else
                ϕ[i, iter] = ϕ[i, iter - 1]
            end
        end

        u = rand(N)
        m = [μ[:, iter] ; ϕ[:, iter] ; ξ[iter - 1]]
        M = length(m)
        cov = δ * Matrix(I, M, M)
        candidates = rand(MvNormal(m, cov))
        
        ll = 0
        threshold = 0

        for k in 1:N
            params_candidates = [candidates[k], candidates[k + N], candidates[M]]
            params_prev = [m[k], m[k + N], candidates[M]]
            
            ll += logL(Y[k], params_candidates) - logL(Y[k], params_prev)
            threshold += log(u[k])
        end


        if ll > threshold
            ξ[iter] = candidates[M]
            acc += 1
        else
            ξ[iter] = ξ[iter - 1]
        end

    end

    return μ, ϕ, ξ
    
end

