"""
    load_station
    
function that allows to load data from a given station.
"""

function load_station(station_id::String)
    path = "C:/Users/leogu/Dropbox/Stage/Perso/Codes/Julia/dat/"*station_id*".csv"
    
    df = CSV.read(path, DataFrame)
    rename!(df,:Année => :Year)
    
    #on met le tableau sous forme tidy (cf cheat sheet de R)
    df_reshape = stack(df, Not(:Year); variable_name=:Duration, value_name=:Pcp)
    dropmissing!(df_reshape,:Pcp)
    
    return df_reshape
end

"""
    GEVparameters

function that returns the optimal parameters of a GEV estimation of a given vector Y
"""
function GEVparameters(Y::Vector{Float64})
    function f(p::Vector{Float64})
        return -logL(Y, p[1], p[2], p[3])
    end
    
    μ₀ = mean(Y)
    σ₀ = std(Y)
    ξ₀ = 0
    p₀ = [μ₀, σ₀, ξ₀]
    p = p₀
    
    try
        res = optimize(f, p₀)
        
         
        if Optim.converged(res)
            p = Optim.minimizer(res)
        else
            @warn "The maximum likelihood algorithm did not find a solution. Maybe try with different initial values or with another method. The returned values are the initial values."
            p = p₀
        end
        
    catch
        println("Error of scale with this vector")
    end
   
    
    return p
end

"""
    logL

fonction donnant la log-vraisemblance d'un vecteur Y suivant une loi GEV de parametre μ, σ et ξ.
"""
function logL(Y::Vector{<:Real},μ::Real,σ::Real,ξ::Real)
    G = GeneralizedExtremeValue(μ, σ, ξ)
    return loglikelihood(G, Y)
end

"""
    BIC_GEV

compute the BIC obtained with a GEV estimation
"""

function BIC_GEV(y::Vector{Float64})
    p = gevfit(y).θ̂
    n = length(y)

    return 3*log(n) - 2logL(y, p[1], exp(p[2]), p[3])
end

