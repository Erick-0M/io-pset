using LinearAlgebra, Statistics, Random, MAT, Plots, StatsPlots

data3 = matread(joinpath(@__DIR__, "100markets3products.mat"))
data5 = matread(joinpath(@__DIR__, "100markets5products.mat"))
figdir = joinpath(@__DIR__, "..", "figs", "Q6")
mkpath(figdir)

gamma = [2, 1, 1]

function outc(data)
    cost = gamma[1] .+ gamma[2] .* vec(data["w"]) .+
           gamma[3] .* vec(data["Z"]) .+ vec(data["eta"])

    price = vec(data["P_opt"])
    markup = price .- cost
    profit = markup .* vec(data["shares"])

    return (; price, markup, profit)
end

out3 = outc(data3)
out5 = outc(data5)

# Summary statistics, pooling product–market observations.
function summarize(x)
    return (
        n=length(x),
        mean=mean(x),
        sd=std(x),
        min=minimum(x),
        p25=quantile(x, 0.25),
        median=median(x),
        p75=quantile(x, 0.75),
        max=maximum(x)
    )
end

for variable in propertynames(out3)
    println("\n", uppercasefirst(string(variable)))
    println("J = 3: ", summarize(getproperty(out3, variable)))
    println("J = 5: ", summarize(getproperty(out5, variable)))
end

# Shared bins and density normalization make the samples comparable.
function comparison_histogram(x3, x5, title; nbins=30)
    lo, hi = extrema(vcat(x3, x5))
    edges = range(lo, hi; length=nbins + 1)

    plt = histogram(
        x3;
        bins=edges,
        normalize=:pdf,
        alpha=0.45,
        label="J = 3",
        color=:steelblue,
        xlabel=title
        # ylabel="Density"
        # title=title
    )

    histogram!(
        plt, x5;
        bins=edges,
        normalize=:pdf,
        alpha=0.45,
        label="J = 5",
        color=:darkorange
    )

    return plt
end

plt_price = comparison_histogram(out3.price, out5.price, "Price")
plt_markup = comparison_histogram(out3.markup, out5.markup, "Markup")
plt_profit = comparison_histogram(out3.profit, out5.profit, "Profit")

plt = plot(
    plt_price, plt_markup, plt_profit;
    layout=(1, 3),
    size=(1200, 350),
    margin=5Plots.mm
)

display(plt)
savefig(plt, joinpath(figdir, "prices_markups_profits.pdf"))

# (b) Common consumer draws, including shocks for the outside option.
beta = [5, 1, 1]
rng = MersenneTwister(2026)
N = 10_000
alpha = 1 .+ exp.(randn(rng, N))
epsilon = -log.(randexp(rng, N, 6, 100))

function consumer_outcomes(data, alpha, epsilon)
    price = data["P_opt"]
    J, T = size(price)
    N = length(alpha)
    value = reshape(data["x1"] * beta .+ vec(data["xi_all"]), J, T)
    choice = zeros(Int, N, T)
    surplus = zeros(N, T)
    shares = zeros(J + 1, T)
    choice_shares = zeros(J + 1, T)

    for t in 1:T
        # Column 1 is the outside option; columns 2:J+1 are inside goods.
        utility = hcat(zeros(N), value[:, t]' .- alpha * price[:, t]')
        shift = maximum(utility; dims=2)
        weight = exp.(utility .- shift)
        denominator = sum(weight; dims=2)
        probability = weight ./ denominator

        # Expected money surplus relative to having only the outside option.
        surplus[:, t] = vec(shift .+ log.(denominator)) ./ alpha
        shares[:, t] = vec(mean(probability; dims=1))
        realized_utility = utility .+ view(epsilon, :, 1:J+1, t)
        choice[:, t] = vec(getindex.(argmax(realized_utility; dims=2), 2)) .- 1
        for j in 0:J
            choice_shares[j + 1, t] = count(==(j), view(choice, :, t)) / N
        end
    end

    return (; choice, surplus, shares, choice_shares)
end

consumer3 = consumer_outcomes(data3, alpha, epsilon)
consumer5 = consumer_outcomes(data5, alpha, epsilon)

println("Expected consumer surplus, pooling consumer–market observations")
println("J = 3: ", summarize(vec(consumer3.surplus)))
println("J = 5: ", summarize(vec(consumer5.surplus)))
println("Mean outside shares: ", mean(consumer3.shares[1, :]),
        " (J = 3), ", mean(consumer5.shares[1, :]), " (J = 5)")

plt_surplus = comparison_histogram(
    vec(consumer3.surplus), vec(consumer5.surplus), "Expected consumer surplus"
)
plot!(plt_surplus; ylabel="Density", size=(700, 400), margin=5Plots.mm)
display(plt_surplus)
savefig(plt_surplus, joinpath(figdir, "consumer_surplus.pdf"))

# (c) Compute the outside probability directly
function check_shares(shares)
    discrepancy = abs.(vec(sum(shares; dims=1)) .- 1)
    @assert all(isfinite, shares) && all(0 .<= shares .<= 1)
    @assert maximum(discrepancy) < 1e-12
    return (; maximum=maximum(discrepancy), mean=mean(discrepancy))
end

for (J, consumer) in ((3, consumer3), (5, consumer5))
    println("\nJ = ", J, " probability-share discrepancies: ",
            check_shares(consumer.shares))
    println("J = ", J, " choice-frequency discrepancies: ",
            check_shares(consumer.choice_shares))
end
