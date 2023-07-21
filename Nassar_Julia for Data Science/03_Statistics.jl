#=
VIDEO 3: STATISTIS
https://juliaacademy.com/courses/937702/lectures/17339512
https://www.youtube.com/watch?v=AAGxWEJ_eWk&t=3s
https://www.youtube.com/watch?v=AAGxWEJ_eWk&list=PLP8iPy9hna6QuDTt11Xxonnfal91JhqjO&index=4
https://github.com/JuliaAcademy/DataScience/blob/master/03.%20Statistics.ipynb
=#
#########################################
using Statistics
using StatsBase
using RDatasets
using Plots
#import Pkg
#Pkg.add("StatsPlots")
using StatsPlots
#Pkg.add("KernelDensity")
using KernelDensity
#Pkg.add("Distributions")
using Distributions
using LinearAlgebra
#Pkg.add("HypothesisTests")
using HypothesisTests
#Pkg.add("PyCall")
#Pkg.build("PyCall")
using PyCall
#Pkg.add("MLBase")
using MLBase

## Eruptions Dataset
D = RDatasets.dataset("datasets","faithful")
Base.@show Base.names(D)
D
RDatasets.describe(D)

# Scatter Plot
eruptions = D[!,:Eruptions]
StatsPlots.scatter(eruptions,label = "eruptions")
waittime = D[!,:Waiting]
StatsPlots.scatter!(waittime, label = "wait time")

## Statistics plots
StatsPlots.boxplot(["eruption length"], eruptions, legend = false, size = (200,400), whisker_width = 1,ylabel = "time in minutes")
StatsPlots.histogram(eruptions, label = "eruptions")
StatsPlots.histogram(eruptions, bins = :sqrt, label = "eruptions")

## Kernel Density Estimates
p = KernelDensity.kde(eruptions)
StatsPlots.histogram(eruptions, label = "eruptions")
StatsPlots.plot!(
    p.x, 
    p.density .* length(eruptions), 
    linewidth = 3, color = 2,label = "kde fit") # nb of elements*bin width

StatsPlots.histogram(eruptions,bins = :sqrt,label = "eruptions")
StatsPlots.plot!(p.x, p.density .* Base.length(eruptions) .*0.2, linewidth = 3, color = 2, label = "kde fit") # nb of elements*bin width

## Normal Distribution
myrandomvector = Base.randn(100_000)
StatsPlots.histogram(myrandomvector)
p = KernelDensity.kde(myrandomvector)
StatsPlots.plot!(
    p.x, 
    p.density .* Base.length(myrandomvector) .*0.1, 
    linewidth = 3, color = 2, label = "kde fit") # nb of elements*bin width

# Probability distributions
d = Distributions.Normal()
myrandomvector = Base.rand(d, 100000)
StatsPlots.histogram(myrandomvector)
p = KernelDensity.kde(myrandomvector)
StatsPlots.plot!(
    p.x, 
    p.density .* Base.length(myrandomvector) .*0.1, 
    linewidth = 3, color = 2, label = "kde fit") # nb of elements*bin width

## Binomial Distribution
b = Distributions.Binomial(40)
myrandomvector = rand(b, 1000000)
StatsPlots.histogram(myrandomvector)
p = KernelDensity.kde(myrandomvector)
StatsPlots.plot!(
    p.x,
    p.density .* Base.length(myrandomvector) .*0.5, 
    color = 2, label = "kde fit") # nb of elements*bin width

## Fit a given set of numbers to a distribution.
x = Base.rand(1000)
d = MLBase.fit(Normal, x)
myrandomvector = Base.rand(d,1000)
StatsPlots.histogram(myrandomvector, nbins = 20, fillalpha = 0.3, label = "fit")
StatsPlots.histogram!(x, nbins = 20, linecolor = :red, fillalpha = 0.3, label = "myvector")

## Fit a given set of numbers to a distribution: Eruptions
x = eruptions
d = MLBase.fit(Normal, x)
myrandomvector = Base.rand(d,1000)
StatsPlots.histogram(myrandomvector, nbins = 20, fillalpha = 0.3)
StatsPlots.histogram!(x, nbins = 20, linecolor = :red, fillalpha = 0.3)

## Hypothesis testing
myrandomvector = Base.randn(1000)
HypothesisTests.OneSampleTTest(myrandomvector)
# Eruptions Data
HypothesisTests.OneSampleTTest(eruptions)

## Using python
#scipy_stats = PyCall.pyimport("scipy_stats")
# @show scipy_stats.spearmanr(eruptions, waittime)
# @show scipy_stats.pearsonr(eruptions, waittime)
# scipy_stats.pearsonr(eruptions, eruptions)

## Correlations
#using Distributions
cortest(x,y) = if Base.length(x) == Base.length(y)
    2 * Distributions.ccdf(Distributions.Normal(),
    Base.atanh(Base.abs(Distributions.cor(x, y))) * Base.sqrt(Base.length(x) - 3))
else
    Base.error("x and y have different lengths")
end
cortest(eruptions, waittime)
StatsBase.corspearman(eruptions, waittime)
Distributions.cor(eruptions, waittime)

# Scatter Plot
StatsPlots.scatter(
    eruptions, waittime, 
    xlabel = "eruption length", 
    ylabel = "wait time between eruptions", 
    legend = false, 
    grid = false, 
    size = (400,300))

## AUC and Confusion Matrix
gt = [1, 1, 1, 1, 1, 1, 1, 2]
pred = [1, 1, 2, 2, 1, 1, 1, 1]
C = MLBase.confusmat(2, gt, pred)   # compute confusion matrix
C ./ Base.sum(C, dims=2)   # normalize per class
Base.sum(LinearAlgebra.diag(C)) / Base.length(gt)  # compute correct rate from confusion matrix
MLBase.correctrate(gt, pred)
C = MLBase.confusmat(2, gt, pred)

# ROC
gt = [1, 1, 1, 1, 1, 1, 1, 0];
pred = [1, 1, 0, 0, 1, 1, 1, 1]
ROC = MLBase.roc(gt,pred)
MLBase.recall(ROC)
MLBase.precision(ROC)
