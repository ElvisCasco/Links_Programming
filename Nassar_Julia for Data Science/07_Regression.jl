#=
VIDEO 7: REGRESSION
https://juliaacademy.com/courses/937702/lectures/17339541
https://www.youtube.com/watch?v=5TCbIK_cpZE
https://www.youtube.com/watch?v=5TCbIK_cpZE&list=PLP8iPy9hna6QuDTt11Xxonnfal91JhqjO&index=8
https://github.com/JuliaAcademy/DataScience/blob/master/07.%20Regression.ipynb
=#
#########################################
using GR
using Plots
using Statistics
using StatsBase
using PyCall
using DataFrames
using GLM
using Tables
using XLSX
using MLBase
using RDatasets
using LsqFit

## Values
xvals = DataFrames.repeat(1:0.5:10, inner = 2)
yvals = 3 .+ xvals .+ 2 .* Base.rand(Base.length(xvals)) .-1
Plots.scatter(
    xvals,
    yvals,
    color =:black,
    leg = false)

## Best fit
function find_best_fit(xvals, yvals)
    meanx = StatsBase.mean(xvals)
    meany = StatsBase.mean(yvals)
    stdx = StatsBase.std(xvals)
    stdy = StatsBase.std(yvals)
    r = StatsBase.cor(xvals, yvals)
    a = r * stdy / stdx
    b = meany - a * meanx
    return a, b
end
a, b = find_best_fit(xvals, yvals)
ynew = a .* xvals .+ b

## Fitting uisng PyCall
np = PyCall.pyimport("numpy");
xdata = xvals
ydata = yvals
Base.@time myfit = np.polyfit(
    xdata,
    ydata, 1);
ynew2 = DataFrames.collect(xdata) .* myfit[1] .+ myfit[2];
Plots.scatter(xvals, yvals)
Plots.plot!(xvals, ynew)
Plots.plot!(xvals, ynew2)

## GLM package
data = DataFrames.DataFrame(X = xdata, Y = ydata)
#ols = GLM.lm(GLM.@formula(Y ~ X), data)
#Plots.plot!(xdata, GLM.predict(ols))

## play around with data for a bit
R = XLSX.readxlsx(wd * "Data/zillow_data_download_april2020.xlsx")
sale_counts = R["Sale_counts_city"][:]
df_sale_counts = DataFrames.DataFrame(sale_counts[2:end, :], Symbol.(sale_counts[1, :]))

monthly_listings = R["MonthlyListings_City"][:]
df_monthly_listings = DataFrames.DataFrame(monthly_listings[2:end, :], Symbol.(monthly_listings[1, :]))
monthly_listings_2020_02 = df_monthly_listings[!,[1,2,3,4,5,end]]
DataFrames.rename!(monthly_listings_2020_02, Symbol("2020-02") .=> Symbol("listings"))

sale_counts_2020_02 = df_sale_counts[!, [1,end]]
DataFrames.rename!(sale_counts_2020_02, Symbol("2020-02") .=> Symbol("sales"))

Feb2020data = DataFrames.innerjoin(monthly_listings_2020_02, sale_counts_2020_02,on=:RegionID) #, type="outer")
DataFrames.dropmissing!(Feb2020data)
sales = Feb2020data[!, :sales]
# prices = Feb2020data[!,:price]
counts = Feb2020data[!, :listings]

using DataStructures
states = Feb2020data[!, :StateName]
C = DataStructures.counter(states)
C.map
countvals = Base.values(C.map)
topstates = Base.sortperm(Base.collect(countvals), rev=true)[1:10]
states_of_interest = DataFrames.collect(DataFrames.keys(C.map))[topstates]

all_plots = Array{Plots.Plot}(undef, 10)
for (i,si) in Base.enumerate(states_of_interest)
    curids = Base.findall(Feb2020data[!, :StateName].==si)
    local data = DataFrames.DataFrame(X = float.(counts[curids]), Y = float.(sales[curids]))
    local ols = GLM.lm(GLM.@formula(Y ~ 0 + X), data)
    all_plots[i] = Plots.scatter(counts[curids], sales[curids], markersize=2,
        xlim = (0, 500), ylim = (0, 500), color =i, aspect_ratio =:equal,
        legend = false, title =si)
    @show si,GLM.coef(ols)
    Plots.plot!(counts[curids], GLM.predict(ols),color=:black)
end
Plots.plot(all_plots...,
    layout = (2,5),
    size = (900,300))

all_plots = Array{Plots.Plot}(undef, 10)
for (i,si) in Base.enumerate(states_of_interest)
    curids = Base.findall(Feb2020data[!, :StateName] .== si)
    local data = DataFrames.DataFrame(X = Base.float.(counts[curids]), Y = Base.float.(sales[curids]))
    local ols = GLM.lm(@formula(Y ~ X), data)
    all_plots[i] = Plots.scatter(counts[curids], sales[curids], markersize=2,
        xlim=(0,500),ylim=(0,500),color=i,aspect_ratio=:equal,
        legend = false, title = si)
    @show si,GLM.coef(ols)
    Plots.plot!(counts[curids], GLM.predict(ols),color=:black)
end
Plots.plot(all_plots...,
    layout = (2,5),
    size = (900,300))

Plots.plot()
#using PyPlot
#Plots.gr()
for (i,si) in Base.enumerate(states_of_interest)
    curids = Base.findall(Feb2020data[!, :StateName] .== si)
    local data = DataFrames.DataFrame(X = Base.float.(counts[curids]), Y = Base.float.(sales[curids]))
    local ols = GLM.lm(GLM.@formula(Y ~ 0 + X), data)
    Plots.scatter!(counts[curids], sales[curids], markersize = 2,
        xlim = (0,500), ylim = (0,500), color = i, aspect_ratio =:equal,
        legend = false, marker = (3,3,Plots.stroke(0)), alpha = 0.2)
        if si == "NC" || si == "CA" || si == "FL"
            #annotate!([(500-20, 10 + coef(ols)[1] * 500, si)])#text(si, 10, :black))])
            #Plots.annotate!([(500-20, 10 + coef(ols)[1] * 500, Plots.text(si, 10, :black))])
            Plots.annotate!([(500-20,10+GLM.coef(ols)[1]*500,Plots.text(si,10))])
        end
    Base.@show si, GLM.coef(ols)
    Plots.plot!(counts[curids], GLM.predict(ols), color = i, linewidth = 2)
end
 Plots.plot(all_plots...,layout=(2,5),size=(900,300))
Plots.xlabel!("listings")
Plots.ylabel!("sales")

## Logistic regression
data = DataFrames.DataFrame(X=[1,2,3,4,5,6,7], Y=[1,0,1,1,1,1,1])
linear_reg = GLM.lm(GLM.@formula(Y ~ X), data)
Plots.scatter(
    data[!,:X],
    data[!,:Y],
    legend = false,
    size = (300, 200))
Plots.plot!(1:7, GLM.predict(linear_reg))

# we will load this data from RDatasets
cats = RDatasets.dataset("MASS", "cats")
#using RData
#R_data = RData.load("https://github.com/JuliaStats/RDatasets.jl/blob/master/data/MASS/cats.rda")
#=
P = download("https://github.com/vincentarelbundock/Rdatasets/blob/master/csv/MASS/cats.csv",
    "Data/cats.csv")
using DelimitedFiles
#P,H = DelimitedFiles.readdlm(P,',';header=true);
#cats = P
cats = DelimitedFiles.readdlm("Data/cats.csv", ','; header = true)
Base.typeof(cats)
cats = DataFrames.DataFrame(cats[1],:auto)
colnames = ["N", "Sex", "BWt", "HWt"]
DataFrames.rename!(cats, Symbol.(colnames))
=#

lmap = MLBase.labelmap(cats[!, :Sex])
ci = MLBase.labelencode(lmap, cats[!,:Sex])
Plots.scatter(cats[!,:BWt], cats[!, :HWt], color = ci, legend = false)
lmap

data = DataFrames.DataFrame(X = cats[!, :HWt], Y = ci .-1)
probit = GLM.glm(
    GLM.@formula(Y ~ X),
    data,
    GLM.Binomial(),
    GLM.LogitLink())
Plots.scatter(
    data[!, :X],
    data[!, :Y],
    label = "ground truth gender",
    color = 6)
Plots.scatter!(
    data[!, :X],
    GLM.predict(probit),
    label = "predicted gender",
    color = 7)

## Non linear regression
xvals = 0:0.05:10
yvals = 1 * Base.exp.(-xvals * 2) + 2 * Base.sin.(0.8 * pi * xvals) + 0.15 * Base.randn(Base.length(xvals));
Plots.scatter(
    xvals,
    yvals,
    legend = false)

@. model2(x, p) = p[1]*Base.exp(-x*p[2]) + p[3]*Base.sin(0.8*pi*x)
p0 = [0.5, 0.5, 0.5]
myfit = LsqFit.curve_fit(
    model2,
    xvals,
    yvals, p0)

p = myfit.param
findyvals = p[1] * Base.exp.(-xvals*p[2]) + p[3]*Base.sin.(0.8*pi*xvals)
Plots.scatter(xvals,yvals,legend=false)
Plots.plot!(xvals,findyvals)

## Linear regression
@. model3(x, p) = p[1]*x
myfit = LsqFit.curve_fit(model3, xvals, yvals, [0.5])
p = myfit.param
findyvals = p[1] * xvals
Plots.scatter(xvals, yvals, legend = false)
Plots.plot!(xvals, findyvals)
