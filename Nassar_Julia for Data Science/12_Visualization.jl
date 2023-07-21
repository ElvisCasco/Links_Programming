#########################################
# VIDEO 12: VISUALIZATION
# https://juliaacademy.com/courses/937702/lectures/17339571
# https://www.youtube.com/watch?v=-XvtUiNrCYM
# https://www.youtube.com/watch?v=-XvtUiNrCYM&list=PLP8iPy9hna6QuDTt11Xxonnfal91JhqjO&index=13
# https://github.com/JuliaAcademy/DataScience/blob/master/12.%20Visualization.ipynb
#########################################
#=
Debe instalarse node.js  y npm primero
https://www.npmjs.com/get-npm
luego, con cmd>
npm install vega
npm install vega-lite
npm install vega-embed

Si no funciona, en REPL:
]rm VegaLite
]add VegaLite
using VegaLite, VegaDatasets
VegaDatasets.dataset("cars") |>
    @vlplot(
        :point,
        x=:Horsepower,
        y=:Miles_per_Gallon,
        color=:Origin,
        width=400,
        height=400
)

dataset
=#
using VegaLite, VegaDatasets
VegaDatasets.dataset("cars") |>
    VegaLite.@vlplot(
        :point,
        x= :Horsepower,
        y= :Miles_per_Gallon,
        color= :Origin,
        width = 400,
        height = 400
)

ENV["GKS_ENCODING"] = "utf-8"
stateabbreviations = Base.Dict("Alabama" => "AL",
    "Alaska" => "AK",
    "Arizona" => "AZ",
    "Arkansas" => "AR",
    "California" => "CA",
    "Colorado" => "CO",
    "Connecticut" => "CT",
    "Delaware" => "DE",
    "Florida" => "FL",
    "Georgia" => "GA",
    "Hawaii" => "HI",
    "Idaho" => "ID",
    "Illinois" => "IL",
    "Indiana" => "IN",
    "Iowa" => "IA",
    "Kansas" => "KS",
    "Kentucky" => "KY",
    "Louisiana" => "LA",
    "Maine" => "ME",
    "Maryland" => "MD",
    "Massachusetts" => "MA",
    "Michigan" => "MI",
    "Minnesota" => "MN",
    "Mississippi" => "MS",
    "Missouri" => "MO",
    "Montana" => "MT",
    "Nebraska" => "NE",
    "Nevada" => "NV",
    "New Hampshire" => "NH",
    "New Jersey" => "NJ",
    "New Mexico" => "NM",
    "New York" => "NY",
    "North Carolina" => "NC",
    "North Dakota" => "ND",
    "Ohio" => "OH",
    "Oklahoma" => "OK",
    "Oregon" => "OR",
    "Pennsylvania" => "PA",
    "Rhode Island" => "RI",
    "South Carolina" => "SC",
    "South Dakota" => "SD",
    "Tennessee" => "TN",
    "Texas" => "TX",
    "Utah" => "UT",
    "Vermont" => "VT",
    "Virginia" => "VA",
    "Washington" => "WA",
    "West Virginia" => "WV",
    "Wisconsin" => "WI",
    "Wyoming" => "WY",
    "District of Columbia"=>"DC")

#=
Pkg.rm("Plots")
Pkg.gc()
Pkg.add("Plots")
Pkg.rm("FFMPEG")
Pkg.gc()
Pkg.add("FFMPEG")
Pkg.build("FFMPEG")
=#
using Plots
using StatsPlots # this package provides stats specific plotting functions
Plots.gr()

using Statistics
using StatsBase
using MLBase

xtickslabels = ["one","five","six","fourteen"]
p = Plots.plot(
    Base.rand(15),
    xticks = ([1,5,6,14], xtickslabels),
    xrotation = 90,
    xtickfont = Plots.font(13))

function pad_empty_plot(p)
    ep = Plots.plot(
        grid = false,
        legend = false,
        axis = false,
        framestyle = :box)#empty plot
    newplot = Plots.plot(
        p,
        ep,layout = Plots.@layout([a{0.99h}; b{0.001h}]))
    return newplot
end
pad_empty_plot(p)

## Let us first get some data that we will use throughout this notebook
using XLSX
using DataFrames
D = DataFrames.DataFrame(
    #XLSX.readtable("data/zillow_data_download_april2020.xlsx", "Sales_median_price_city")...);
    XLSX.readtable(wd * "data/zillow_data_download_april2020.xlsx", "Sales_median_price_city"));
DataFrames.dropmissing!(D)
states = D[:, :StateName]

NYids = Base.findall(states .== "New York")
NYframe = DataFrames.dropmissing(D[NYids, :])
CAids = Base.findall(states .== "California")
CAframe = DataFrames.dropmissing(D[CAids,:])
FLids = Base.findall(states .== "Florida")
FLframe = DataFrames.dropmissing(D[FLids,:])

## Plot 1: Symmetric violin plots and annotations
# pick a year: 2020-02
ca = CAframe[!, Symbol("2020-02")]
ny = NYframe[!, Symbol("2020-02")]
fl = FLframe[!, Symbol("2020-02")]

StatsPlots.violin(
    ["New York"],
    ny,
    legend = false,
    alpha = 0.8)
StatsPlots.violin!(
    ["California"],
    ca,
    alpha = 0.8)
StatsPlots.violin!(
    ["Florida"],
    fl,
    alpha = 0.8)

# 2020 data
ca = CAframe[!, Symbol("2020-02")]
ny = NYframe[!, Symbol("2020-02")]
fl = FLframe[!, Symbol("2020-02")]
StatsPlots.violin(
    ["New York"],
    ny,
    legend = false,
    alpha = 0.8,
    side = :right)
StatsPlots.violin!(
    ["California"],
    ca,
    alpha = 0.8,
    side = :right)
StatsPlots.violin!(
    ["Florida"],
    fl,
    alpha = 0.8,
    side = :right)

### get the February 2010 data
ca10 = CAframe[!, Symbol("2010-02")]
ny10 = NYframe[!, Symbol("2010-02")]
fl10 = FLframe[!, Symbol("2010-02")]

StatsPlots.violin!(
    ["New York"],
    ny10,
    legend = false,
    alpha = 0.8,
    side = :left)
StatsPlots.violin!(
    ["California"],
    ca10,
    alpha = 0.8,
    side = :left)
StatsPlots.violin!(
    ["Florida"],
    fl10,
    alpha = 0.8,
    side = :left)

# No need for using many colors, let's just use one color for 2010, and one color for 2020

# pick a year: 2019-02
ca = CAframe[!, Symbol("2010-02")]
ny = NYframe[!, Symbol("2010-02")]
fl = FLframe[!, Symbol("2010-02")]
StatsPlots.violin(
    ["New York"],
    ny,
    alpha = 0.8,
    side = :left,
    color = 6,
    label = "2010-02")
StatsPlots.violin!(
    ["California"],
    ca,
    alpha = 0.8,
    side = :left,
    color = 6,
    label = "")
StatsPlots.violin!(
    ["Florida"],
    fl,
    alpha = 0.8,
    side = :left,
    color = 6,
    label = "")

# pick a year: 2020-02
ca = CAframe[!, Symbol("2020-02")]
ny = NYframe[!, Symbol("2020-02")]
fl = FLframe[!, Symbol("2020-02")]
StatsPlots.violin!(
    ["New York"],
    ny,
    alpha = 0.8,
    side = :right,
    color = 7,
    label = "2020-02")
StatsPlots.violin!(
    ["California"],
    ca,
    alpha = 0.8,
    side = :right,
    color = 7,
    label = "")
StatsPlots.violin!(
    ["Florida"],
    fl,
    alpha = 0.8,
    side = :right,
    color = 7,
    label = "")

# pick a year: 2019-02
ca = CAframe[!, Symbol("2010-02")]
ny = NYframe[!, Symbol("2010-02")]
fl = FLframe[!, Symbol("2010-02")]
StatsPlots.violin(
    ["New York"],
    ny,
    alpha = 0.8,
    side = :left,
    color = 6,
    label = "2010-02")
StatsPlots.violin!(
    ["California"],
    ca,
    alpha = 0.8,
    side = :left,
    color = 6,
    label = "")
StatsPlots.violin!(
    ["Florida"],
    fl,
    alpha = 0.8,
    side = :left,
    color = 6,
    label = "")

# pick a year: 2020-02
ca = CAframe[!, Symbol("2020-02")]
ny = NYframe[!, Symbol("2020-02")]
fl = FLframe[!, Symbol("2020-02")]
StatsPlots.violin!(
    ["New York"],
    ny,
    alpha = 0.8,
    side = :right,
    color = 7,
    label = "2020-02")
StatsPlots.violin!(
    ["California"],
    ca,
    alpha = 0.8,
    side = :right,
    color = 7,
    label = "")
StatsPlots.violin!(
    ["Florida"],
    fl,
    alpha = 0.8,
    side = :right,
    color = 7,
    label = "")

m = Statistics.median(ny)
ep = 0.1
StatsPlots.annotate!(
    [(0.5+ep, m+0.05,
    StatsPlots.text(m/1000,10,:left))])

m = Statistics.median(ca)
ep = 0.1
StatsPlots.annotate!(
    [(1.5 + ep, m + 0.05,
    StatsPlots.text(m/1000,10, :left))])

m = Statistics.median(fl)
ep = 0.1
StatsPlots.annotate!(
    [(2.5 + ep, m + 0.05,
    StatsPlots.text(m/1000,10, :left))])

StatsPlots.plot!(
    xtickfont = StatsPlots.font(10),
        size=(500,300))

# putting it together.

ep = 0.05 # will later be used in padding for annotations

# set up the plot
StatsPlots.plot(xtickfont = StatsPlots.font(10))

states_of_interest = ["New York", "California", "Florida", "Ohio","Idaho"]
years_of_interst = [Symbol("2010-02"), Symbol("2020-02")]

# year 1
global xstart = 0.5
yi = years_of_interst[1]
for si in states_of_interest
    global xstart
    curids = Base.findall(states.==si)
    curFrame = D[curids,:]
    curprices = curFrame[!,yi]
    local m = Statistics.median(curprices)
    StatsPlots.annotate!(
        [(xstart-ep,m+0.05,
        StatsPlots.text(m/1000,8,:right))])
    xstart += 1
    StatsPlots.violin!(
        [si],
        curprices,
        alpha = 0.8,
        side = :left,
        color = 6,
        label = "")
end
StatsPlots.plot!(
    #Shape([],[]), 
    color = 6, label = yi)

# year 2
global xstart = 0.5
yi = years_of_interst[2]
for si in states_of_interest
    global xstart
    curids = Base.findall(states.==si)
    curFrame = D[curids,:]
    curprices = curFrame[!,yi]
    local m = Statistics.median(curprices)
    StatsPlots.annotate!(
        [(xstart+ep,m+0.05,
        StatsPlots.text(m/1000,8,:left))])
    xstart += 1
    StatsPlots.violin!(
        [si],
        curprices,
        alpha = 0.8,
        side = :right,
        color = 7,
        label = "")
end
StatsPlots.plot!(
    #Shape([],[]),
    color = 7,
    label = yi)
StatsPlots.ylabel!("housing prices")

## Plot 2: Bar charts, histograms, and insets
mapstates = MLBase.labelmap(states)
stateids = MLBase.labelencode(mapstates, states)
StatsPlots.histogram(
    stateids,
    nbins = Base.length(mapstates))

# first we'll start with sorting
h = MLBase.fit(
    Histogram,
    stateids,
    nbins = Base.length(mapstates))
sortedids = Base.sortperm(h.weights, rev = true)
StatsPlots.bar(h.weights[sortedids], legend = false)

StatsPlots.bar(
    h.weights[sortedids],
    legend = false,
    orientation = :horizontal,
    yflip = true)

# just an example of annotations
StatsPlots.bar(
    h.weights[sortedids],
    legend = false,
    orientation = :horizontal,
    yflip = true,
    size = (400,500))
stateannotations = mapstates.vs[sortedids]
for i = 1:3
    StatsPlots.annotate!(
        [(h.weights[sortedids][i] - 5,
        i,
        StatsPlots.text(stateannotations[i], 10, :left))])
end
StatsPlots.plot!()

StatsPlots.bar(
    h.weights[sortedids],
    legend = false,
    orientation = :horizontal,
    yflip = true,
    linewidth = 0,
    width = 0,
    size = (400, 500))
stateannotations = mapstates.vs[sortedids]
for i = 1:Base.length(stateannotations)
    StatsPlots.annotate!(
        [(h.weights[sortedids][i] - 5,
        i,
        StatsPlots.text(stateabbreviations[stateannotations[i]], 5, :left))])
end
StatsPlots.plot!()

StatsPlots.bar(
    h.weights[sortedids],
    legend = false,
    orientation = :horizontal,
    yflip = true,
    linewidth = 0,
    width = 0,
    color = :gray,
    alpha = 0.8)
stateannotations = mapstates.vs[sortedids]
for i = 20:20:200
    StatsPlots.plot!(
        [i, i],
        [50, 0],
        color = :white)
end
for i = 1:Base.length(stateannotations)
    StatsPlots.annotate!(
        [(h.weights[sortedids][i] - 5,
        i,
        StatsPlots.text(stateabbreviations[stateannotations[i]], 6, :left))])
end
StatsPlots.plot!(
    grid = false,
    yaxis = false,
    xlim = (0, Base.maximum(h.weights)),
    xticks = 0:20:200)
StatsPlots.xlabel!("number of listings")

StatsPlots.bar(
    h.weights[sortedids],
    legend = false,
    orientation = :horizontal,
    yflip = true,
    linewidth = 0,
    color = :gray,
    alpha = 0.8,
    size = (300, 500))
stateannotations = mapstates.vs[sortedids]
ht = Base.length(h.weights)
for i = 20:20:200
    StatsPlots.plot!(
        [i, i],
        [ht, 0],
        color = :white)
end
for i = 1:Base.length(stateannotations)
    StatsPlots.annotate!(
        [(h.weights[sortedids][i] + 2,
        i,
        StatsPlots.text(stateabbreviations[stateannotations[i]], 6, :left))])
end
StatsPlots.plot!(
    grid = false,
    yaxis = false,
    xlim = (0, StatsPlots.maximum(h.weights) + 5),
    xticks = 0:20:200)
StatsPlots.xlabel!("number of listings")

f = Plots.plot!(inset = Plots.bbox(0.7,0.15,0.25,0.6,:top,:left))
Plots.bar!(
    f[2],
    h.weights[sortedids][21:end],
    legend = false,
    orientation = :horizontal,
    yflip = true,
    linewidth = 0,
    width = 0,
    color = :gray,
    alpha = 0.8)
for i = 21:Base.length(stateannotations)
    Plots.annotate!(
        f[2],
        [(h.weights[sortedids][i] + 1,
        i - 20,
        Plots.text(stateabbreviations[stateannotations[i]], 6, :left))])
end
Plots.plot!(
    f[2],
    [10, 10],
    [20, 0],
    color = :white,
    xticks = 0:10:20,
    yaxis = false,
    grid = false,
    xlim = (0, 20))
Plots.plot!()

## Plot 3: Plots with error bars
M = Base.Matrix(NYframe[:,5:end])
xtickslabels = Base.string.(Base.names(NYframe[!,5:end]))

Plots.plot()
for i = 1:Base.size(M,1)
    Plots.plot!(
        M[i, :],
        legend = false)
end
Plots.plot!()
p = Plots.plot!(
    xticks = (1:4:Base.length(xtickslabels),
        xtickslabels[1:4:end]),
        xrotation = 90,
        xtickfont = Plots.font(8),
        grid=false)
pad_empty_plot(p)

function find_percentile(M, pct)
    r = Base.zeros(Base.size(M,2))
    for i = 1:Base.size(M,2)
        v = M[:,i]
        len = Base.length(v)
        ind = Base.floor(Int64,pct*len)
        newarr = Base.sort(v);
        r[i] = newarr[ind];
    end
    return r
end

md = find_percentile(M, 0.5)
mx = find_percentile(M, 0.8)
mn = find_percentile(M, 0.2)
Plots.plot(
    md,
    ribbon = (md .-mn, mx.-md),
    color = :blue,
    label = "NY",
    grid = false)
p = Plots.plot!(
    xticks = (1:4:Base.length(xtickslabels),
        xtickslabels[1:4:end]),
        xrotation = 90,
        xtickfont = Plots.font(8))
pad_empty_plot(p)

function plot_individual_state!(plotid,statevalue,colorid)
    curids = Base.findall(states.==statevalue)
    curFrame = D[curids,:]
    M = Base.Matrix(curFrame[:, 5:end])
    md = find_percentile(M, 0.5)
    mx = find_percentile(M, 0.8)
    mn = find_percentile(M, 0.2)
    Plots.plot!(
        plotid,
        md,
        ribbon =(md .- mn, mx .- md),
        color = colorid,
        label = stateabbreviations[statevalue],
        grid = false)
    Plots.plot!(
        plotid,xticks = (1:4:Base.length(xtickslabels),
            xtickslabels[1:4:end]),
            xrotation = 90,
            xtickfont = font(8))
end

plotid = Plots.plot()
plot_individual_state!(plotid, "Indiana", 1)
plot_individual_state!(plotid, "Ohio", 2)
plot_individual_state!(plotid, "Idaho", 3)
# plot_individual_state!(plotid,"California",4)
Plots.ylabel!("prices")
pad_empty_plot(plotid)

## Plot 4: Plots with double axes
vector1 = Base.rand(10)
vector2 = Base.rand(10)*100
Plots.plot(
    vector1,
    label = "b",
    size = (300, 200))
Plots.plot!(
    Plots.twinx(),
    vector2,
    color = 2,
    axis = false)

xtickslabels = NYframe[!,:RegionName]

sz = NYframe[!, :SizeRank]
pc = NYframe[!, end]
M = Base.Matrix(NYframe[:, 5:end])
M = Base.copy(M')
md = find_percentile(M, 0.9)

md = find_percentile(M, 0.5)
mx = find_percentile(M, 0.9)
mn = find_percentile(M, 0.1)
vector1 = sz

Plots.plot()
Plots.plot!(
    md,
    ribbon = (md .- mn, mx .- md),
    color = 1,
    grid = false,
    label = "")

Plots.plot!(
    xticks = (1:Base.length(xtickslabels),
        xtickslabels),
    xrotation = 90,
    xtickfont = font(10))
Plots.plot!(
    Plots.twinx(),
    vector1,
    color = 2,
    label = "",
    ylabel = "rank",
    grid = false,
    xticks = [],
    linewidth = 2)
Plots.plot!(
    #Plots.Shape([],[]),
    color = 1,
    label = "Prices (left)")
p = Plots.plot!(
    [],
    [],
    color = 2,
    label = "Rank (right)")
ep = Plots.plot(
    grid = false,
    legend = false,
    axis = false,
    framestyle = :box)#empty plot
Plots.plot(
    p,
    ep,
    layout = Plots.@layout([a{0.85h}; b{0.001h}]))

## Plot 5: High-dimensional data in a 2D plot
CA202002 = CAframe[!, Symbol("2020-02")]
CA201002 = CAframe[!, Symbol("2010-02")]
Plots.scatter(CA201002, CA202002)

CA202002 = CAframe[!, Symbol("2020-02")]
CA201002 = CAframe[!, Symbol("2010-02")]
CAranks = CAframe[!, :SizeRank]
Plots.scatter(
    CA201002,
    CA202002,
    legend = false,
    markerstrokewidth = 0,
    markersize = 3,
    alpha = 0.6,
    grid = false)

using ColorSchemes
# normalize the ranks to be between 0 and 1
continuousranks = CAranks ./ Base.maximum(CAranks)

# create a placeholder vector that will store the color of each value
colorsvec = Base.Vector{RGB{Float64}}(undef, Base.length(continuousranks))

# and finally map the colors according to ColorSchemes.autumn1, there are many other schemes you can choose from
Base.map(
    i -> colorsvec[i] = get(ColorSchemes.autumn1, continuousranks[i]),
        1:Base.length(colorsvec))

continuousdates = CAranks ./ Base.maximum(CAranks)
colorsvec = Vector{RGB{Float64}}(undef,length(continuousdates))
Base.map(
    i -> colorsvec[i] = get(ColorSchemes.autumn1,
        continuousdates[i]),
    1:length(colorsvec))
StatsPlots.scatter(
    CA201002,
    CA202002,
    color = colorsvec,
    legend = false,
    markerstrokewidth = 0,
    markersize = 3,
    grid = false)
StatsPlots.xlabel!(
    "2010-02 prices",
    xguidefontsize = 10)
StatsPlots.ylabel!(
    "2020-02 prices",
    yguidefontsize = 10)
p1 = StatsPlots.plot!()

#set up the plot canvas
xvals = 0:100
s = StatsPlots.Shape([0,1,1,0], [0,0,1,1])
StatsPlots.plot(
    s,
    color = ColorSchemes.autumn1[1],
    grid = false,
    axis = false,
    legend = false,
    linewidth = 0,
    linecolor = nothing)

for i = 2:101
    s = StatsPlots.Shape([xvals[i],
        xvals[i] + 1,
        xvals[i] + 1,
        xvals[i]],
        [0,0,1,1])
    StatsPlots.plot!(
        s,
        color = ColorSchemes.autumn1[i],
        grid = false,
        axis = false,
        legend = false,
        linewidth = 0,
        linecolor = nothing)
end

mynormalizer = Base.maximum(CAranks)
xtickslabels = 0:Base.div(mynormalizer,10):mynormalizer
continuousdates = xtickslabels ./ mynormalizer
xticksloc = Base.round.(Int,continuousdates.*101)

# annotate using the ranks
rotatedfont = StatsPlots.font(
    10,
    "Helvetica",
    rotation=90)
for i = 1:Base.length(xtickslabels)
    StatsPlots.annotate!(
        xticksloc[i],
        0.5,
        StatsPlots.text(
            xtickslabels[i], rotatedfont))
end
p2 = StatsPlots.plot!()

mylayout = StatsPlots.@layout([a{0.89h}; b{0.1h}])
StatsPlots.plot(
    p1,
    p2,
    layout = mylayout
    )
