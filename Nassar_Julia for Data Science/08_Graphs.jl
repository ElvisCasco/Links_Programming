#=
VIDEO 8: GRAPHS
https://juliaacademy.com/courses/937702/lectures/17339543
https://www.youtube.com/watch?v=1AgFyLpM3_4
# https://www.youtube.com/watch?v=1AgFyLpM3_4&list=PLP8iPy9hna6QuDTt11Xxonnfal91JhqjO&index=9
# https://github.com/JuliaAcademy/DataScience/blob/master/08.%20Graphs.ipynb
=#
#########################################
using LightGraphs
using MatrixNetworks
using VegaDatasets
using DataFrames
using SparseArrays
using LinearAlgebra
using Plots
using VegaLite # only for jupyter notebooks

## Get and convert the data
airports = VegaDatasets.dataset("airports")
flightsairport = VegaDatasets.dataset("flights-airport")
Base.typeof(flightsairport)
flightsairportdf = DataFrames.DataFrame(flightsairport)
allairports = DataFrames.vcat(flightsairportdf[!, :origin], flightsairportdf[!, :destination])
uairports = DataFrames.unique(allairports)

# create an airports data frame that has a subset of airports that are only included in the routes dataset
airportsdf = DataFrames.DataFrame(airports)
subsetairports = DataFrames.map(
    i -> Base.findfirst(airportsdf[!, :iata] .== uairports[i]),
    1:Base.length(uairports))
airportsdf_subset = airportsdf[subsetairports, :]

## build the adjacency matrix
ei_ids = DataFrames.findfirst.(Base.isequal.(flightsairportdf[!, :origin]), [uairports])
ej_ids = DataFrames.findfirst.(Base.isequal.(flightsairportdf[!, :destination]), [uairports])
edgeweights = flightsairportdf[!, :count]
A = SparseArrays.sparse(ei_ids, ej_ids, 1, Base.length(uairports), Base.length(uairports))
A = Base.max.(A, A')
Plots.spy(A)
LinearAlgebra.issymmetric(A)

## build a graph via LightGraphs
L = LightGraphs.SimpleGraph(A)
G = LightGraphs.SimpleGraph(10) #SimpleGraph(nnodes,nedges)
LightGraphs.add_edge!(G, 7,5)#modifies graph in place.
LightGraphs.add_edge!(G, 3, 5)
LightGraphs.add_edge!(G, 5, 2)

## scomponents
cc = MatrixNetworks.scomponents(A)
degrees = Base.sum(A, dims = 2)[:]
p1 = Plots.plot(
    Base.sort(degrees, rev = true),
    ylabel = "log degree",
    legend = false,
    yaxis =:log)
p2 = Plots.plot(
    Base.sort(degrees, rev = true),
    ylabel = "degree",
    legend = false)
Plots.plot(p1, p2, size = (600, 300))

## let's find the airport that has the most connections
maxdegreeid = Base.argmax(degrees)
uairports[maxdegreeid]
us10m = VegaDatasets.dataset("us-10m")
VegaLite.@vlplot(
    width=500,
    height=300) +
    VegaLite.@vlplot(
    mark = {
            :geoshape,
            fill = :lightgray,
            stroke = :white
        },
        data = {
            values = us10m,
            format = {
                type = :topojson,
                feature = :states
            }
        },
        projection = {type = :albersUsa},
    ) +
    VegaLite.@vlplot(
        :circle,
        data = airportsdf_subset,
        projection = {type = :albersUsa},
        longitude = "longitude:q",
        latitude = "latitude:q",
        size = {value = 10},
        color = {value = :steelblue}
    )+
    VegaLite.@vlplot(
        :rule,
        data = flightsairport,
        transform = [
            {filter = {field = :origin, equal = :ATL}},
            {
                lookup = :origin,
                from = {
                    data = airportsdf_subset,
                    key = :iata,
                    fields = ["latitude", "longitude"]
                },
                as = ["origin_latitude", "origin_longitude"]
            },
            {
                lookup = :destination,
                from = {
                    data = airportsdf_subset,
                    key = :iata,
                    fields = ["latitude", "longitude"]
                },
                as = ["dest_latitude", "dest_longitude"]
            }
        ],
        projection = {type = :albersUsa},
        longitude = "origin_longitude:q",
        latitude = "origin_latitude:q",
        longitude2 = "dest_longitude:q",
        latitude2 = "dest_latitude:q"
    )

## Shortest path problem
ATL_paths = MatrixNetworks.dijkstra(A, maxdegreeid)
ATL_paths[1][maxdegreeid]
Base.maximum(ATL_paths[1])
Base.@show stop1 = Base.argmax(ATL_paths[1])
Base.@show uairports[stop1]
Base.@show stop2 = ATL_paths[2][stop1]
Base.@show uairports[stop2]
Base.@show stop3 = ATL_paths[2][stop2]
Base.@show uairports[stop3]
Base.@show stop4 = ATL_paths[2][stop3]
Base.@show uairports[stop4]

using VegaLite, VegaDatasets
us10m = VegaDatasets.dataset("us-10m")
airports = VegaDatasets.dataset("airports")
VegaLite.@vlplot(
    width = 800,
    height = 500) +
    VegaLite.@vlplot(
        mark = {
            :geoshape,
            fill = "#eee",
            stroke = :white
        },
        data = {
            values = us10m,
            format = {
                type = :topojson,
                feature = :states
            }
        },
        projection = {type = :albersUsa},
    ) +
    VegaLite.@vlplot(
        :circle,
        data = airportsdf_subset,
        projection = {type = :albersUsa},
        longitude = "longitude:q",
        latitude = "latitude:q",
        size = {value = 5},
        color = {value = :gray}
    ) +
    VegaLite.@vlplot(
        :line,
        data = {
            values = [
                {airport = :ATL, order = 1},
                {airport = :SEA, order = 2},
                {airport = :JNU, order = 3},
                {airport = :GST,order = 4}
            ]
        },
        transform = [{
            lookup = :airport,
            from = {
                data = airports,
                key = :iata,
                fields = ["latitude","longitude"]
            }
        }],
        projection = {type = :albersUsa},
        longitude = "longitude:q",
        latitude = "latitude:q",
        order = {field = :order, type = :ordinal}
    )

## We can look at distances from other airports too, and put the whole thing in a function.
nodeid = Base.argmin(degrees)
Base.@show uairports[nodeid]
d = MatrixNetworks.dijkstra(A, nodeid)
Base.argmax(d[1]), uairports[Base.argmax(d[1])]
function find_path(d, id)
    shortestpath = Base.zeros(Int, 1 + Base.Int.(d[1][id]))
    shortestpath[1] = id
    for i = 2:Base.length(shortestpath)
        shortestpath[i] = d[2][shortestpath[i-1]]
    end
    return shortestpath
end
p = find_path(d, 123)
uairports[p]

## Minimum Spanning Tree (MST)
ti, tj, tv, nverts = MatrixNetworks.mst_prim(A)
df_edges = DataFrames.DataFrame(:ei => uairports[ti], :ej => uairports[tj])
VegaLite.@vlplot(width = 800, height = 500) +
    VegaLite.@vlplot(
        mark = {
            :geoshape,
            fill = "#eee",
            stroke = :white
        },
        data = {
            values = us10m,
            format = {
                type = :topojson,
                feature = :states
            }
        },
        projection = {type = :albersUsa},
    ) +
    VegaLite.@vlplot(
        :circle,
        data = airportsdf_subset,
        projection = {type = :albersUsa},
        longitude = "longitude:q",
        latitude = "latitude:q",
        size = {value = 20},
        color = {value = :gray}
    ) +
    VegaLite.@vlplot(
        :rule,
        data = df_edges, #data=flightsairport,
        transform = [
            {
                lookup = :ei,
                from={
                    data = airportsdf_subset,
                    key = :iata,
                    fields = ["latitude", "longitude"]
                },
                as = ["originx", "originy"]
            },
            {
                lookup = :ej,
                from = {
                    data = airportsdf_subset,
                    key = :iata,
                    fields = ["latitude", "longitude"]
                },
                as = ["destx", "desty"]
            }
        ],
        projection = {type = :albersUsa},
        longitude = "originy:q",
        latitude = "originx:q",
        longitude2 = "desty:q",
        latitude2 = "destx:q"
    )

## PageRank
v = MatrixNetworks.pagerank(A, 0.85)
Base.sum(v)
DataFrames.insertcols!(airportsdf_subset, 7 , :pagerank_value => v)
VegaLite.@vlplot(width = 500, height = 300) +
    VegaLite.@vlplot(
        mark = {
            :geoshape,
            fill = "#eee",
            stroke = :white
        },
        data = {
            values = us10m,
            format = {
                type = :topojson,
                feature = :states
            }
        },
        projection = {type = :albersUsa},
    ) +
    VegaLite.@vlplot(
        :circle,
        data = airportsdf_subset,
        projection = {type = :albersUsa},
        longitude = "longitude:q",
        latitude = "latitude:q",
        size = "pagerank_value:q",
        color = {value = :steelblue}
    )

## Clustering Coefficients
cc = MatrixNetworks.clustercoeffs(A)
cc[Base.findall(cc .<= Base.eps())] .= 0
cc
DataFrames.insertcols!(airportsdf_subset, 7, :ccvalues => cc)
VegaLite.@vlplot(width = 500, height = 300) +
    VegaLite.@vlplot(
        mark = {
            :geoshape,
            fill = "#eee",
            stroke = :white
        },
        data = {
            values = us10m,
            format = {
                type = :topojson,
                feature = :states
            }
        },
        projection = {type = :albersUsa},
    ) +
    VegaLite.@vlplot(
        :circle,
        data = airportsdf_subset,
        projection = {type = :albersUsa},
        longitude = "longitude:q",
        latitude = "latitude:q",
        size = "ccvalues:q",
        color = {value = :gray}
    )
