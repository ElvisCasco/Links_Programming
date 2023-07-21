#=
VIDEO 5: CLUSTERING
https://juliaacademy.com/courses/937702/lectures/17339538
https://www.youtube.com/watch?v=cwurgt7cn5s
https://www.youtube.com/watch?v=cwurgt7cn5s&list=PLP8iPy9hna6QuDTt11Xxonnfal91JhqjO&index=6
https://github.com/JuliaAcademy/DataScience/blob/master/05.%20Clustering.ipynb
=#
#########################################
# Packages we will use throughout this notebook
#import Pkg;
#Pkg.add("Clustering")
using Clustering
#Pkg.add("VegaLite")
using VegaLite
using VegaDatasets
using DataFrames
using Statistics
#Pkg.add("JSON")
using JSON
using CSV
using Distances

wd = "C:/Directorio_Trabajo/Aplicaciones_Actual/Nassar_Julia for Data Science/"
# Getting some data
Base.download(
    "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv",
    "Data/newhouses.csv")
houses = CSV.read(
    "Data/newhouses.csv",
    DataFrames.DataFrame)
Base.names(houses)

# JSON data
cali_shape = JSON.parsefile(
    wd * "Data/california-counties.json")
VV = VegaDatasets.VegaJSONDataset(cali_shape,
    wd * "Data/california-counties.json")
VegaLite.@vlplot(width=500, height=300) +
    VegaLite.@vlplot(
        mark = {
            :geoshape,
            fill = :black,
            stroke = :white
            },
        data = {
            values = VV,
        format = {
            type = :topojson,
            feature = :cb_2015_california_county_20m
        }
    },
    projection = {type = :albersUsa},
    )+
    VegaLite.@vlplot(
        :circle,
        data = houses,
        projection = {type = :albersUsa},
        longitude = "longitude:q",
        latitude = "latitude:q",
        size = {value = 12},
        color = "median_house_value:q"
)

# Bucketing
bucketprice = Core.Int.(
    Base.div.(houses[!, :median_house_value],
        50000))
DataFrames.insertcols!(houses,
    3,
    :cprice => bucketprice)
VegaLite.@vlplot(width=500, height=300) +
    VegaLite.@vlplot(
        mark = {
            :geoshape,
            fill = :black,
            stroke = :white
            },
        data = {
            values = VV,
            format = {
                type = :topojson,
                feature = :cb_2015_california_county_20m
                }
            },
        projection = {type = :albersUsa},
    ) +
    VegaLite.@vlplot(
        :circle,
        data = houses,
        projection = {type = :albersUsa},
        longitude = "longitude:q",
        latitude = "latitude:q",
        size = {value = 12},
        color = "cprice:n"
)

## K-Means Clustering
X = houses[!,
    [:latitude, :longitude]]
C = Clustering.kmeans(Matrix(X)',
    10)
DataFrames.insertcols!(houses,
    3,
    :cluster10 => C.assignments)

cali_shape = JSON.parsefile(wd * "Data/california-counties.json")
VegaLite.@vlplot(width=500, height=300) +
    VegaLite.@vlplot(
        mark = {
            :geoshape,
            fill = :black,
            stroke = :white
        },
        data = {
            values = VV,
            format = {
                type = :topojson,
                feature = :cb_2015_california_county_20m
            }
        },
        projection = {type = :albersUsa},
    ) +
    VegaLite.@vlplot(
        :circle,
        data = houses,
        projection = {type = :albersUsa},
        longitude = "longitude:q",
        latitude = "latitude:q",
        size = {value = 12},
        color = "cluster10:n"
)

## K-medoids clustering
xmatrix = Base.Matrix(X)'
D = Distances.pairwise(Distances.Euclidean(),
    xmatrix,
    xmatrix,
    dims = 2)
K = Clustering.kmedoids(D,
    10)
DataFrames.insertcols!(houses,
     3,
     :medoids_clusters => K.assignments)
cali_shape = JSON.parsefile("Data/california-counties.json")
VV = VegaDatasets.VegaJSONDataset(cali_shape,
    "Data/california-counties.json")
VegaLite.@vlplot(width=500,height=300) +
    VegaLite.@vlplot(
        mark = {
            :geoshape,
            fill = :black,
            stroke = :white
        },
        data = {
            values = VV,
            format = {
                type = :topojson,
                feature = :cb_2015_california_county_20m
            }
        },
        projection = {type = :albersUsa},
    ) +
    VegaLite.@vlplot(
        :circle,
        data = houses,
        projection = {type = :albersUsa},
        longitude = "longitude:q",
        latitude = "latitude:q",
        size = {value = 12},
        color = "cluster10:n"
)

## Hierarchial Clustering
K = Clustering.hclust(D)
L = Clustering.cutree(K;
    k =10)
DataFrames.insertcols!(houses,
    3,
    :hclust_clusters => L)
VegaLite.@vlplot(width=500, height=300) +
    VegaLite.@vlplot(
        mark = {
            :geoshape,
            fill = :black,
            stroke = :white
        },
        data = {
            values = VV,
            format = {
                type = :topojson,
                feature = :cb_2015_california_county_20m
            }
        },
        projection = {type = :albersUsa},
    ) +
    VegaLite.@vlplot(
        :circle,
        data = houses,
        projection = {type = :albersUsa},
        longitude = "longitude:q",
        latitude = "latitude:q",
        size = {value = 12},
        color = "hclust_clusters:n"
    )

## DBscan
using Distances
dclara = Distances.pairwise(Distances.SqEuclidean(),
    Base.Matrix(X)',
    dims = 2)
L = Clustering.dbscan(dclara, 0.05, 10)
Base.@show Base.length(Base.unique(L.assignments))
DataFrames.insertcols!(houses,
    3,
    :dbscanclusters3 => L.assignments)
VegaLite.@vlplot(
    width = 500,
    height = 300) +
    VegaLite.@vlplot(
        mark = {
            :geoshape,
            fill = :black,
            stroke = :white
        },
        data = {
            values = VV,
            format = {
                type = :topojson,
                feature = :cb_2015_california_county_20m
            }
        },
        projection = {type = :albersUsa},
    ) +
    VegaLite.@vlplot(
        :circle,
        data = houses,
        projection = {type = :albersUsa},
        longitude = "longitude:q",
        latitude = "latitude:q",
        size = {value = 12},
        color = "dbscanclusters3:n"
    )
