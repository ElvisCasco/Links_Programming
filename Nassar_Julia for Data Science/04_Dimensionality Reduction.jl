#=
VIDEO 4: DIMENSIONALITY REDUCTION
https://juliaacademy.com/courses/937702/lectures/17339513
https://www.youtube.com/watch?v=hIsYy04zO7U
https://www.youtube.com/watch?v=hIsYy04zO7U&list=PLP8iPy9hna6QuDTt11Xxonnfal91JhqjO&index=5
https://github.com/JuliaAcademy/DataScience/blob/master/04.%20Dimensionality%20Reduction.ipynb
=#
#########################################
# Packages we will use throughout this notebook
# Pkg.add("UMAP")
using UMAP
#Pkg.add("Makie")
using Makie
using XLSX
#Pkg.add("VegaDatasets")
using VegaDatasets
using DataFrames
#Pkg.add("MultivariateStats")
using MultivariateStats
using RDatasets
using StatsBase
using Statistics
using LinearAlgebra
using Plots
#Pkg.add("ScikitLearn")
using ScikitLearn
using MLBase
#Pkg.add("Distances")
using Distances

## Data
C = DataFrames.DataFrame(VegaDatasets.dataset("cars"))
DataFrames.dropmissing!(C)
M = Base.Matrix(C[:,2:7])
Base.names(C)
car_origin = C[:,:Origin]
carmap = MLBase.labelmap(car_origin) #from MLBase
uniqueids = MLBase.labelencode(carmap, car_origin)

## PCA
# center and normalize the data
data = M
data = (data .- StatsBase.mean(data, dims = 1))./ StatsBase.std(data, dims = 1)
# each car is now a column, PCA takes features - by - samples matrix
data'

# First, we will fit the model via PCA. maxoutdim is the output dimensions, we want it to be 2 in this case.
p = MultivariateStats.fit(PCA, data', maxoutdim = 2)
# Projection Matrix
P = MultivariateStats.projection(p)
# Apply to one car
P' * (data[1,:] - StatsBase.mean(p))

# Transform All the Data
Yte = MultivariateStats.transform(p, data') #notice that Yte[:,1] is the same as P'*(data[1,:]-mean(p))

# We can also go back from two dimensions to 6 dimensions (approximation)
# reconstruct testing observations (approximately)
Xr = MultivariateStats.reconstruct(p, Yte)
LinearAlgebra.norm(Xr - data') # this won't be zero

# Scatter Plot
Plots.scatter(Yte[1,:], Yte[2,:])
Plots.scatter(Yte[1, car_origin .== "USA"], Yte[2, car_origin .== "USA"], color = 1, label = "USA")
Plots.xlabel!("pca component1")
Plots.ylabel!("pca component2")
Plots.scatter!(Yte[1, car_origin .== "Japan"], Yte[2, car_origin .== "Japan"], color = 2, label = "Japan")
Plots.scatter!(Yte[1, car_origin .== "Europe"], Yte[2, car_origin .== "Europe"], color = 3, label = "Europe")

# 3 Dimensions
p = MLBase.fit(PCA,data', maxoutdim = 3)
Yte = MultivariateStats.transform(p, data')
Plots.scatter3d(Yte[1,:], Yte[2,:], Yte[3,:], color =uniqueids, legend = false)
# using Makie
#scene = Makie.scatter(Yte[1,:], Yte[2,:], Yte[3,:], color = uniqueids)
#display(scene)
#scene

## t-SNE
# www.github.com/nassarhuda/JuliaTutorials/blob/master/TSNE/TSNE.ipynb
ScikitLearn.@sk_import manifold : TSNE
tfn = TSNE(n_components = 2) #,perplexity=20.0,early_exaggeration=50)
Y2 = tfn.fit_transform(data);
Plots.scatter(
    Y2[:,1],Y2[:,2],
    color=uniqueids,
    legend=false,size=(400,300),markersize=3)

# UMAP
L = StatsBase.cor(data, data, dims = 2)
embedding = UMAP.umap(L, 2)
Plots.scatter(
    embedding[1,:], embedding[2,:], 
    color = uniqueids)

# Euclidean distances
L = Distances.pairwise(
    Distances.Euclidean(), data, data, dims = 1)
embedding = UMAP.umap(-L, 2)
Plots.scatter(
    embedding[1,:], embedding[2,:], 
    color = uniqueids)
