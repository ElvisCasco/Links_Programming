## Libros para descargar (5 diarios)
#https://b-ok.lat/dl/3685042/c217ab?dsource=recommend
#https://www.universitygames.com/instructions
# Introduction to Computational Thinking
# https://github.com/mitmath/6S083
# https://github.com/nassarhuda/PyDataChi2016
# https://www.youtube.com/watch?v=I6cfml5VLRg   # Data Analysis of Coronavirus Outbreak
# http://jcharistech-institute.thinkific.com/courses/take/learn-julia-programming-fundamentals-of-julia-lang/lessons/5081064-strings-characters
# https://riptutorial.com/julia-lang/awesome-learning/youtube
# www.github.com/nassarhuda/JuliaTutorials/blob/master/TSNE/TSNE.ipynb
# https://vega.github.io/vega-lite/

## Intro to Julia
# https://www.youtube.com/watch?v=8mZRIRHAZfo&list=PLP8iPy9hna6RcodntjrZmz7rjL-gVXl_K&index=13&app=desktop
# Julia for Data Science by Huda Nassar

#=
VIDEO 1: DATA
https://juliaacademy.com/courses/937702/lectures/17339299
https://www.youtube.com/watch?v=iG1dZBaxS-U
https://www.youtube.com/watch?v=iG1dZBaxS-U&list=PLP8iPy9hna6QuDTt11Xxonnfal91JhqjO&index=2
https://github.com/JuliaAcademy/DataScience/blob/master/01.%20Data.ipynb
=#
#########################################

wd = "C:/Directorio_Trabajo/Aplicaciones_Actual/Nassar_Julia for Data Science/"

## Data
#t = rmprocs(2, waitfor=0)
#workers()
using BenchmarkTools
using DataFrames
using DelimitedFiles
using CSV
using XLSX

## Get some Data, using download
#? download # digitar en REPL
P = Base.download("https://raw.githubusercontent.com/nassarhuda/easy_Data/master/programming_languages.csv",
    "Data/programminglanguages.csv")
P,H = DelimitedFiles.readdlm(P,','; header = true)
H
P
Base.@show Base.typeof(P)
P[1:10,:]

# To write to a text file, you can:
DelimitedFiles.writedlm(
    "Data/programminglanguages_dlm.txt",
    P,
    '-')

## Read csv Files
#C = CSV.read("programminglanguages.csv", DataFrames.DataFrame);
C = CSV.read(
    "Data/programminglanguages.csv",
    DataFrames.DataFrame)
CSV.@show Base.typeof(C)
C[1:10,:]
Base.names(C)
C.year
C[:,:year]
C.language
DataFrames.describe(C)

# To write to a *.csv file using the CSV package
CSV.write(
    "Data/programminglanguages_CSV.csv",
    DataFrames.DataFrame(P))

# Compare Performance
BenchmarkTools.@btime P,H = DelimitedFiles.readdlm(
    "Data/programminglanguages.csv",
    ',';
    header=true);
BenchmarkTools.@btime C = CSV.read(
    "Data/programminglanguages.csv",
    DataFrames.DataFrame);

## Read xlsx Files
T = XLSX.readdata(
    wd * "Data/zillow_data_download_april2020.xlsx", #file name
    "Sale_counts_city", #sheet name
    "A1:F9" #cell range
    )
# taking longer time
G = XLSX.readtable(
    wd * "Data/zillow_data_download_april2020.xlsx",
    "Sale_counts_city")
#G[1]
#G[1][1][1:10]
#G[2][1:6]
#D = DataFrames.DataFrame(G...) # equivalent to DataFrame(G[1],G[2])
D = DataFrames.DataFrame(T) # equivalent to DataFrame(G[1],G[2])

## Fill info from DataFrame
foods = ["apple", "cucumber", "tomato", "banana"]
calories = [105,47,22,105]
prices = [0.85,1.6,0.8,0.6,]
Dataframe_calories = DataFrames.DataFrame(item = foods, calories = calories)
Dataframe_prices = DataFrames.DataFrame(item = foods, price = prices)

## Join DataFrames
#DF = DataFrames.join(Dataframe_calories,Dataframe_prices,on=:item,kind=:inner)
DF = DataFrames.innerjoin(Dataframe_calories, Dataframe_prices, on = :item)

## we can also use the DataFrame constructor on a Matrix
DataFrames.DataFrame(T)

## if you already have a Dataframe: inefficient
# XLSX.writetable("filename.xlsx", collect(DataFrames.eachcol(df)), DataFrames.names(df))
#XLSX.writetable("writefile_using_XLSX.xlsx",G[1],G[2])

## Importing your Data
## JLD
#import Pkg
#Pkg.add("JLD")
#Pkg.build("HDF5")
using JLD
jld_Data = JLD.load(
    wd * "Data/mytempData.jld")
JLD.save(
    wd * "Data/mywrite.jld", 
    "A", 
    jld_Data)

## NPZ
#Pkg.add("NPZ")
using NPZ
npz_Data = NPZ.npzread(
    wd * "Data/mytempData.npz")
NPZ.npzwrite(
    wd * "Data/mywrite.npz", 
    npz_Data)

## RData
#Pkg.rm("RData")
#Pkg.add("RData")
##Pkg.build("CodecZlib")
using RData
R_Data = RData.load(
    wd * "Data/mytempData.rda")
# We'll need RCall to save here. https://github.com/JuliaData/RData.jl/issues/56
#Pkg.add("RCall")
using RCall
RCall.@rput R_Data
RCall.R"save(R_Data, file=\"Data/mywrite.rda\")"

## MAT
#Pkg.add("MAT")
using MAT
Matlab_Data = MAT.matread(
    wd * "Data/mytempData.mat")
MAT.matwrite(
    wd * "Data/mywrite.mat",
    Matlab_Data)

Base.@show Base.typeof(jld_Data)
Base.@show Base.typeof(npz_Data)
Base.@show Base.typeof(R_Data)
Base.@show Base.typeof(Matlab_Data)

Matlab_Data

# Time to Process the Data from Julia
P

## Q1: Which year was was a given language invented?
function year_created(P,language::String)
    loc = DataFrames.findfirst(P[:,2] .== language)
    return P[loc,1]
end
year_created(P,"Julia")
#year_created(P,"W")

function year_created_handle_error(P,language::String)
    loc = DataFrames.findfirst(P[:,2] .== language)
    !Base.isnothing(loc) && return P[loc,1]
    Base.error("Error: Language not found.")
end
#year_created_handle_error(P,"W")

## Q2: How many languages were created in a given year?
function how_many_per_year(P,year::Int64)
    year_count = Base.length(Base.findall(P[:,1].==year))
    return year_count
end
how_many_per_year(P,2011)

P_df = C #DataFrame(year = P[:,1], language = P[:,2]) # or DataFrame(P)

## Q1: Which year was was a given language invented?
# it's a little more intuitive and you don't need to remember the column ids
function year_created(P_df,language::String)
    loc = Base.findfirst(P_df.language .== language)
    return P_df.year[loc]
end
year_created(P_df,"Julia")

## Q2: How many languages were created in a given year?
function how_many_per_year(P_df,year::Int64)
    year_count = Base.length(Base.findall(P_df.year.==year))
    return year_count
end
how_many_per_year(P_df,2011)

## Dictionaries
# A quick example to show how to build a dictionary
Base.Dict([("A", 1), ("B", 2),(1,[1,2])])

P_dictionary = Base.Dict{Integer,Vector{String}}()
dict = Base.Dict{Integer,Vector{String}}()
for i = 1:Base.size(P,1)
    year,lang = P[i,:]
    if year in Base.keys(dict)
        dict[year] = Base.push!(dict[year],lang)
        # note that push! is not our favorite thing do in Julia,
        # but we're focusing on correctness rather than speed here
    else
        dict[year] = [lang]
    end
end

# Though a smarter way to do this is:
curyear = P_df.year[1]
P_dictionary[curyear] = [P_df.language[1]]
for (i,nextyear) in Base.enumerate(P_df.year[2:end])
    #if nextyear == curyear
    #    same key
    #    P_dictionary[curyear] = push!(P_dictionary[curyear],P_df.language[i+1])
        # note that push! is not our favorite thing do in Julia,
        # but we're focusing on correctness rather than speed here
    #else
        local curyear = nextyear
        P_dictionary[curyear] = [P_df.language[i+1]]
    #end
end
Base.length(Base.keys(P_dictionary))
Base.length(Base.unique(P[:,1]))

## Q1: Which year was was a given language invented?
# now instead of looking in one long vector, we will look in many small vectors
function year_created(P_dictionary,language::String)
    keys_vec = Base.collect(Base.keys(P_dictionary))
    lookup = Base.map(keyid -> Base.findfirst(P_dictionary[keyid].==language),keys_vec)
    # now the lookup vector has `nothing` or a numeric value. We want to find the index of the numeric value.
    return keys_vec[Base.findfirst((!isnothing).(lookup))]
end
year_created(P_dictionary,"Julia")


## Q2: How many languages were created in a given year?
how_many_per_year(P_dictionary,year::Int64) = Base.length(P_dictionary[year])
how_many_per_year(P_dictionary,2011)

## A note about missing Data
# assume there were missing values in our Dataframe
P[1,1] = missing
P_df = DataFrames.DataFrame(year = P[:,1], language = P[:,2])
DataFrames.dropmissing(P_df)
