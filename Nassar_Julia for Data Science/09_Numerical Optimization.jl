#=
VIDEO 9: OPTIMIZATION
https://juliaacademy.com/courses/937702/lectures/17339544
https://www.youtube.com/watch?v=7b9b6glGnjA
https://www.youtube.com/watch?v=7b9b6glGnjA&list=PLP8iPy9hna6QuDTt11Xxonnfal91JhqjO&index=10
https://github.com/JuliaAcademy/DataScience/blob/master/09.%20Numerical%20Optimization.ipynb
=#
#########################################
using Pkg
#=
en Julia
]add https://github.com/JuliaIO/CodecZlib.jl
=#
using Convex
using SCS
using XLSX
using DataFrames
using Plots
using CSV
using Statistics
using Images
using DelimitedFiles

## Problem 1 Portfolio investment.
#T = DataFrames.DataFrame(XLSX.readtable("Data/stock_prices.xlsx","Sheet2")...)
T = XLSX.readtable(
    "Data/stock_prices.xlsx",
    "Sheet2")
T = DataFrames.DataFrame(T)
Plots.plot(
    T[!,:MSFT],
    label="Microsoft")
Plots.plot!(
    T[!,:AAPL],
    label="Apple")
Plots.plot!(
    T[!,:FB],
    label="FB")

# convert the prices to a Matrix to be used later in the optimization problem
prices_matrix = Base.Matrix(T)

M1 = prices_matrix[1:end-1,:]
M2 = prices_matrix[2:end,:]
R = (M2.-M1)./M1
risk_matrix = Statistics.cov(R)

# note that the risk matrix is positive definite
Convex.isposdef(risk_matrix)

r = Statistics.mean(R,dims=1)[:]

x = Convex.Variable(Base.length(r))
problem = Convex.minimize(x'*risk_matrix*x,[Base.sum(x)==1;r'*x>=0.02;x.>=0])

# make the problem DCP compliant
problem = Convex.minimize(Convex.quadform(x,risk_matrix),[Base.sum(x)==1;r'*x>=0.02;x.>=0])

Convex.solve!(problem, SCS.Optimizer)
x
Base.sum(x.value)
# return
r'*x.value
x.value .* 1000

## Problem 2 Image recovery.
Kref = Images.load("Data/khiam-small.jpg")

K = Base.copy(Kref)
p = Base.prod(Base.size(K))
missingids = Base.rand(1:p,400)
K[missingids] .= Images.RGBX{N0f8}(0.0,0.0,0.0)
K
Images.Gray.(K)
Y = Float64.(Images.Gray.(K));
correctids = Base.findall(Y[:].!=0)
X = Convex.Variable(Base.size(Y))
problem = Convex.minimize(Convex.nuclearnorm(X))
problem.constraints += X[correctids]==Y[correctids]

Convex.solve!(problem, SCS.Optimizer())#eps=1e-3, alpha=1.5))

Base.@show Convex.norm(Base.float.(Images.Gray.(Kref))-X.value)
Base.@show Convex.norm(-X.value)
Images.colorview(Gray, X.value)

Base.@show Convex.norm(Base.float.(Images.Gray.(Kref))-X.value)
Base.@show Convex.norm(-X.value)
Images.colorview(Gray, X.value)

## Problem 3 Diet optimization problem.
using JuMP
using GLPK

category_data = JuMP.Containers.DenseAxisArray(
    [1800 2200;
     91   Inf;
     0    65;
     0    1779],
    ["calories", "protein", "fat", "sodium"],
    ["min", "max"])

Base.@show category_data["calories","max"]
Base.@show category_data["fat","min"]

foods = ["hamburger", "chicken", "hot dog", "fries", "macaroni", "pizza","salad", "milk", "ice cream"]

# we will use the same concept we used above to create an array indexed
# by foods this time to record the cost of each of these items
cost = JuMP.Containers.DenseAxisArray(
    [2.49, 2.89, 1.50, 1.89, 2.09, 1.99, 2.49, 0.89, 1.59],
    foods)

food_data = JuMP.Containers.DenseAxisArray(
    [410 24 26 730;
     420 32 10 1190;
     560 20 32 1800;
     380  4 19 270;
     320 12 10 930;
     320 15 12 820;
     320 31 12 1230;
     100  8 2.5 125;
     330  8 10 180],
    foods,
    ["calories", "protein", "fat", "sodium"])

Base.@show food_data["chicken", "fat"]
Base.@show food_data["milk", "sodium"]

# set up the model
model = JuMP.Model(GLPK.Optimizer)

categories = ["calories", "protein", "fat", "sodium"]

# add the variables
JuMP.@variables(model, begin
    # Variables for nutrition info
    category_data[c, "min"] <= nutrition[c = categories] <= category_data[c, "max"]
    # Variables for which foods to buy
    buy[foods] >= 0
end)

# Objective - minimize cost
JuMP.@objective(model, Min, sum(cost[f] * buy[f] for f in foods))

# Nutrition constraints
JuMP.@constraint(model, [c in categories],
    sum(food_data[f, c] * buy[f] for f in foods) == nutrition[c]
)

JuMP.optimize!(model)
term_status = JuMP.termination_status(model)
is_optimal = term_status == MOI.OPTIMAL
Base.@show JuMP.primal_status(model) == MOI.FEASIBLE_POINT
Base.@show JuMP.objective_value(model) ≈ 11.8288 atol = 1e-4

Base.hcat(buy.data,JuMP.value.(buy.data))

## How many passports do you need to travel the world without obtaining a visa in advance?
#git clone https://github.com/ilyankou/passport-index-dataset.git
passportdata = DelimitedFiles.readdlm(
    #joinpath("passport-index-dataset-master","passport-index-matrix.csv"),',')
    "Data/passport-index-matrix.csv")
cntr = passportdata[2:end,1]
vf = (x -> Base.typeof(x)==Int64 || x == "VF" || x == "VOA" ? 1 : 0).(passportdata[2:end,2:end])

model = JuMP.Model(GLPK.Optimizer)
#=
JuMP.@variable(model, pass[1:Base.length(cntr)], Bin)
JuMP.@constraint(model, [j=1:Base.length(cntr)], sum( vf[i,j]*pass[i] for i in 1:Base.length(cntr)) >= 1)
JuMP.@objective(model, Min, Base.sum(pass))

JuMP.optimize!(model)

Base.print(JuMP.objective_value(model)," passports: ",Base.join(cntr[Base.findall(JuMP.value.(pass) .== 1)],", "))
=#