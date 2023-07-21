#########################################
# VIDEO 11: FROM OTHER LANGUAGES
# https://juliaacademy.com/courses/937702/lectures/17370113
# https://www.youtube.com/watch?v=3DYJiAgApdk&list=PLP8iPy9hna6QuDTt11Xxonnfal91JhqjO&index=12
# https://github.com/JuliaAcademy/DataScience/blob/master/11.%20from%20other%20languages.ipynb
#########################################
## Python
using PyCall
math = PyCall.pyimport("math")
math.sin(math.pi / 4) # returns ≈ 1/√2 = 0.70710678...

# https://github.com/networkx/networkx
#python_networkx = pyimport("networkx")

PyCall.py"""
    import numpy
    def find_best_fit_python(xvals,yvals):
        meanx = numpy.mean(xvals)
        meany = numpy.mean(yvals)
        stdx = numpy.std(xvals)
        stdy = numpy.std(yvals)
        r = numpy.corrcoef(xvals,yvals)[0][1]
        a = r*stdy/stdx
        b = meany - a*meanx
        return a,b
    """

xvals = Base.repeat(1:0.5:10, inner = 2)
yvals = 3 .+ xvals .+ 2 .* Base.rand(Base.length(xvals)) .-1
find_best_fit_python = PyCall.py"find_best_fit_python"
a,b = find_best_fit_python(xvals,yvals)

## R code
using RCall
# we can use the rcall function
r = RCall.rcall(:sum, Float64[1.0, 4.0, 6.0])
Base.typeof(r[1])
z = 1
RCall.@rput z
r = RCall.R"z+z"
r[1]
x = Base.randn(10)

RCall.@rimport base as rbase
rbase.sum([1, 2, 3])

#=
R"""
library("usethis")
library("devtools")
install_github("cran/boot")
#install.packages("boot")
"""
=#
RCall.@rlibrary boot # install package boot in R before this
RCall.R"t.test($x)"

using HypothesisTests
HypothesisTests.OneSampleTTest(x)

## C code
t = ccall(:clock, Int32, ())

## JuliaCon
# https://github.com/xorJane/Excelling-at-Julia-Basics-and-Beyond/blob/master/JuliaCon2019_Huda/Julia%20Wrappers.ipynb
# wrapping functions:
using PyCall #if calling this for the first time, add the package first by typing Pkg.add("PyCall")

# 1. take your existing python function and wrap it into a julia function
# 2. You can also write quick and easy python code from within julia
PyCall.py"""
    def py_sum2(A):
        s = 0.0
        for a in A:
            s += a
        return s
    """
sum_py = PyCall.py"py_sum2"

# 3. You can also import your favorite python package
#pyimport("cvxpy")

## Performance tips for Julia
# https://github.com/xorJane/Excelling-at-Julia-Basics-and-Beyond/blob/master/JuliaCon2019_Huda/Performance%20Tips%20for%20Julia.ipynb
# 1. As a rule of thumb, typed functions run faster.
function _show_subtype_tree(mytype, printlevel)
    allsubtypes = subtypes(mytype)
    for cursubtype in allsubtypes
        print("\t"^printlevel)
        println("|___",cursubtype)
        printlevel += 1
        _show_subtype_tree(cursubtype,printlevel)
        printlevel -= 1
    end
end
function show_type_tree(T)
    println(T)
    _show_subtype_tree(T,0)
end
show_type_tree(Number)

function square_plus_one(v::T) where T <:Number
    g = v*v
    return g+1
end
v = Base.rand()
Base.typeof(v)
InteractiveUtils.@code_warntype square_plus_one(v)

w = 5
Base.typeof(w)
InteractiveUtils.@code_warntype square_plus_one(w)

mutable struct Cube
    length
    width
    height
end
volume(c::Cube) = c.length*c.width*c.height

mutable struct Cube_typed
    length::Float64
    width::Float64
    height::Float64
end
volume(c::Cube_typed) = c.length*c.width*c.height

mutable struct Cube_parametric_typed{T <: Real}
    length::T
    width::T
    height::T
end
volume(c::Cube_parametric_typed) = c.length*c.width*c.height

c1 = Cube(1.1,1.2,1.3)
c2 = Cube_typed(1.1,1.2,1.3)
c3 = Cube_parametric_typed(1.1,1.2,1.3)
Base.@show volume(c1) == volume(c2) == volume(c3)

using BenchmarkTools
BenchmarkTools.@btime volume(c1) # not typed
BenchmarkTools.@btime volume(c2) # typed float
BenchmarkTools.@btime volume(c3) # typed parametric

InteractiveUtils.@code_warntype volume(c1)
InteractiveUtils.@code_warntype volume(c2)
InteractiveUtils.@code_warntype volume(c3)

# Types matter, when you know anything about the types of your variables, include them in your code to make it run faster
function zero_or_val(x::Real)
    if x >= 0
        return x
    else
        return 0
    end
end
InteractiveUtils.@code_warntype zero_or_val(0.2)

function zero_or_val_stable(x::Real)
    T = promote_type(typeof(x),Int)
    if x >= 0
        return T(x)
    else
        return T(0)
    end
end
InteractiveUtils.@code_warntype zero_or_val_stable(0.2)

function flipcoin_then_add(v::Vector{T}) where T <: Real
    s = 0
    for vi in v
        r = rand()
        if r >=0.5
            s += 1
        else
            s += vi
        end
    end
end

function flipcoin_then_add_typed(v::Vector{T}) where T <: Real
    s = Base.zero(T)
    for vi in v
        r = Base.rand()
        if r >=0.5
            s += Base.one(T)
        else
            s += vi
        end
    end
end
myvec = Base.rand(1000)
Base.@show flipcoin_then_add(myvec) == flipcoin_then_add_typed(myvec)

BenchmarkTools.@btime flipcoin_then_add(rand(1000))
BenchmarkTools.@btime flipcoin_then_add_typed(rand(1000))

# 2. As a rule of thumb, functions with preallocated memory run faster
function build_fibonacci_preallocate(n::Int)
    Base.@assert n >= 2
    v = Base.zeros(Int64,n)
    v[1] = 1
    v[2] = 1
    for i = 3:n
        v[i] = v[i-1] + v[i-2]
    end
    return v
end

function build_fibonacci_no_allocation(n::Int)
    Base.@assert n >= 2
    v = Base.Vector{Int64}()
    Base.push!(v,1)
    Base.push!(v,1)
    for i = 3:n
        Base.push!(v,v[i-1]+v[i-2])
    end
    return v
end

Base.@show Base.isequal(build_fibonacci_preallocate(10),build_fibonacci_no_allocation(10))
n = 100
BenchmarkTools.@btime build_fibonacci_no_allocation(n)
BenchmarkTools.@btime build_fibonacci_preallocate(n)

# Let's say, for some reason you want to access all the elements of a matrix once
# Create a random matrix A of size m-by-n
m = 10000
n = 10000
A = Base.rand(m,n)

function matrix_sum_rows(A::Matrix)
    m,n = Base.size(A)
    mysum = 0
    for i = 1:m # fix a row
        for j = 1:n # loop over cols
            mysum += A[i,j]
        end
    end
    return mysum
end

function matrix_sum_cols(A::Matrix)
    m,n = Base.size(A)
    mysum = 0
    for j = 1:n # fix a column
        for i = 1:m # loop over rows
            mysum += A[i,j]
        end
    end
    return mysum
end

function matrix_sum_index(A::Matrix)
    m,n = Base.size(A)
    mysum = 0
    for i = 1:m*n
        mysum += A[i]
    end
    return mysum
end

Base.@show matrix_sum_cols(A) ≈ matrix_sum_rows(A) ≈ matrix_sum_index(A)

BenchmarkTools.@btime matrix_sum_rows(A)
BenchmarkTools.@btime matrix_sum_cols(A)
BenchmarkTools.@btime matrix_sum_index(A)

# The experiment is to find the hypotenuse value of all triangles
b = Base.rand(1000)*10
h = Base.rand(1000)*10
function find_hypotenuse(b::Vector{T},h::Vector{T}) where T <: Real
    return Base.sqrt.(b.^2+h.^2)
end
# Let's time it
BenchmarkTools.@btime find_hypotenuse(b,h)

function find_hypotenuse_optimized(b::Vector{T},h::Vector{T}) where T <: Real
    accum_vec = Base.similar(b)
    for i = 1:Base.length(b)
        accum_vec[i] = b[i]^2
        accum_vec[i] = accum_vec[i] + h[i]^2 # here, we used the same space in memory to hold the sum
        accum_vec[i] = Base.sqrt(accum_vec[i]) # same thing here, to hold the sqrt
        # or:
        # accum_vec[i] = sqrt(b[i]^2+h[i]^2)
    end
    return accum_vec
end
BenchmarkTools.@btime find_hypotenuse_optimized(b,h)

# Vectorized operations are not necessarily faster.
function function_inplace!(v::Vector{T},myfn::Function) where T
    for i = 1:Base.length(v)
        v[i] = myfn(v[i])
    end
    v
end

function function_not_inplace(v::Vector{T},myfn::Function) where T
    w = Base.zeros(Base.eltype(v),Base.length(v))
    for i = 1:Base.length(v)
        w[i] = myfn(v[i])
    end
    w
end

v = Base.rand(100)
BenchmarkTools.@btime function_inplace!(v,x->x^2)
BenchmarkTools.@btime function_not_inplace(v,x->x^2)
BenchmarkTools.@btime v.^2;

# What are iterators and why do we care about them?
struct fib_iterator
    n::Int
end

function Base.iterate(f::fib_iterator,state=(0,0,1))
    prev1,prev2,stepid = state
    # state the ending conditions first
    if stepid == 1
        return (1,(0,1,2))
    end
    if f.n < stepid
        return nothing
    end
    # else
    y = prev1+prev2
    stepid += 1
    return (y,(prev2,y,stepid))
end

function myfib(n)
    v = Base.zeros(Int,n+1)
    v[1] = 1
    v[2] = 1
    for i = 3:n+1
        v[i] = v[i-1] + v[i-2]
    end
    return v
end

function test_iterator(n)
    f = fib_iterator(n)
    s = 0
    for i in f
        s += i
    end
end
function test_allocate(n)
    s = 0
    for i in myfib(n)
        s += i
    end
end

BenchmarkTools.@btime test_iterator(10)
BenchmarkTools.@btime test_allocate(10)

# Iterators are a powerful tool, use them when you don't need to store the values.
# Always think about memory... Do you really need A[row_ids,col_ids]
using SparseArrays
using LinearAlgebra
A = SparseArrays.sprand(500,500,0.1)
function set_sum(A,rowids,colids)
    s = Base.sum(A[rowids,colids])
end
function set_sum_view(A,rowids,colids)
    s = Base.sum(Base.view(A,rowids,colids))
end

using Random
BenchmarkTools.@btime set_sum(A,Random.randperm(10), Random.randperm(10))
BenchmarkTools.@btime set_sum_view(A,Random.randperm(10), Random.randperm(10))

# 3. There are many tools in Julia that helps you write faster code
# @profile
function myfunc()
    A = Base.rand(200, 200)
    Base.sum(A)
end
using Profile
Profile.@profile myfunc()
Profile.print()
Profile.clear()

# @inbounds
# Let us say we want to find the sum of all elements in a vector
function new_sum(myvec::Vector{Int})
    s = 0
    for i = 1:Base.length(myvec)
        s += myvec[i]
    end
    return s
end

function new_sum_inbounds(myvec::Vector{Int})
    s = 0
    Base.@inbounds for i = 1:Base.length(myvec)
        s += myvec[i]
    end
    return s
end
myvec = Base.collect(1:1000000)
BenchmarkTools.@btime new_sum(myvec)
BenchmarkTools.@btime new_sum_inbounds(myvec)
Base.@show isequal(new_sum(myvec),new_sum_inbounds(myvec))

#=
# Be careful though!
function new_sum_WRONG(myvec::Vector{Int})
    s = 0
    for i = 1:Base.length(myvec)+1
        s += myvec[i]
    end
    return s
end

function new_sum_inbounds_WRONG(myvec::Vector{Int})
    s = 0
    Base.@inbounds for i = 1:Base.length(myvec)+1
        s += myvec[i]
    end
    return s
end

myvec = Base.collect(1:1000000)
BenchmarkTools.@btime new_sum_WRONG(myvec)
BenchmarkTools.@btime new_sum_inbounds_WRONG(myvec) # this actually exectued!
=#

# @code_XXX
# @code_llvm
# @code_lowered
# @code_native
# @code_typed
# @code_warntype
function flipcoin(randval::Float64)
    if randval<0.5
        return "H"
    else
        return "T"
    end
end
InteractiveUtils.@code_lowered flipcoin(rand()) # syntax tree
InteractiveUtils.@code_warntype flipcoin(rand()) # try @code_typed
InteractiveUtils.@code_llvm flipcoin(rand()) # this and code_warntype are probably the most relevant
InteractiveUtils.@code_native flipcoin(rand())
