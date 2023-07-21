#=
VIDEO 2: LINEAR ALGEBRA
https://juliaacademy.com/courses/937702/lectures/17339511
https://www.youtube.com/watch?v=bndXPsRHPg0&t=10s
https://www.youtube.com/watch?v=bndXPsRHPg0&list=PLP8iPy9hna6QuDTt11Xxonnfal91JhqjO&index=3
https://github.com/JuliaAcademy/DataScience/blob/master/02.%20Linear%20Algebra.ipynb
=#
#########################################
# some packages we will use
using LinearAlgebra
using SparseArrays
#Pkg.add("Images")
using Images
using MAT

## Getting Started
A = Base.rand(10,10); # created a random matrix of size 10-by-10
Atranspose = A' # matrix transpose
A = A*Atranspose; # matrix multiplication
Base.@show A[11] == A[1,2]
b = Base.rand(10) #created a random vector of size 10
# \ is always the recommended way to solve a linear system.
# You almost never want to call the inv function
x = A\b #x is the solutions to the linear system Ax=b
Base.@show LinearAlgebra.norm(A*x-b)
Base.@show Base.typeof(A)
Base.@show Base.typeof(b)
Base.@show Base.typeof(Base.rand(1,10))
Base.@show Base.typeof(Atranspose)

Matrix{Float64} == Array{Float64,2}
Vector{Float64} == Array{Float64,1}
LinearAlgebra.adjoint(A)
Atranspose.parent
Base.sizeof(A)

# To actually copy the matrix:
B = Base.copy(Atranspose)
Base.sizeof(B)

## Factorizations

# LU factorization
# L * U = P * A
luA = LinearAlgebra.lu(A)
luA.L
luA.P
luA.U
luA.p
# Prueba
LinearAlgebra.norm(luA.L * luA.U - luA.P * A)

# QR factorization
# Q * R = A
qrA = LinearAlgebra.qr(A)
qrA.Q
qrA.R
# Prueba
LinearAlgebra.norm(qrA.Q * qrA.R - A)

# Cholesky factorization, note that A needs to be symmetric positive definite
# L * L' = A
LinearAlgebra.isposdef(A)
cholA = LinearAlgebra.cholesky(A)
cholA.L
cholA.U
cholA.UL
LinearAlgebra.factorize(A)
# Prueba
LinearAlgebra.norm(cholA.L * cholA.U - A)

# convert(Diagonal{Int64,Array{Int64,1}},diagm([1,2,3]))
LinearAlgebra.Diagonal([1,2,3])
LinearAlgebra.I(3)

## Sparse Linear Algebra
using SparseArrays
S = SparseArrays.sprand(5,5,2/5)
S.rowval
S.n
S.nzval
Base.Matrix(S)
S.colptr
S.m

## Images as matrices
#Pkg.add("QuartzImageIO")
Base.@static if Sys.isapple()
    using QuartzImageIO
end
#Pkg.add("ImageMagick")
using ImageMagick
X1 = Images.load(wd * "Data/khiam-small.jpg")
Base.@show Base.typeof(X1)
X1[1,1] # this is pixel [1,1]

# Adjust colors
R = Base.map(i->X1[i].r,1:Base.length(X1))
R = Float64.(Base.reshape(R,Base.size(X1)...))
G = Base.map(i->X1[i].g,1:Base.length(X1))
G = Float64.(Base.reshape(G,Base.size(X1)...))
B = Base.map(i->X1[i].b,1:Base.length(X1))
B = Float64.(Base.reshape(B,Base.size(X1)...))
Z = Base.zeros(Base.size(R)...) # just a matrix of all zeros of equal size as the image
Images.RGB.(Z,G,B)

# Gray scale
Xgray = Images.Gray.(X1)
Xgrayvalues = Float64.(Xgray)
SVD_V = LinearAlgebra.svd(Xgrayvalues)
SVD_V.S
SVD_V.U
SVD_V.V
SVD_V.Vt
LinearAlgebra.norm(SVD_V.U * LinearAlgebra.diagm(SVD_V.S) * SVD_V.V' - Xgrayvalues)

# use the top 4 singular vectors/values to form a new image
u1 = SVD_V.U[:,1]
v1 = SVD_V.V[:,1]
img1 = SVD_V.S[1]*u1*v1'

i = 2
u1 = SVD_V.U[:,i]
v1 = SVD_V.V[:,i]
img1 += SVD_V.S[i]*u1*v1'

i = 3
u1 = SVD_V.U[:,i]
v1 = SVD_V.V[:,i]
img1 += SVD_V.S[i]*u1*v1'

Images.Gray.(img1)

# Using 100 singular vectors/values
i = 1:100
u1 = SVD_V.U[:,i]
v1 = SVD_V.V[:,i]
img1 = u1*SparseArrays.spdiagm(0=>SVD_V.S[i])*v1'
Images.Gray.(img1)
LinearAlgebra.norm(Xgrayvalues - img1)

## Face Recognition Problem
M = MAT.matread(wd * "Data/face_recog_qr.mat")
q = Base.reshape(M["V2"][:,1],192,168)
Images.Gray.(q)
b = q[:]
A = M["V2"][:,2:end]
x = A\b #Ax=b
Images.Gray.(Base.reshape(A * x,192,168))
LinearAlgebra.norm(A * x - b)

# Let's try to make the picture harder to recover. We will add some random error.
qv = q+Base.rand(Base.size(q,1), Base.size(q,2))*0.5
qv = qv./Base.maximum(qv)
Images.Gray.(qv)
b = qv[:];
x = A\b
# The error is so much bigger this time.
LinearAlgebra.norm(A * x - b)
Images.Gray.(Base.reshape(A * x,192,168))
