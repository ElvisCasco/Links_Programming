#=
VIDEO 6: CLASSIFICATION
https://juliaacademy.com/courses/937702/lectures/17339540
https://www.youtube.com/watch?v=OQRPeIQasdo
https://www.youtube.com/watch?v=OQRPeIQasdo&list=PLP8iPy9hna6QuDTt11Xxonnfal91JhqjO&index=7
https://github.com/JuliaAcademy/DataScience/blob/master/06.%20Classification.ipynb
=#
#########################################
#import Pkg
#Pkg.add("GLMNet")
using GLMNet
using RDatasets
using MLBase
using Plots
using DecisionTree
using Distances
using NearestNeighbors
using Random
using LinearAlgebra
using DataStructures
using LIBSVM
# Function
findaccuracy(predictedvals, groundtruthvals) =
    Base.sum(predictedvals .== groundtruthvals) /
    Base.length(groundtruthvals)

# Get the data first
iris = RDatasets.dataset("datasets", "iris")
X = Base.Matrix(iris[:,1:4])
irislabels = iris[:, 5]
irislabelsmap = MLBase.labelmap(irislabels)
y = MLBase.labelencode(irislabelsmap, irislabels)

# training and testing data
function perclass_splits(y,at)
    uids = Base.unique(y)
    keepids = []
    for ui in uids
        curids = Base.findall(y.==ui)
        rowids = Random.randsubseq(curids, at)
        push!(keepids,rowids...)
    end
    return keepids
end
trainids = perclass_splits(y, 0.7)
testids = Base.setdiff(1:Base.length(y),
    trainids)

# assign classes based on the predicted values when the predicted values are continuous
assign_class(predictedvalue) = Base.argmin(Base.abs.(predictedvalue .- [1,2,3]))

## Method 1: Lasso
# choose the best lambda to predict with.
path = GLMNet.glmnet(
    X[trainids,:],
    y[trainids])
cv = GLMNet.glmnetcv(
    X[trainids,:],
    y[trainids])
mylambda = path.lambda[Base.argmin(cv.meanloss)]
path = GLMNet.glmnet(
    X[trainids,:],
    y[trainids],
    lambda=[mylambda]);
q = X[testids, :];
predictions_lasso = GLMNet.predict(path, q)
predictions_lasso = assign_class.(predictions_lasso)
findaccuracy(predictions_lasso, y[testids])

## Method 2: Ridge
# We will use the same function but set alpha to zero.
# choose the best lambda to predict with.
path = GLMNet.glmnet(
    X[trainids,:],
    y[trainids],
    alpha=0);
cv = GLMNet.glmnetcv(
    X[trainids,:],
    y[trainids],
    alpha=0)
mylambda = path.lambda[Base.argmin(cv.meanloss)]
path = GLMNet.glmnet(
    X[trainids,:],
    y[trainids],
    alpha = 0,
    lambda = [mylambda]);
q = X[testids, :];
predictions_ridge = GLMNet.predict(path, q)
predictions_ridge = assign_class.(predictions_ridge)
findaccuracy(predictions_ridge, y[testids])

## Method 3: Elastic Net
# We will use the same function but set alpha to 0.5 (it's the combination of lasso and ridge).
# choose the best lambda to predict with.
path = GLMNet.glmnet(
    X[trainids,:],
    y[trainids],
    alpha = 0.5);
cv = GLMNet.glmnetcv(
    X[trainids,:],
    y[trainids],
    alpha = 0.5)
mylambda = path.lambda[Base.argmin(cv.meanloss)]
path = GLMNet.glmnet(
    X[trainids,:],
    y[trainids],
    alpha = 0.5,
    lambda = [mylambda]);
q = X[testids, :];
predictions_EN = GLMNet.predict(path, q)
predictions_EN = assign_class.(predictions_EN)
findaccuracy(predictions_EN, y[testids])

## Method 4: Decision Trees
model = DecisionTree.DecisionTreeClassifier(max_depth = 2)
DecisionTree.fit!(
    model,
    X[trainids,:],
    y[trainids])
q = X[testids,:];
predictions_DT = DecisionTree.predict(
    model,
    q)
findaccuracy(predictions_DT, y[testids])

## Method 5: Random Forests
# The RandomForestClassifier is available through the DecisionTree package as well.
model = DecisionTree.RandomForestClassifier(n_trees = 20)
DecisionTree.fit!(
    model,
    X[trainids,:],
    y[trainids])
q = X[testids, :];
predictions_RF = DecisionTree.predict(model, q)
findaccuracy(predictions_RF, y[testids])

## Method 6: Using a Nearest Neighbor method
# We will use the NearestNeighbors package here.
Xtrain = X[trainids, :]
ytrain = y[trainids]
kdtree = NearestNeighbors.KDTree(Xtrain')
queries = X[testids, :]
idxs, dists = NearestNeighbors.knn(
    kdtree,
    queries',
    5,
    true)
c = ytrain[Base.hcat(idxs...)]
possible_labels = Base.map(
    i -> DataStructures.counter(c[:,i]), 1:Base.size(c, 2))
predictions_NN = Base.map(
    i -> Base.parse(
        Int,
        Base.string(
            Base.argmax(DataFrames.DataFrame(possible_labels[i])[1,:]))), 1:Base.size(c,2))
findaccuracy(predictions_NN,y[testids])

## Method 7: Support Vector Machines
# We will use the LIBSVM package here.
Xtrain = X[trainids, :]
ytrain = y[trainids]
model = LIBSVM.svmtrain(Xtrain', ytrain)
predictions_SVM, decision_values = LIBSVM.svmpredict(
    model,
    X[testids, :]')
findaccuracy(predictions_SVM, y[testids])

## Putting all the results together:
overall_accuracies = Base.zeros(7)
methods = ["lasso","ridge","EN", "DT", "RF","kNN", "SVM"]
ytest = y[testids]
overall_accuracies[1] = findaccuracy(predictions_lasso,ytest)
overall_accuracies[2] = findaccuracy(predictions_ridge,ytest)
overall_accuracies[3] = findaccuracy(predictions_EN,ytest)
overall_accuracies[4] = findaccuracy(predictions_DT,ytest)
overall_accuracies[5] = findaccuracy(predictions_RF,ytest)
overall_accuracies[6] = findaccuracy(predictions_NN,ytest)
overall_accuracies[7] = findaccuracy(predictions_SVM,ytest)
Base.hcat(methods, overall_accuracies)
