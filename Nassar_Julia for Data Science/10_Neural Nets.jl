#########################################
# VIDEO 10: NEURAL NETS
# https://juliaacademy.com/courses/937702/lectures/17370102
# https://www.youtube.com/watch?v=Oxi0Pfmskus
# https://www.youtube.com/watch?v=Oxi0Pfmskus&list=PLP8iPy9hna6QuDTt11Xxonnfal91JhqjO&index=11
# https://github.com/JuliaAcademy/DataScience/blob/master/10.%20Neural%20Nets.ipynb
#########################################
using Flux, MLDatasets 
#using Flux.Data.MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated
using Images

imgs = MLDatasets.MNIST.traindata()#Flux.Data.MNIST.images()
Images.colorview(Gray, imgs[100])

Base.typeof(imgs[3])
myFloat32(X) = Float32.(X)
fpt_imgs = myFloat32.(imgs)
Base.typeof(fpt_imgs[3])
JuMP.vectorize(x) = x[:]
vectorized_imgs = JuMP.vectorize.(fpt_imgs)

Base.typeof(vectorized_imgs)
X = Base.hcat(vectorized_imgs...)
Base.size(X)
onefigure = X[:,3]
t1 = Base.reshape(onefigure,28,28)
Images.colorview(Gray,t1)
labels = MNIST.labels()
labels[1]
Y = onehotbatch(labels, 0:9)

m = Flux.Chain(
  Flux.Dense(28^2, 32, relu),
  Flux.Dense(32, 10),
  softmax)

m(onefigure)
loss(x, y) = Flux.crossentropy(m(x), y)
accuracy(x, y) = Base.mean(Base.argmax(m(x)) .== Base.argmax(y))
datasetx = repeated((X, Y), 200)
C = Base.collect(datasetx)
evalcb = () -> Base.@show(loss(X, Y))

ps = Flux.params(m)
opt = Flux.ADAM()
Flux.train!(loss, ps, datasetx, opt, cb = throttle(evalcb, 10))
tX = Base.hcat(float.(Base.reshape.(MNIST.images(:test), :))...);
test_image = m(tX[:,1])

Base.argmax(test_image) - 1
t1 = Base.reshape(tX[:,1],28,28)
Images.colorview(Gray, t1)

onefigure = X[:,2]
m(onefigure)
Y[:,2]
