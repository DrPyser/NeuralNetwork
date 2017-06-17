# NeuralNetwork

A neural network framework in Haskell, and a MNIST classifier using that framework.

The design is currently based on two main Haskell features to offer flexibility, extensibility:

* Typeclasses for most things, e.g. A "Layer" typeclass, a "Network" typeclass, a "Backpropagation" and "GradientDescent" type class.
  This allows flexibility for minimal code redundancy.

* Pipes-based training framework, inspired by [this library](https://hackage.haskell.org/package/neural-0.3.0.0/docs/Numeric-Neural-Pipes.html).
  Each step of the training process is separated and encapsulated in components designed to produce, consume or transform a stream of values,
  e.g. input samples(or batches of samples), training states, ...
  This modularity allows for easy modification of individual components(separation of concerns),
  easy introspection and feedback at any point of the training process(down to the level of each step of the gradient descent),
  easy extensibility, and perhaps better performance(i.e. no list of samples are passed at any point in the training process).
  

