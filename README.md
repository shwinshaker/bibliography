# bibliography
Abstracts and keywords for a list of papers and extensive readings

## Machine Learning
* Discriminative vs. Generative classifiers
    - Discriminative classifiers (Eg. Logistic; SVM) learn the decision boundary \[![equation](http://latex.codecogs.com/gif.latex?p(y|x))\]; Generative classifiers (Eg. Naive bayes; HMM) model the distribution ($P(x,y)$ -> $P(y|x)$). ([Nice story](https://medium.com/@mlengineer/generative-and-discriminative-models-af5637a66a3))
    - Generative learners learn faster as the number of training examples grows, but converge to higher asymtotic error, compared to Discriminative learner. This suggests that generative learners may outperform discriminative learners when data are not sufficient. ([Supported by theoretical and empirical analyses by Andrew Ng](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf))

* Batch Normalization
    - Batch normalization is a way to reduce second-order relationships between different layers. It's like a checkpoint to isolate subsequent layers to make the training of very deep networks easier. Batch normalization prefers large datasets. It brings a minor effect of regularization. Watch [Ian Goodfellow's great talk](https://www.youtube.com/watch?v=Xogn6veSyxA&feature=youtu.be) about batch normalization.
    - BN before or after non-linear activations? (AKA. BN the inputs or the outputs of NN layers?) It doesn't really matter and depends. See relavant [discussion and experiments](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/).

* Covariance Shift
    - Covariance shift or dataset shift refer to the issue that the distributions of training and testing data differ.
    - Covariance shift can be detected by treating training and testing data as a binary classification task and see if they are distinguishable. See [a simple demo](https://blog.bigml.com/2014/01/03/simple-machine-learning-to-detect-covariate-shift/).

* Matthews correlation coefficient
    - A measure of the quality of binary classification

## Computer Vision

* [Center loss](https://ydwen.github.io/papers/WenECCV16.pdf)
    - The joint supervision of softmax loss and center loss enables the inter-class dispension and intra-class compactness simultaneously.
    - During training, centers are computed in a mini-batch flavour to ensure efficiency. Updates of centers are controlled by a learning rate to avoid perturbations of mislabeled examples.

* [A-Softmax loss](https://arxiv.org/abs/1704.08063) (SphereFace) **#Face-Recognition #Metric-Learning**
    - Features learned by softmax loss have intrinsic angular distribution. Euclidean margin based losses may not be compatible.
    - By normalizing weights and imposing stringent angular criterion, learned features can be constrained to be discriminative on a hypersphere manifold, which inherently matches the prior that face images lie on a manifold.
    - Instead of weights, the learned now has to depend on the features. Thus the learning of discriminative features is made explicit.
    - A scalar parameter $m$ can be used to adjust the intra-class compactness of the learned features, and therefore the difficulty of the learning task. Lower bound of $m$ can be quantitatively derived to ensure the maximal intra-class distance is smaller than the minimal inter-class distance.
