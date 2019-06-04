# bibliography
Abstracts and keywords for a list of papers and extensive readings

## Machine Learning
* Discriminative vs. Generative classifiers
    - Discriminative classifiers (Eg. Logistic; SVM) learn the decision boundary \[![equation](http://latex.codecogs.com/gif.latex?p(y|x))\]; Generative classifiers (Eg. Naive bayes; HMM) model the distribution ($P(x,y)$ -> $P(y|x)$). ([Nice story](https://medium.com/@mlengineer/generative-and-discriminative-models-af5637a66a3))
    - Generative learners learn faster as the number of training examples grows, but converge to higher asymtotic error, compared to Discriminative learner. This suggests that generative learners may outperform discriminative learners when data are not sufficient. ([Supported by theoretical and empirical analyses by Andrew Ng](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf))

* Batch Normalization ([Original paper](https://arxiv.org/pdf/1502.03167.pdf)) ([Deep learning book Section 8.7.1](http://www.deeplearningbook.org/contents/optimization.html))
    - Batch normalization is a way to reduce second-order relationships between different layers. It's like a checkpoint to isolate subsequent layers to make the training of very deep networks easier. Batch normalization prefers large datasets. It brings a minor effect of regularization. Watch [Ian Goodfellow's great talk](https://www.youtube.com/watch?v=Xogn6veSyxA&feature=youtu.be) about batch normalization.
    - BN before or after non-linear activations? (AKA. BN the outputs $XW+b$ or the inputs $X$ of linear transformation layers?) It is not quite clear and depends. In the original paper the former was recommended, which is confirmed in Deep learning book. See relavant [discussion and experiments](https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/).

* Covariance Shift
    - Covariance shift or dataset shift refer to the issue that the distributions of training and testing data differ.
    - Covariance shift can be detected by treating training and testing data as a binary classification task and see if they are distinguishable. See [a simple demo](https://blog.bigml.com/2014/01/03/simple-machine-learning-to-detect-covariate-shift/).

* Matthews correlation coefficient
    - A measure of the quality of binary classification

* Explanation
    - LIME: Say a text classification task with only unigrams as features. Given the change of prediction probability (confidence) after removing a word from the document, we can see the the weight (importance) of this word contributed to the prediction. This is essentially same as what we do in decision tree (or statistical learning in general), that each word is a feature in a node, and the decision rule is whether this word (feature) exists or not.

## Machine Learning algorithms
* Boosting
    - A boosting procedure can only stop if the weak learner's prediction is pure (It's perfect)
    - Boosting is robust to overfitting. But [why](https://www.quora.com/Why-is-the-boosting-algorithm-robust-to-overfitting)? My understanding: Because each *weak* learner has to make *global* decision on the entire data, can't be localized around outliers. (Weak yet global) Not like decision trees, which each node is more localized as it goes deeper. Yoav (CS255) said it's related to the margin, but I didn't fully understand.
    - Boosting works well when the classifier depends on just a small subset of the features (Sparse detector). In contrast, support vector machine (SVM) is a dense detector.
    - Boosting (or SVM) indicates the confidence of prediction by the margin between classes, which implies the stability of the classifier's predictions against small changes in the training sample. In contrast, the logistic regression (or multi-layer perceptrons) produces a probability of the prediction indicting it's confidence, but that is not the confidence that how much we should trust the model.
    - Adaboost (see [derivations](http://www.inf.fu-berlin.de/inst/ag-ki/adaboost4.pdf))
    - Anyboost: a general framework for boosting algorithms, viewing the weights of examples as the gradients of loss function (see [**Boosting algorithms as gradient descent**](https://papers.nips.cc/paper/1766-boosting-algorithms-as-gradient-descent.pdf)) (XGBoost: An efficient and flexible gradient boosting library)
    - Reference: [A comprehensible introduction to boosted trees](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)

## Computer Vision

* [DeepFace](https://ieeexplore.ieee.org/document/6909616) **#Face-Recognition**
    - Perform 3D alignment in order to align faces undergoing out-of-plane rotations (pitch and yaw). 67 fiducial points are first detected in 2D images, which are then mapped to 3D points following an affine transformation.
    - Locally connected layers are stacked on normal convolutional layers to extract spatially independent features. The reason states that after proper alignment, different regions on a feature map should have different statistics, thus spatial invariance of convolutions no longer holds.
    - Different verificaiton metrics are experimented to measure the similarity between feature representations, including naive inner product (unsupervised), weighted $\chi^2$ similarity and siamese network (supervised).

* [DeepID2](https://arxiv.org/abs/1406.4773) **#Face-Recognition**
    - Signals of face identification and verfication are joined to supervise the model, which reduces the intra-class variations and enlarge the inter-class variations simultaneously
    - A gradient descent algorithm for the joint learning is well designed.
    - Detailed analyses on how the trade-off between identification and verificaiton losses affect the performance

* [Center loss](https://ydwen.github.io/papers/WenECCV16.pdf)
    - The joint supervision of softmax loss and center loss enables the inter-class dispension and intra-class compactness simultaneously.
    - During training, centers are computed in a mini-batch flavour to ensure efficiency. Updates of centers are controlled by a learning rate to avoid perturbations of mislabeled examples.

* [A-Softmax loss](https://arxiv.org/abs/1704.08063) (SphereFace) **#Face-Recognition #Metric-Learning**
    - Features learned by softmax loss have intrinsic angular distribution. Euclidean margin based losses may not be compatible.
    - By normalizing weights and imposing stringent angular criterion, learned features can be constrained to be discriminative on a hypersphere manifold, which inherently matches the prior that face images lie on a manifold.
    - Instead of weights, the learned now has to depend on the features. Thus the learning of discriminative features is made explicit.
    - A scalar parameter $m$ can be used to adjust the intra-class compactness of the learned features, and therefore the difficulty of the learning task. Lower bound of $m$ can be quantitatively derived to ensure the maximal intra-class distance is smaller than the minimal inter-class distance.

* [Large Margin Cosine Loss](https://arxiv.org/abs/1801.09414) (CosFace) **#Face-Recognition #Metric-Learning**
    - The decision boundary of A-Softmax depends on $\theta$, which leads to different margins for different classes. And it also poses a difficulty to optimization due to the non-linearity of Cosine Function.
    - It should be favored by directly pushing the margins based on cosine similarity.

* [Additive Angular Margin Loss](https://arxiv.org/abs/1801.07698) (ArcFace) **#Face-Recognition #Metric-Learning**
    - Directly push the margin in terms of goedesic distance, which is consistent with the nature of angular distribution of classes in the normalized hypersphere.
    - 
