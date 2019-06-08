## Glossary and Concepts
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

## Algorithms
* Boosting
    - A boosting procedure can only stop if the weak learner's prediction is pure (It's perfect)
    - Boosting is robust to overfitting. But [why](https://www.quora.com/Why-is-the-boosting-algorithm-robust-to-overfitting)? My understanding: Because each *weak* learner has to make *global* decision on the entire data, can't be localized around outliers. (Weak yet global) Not like decision trees, which each node is more localized as it goes deeper. Yoav (CS255) said it's related to the margin, but I didn't fully understand.
    - Boosting works well when the classifier depends on just a small subset of the features (Sparse detector). In contrast, support vector machine (SVM) is a dense detector.
    - Boosting (or SVM) indicates the confidence of prediction by the margin between classes, which implies the stability of the classifier's predictions against small changes in the training sample. In contrast, the logistic regression (or multi-layer perceptrons) produces a probability of the prediction indicting it's confidence, but that is not the confidence that how much we should trust the model.
    - Adaboost (see [derivations](http://www.inf.fu-berlin.de/inst/ag-ki/adaboost4.pdf))
    - Anyboost: a general framework for boosting algorithms, viewing the weights of examples as the gradients of loss function (see [**Boosting algorithms as gradient descent**](https://papers.nips.cc/paper/1766-boosting-algorithms-as-gradient-descent.pdf)) (XGBoost: An efficient and flexible gradient boosting library)
    - Reference: [A comprehensible introduction to boosted trees](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)

