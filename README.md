# bibliography
Abstracts and keywords for a list of papers and extensive readings

## Machine Learning
* Discriminative vs. Generative classifiers
    - Discriminative classifiers (Eg. Logistic; SVM) learn the decision boundary ($P(y|x)$); Generative classifiers (Eg. Naive bayes; HMM) model the distribution ($P(x,y)$ -> $P(y|x)$). ([Nice story](https://medium.com/@mlengineer/generative-and-discriminative-models-af5637a66a3))
    - Generative learners learn faster as the number of training examples grows, but converge to higher asymtotic error, compared to Discriminative learner. This suggests that generative learners may outperform discriminative learners when data are not sufficient. ([Supported by theoretical and empirical analyses by Andrew Ng](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf))

## Computer Vision

* [Center loss](https://ydwen.github.io/papers/WenECCV16.pdf)
    - The joint supervision of softmax loss and center loss enables the inter-class dispension and intra-class compactness simultaneously.
    - During training, centers are computed in a mini-batch flavour to ensure efficiency. Updates of centers are controlled by a learning rate to avoid perturbations of mislabeled examples.

