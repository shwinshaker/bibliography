## Face recognition

* [DeepFace](https://ieeexplore.ieee.org/document/6909616) **#Face-Recognition**
    - Perform 3D alignment in order to align faces undergoing out-of-plane rotations (pitch and yaw). 67 fiducial points are first detected in 2D images, which are then mapped to 3D points following an affine transformation.
    - Locally connected layers are stacked on normal convolutional layers to extract spatially independent features. The reason states that after proper alignment, different regions on a feature map should have different statistics, thus spatial invariance of convolutions no longer holds.
    - Different verificaiton metrics are experimented to measure the similarity between feature representations, including naive inner product (unsupervised), weighted $\chi^2$ similarity and siamese network (supervised).

* [DeepID2](https://arxiv.org/abs/1406.4773) **#Face-Recognition**
    - Signals of face identification and verfication are joined to supervise the model, which reduces the intra-class variations and enlarge the inter-class variations simultaneously
    - A gradient descent algorithm for the joint learning is well designed.
    - Detailed analyses on how the trade-off between identification and verificaiton losses affect the performance

* [FaceNet](https://arxiv.org/abs/1503.03832) **#Face-Recognition #Metric-Learning**
    - Triplet loss
    - Online hard example mining

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

* [3DMM](https://dl.acm.org/citation.cfm?id=311556) **#Face-Alignment #Face-Reconstruction**
    - A morphable face model is derived by transforming the shape (coordinates of vertices) and texture (colors) into a vector space representation
    - Facial attributes (Eg. expression, gender, weight, hooked or concave nose) can be concluded from the statistics (PCA) of example faces, by means of which automate matching is possible.

* [3D-DFA](https://arxiv.org/abs/1511.07212) **#Face-Alignment**
    - Use a 3D morphable model (3DMM) to construct the 3D face. 2D images are derived from a Weal Perspective Projection.
    - Use a regression based CNN to estimate the residuals of parameters of the face model
    - Design a special feature called Projected Normalized Coordinate Code (PNCC) as an additional input to the network besides the original image. PNCC encodes the information of visible 3D vertexes through z-buffer thus benefits the learning.
    - Faces with large pose variations can now be synthesized by manipulating estimated parameters

* Recipes for face recognition to work (see Lecture 11)

* [DeepPose](https://arxiv.org/abs/1312.4659) **#Human-Pose-Estimation**
    - Use a regression based on deep CNN to estimate human pose, save the effort to manually design features for body parts or model the interactions between joints
    - Use a cascade of CNN to gradually look into the local details of joints based on previous predicted localizations, such that poses can be refined
    - Metrics including Percentage of Correct Parts (PCP) and Percentage of Detected Joints (PDJ). The former one has certain drawback in that it penalize shorter limbs.

* [Convolutional Pose Machines](https://arxiv.org/abs/1602.00134) **#Human-Pose-Estimation**
    - Use belief maps of the joints instead of landmark coordinates to supervise the learning, which perserves spatial uncertainty and can be informative.
    - Concatenation of stages based on preceding belief maps of parts allows subsequent networks to infer from pther parts that are highly confident, capturing long-range interations between parts. This also contributes in view of large receptive field and rich contextual information.
    - Deep layers lead to vanishing gradients, which is resolved by intermediate supervision signals.

* [Hourglass networks]()

* Recipes for Human Pose Estimation (see Lecture 13)

* [Fully convolutional network]() (FCN)
    - Cast fully conneted (fc) layers in classic achitectures into convolutional layers, thus image of any size can now be handled (otherwise we have to deepen the network). Final layer is upsampled (deconv) into a coarse heatmap compared to the original image resolution.
    - To refine the spatical precision of the output, lower but finer layer are upsampled (bilinear interpolation) and combined into the original output (like skip connection). This fashion can be iteratively done.

## DimensionReduction&Visualization
* t-SNE
    - Use a joint probability distribution in minimizing Kullback-Leibler divergences. SNE is now symmetric ($P_{i,j}=P_{j,i}$)
    - Use a student t-distribution in the low-dimensional map to allow much heavier tail than Gaussian, to alleviate the crowding problem, which points out that a moderate distance in high dimension will be mapped to much far away in low dimension.
    - t-SNE gradient stongly repels dissimilar points (large distance in high dimension) in low-dimensional space.

