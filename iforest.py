# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

from core import *


# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

def walk_tree(forest, node, treeIdx, obsIdx, X, depth=0):
    '''
    Recursive function that walks a tree from an already fitted forest to compute the path length
    of the new observations.

    :param forest: Already fited forest.
    :param node: current node.
    :param treeIdx: index of the tree that is being walked.
    :param obsIdx: 1D array of length n_obs. 1/0 if the obs has reached / has not reached the node.
    :param X: 2D array. observations.
    :param depth: current depth.
    :return: None
    '''
    if isinstance(node, LeafNode):

        forest.L[obsIdx, treeIdx] = node.path_length

    else:
        idx = (X[:, node.splitAtt] <= node.splitValue) * obsIdx
        walk_tree(forest, node.left, treeIdx, idx, X, depth + 1)

        idx = (X[:, node.splitAtt] > node.splitValue) * obsIdx
        walk_tree(forest, node.right, treeIdx, idx, X, depth + 1)


def create_tree(X, sample_size, max_height, improved):
    '''
    Creates an isolation tree using a sample of size sample_size of the original data.

    :param X: 2D array with the observations. Dimensions should be (n_obs, n_features)
    :param sample_size: Sample size to run the algorithm.
    :param max_height: Maximum height of the tree.
    :param improved: Choose if using the improved version of the tree or not.
    :return: returns an isolation tree.
    '''
    rows = np.random.choice(len(X), sample_size, replace=False)
    X = X[rows, :]
    return IsolationTree(max_height).fit(X, improved)


class IsolationTreeEnsemble:
    '''
    Isolation Forest.

    Even though all the methods are thought to be public the main functionality of the class is given by:

    - __init__
    - __fit__
    - __predict__

    '''
    def __init__(self, sample_size:int, n_trees:int=10):
        '''
        Creates the algorithm object.

        :param sample_size: sample size from the training observations to fit the model.
        :param n_trees: number of trees in the forest. The higher the less variance.
        '''

        self.sample_size = sample_size
        self.n_trees = n_trees

    def fit(self, X: (np.ndarray, pd.DataFrame), improved: bool=False, n_jobs:int=4):
        """

        Fits the algorithm into a model.

        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.

        Uses parallel computing.

        :param X: 2D array of size (n_obs,n_features). Training observations to fit the algorithm.
        :param improved: If useing the more resilient version or not (True/False respectively)
        :param n_jobs: n_jobs/threads to use in parallel computing.
        :return: the object itself.
        """

        X = X.values if isinstance(X, pd.DataFrame) else X

        limit_height = ceil(np.log2(self.sample_size))

        create_tree_partial = partial(create_tree, sample_size=self.sample_size,
                                      max_height=limit_height, improved=improved)

        with Pool(n_jobs) as p:
            self.trees = p.map(create_tree_partial,
                               [X for _ in range(self.n_trees)]
                               )

        return self

    def path_length(self, X: (np.ndarray, pd.DataFrame)) -> np.ndarray:
        """

        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).

        :param X: Observations from which the path length should be computed.
        :return: Path length for each of the observations.
        """

        X = X.values if isinstance(X, pd.DataFrame) else X

        if not hasattr(self, 'L'):
            self.L = np.zeros((len(X), self.n_trees))

        for treeIdx, itree in enumerate(self.trees):
            obsIdx = np.ones(len(X)).astype(bool)
            walk_tree(self, itree, treeIdx, obsIdx, X)

    def anomaly_score(self, X: (np.ndarray, pd.DataFrame)) -> np.ndarray:
        """

        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.

        :param X: 2D array. Observations from which the anomaly score will be computed.
        :return: 1D array. scores.
        """

        X = X.values if isinstance(X, pd.DataFrame) else X

        # Get the path length for each of the observations.
        self.path_length(X)

        # Compute the scores from the path lengths (self.L)
        if self.sample_size > 2:
            return 2 ** (-self.L.mean(1) / (2 * (np.log(self.sample_size - 1) +
                                                 np.euler_gamma) - 2 * (self.sample_size - 1) / self.sample_size))
        if self.sample_size == 2:
            return 2 ** (-self.L.mean(1))
        else:
            return 0

    def predict_from_anomaly_scores(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        """

        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.

        :param scores: 1D array. Scores produced by the random forest.
        :param threshold: Threshold for considering a observation an anomaly, the higher the less anomalies.
        :return: Return predictions
        """

        return scores >= threshold

    def predict(self, X: (np.ndarray, pd.DataFrame), threshold: float) -> np.ndarray:
        """
        A shorthand for calling anomaly_score() and predict_from_anomaly_scores().

        :param X: Observations to be predicted
        :param threshold: Threshold for considering a observation an anomaly, the higher the less anomalies.
        :return: return the predictions 1/0 if anomaly/not anomaly respectively.
        """

        X = X.values if isinstance(X, pd.DataFrame) else X

        scores = self.anomaly_score(X)
        return self.predict_from_anomaly_scores(scores, threshold)


class IsolationTree:
    '''
    Construct a tree via randomized splits with maximum height height_limit.
    '''
    def __init__(self, height_limit):
        '''

        :param height_limit: Maximum height of the tree.
        '''

        self.height_limit = height_limit

    def fit(self, X: np.ndarray, improved=False):
        """

        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.

        :param X: Observations to fit the tree.
        :param improved: Choose if using the improved version of not (More resilient, less time efficient).
        :return: Returns the tree itself.
        """

        if improved:
            self.root = InNode_improved(X, self.height_limit, 0)
        else:
            self.root = InNode(X, self.height_limit, 0)

        return self.root


class InNode:
    '''
    Node of the tree that is not a leaf node.

    The functionality of the class is:

    - Do the best split from a sample of randomly chosen
        dimensions and split points.

    - Partition the space of observations according to the
    split and send the along to two different nodes

    The method usually has a higher complexity than doing it for every point.
    But because it's using NumPy it's more efficient time-wise.
    '''
    def __init__(self, X, height_limit, current_height):
        '''

        :param X: Observations that have reached the node.
        :param height_limit: height_limit of the tree
        :param current_height: current_height of that node.
        '''
        # declare variables to be used
        n_obs, n_features = X.shape
        next_height = current_height + 1
        limit_not_reached = height_limit > next_height

        # constructing search space (Vectorized)
        self.splitAtt = np.random.randint(0, n_features, 1)[0]
        splittingCol = X[:, self.splitAtt]
        self.splitValue = np.random.uniform(splittingCol.min(), splittingCol.max())

        idx = splittingCol <= self.splitValue

        X_aux = X[idx, :]
        self.left = (InNode(X_aux, height_limit, next_height)
                     if limit_not_reached and X_aux.shape[0] > 1 and np.all(X_aux.max(0) != X_aux.min(0)) else LeafNode(
            X_aux, next_height))

        idx = np.invert(idx)
        X_aux = X[idx, :]
        self.right = (InNode(X_aux, height_limit, next_height)
                      if limit_not_reached and X_aux.shape[0] > 1 and np.all(
            X_aux.max(0) != X_aux.min(0)) else LeafNode(X_aux, next_height))

        self.n_nodes = 1 + self.left.n_nodes + self.right.n_nodes


class InNode_improved:
    '''
    Node of the tree that is not a leaf node. More efficient version of InNode
    in terms of resilient to random choices. Takes the best split from multiple
    possible splits.

    The functionality of the class is:

    - Do the best split from a sample of randomly chosen
        dimensions and split points.

    - Partition the space of observations according to the
    split and send the along to two different nodes

    The method usually has a higher complexity than doing it for every point.
    But because it's using NumPy it's more efficient time-wise.
    '''
    def __init__(self, X, height_limit, current_height):
        '''

        :param X: Observations that have reached the node.
        :param height_limit: height_limit of the tree
        :param current_height: current_height of that node.
        '''
        # declare variables to be used
        n_obs, n_features = X.shape
        next_height = current_height + 1
        limit_not_reached = height_limit > next_height

        # constructing search space (Vectorized)
        check_features = min(20, n_features)
        splitAtts = np.random.choice(n_features, check_features, replace=False)
        splitting_cols = X[:, splitAtts]
        values = np.random.uniform(splitting_cols.min(axis=0),
                                   splitting_cols.max(axis=0),
                                   (5, 1, check_features))

        # selecting best pair feature-value
        n_split = (splitting_cols <= values).sum(-2)
        idx = np.argmax(np.abs(len(X) - 2 * n_split))
        splittingCol, self.splitValue, self.splitAtt = (X[:, splitAtts[idx % check_features]],
                                                        values[idx // check_features, 0, idx % check_features],
                                                        splitAtts[idx % check_features])

        # create children
        idx = splittingCol <= self.splitValue
        X_aux = X[idx, :]
        self.left = (InNode_improved(X_aux, height_limit, next_height)
                     if limit_not_reached and X_aux.shape[0] > 1 and np.all(X_aux.max(0) != X_aux.min(0)) else LeafNode(
            X_aux, next_height))

        idx = np.invert(idx)
        X_aux = X[idx, :]
        self.right = (InNode_improved(X_aux, height_limit, next_height)
                      if limit_not_reached and X_aux.shape[0] > 1 and np.all(
            X_aux.max(0) != X_aux.min(0)) else LeafNode(X_aux, next_height))

        self.n_nodes = 1 + self.left.n_nodes + self.right.n_nodes


class LeafNode:
    '''
    Leaf node

    The base funcitonality is storing the path lenght of the observations in that node.

    For that we use the expectd path lenght + the current height if there are more
    than two observations (self.size) in the node.

    '''
    def __init__(self, X, height):
        '''

        :param X: Observations that have reached the final node.
        :param height: Height / deepness of the leaf node.
        '''

        self.size = len(X)
        self.n_nodes = 1

        if self.size > 2:
            self.path_length = (height + 2 * (np.log(self.size - 1) + np.euler_gamma) - 2 * (self.size - 1) / self.size)
        elif self.size == 2:
            self.path_length = height + 1
        else:
            self.path_length = height


def find_TPR_threshold(y, scores, desired_TPR):
    """

    Evaluate the algorithm via the True Positive Rate (TPR) and
    False Positive Rate (FPR). Computes the threshold
    needed to reach a specific TPR. Returns threshold and FPR.

    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.

    Function provided by instructor.

    :param y: ground truth.
    :param scores: model prediction.
    :param desired_TPR: TPR in 0-1 scale
    :return: threshold, FPR. Threshold for the given TPR and FPR at that threshold.
    """
    threshold = 1.0
    TPR = 0

    while TPR < desired_TPR and threshold >= 0:
        threshold -= .01

        hard_scores = scores > threshold

        TN, FP, FN, TP = confusion_matrix(y, hard_scores).flat

        TPR = TP / (TP + FN)
        FPR = FP / (TN + FP)

    return threshold, FPR
