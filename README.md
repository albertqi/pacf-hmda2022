# An Audit of Historical Mortgage Approval Data in Vermont Using PAC and Metric-Fair Learning
## Albert Qi and Ronak Malik
### Spring 2024

Decision-making algorithms are being used more often in salient situations. One area in which there is rising concern that algorithms might perpetuate or augment existing biases is evaluating credit risk and lending. The loan approval process is usually opaque and functions with impunity. We want to further investigate this topic by building a fair model that determines the risk taken on by a bank and ultimately predicts an approval or denial of a mortgage using mortgage application information.

We develop a (dis)similarity metric that uses a human arbiter. We then train an unconstrained linear classifier on historical mortgage approval data in Vermont, reaching convergence at 83% accuracy. Using our metric, we then retrain our linear classier under a set of fairness constraints, showing that there is no significant difference in accuracy on the same data. This implies a possibly fair mortgage approval process in Vermont.
