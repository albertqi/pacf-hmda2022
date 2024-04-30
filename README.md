# CS 226r
## Albert Qi and Ronak Malik
### Spring 2024

Steps:
- Make the violation loss differentiable
- Show that the new violation loss holds the same properties as the one proposed in PACF
- Implement projected gradient descent by differentiating on the violation loss:
    - If we're not fair (i.e. not in the set of valid hypothesis, perform gsd on violation loss until we are back in the set)
    - otherwise, optimize error loss
- Show that this version of PGSD is valid
- Implement Ilvento's algorithm to create fairness metric.
