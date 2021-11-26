# Sphere Fitting with Neural Guided RANSAC

## About

This project is a demonstration of a simple sphere fitting algorithm using a neural network to guide the RANSAC algorithm. The algorithm is based on the [NG-RANSAC algorithm](https://arxiv.org/abs/1905.04132) and is used to find the best fitting sphere to a set of points.

<p align="center">
    <img src="./media/gt.png" alt="Ground truth"/>
    <img src="./media/probs.gif" alt="Probability Distribution learned by Neural Network"/>
</p>
<p align="center">
    <em>Fig 1-2. On the left plot you can observe the ground truth shape, and on the right plot you can see the probability distribution that is learned by NN to guide inlier selection for sphere fitting.</em>
</p>