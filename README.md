# nn-cert-individual-fairness

This repository contains the code to reproduce the results of the paper [Certifiable Individual Fairness Guarantees for Neural Networks](...), which presents techniques to (1) certify individual fairness of fully connected neural networks, and (2) train networks that are fair by construction.

### Install

In the paper we use the following datasets: [Adult](https://archive.ics.uci.edu/ml/datasets/Adult),
[Credit](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#), [German](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data\)) and [Crime](http://archive.ics.uci.edu/ml//datasets/Communities+and+Crime). However, some preprocessing is required on the raw data to be used in our code. For convenience we share the cleaned data, which can be accessed using [git lfs](https://git-lfs.github.com/)(please follow installation instructions on their website).

Install Python dependencies found in the `requirements.txt` file.

Finally, this repository relies on using optimisation solvers. Ensure you have CBC and Gurobi installed. Note, Gurobi is not strictly necessary, but you will have to substitute references to it in the experiment scripts to fall back to using just CBC.

### Reproduce experiments

To simply reproduce the results follow the instructions in `plots/ReproducePlots.ipynb`.

We use [Mlflow](https://www.mlflow.org/) to track the experiments results. The flow is you run a script, that executes the experiments and tracks the results in mlflow. You then fetch parameters and metrics to plot the results.

If you want to view the experiments results through the mlflow UI, you can run `mlflow ui` on the terminal.



