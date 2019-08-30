## mvatrain

mva misc. for signal/(mis-reconstruction, background) discrimination.

also includes work done over the summer of 2019 to investigate alternate model trainings and BDT improvements.

for an index of folders and files, see index.txt.

large data files have been compressed into tar.gz files in order to push to GitHub.

the primary notebook used during the project was **chemex.ipynb**.

Dependencies:

- [`scikit-learn`](https://scikit-learn.org) General ML package, various tools, metrics
- [`xgboost`](https://xgboost.ai/) Powerful state-of-art GBRT implementation
- [`bayesian-optimization`](https://github.com/fmfn/BayesianOptimization) Hyperparameter optimization

---

### How to add

```bash
conda install scikit-learn
conda install xgboost
conda install bayesian-optimization
```
