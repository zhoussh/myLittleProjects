###Abstract
For very large dataset or sparse dataset
1. PMF model, linearly scale with observations, large, sparse, imbalance dataset
2. adaptive and prior on model parameters, control model automatically
3. constrained version PMF model based on the assumption that : user who have rated similar sets of movies are likely to have similar preferences.
4. multiple PMF models linearly combined with predictions of RBM model, get better result

###Introduction
1.low-dimensional factor models
2.unobserverd factors determine the users' preferences and attitudes
3.linear factor model: user's preference linearly combining item factor vectors using user-specific confficients
4.fail in netfilx prize for two reasons : 
5. reason1 : can not sclae linearly with large datasets
6. reason2 : can not handle with sparse datasets
7. the whole structure of this paper : chapter2 -> present PMF model : user preference matrix with product of lower-rank user and movie matrix. chapter3 -> extend PMF model to include adaptive priors over movie and user feature vectors, show how these priors control model complexity automatically. chapter4 -> show constrained version of PMF model based on assumption the user who rate similar sets of movies have similar preferences. chapter5->report experimental result that PMF considerably outperforms standard SVD model, constrained PMF and PMF with learnable priors improve model performance.
8. summary: three models: basic PMF, constrained PMF, bayes PMF. performace 
9. result compare: constrained PMF > bayes PMF > basic PMF > SVD