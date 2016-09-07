input file format:
    0 qid:1 docid 1:3 2:0 ... 
    2 qid:1 docid 1:2 2:2 ...  
 
parameters:
    M : total boosting iteration number, M rounds
    N : total documents number
    L : number of leaf nodes for each regression
    v : shrinkage coefficient
LambdaMART algorithm:
1:  for i = 0 to N do           //for every document
2:      F0(x_i) = BaseModels(x_i)
3:  end for
4:  for m = 1 to M do         
5:        for i = 0 to N do
6:          y_i = lambda_i          //calculate for every documents
7:          omega_i = dy_i/dF(x_i)  //calculate second derivative using lambda-gradient
8:      end for
9:      {R_lm}L_l=1                 //L terminal nodes for per tree
10:     for l = 0 to L do
11:         lambda_lm = sum(y_i) 
12:     end for
13:     for i = 0 to N do
14:         F_m(x_i) = F_m-1(x_i) + vsum()
15:     end for
16: end for

compare GBDT split algorithm:
sklearn version:
    Input : train data, ordered features using index
    many criterons, including and default is MSE, so we select this version
xgboost version:
    Input : train data, ordered features using index
    using gain, enumerate all possible trees, every time optimize a tree
    pruning trees to speed up

**data structure** 
leaves{

}


**work flow** :
data_list

group_by_docid():
    input format --> label qid docid features --> samples -->

for tree in trees:
    for document in documents:
        calculate lambda, omega per document
        (lambda gradient, and second derivatives omega)
    1.find best split in the regression trees using data(train data labels, lambda)
    using features return features orders, MSE criterion
    2.update leaves values using Newton method
      input : regression tree, samples 
      leaves = regression_tree.leaves
      for leave in leaves:
        for doc in docs:(for sample in samples:)
           s1 += pseudoResponse[idx]
           s2 += weight[idx]
        if s2 == 0:
            leave = 0
        else:
            leave = s1/s2
    3.update model scores using Newton method
      for leave in leaves:
        for doc in docs:((for sample in samples:)
            sum += leaves_value *
    4. CV:(TO DO)
        training data --> model -->validation scores
        rolling back to add or remove tree
       
Ref:
Q. Wu, C.J.C Burges, K. Svore and J. Gao. Adapting Boosting For Information Retrieval Measures. 2009


