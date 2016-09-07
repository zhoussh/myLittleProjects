##Theory
algorithm from paper[Ad Click Prediction: a view from the trenches], this paper show the
details of the Per-Coordinate FRTL-Proximal with L1 and L2 Regularization for LR

Data Format:
    csv data format
    id  click:1 1:val1  2:val2  3:val3 .... 


>Input : alpha, beta, lambda1, lambda2, features vector **xi**
>Initial : coordinate or dimension i = [1, 2, ,3 ... d], zi = 0, ni = 0

>for t = 1 to T do:
>   get a set I = {i|xi != 0} from vector **xi**
>   for i in I.item:
>       if abs(zi)<=lambda1  wt,i = 0 
>       else wt,i-(((beta+sqrt(ni))/alpha + lambda2)^(-1))*(zi-sgn(zi)*lambda1)
>   Predict probability pt = sigmoid(**xt** * **w**) 
>   
>   for all i 