The update function for the neurons is defined as:
![εικόνα](https://github.com/Orepap/TMDC/assets/93657525/6026bdde-6ead-4ab5-8981-a94d7a831fed)  

where W_j (i+1) is the matrix of neuron with index j at time (i+1), with i being the current iteration, X is the matrix of the input sample, W_q is the BMU, and y is the learning rate, which follows an exponential reduction:
              y= y_o  exp⁡〖((-i)/t_1 ),〗	(2)
              
where y_o is the initial learning rate, t_1 is a user defined constant which controls the exponential decrease of the learning rate, and U(W_j,W_q,i) is the neighborhood function, which dictates the cooperation between neurons. It decreases exponentially and includes a reducing Gaussian distance function [14]:
         U = exp((-〖〖 d〗_jq〗^2)/(2 〖(σ_(0 ) exp((-i log⁡(σ_0 ))/t_2 ))〗^2 )),	(3)
         
where σ_0 is the standard deviation of the initial Euclidean distances of the randomly initiated neurons, t_2  is a user defined constant which controls the exponential decrease of the neighborhood function and lastly, d_jq is the Euclidean distance between a neighbor neuron and the BMU, which is calculated using the Frobenius norm of the neuron matrices difference:
        ‖W_j-W_q ‖_F= ‖D‖_F=√(∑_(k=1)^m▒∑_(l=1)^n▒|D_kl |^2 ) 	(4)
