# learning-neural-network
learning a neural network

about 20% made by AI, 80% by me.

# the NN folder:
## from the basic linear to the MLP model
### liner.cpp: 
- the basic linear model
### layer.cpp: 
- MLP but hard-coded layer and input output size
- used to predict the sine function
### layer2.cpp:
- separated layer of MLP part as a struct called "Layer", input and output size can be easily changed
- input x, output sin(x*2pi) and sin(x*4pi) in the same time
### layer3.cpp:
- no special change in the model structure
- Use a Fourier-like input to recreate the original value
### layer4.cpp:
- training as a function f(x), which f(f(x)) = (2*x*x+1)/3
- make a step closer to RNN
- layer4_graph.py is used to see how layer4.cpp acts when dealing with a strange function
- an interesting experiment

# the RNN folder:
## from MLP to multi-layer fake RNN
### A_pre.cpp:
- fully rewrite the MLP structure
- train more frequently on large-loss testcases
- as a stracture test, input x,y and output (x+y)/2
- added a simple gradient fixer
### B_fib.cpp:
- a fake RNN, which is just a multi-layer MLP add a single memory, not to every single layer
- Use one-hot code to train this model
### C_sinwave.cpp:
- file-io added, though this part is mostly made by al
- Try to predict the whole sine wave by just giving the first few points
- doesn't work well, the AI fixed version(C_sinwave_AI.cpp) also doesn't work well
- Maybe it's due to the incorrect stacked-RNN implementation, and this is somehow too hard for this model

# the RNN2 folder
## from MLP to stacked-RNN
### A_st.cpp:
- fully rewrite the model structure
- train all parameters in a single for loop
- can change active function easily
### B_fib.cpp:
- single-layer RNN
### C_fib2.cpp:
- stacked RNN
