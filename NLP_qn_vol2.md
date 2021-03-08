## Volume 2

1. **Perplexity?** is e^Entropy. Hence the lower the better a LM is.

2. **ReLU issues?** 3 issues: exploding gradient (clip it), no learning at zero (leaky), not normalized. 

3. **SVD vs NN** for embedding of latent? SVD uses linear combination of input, NN uses non linear. 

4. **hidden and cell** hidden state is all the info thus far, it's the "Short term" part of lstm. Cell is selectively updated by the gates to retain info that may be important in the future (e.g grammar syntax). Hence "Long term". 

5. **LSTM** parameters? 4(𝑚h+h²+h). Let's break it down: 4 is the number of gates (input gates, forget,output). mh is for input weights W, h^2 is for hidden weights U, h is for the bias. 

6. **Time complexity of LSTM** ?  O(mh^2)

7. **Time complexity of Transformer** ?  O(m^2h). If h<< m then Transformer faster than LSTM. 

8. **What is attention** ? It sees how relevant itself is to other words. In terms of computation, it's better as well.

8.1 **Why is it better** ? Transformer see the whole sequence at once rather than wait like RNN, so it can model dependencies better. Each attention head captures a certain type of dependency, so multi head is needed to capture a variety of them. 

8.2 **LM without Dropout** Albertv2 due to its parameter sharing regularization effect. It has 18x less params, so the sharing has an effect. 

8.3 **Bert varieties** Bert base and large. DistilBert infer faster, with minor drop in accuracy. Accuracy wise Roberta should be better. Embedding is also factorized.

9. **Adams' issues**? it doesnt generalize as well as SGD with momentum, and the starting LR shud be small to avoid divergence -> CosineAnnealing with Warm Restart. 

10. **LAMB and LARS**? They are both SGD. LARS adjust lr by layerwise trust ratio. LAMBS clip it. trust ratio is similar to leap of faith, when u r confident then u make larger steps. It's norm of layer weights/ gradients update. 

11. **Lr scheduler**? CosineAnnealing and One Cycle Policy. 

12. **Text Encoder**? CBOW, ELmo, 

13. **ULMFiT** how to do transfer learning effectively. 

14. **Bert vs GPT** Bert is bi and GPT is straight. GPT uses decoder and Bert uses encoders. 

15. **BPE** ? A smart tokenizer by GPT