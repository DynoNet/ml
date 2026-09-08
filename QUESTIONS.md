Questions:

1. Using embbedings loses the spatial awareness and the model has to relearn it by itself compared to continuous normalized values?

2. We use nn.TransformerDecoder rather than an Encoder because autoregressive generation requires causal masking. Token $t$ is only allowed to attend to tokens $\le t$, preventing the model from looking ahead at future holds. Explain ?

3. In the original code: - Features were embedded to smaller dimensions ($32, 64, 32$), concatenated into a $128\text{D}$ vector, and passed through an `nn.Linear` projection layer. - **Effect:** Adds extra trainable parameters (`nn.Linear`) to explicitly mix the three features before the Transformer layers. In the current code: - Features are embedded directly to $128\text{D}$ and added together ($X + Y + \text{Role}$). - **Effect:** Eliminates the `nn.Linear` projection layer, saving parameters and compute. This mirrors how standard Transformers combine token and positional embeddings. Both methods work in practice:- **Summation** is lighter on parameters and relies on the Transformer's self-attention layers to disentangle feature interactions. - **Concat + Projection** gives the model a dedicated linear layer to mix features before self-attention, at the cost of slightly more parameters.

4. Why is the time per epoch for the TransformerDecoder ~90seconds on my hardware while the TransformerEncoder one is ~60s ? Update to the code now Encoder is also 100s.

5. Why can we freely swap encoders for decoders as if their fundamental mode of working isn't different ?

6. What did the project EDA's tell us about the way the model works ? Give me a list of problems and how EDA nudges us into finding them.

7. How did the model peek into the future ?

8. Why did the model output 99% middle holds ?  

------------------------------
Hold   | X      | Y      | Role  
------------------------------
1      | 28     | 2      | 5     
2      | 6      | 2      | 5     
3      | 6      | 2      | 5     
4      | 10     | 2      | 5     
5      | 6      | 2      | 5     
6      | 28     | 2      | 5     
7      | 28     | 2      | 5     
8      | 28     | 2      | 5     
9      | 6      | 2      | 5     
10     | 10     | 1      | 5     
11     | 6      | 2      | 5     
12     | 6      | 10     | 5     
13     | 6      | 10     | 5     
14     | 6      | 2      | 5     
15     | 6      | 2      | 5     
16     | 6      | 2      | 5     
17     | 28     | 2      | 5     
18     | 6      | 2      | 5     
19     | 6      | 2      | 5     
20     | 6      | 2      | 5     
21     | 6      | 2      | 5     
22     | 6      | 34     | 5     
23     | 28     | 2      | 5     
24     | 6      | 2      | 5     
25     | 6      | 2      | 5     
26     | 28     | 2      | 5     
27     | 6      | 2      | 5     
28     | 6      | 2      | 5     
29     | 6      | 2      | 5     
30     | 6      | 2      | 5     
------------------------------
Total holds: 30

This Actually changed to 100% holds of role 3 and then fixed by nroamlizing angle and difficulty before putting them in.

9. why did the model repeat so many hold positions ? it's like it doesn't see the flow of these climbs
Not sure if we changed the generation too much what changed was normalizing the input
I don't even think we have anti dupplicates in the current LLM-written code for generation and it works :)

10. How would 10 more epochs affect the results ? 
Not too well lol.

11. How would more layers affect the results ? 

12. How would different hyperparameters affect the results ?

13. recap how these work
        #How can you reproject the output of the decoder on these?
        self.x_ll = nn.Linear(d_model, vocab_x)
        self.y_ll = nn.Linear(d_model, vocab_y)
        self.r_ll = nn.Linear(d_model, vocab_r)

14. #TODO: Understand the decoding
def decode_token(token_id: int, vocab_y: int, vocab_r: int) -> tuple[int, int, int]:
    r = token_id % vocab_r
    temp = token_id // vocab_r
    y = temp % vocab_y
    x = temp // vocab_y

    return x, y, r

    where token_id = x * (vocab_y * vocab_r) + y * vocab_r + r
