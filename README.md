# DynoNet
deep learning route generator for the Kilter Board.

An autoregressive Transformer model in PyTorch that generates custom climbing routes for the standard 12x12 Original Kilter Board layout, conditioned on wall angle and difficulty grade.

---

## Technical Overview

* **Target Layout**: 12x12 Original Kilter Board (`product_id = 1` in `kilter.db`).
* **Grid Bounds**: $X \in [1, 35]$, $Y \in [1, 38]$ (includes kickboard offsets $KB1 \to 1, KB2 \to 2$, main board $+ 3$).
* **Role Map**: `0: Padding`, `1: Metadata`, `2: Start`, `3: Middle`, `4: Finish`, `5: Foot`.
* **Architecture**: Decoder-Only Causal Transformer (implemented via PyTorch's `nn.TransformerEncoder` with a square causal mask).

---

## Vocabulary & Mapping

To model the joint distribution $P(X, Y, R \mid \text{context})$ without assuming conditional independence between coordinate heads, $(X, Y, R)$ tuples are flattened into a single unified token ID:

$$\text{Token ID} = X \times (\text{vocab\_y} \times \text{vocab\_r}) + Y \times \text{vocab\_r} + \text{Role}$$

* **Vocab Specs**: $\text{vocab\_x} = 36$, $\text{vocab\_y} = 39$, $\text{vocab\_r} = 6$.
* **Total Unified Vocabulary**: $36 \times 39 \times 6 = 8,424$ tokens.

---

## Sequence Structure

Sequences are fixed to a max length of 41 steps:

1. **Step 0 (Metadata)**: `[Angle, Grade]` vector projected via `nn.Linear(2, d_model)`.
2. **Steps 1..N (Holds)**: Generated hold tokens sorted strictly by $(Y, X)$ ascending.
3. **Steps N+1..41 (Padding)**: Token `0` padded to max length.

```text
[ [Angle, Grade],  Hold_1,  Hold_2,  ...,  Hold_N,  0,  0, ... ]
