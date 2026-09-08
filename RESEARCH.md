From a technical standpoint, the Kilter Board uses standard Bluetooth Low Energy (BLE) GATT services to write LED color state arrays to the board controller. While reverse-engineering the byte payload format is straightforward, official API/BLE documentation from Kilter / Setter Closet is required if you intend to commercialize or distribute an app to avoid breaking changes or policy violations.

# Research Directions: Spatial Attention for Climbing Models

## 1. Distance-Aware Attention Biases (Euclidean Attention)

Standard self-attention computes connection weights purely from learned semantics:

$$
\text{Attention}(Q, K, V)
=
\text{softmax}\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

You can introduce an explicit spatial distance penalty directly into the attention matrix before the softmax step:

$$
\text{Attention}(Q, K, V)
=
\text{softmax}\left(
\frac{QK^T}{\sqrt{d_k}}
-
\gamma \cdot D_{\text{Euclidean}}(i,j)
\right)V
$$

where the Euclidean distance between holds $i$ and $j$ is:

$$
D(i,j)
=
\sqrt{
(x_i-x_j)^2 + (y_i-y_j)^2
}
$$

### Research Goal

Test whether explicitly penalizing or favoring attention weights based on physical coordinate distance helps the model learn human reach limitations faster than relying purely on learned embeddings.

---

## 2. 2D Rotary Positional Embeddings (2D RoPE)

Standard 1D positional encodings tell the model about token sequence order, but fail to explicitly represent 2D spatial relationships such as:

> "Hold B is 2 units above and 1 unit right of Hold A."

### Approach

Adapt 2D Rotary Positional Embeddings (2D RoPE), commonly used in Vision Transformers, to split embedding dimensions between the $X$ and $Y$ coordinates.

The model can then encode spatial position directly into the query and key representations, allowing their dot product to naturally reflect relative 2D relationships regardless of where the holds appear in the sequence.

### Research Goal

Test whether 2D RoPE improves the model's ability to learn spatial relationships between climbing holds compared with standard 1D positional encodings.

---

## 3. Permutation-Invariant Hold Grouping

In climbing datasets, human setters may record holds in varying sequence orders. For example:

* Foot holds may be placed before start holds.
* Hand holds may be listed left-to-right or right-to-left.
* Intermediate holds may appear in different orders across routes.

This introduces arbitrary ordering information that does not necessarily correspond to the physical structure of the problem.

### Approach

Explore order-invariant attention mechanisms for holds that share the same vertical height ($Y$-level).

The goal is to force the Transformer to treat intermediate hold choices as **spatial sets** rather than arbitrary 1D sequences.

### Research Goal

Test whether permutation-invariant grouping improves generalization across routes with different hold-order conventions and reduces the model's dependence on arbitrary dataset ordering.

