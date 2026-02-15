# PyTorch Fundamentals: Tensors & Autograd

A practical guide to understanding the core mechanics of PyTorch. Even if you're familiar with these topics, a quick review can clarify concepts you use daily.

---

## 1. Tensor Basics

**Start here:** [PyTorch Tensor Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)

**Key concepts to understand:**
- Tensor attributes (shape, dtype, device)
- Operations: in-place vs. creating copies
- Moving tensors between CPU/GPU
- What `requires_grad` does

---

## 2. How Tensors Work Under the Hood (Optional but Valuable)

**Read:** [PyTorch Internals - Views, Strides, and Memory](https://blog.ezyang.com/2019/05/pytorch-internals/)

**Why this matters:**  
You'll frequently use `.view()`, `.reshape()`, and `.contiguous()` when training models. Understanding views vs. copies helps you avoid subtle bugs—like when shifted logits need to be contiguous during LLM fine-tuning.

**Interactive tool:** [Stride Visualizer](https://ezyang.github.io/stride-visualizer/index.html) - See how tensors are actually laid out in memory

---

## 3. Autograd: How Gradients Flow

Autograd automatically computes gradients during backpropagation. Here's what you need to know:

### Core tutorials (read in order):
1. [A Gentle Introduction to torch.autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
2. [Automatic Differentiation Basics](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)  
   - Pay attention to leaf nodes, `retain_graph`, and the side notes
   - **Important sections:**
     - [Disabling gradient tracking](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#disabling-gradient-tracking) (you'll use this constantly)
     - [Computational graphs](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#more-on-computational-graphs)
     - [Tensor gradients and Jacobian products](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#optional-reading-tensor-gradients-and-jacobian-products) (generalized backprop)

3. [Leaf vs. Non-leaf Tensors](https://docs.pytorch.org/tutorials/beginner/understanding_leaf_vs_nonleaf_tutorial.html)  
   **Key insight:** Only parameters (leaf tensors) store gradients by default. Intermediate activations don't.

4. [Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)  
   Read all sections except "Complex Calculus" (rarely needed in practice)

---

## 4. Practical Topics

- [Zeroing Gradients](https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html) - Why and when to call `optimizer.zero_grad()`
- [What is torch.nn really?](https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html) - Why we use `nn.Module` instead of raw tensors

---

**Bottom line:** Understanding these fundamentals will save you hours of debugging and help you write more efficient training loops.