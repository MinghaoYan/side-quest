TRIMUL_SYSTEM_MESSAGE = '''
You are an expert Triton engineer tasked with translating PyTorch code into highly optimized Triton kernel code.

You will be implementing a Triangle Multiplicative Update (TriMul) module that is a core operation
for AlphaFold3, Chai, Protenix, and other protein structure prediction models in BioML.

The TriMul operator operates over a 4D tensor of shape [B, N, N, C].

Your task:
- Implement the "outgoing" version of the TriMul operator from the AlphaFold3 paper.
- You will not have to compute or store gradients for this version. You will only need to implement the forward pass.

Your function should be defined as 'custom_kernel' with the following signature:
Input:
- `data`: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
    - input: Input tensor of shape [bs, seq_len, seq_len, dim]
    - mask: Mask tensor of shape [bs, seq_len, seq_len]
    - weights: Dictionary containing model weights
    - config: Dictionary containing model configuration parameters

Output:
- output: Processed tensor [bs, seq_len, seq_len, dim]

**Problem Constraints:**
- B ∈ {1,2}, N ∈ {128,256,512,1024}, c ∈ {128}, c_z ∈ {128,384,768}
- The input distribution will be sampled from a standard Normal distribution, or a heavy-tailed Cauchy distribution (gamma = 2).
- There will either be no mask, or a randomly sampled mask over the inputs.

**Remarks.** So why is this problem so annoying? Because you have to choose whether to load / deal with either the channel dimensions c,c_z that the LayerNorms require (otherwise you have to do a synchronize to compute the statistics like mean / variance) or the sequence dimension N.
The sequence dimension is particularly annoying because it's quite large, but also because we compute pair-wise operations at the last operation that sum over another sequence dimension (this is N^3!).
However, I really like this kernel because it only consists of "simple" operations, and is really easy to understand. It is a true test of "fusions" that torch.compile() doesn't do that well.

Here is a pytorch implementation of the TriMul module. You will want to implement a kernel for the operations in the forward call:

```python
import torch
from torch import nn, einsum
import math

# Reference code in PyTorch
class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [bs, seq_len, seq_len, dim]
        mask: [bs, seq_len, seq_len]

        Returns:
            output: [bs, seq_len, seq_len, dim]
        """
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left, right)
        # This einsum is the same as the following:
        # out = torch.zeros(batch_size, seq_len, seq_len, dim, device=x.device)

        # # Compute using nested loops
        # for b in range(batch_size):
        #     for i in range(seq_len):
        #         for j in range(seq_len):
        #             # Compute each output element
        #             for k in range(seq_len):
        #                 out[b, i, j] += left[b, i, k, :] * right[b, j, k, :]

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)
```

Here is some example skeleton code of the entrypoint function you will create:
```python
def custom_kernel(data)
    input_tensor, mask, weights, config = data
    dim, hidden_dim = config["dim"], config["hidden_dim"]

    # Access the given weights of the model
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    left_proj_weight = weights["left_proj.weight"]
    right_proj_weight = weights["right_proj.weight"]
    left_gate_weight = weights["left_gate.weight"]
    right_gate_weight = weights["right_gate.weight"]
    out_gate_weight = weights["out_gate.weight"]
    to_out_norm_weight = weights["to_out_norm.weight"]
    to_out_norm_bias = weights["to_out_norm.bias"]
    to_out_weight = weights["to_out.weight"]

    # Perform TriMul

    return out
```

To help you understand which triton version we are using, here is some example triton code for an unrelated task:
```python
import triton
import triton.language as tl

@triton.jit
def matmul_persistent_ws_kernel(
   a_ptr, b_ptr, c_ptr, M, N, K,
   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
   pid = tl.program_id(axis=0) # async_task 0, 1, 2
   num_pid_m = tl.cdiv(M, BLOCK_M) # async_task 0, 1, 2
   num_pid_n = tl.cdiv(N, BLOCK_N) # async_task 0, 1, 2
   pid_m = pid // num_pid_m # async_task 0, 1, 2
   pid_n = pid % num_pid_n # async_task 0, 1, 2
   offs_m_1 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M // 2) # async_task 0, 1, 2
   offs_m_2 = pid_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M) # async_task 0, 1, 2
   offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_N) # async_task 0, 1, 2
   offs_k = tl.arange(0, BLOCK_K) # async_task 0
   a_ptrs_1 = a_ptr + (offs_m_1[:, None] * stride_am + offs_k[None, :] * stride_ak) # async_task 0
   a_ptrs_2 = a_ptr + (offs_m_2[:, None] * stride_am + offs_k[None, :] * stride_ak) # async_task 0
   b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn) # async_task 0
   acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 1
   acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 2
   for k in range(0, tl.cdiv(K, BLOCK_K)): # async_task 0, 1, 2
       a_1 = tl.load(a_ptrs_1)   # async_task 0
       a_2 = tl.load(a_ptrs_2)   # async_task 0
       b = tl.load(b_ptrs)   # async_task 0
       acc_1 += tl.dot(a_1, b)   # async_task 1
       acc_2 += tl.dot(a_2, b)   # async_task 2
       a_ptrs_1 += BLOCK_K * stride_ak # async_task 0
       a_ptrs_2 += BLOCK_K * stride_ak # async_task 0
       b_ptrs += BLOCK_K * stride_bk # async_task 0
   c_1 = acc_1.to(tl.float16) # async_task 1
   c_2 = acc_2.to(tl.float16) # async_task 2
   c_ptrs_1 = c_ptr_1 + stride_cm * offs_m_1[:, None] + stride_cn * offs_n[None, :] # async_task 1
   c_ptrs_2 = c_ptr_2 + stride_cm * offs_m_2[:, None] + stride_cn * offs_n[None, :] # async_task 2
   tl.store(c_ptrs_1, c_1) # async_task 1
   tl.store(c_ptrs_2, c_2) # async_task 2
```

A few general triton tips:
- tl.arange only takes in constexpr arguments (static or tl.constexpr)
- You cannot use continue in your kernel code
- tl.dot can only take in two input tensors
- There is no tl.mean

Here are the different configs that your kernel will be tested on ("nomask" sets whether there will be no mask, or a randomly sampled mask over the inputs):

Test Cases for correctness and runtime (optimize runtime for these):
  - {"seqlen": 256, "bs": 2, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
  - {"seqlen": 768, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
  - {"seqlen": 256, "bs": 2, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
  - {"seqlen": 512, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
  - {"seqlen": 1024, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
  - {"seqlen": 768, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
  - {"seqlen": 1024, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": True, "distribution": "normal"}

Rules:
- The tensors arguments passed in will be already on your cuda device.
- Define all of your code in one final ```python ``` block.
- We will test the correctness of your kernel on multiple input shapes, make sure to support different potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is in float32.
- You must use trition 3.3.1 and these kernels will be run on an H100.
- You do not have to implement everything in triton, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your algorithm.
'''.strip()

SYSTEM_PROMPT = TRIMUL_SYSTEM_MESSAGE

TRIMUL = '''
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output
'''

CODING_REQ = """
While completing your task, you MUST:
- Enclose your code in triple backticks to properly format the code in Markdown.
- Your code will replace the content between RegexTagCustomPruningAlgorithmStart and RegexTagCustomPruningAlgorithmEnd in trimul_1.py.
- You MUST keep the public entrypoint `custom_kernel(data)`.
- The evaluator will call `custom_kernel(data)` where `data` is `(input_tensor, mask, weights, config)`.
- You MAY add helper functions, Triton kernels, or helper classes inside the editable block.
- Your final output tensor must be `torch.float32`.
- Each triple-backtick enclosed code block must contain valid Python and a valid implementation.
"""

BACKGROUND = """
### Background on Triangle Multiplicative Update (TriMul)

TriMul is a core AlphaFold-style operator over a 4D tensor `[B, N, N, C]`. The forward pass applies layer normalization, gated projections, a triangle multiplicative interaction over the sequence axis, then output normalization and projection.

This task is evaluated on both correctness and runtime. The evaluator first checks all reference test cases and then benchmarks the candidate on 7 benchmark cases. The final score is:

`Kernel speedup = SCORE_SCALE / geom_mean_us`

Higher is better, so faster kernels produce larger scores.
"""

TASK_INTRO = """
You are an expert Triton and PyTorch kernel engineer. Your goal is to optimize the TriMul forward pass while preserving correctness across multiple shapes, masks, and input distributions.
"""


def construct_mutation_prompt(sota_algorithm, ablation_list):
    ablation_descriptions = "\n".join(ablation_list)
    return f"""
We are conducting an evolutionary optimization process for the Triangle Multiplicative Update (TriMul) kernel.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### System guidance
{SYSTEM_PROMPT}

### Current state-of-the-art
```python
{sota_algorithm}
```

# Knowledge base

{ablation_descriptions}

## Your Task

Brainstorm 5 concrete implementation options, explain the tradeoffs, then select one option to implement. Balance ambitious kernel fusion ideas with practical correctness-preserving changes. Avoid repeating experiments that have already been tried.
"""


def construct_idea_gen_prompt(sota_algorithm, idea_repo):
    return f"""
We are conducting an evolutionary optimization process for the Triangle Multiplicative Update (TriMul) kernel.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### System guidance
{SYSTEM_PROMPT}

### Current state-of-the-art
```python
{sota_algorithm}
```

### Idea Repo
{idea_repo}

## Your Task

Generate 3 distinct ideas that could improve TriMul runtime without breaking correctness. Use this format:

Idea 1
Hypothesis: <idea>
Reasoning: <reasoning>

Idea 2
Hypothesis: <idea>
Reasoning: <reasoning>

Idea 3
Hypothesis: <idea>
Reasoning: <reasoning>
"""


def construct_idea_select_prompt(sota_algorithm, idea_repo):
    return f"""
We are conducting an evolutionary optimization process for the Triangle Multiplicative Update (TriMul) kernel.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
```python
{sota_algorithm}
```

### Idea Repo
{idea_repo}

## Your Task

Select one idea to test next and describe the exact experiment to implement.

Idea ID: <Idea ID>
Experiment description: <concrete implementation change to test>
"""


def construct_idea_select_no_code_prompt(sota_algorithm, idea_repo):
    return construct_idea_select_prompt(sota_algorithm, idea_repo)


def construct_code_impl_prompt(
    sota_algorithm,
    idea_id,
    exp_description,
    selected_idea_text=None,
):
    idea_section = f"""Idea ID: {idea_id}
Experiment description: {exp_description}"""
    if selected_idea_text:
        idea_section = f"""Selected idea:
{selected_idea_text}

Idea ID: {idea_id}
Experiment description: {exp_description}"""

    return f"""
We are conducting an evolutionary optimization process for the Triangle Multiplicative Update (TriMul) kernel.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
```python
{sota_algorithm}
```

## Your Task

Implement the selected experiment.

{idea_section}
"""


def construct_gen_hypothesis_prompt(sota_algorithm, idea_repo, idea):
    return f"""
We are conducting an evolutionary optimization process for the Triangle Multiplicative Update (TriMul) kernel.

{BACKGROUND}

{TASK_INTRO}

{CODING_REQ}

### Current state-of-the-art
```python
{sota_algorithm}
```

### Idea Repo
{idea_repo}

## Your Task

Propose the next experiment for idea {idea.id}.

Idea ID: {idea.id}
Experiment description: <concrete implementation change to test>
"""


TOURNAMENT_PROMPT = "\n"

SUMMARIZE_EVAL_PROMPT = """
## Your Task

Provide a concise summary of this TriMul experiment. First write a short paragraph, then provide exactly 1 bullet starting with `-`. In the bullet, begin with `Results: Kernel speedup: <score>` and then state the most useful lesson from the experiment.
"""

EVAL_DESCRIPTION_PROMPT = """
### Candidate results
The table presents the TriMul evaluation results.

### Understanding metrics
`Kernel speedup = SCORE_SCALE / geom_mean_us`, so higher is better.
"""

HPARAM_PROMPT = ""
