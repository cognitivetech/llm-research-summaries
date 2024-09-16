# A Hierarchical Bayesian Language Model based on Pitman-Yor Processes (Summary)

by Yee Whye Teh

https://www.stats.ox.ac.uk/~teh/research/compling/acl2006.pdf

## Contents
- [Introudction](#introudction)
- [2 Pitman-Yor Process](#2-pitman-yor-process)
- [3 Hierarchical Pitman-Yor Language](#3-hierarchical-pitman-yor-language)
- [4 Hierarchical Chinese Restaurant Processes](#4-hierarchical-chinese-restaurant-processes)
- [5 Inference Schemes](#5-inference-schemes)
- [6 Experimental Results](#6-experimental-results)
- [7 Discussion](#7-discussion)

## Introudction
**Background**:
- Proposed new hierarchical Bayesian n-gram language model using Pitman-Yor processes
- Makes use of Pitman-Yor processes, which produce power-law distributions closer to those in natural languages
- Shows that an approximation to the hierarchical Pitman-Yor language model recovers the exact formulation of interpolated Kneser-Ney (IKN)

**Importance**:
- Previously, Bayesian probabilistic models had poor performance compared to other smoothing methods
- This paper addresses that issue by proposing a novel hierarchical Pitman-Yor language model
- Demonstrates that IKN can be interpreted as an approximate inference scheme in the hierarchical Pitman-Yor language model, producing superior results

**Key Findings**:
1. Proposed a new hierarchical Bayesian n-gram language model using Pitman-Yor processes
2. Shows that IKN can be interpreted as an approximate inference scheme in the hierarchical Pitman-Yor language model
3. Demonstrates superior performance compared to interpolated and modified Kneser-Ney, and the hierarchical Dirichlet language model

**Methodology**:
1. Introduces the Pitman-Yor process as a generalization of the Dirichlet distribution
2. Proposes a hierarchical Pitman-Yor language model where each hidden variable is distributed according to a Pitman-Yor process
3. Describes an efficient Markov chain Monte Carlo sampling scheme for inference
4. Verifies the correspondence between interpolated Kneser-Ney and the Bayesian approach
5. Provides experimental comparisons with interpolated, modified Kneser-Ney, and the hierarchical Dirichlet language model

## 2 Pitman-Yor Process

**Pitman-Yor Process**

**Description**: Non-parametric Bayesian model used to estimate probabilities of words in a language model.

**Parameters**:
- **d**: Discount parameter (0 <= d < 1)
- **θ**: Strength parameter (theta > -d)
- **G0**: Mean vector, usually set uniformly for all words

**Prior Distribution**: G ~ PY(d, θ, G0)

**Distribution over Sequences**: Used for language modeling

**Generative Procedure**:
1. First word x1 is assigned the value of the first draw y1 from G0
2. Subsequent words xc+1 are either:
   - Assigned the value of a previous draw yk with probability θ + ck (increment ck)
   - Or assigned the value of a new draw from G0 with probability θ + dt and draw yt
3. Repeat until sequence of words is generated

**Properties**:
1. Rich-gets-richer clustering: More words assigned to a draw, more likely subsequent words will be assigned to it.
2. Proportion of rare words increases with number of draws from G0 and d value.
3. Number of unique words grows as O(θTd) for large T.
4. For d = 0, distribution is Dirichlet, growth is slower at O(theta log T).
5. Metaphor: Chinese restaurant process, customers sit at tables (draws from G), each new one may join an existing table or create a new one (draw from G0).

## 3 Hierarchical Pitman-Yor Language

**Hierarchical Pitman-Yor Language Model**

**Background:**
- Describes an n-gram language model based on a hierarchical extension of the Pitman-Yor process
- Pitman-Yor process used as prior for Gu (probability current word given context)
  - Strength and discount parameters depend on length of context |u|
  - Mean vector is Gπ(u), probabilities of current word given all but earliest word in context

**Model Structure:**
- Recursive placement of prior over Gπ(u) using (3) until reaching empty context ∅
- Global mean vector G∅ has uniform prior, total parameters = 2n
- Suffix tree structure with nodes representing contexts and children adding words to beginning

## 4 Hierarchical Chinese Restaurant Processes
**Generative Procedure:**
1. Draw words from each Gu using Chinese restaurant process (CRP) based on Pitman-Yor distribution
2. Recursively draw words from parent distributions Gπ(u) until reaching global mean distribution G0
3. Equivalent to hierarchical Pitman-Yor language model but with Gu's marginalized out
4. In next section, tractable inference schemes derived for this model based on seating arrangements from CRP.

## 5 Inference Schemes

**Markov Chain Monte Carlo Sampling Inference Scheme for Hierarchical Pitman-Yor Language Model**

**Training Data**:
- Consists of number of occurrences **cuw** of each word **w** after contexts of length exactly **n-1** (context **u**)
- Corresponds to observing word **w** drawn **cuw** times from Gu

**Interest**:
- Posterior distribution over latent vectors **G** and parameters **Θ** given training data **D**
- Equivalent to posterior distribution over seating arrangements **S** and parameters **Θ**

**Predictive Probabilities**:
- Given test word **w**, probability of word after context **u** is:
  - **p(w|u, S, Θ)** = **WordProb(u,w)** **p(S, Θ|D)**
- Approximated using samples from posterior distribution

**Word Probability Function**:
- **WordProb(u,w)**: predictive probability under a particular setting of seating arrangements **S** and parameters **Θ**
- Formula (12) in text

**Gibbs Sampling**:
- Used to obtain posterior samples for variables **S** and **Θ**
- Variables consist of indexes of draws from each Gu assigned to a word
- Probabilities given by functions **DrawWord(u)** and **WordProb(u,w)**

**Sampling Scheme**:
- O(nT) time, O(M) space per iteration for hierarchical Pitman-Yor language model with n-grams
- O(nI) time during test time to calculate predictive probabilities

**Discounts**:
- Gradually growing as a function of n-gram counts
- Average discount grows slowly as **cuw** grows
- Interpolated Kneser-Ney produces the same discounts as hierarchical Pitman-Yor, but with different discount values for tuw

**Modified Kneser-Ney**:
- Uses same counts as interpolated Kneser-Ney, but uses different discounts up to a maximum of **c(max)**
- Not an approximation of hierarchical Pitman-Yor language model due to differing discounts.

## 6 Experimental Results

**Experimental Results on Hierarchical Pitman-Yor Language Model**

**Dataset**:
- APNews corpus of approximately 16 million words used for training, validation, and testing
- Vocabulary size: 17964

**Training Set Size and N-grams**:
- Experimented with trigrams (n = 3) on varying training set sizes between 2 million and 14 million words in increments
- Also experimented with n = 2, 3, and 4 on the full 14 million word training set

**Methods Compared**:
- **Interpolated Kneser-Ney (IKN)**: trained using conjugate gradient descent in cross-entropy on validation set, then folded into final probability estimates
- **Modified Kneser-Ney (MKN)** with maximum discount cut-off c(max) = 3
- **Hierarchical Dirichlet language model (HDLM)**
- **Hierarchical Pitman-Yor language model (HPYLM)**: posterior distribution over latent variables and parameters inferred using proposed Gibbs sampler

**Perplexities on Test Set**:
- HDLM performed worst, while HPYLM outperformed IKN
- HPYLM slightly underperformed MKN due to not being a perfect language model for optimization of predictive performance
- Kneser-Ney variants were optimized using cross-validation and had best performance overall

**HPYCV Model**:
- A hierarchical Pitman-Yor language model with parameters obtained by fitting in a generalization of IKN using Gibbs sampling
- Performed better than MKN (except marginally on small problems) and had best performance overall
- Still not optimized due to the cost of cross-validation using a hierarchical Pitman-Yor language model inferred by Gibbs sampling

**Contributions to Cross-Entropies**:
- Differences between methods appeared as differences among rare words, with common words having negligible impact
- HPYLM performed worse on words occurring only once and better on other words, while HPYCV was reversed and performed better on low-frequency words and worse on more common words.

## 7 Discussion

**Hierarchical Pitman-Yor Process**

**Benefits of Hierarchical Model**:
- Superior performance compared to state-of-the-art methods
- Interpolated Kneser-Ney (IKN) method can be interpreted as approximate inference in the hierarchical Pitman-Yor language model
- Fully Bayesian model allows for:
  - Coherent probabilistic model
  - Ease of improvements by building in prior knowledge
  - Integration into more complex models

**Comparison with Kneser-Ney Variants**:
- **Hierarchical Dirichlet language model** was an inspiration, but the use of Dirichlet distribution led to non-competitive results
- The hierarchical Pitman-Yor process is a generalization that gives state-of-the-art performance
- The hierarchical Pitman-Yor process is a natural extension of the **hierarchical Dirichlet process** proposed for clustering

**Advantages of Bayesian Nonparametric Processes**:
- Can relax strong assumptions on parametric forms while retaining computational efficiency
- Handle model selection and structure learning in graphical models elegantly

**Acknowledgements**:
- **Lee Kuan Yew Endowment Fund** for funding
- Joshua Goodman for insights on IKN, MKN, and smoothing techniques
- John Blitzer and Yoshua Bengio for dataset support
- Anoop Sarkar for discussions
- Hal Daume III, Min Yen Kan, and anonymous reviewers for helpful comments

