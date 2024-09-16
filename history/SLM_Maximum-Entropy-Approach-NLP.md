# A Maximum Entropy Approach to Natural Language Processing

by Adam L. Berger, Vincent J. Della Pietra, Stephen A. Della Pietra 

https://aclanthology.org/J96-1002.pdf

## Contents
- [1. Introduction](#1-introduction)
- [2. A Maximum Entropy Overview](#2-a-maximum-entropy-overview)
- [3. Maximum Entropy Modeling](#3-maximum-entropy-modeling)
  - [3.1 Training Data](#31-training-data)
  - [3.2 Statistics, Features and Constraints](#32-statistics-features-and-constraints)
  - [3.3 Maximum Entropy Principle](#33-maximum-entropy-principle)
  - [3.4 Parametric Form The maximum entropy principle presents us with a problem in constrained optimization: find the p](#34-parametric-form-the-maximum-entropy-principle-presents-us-with-a-problem-in-constrained-optimization-find-the-p)
  - [3.5 Relation to Maximum Likelihood](#35-relation-to-maximum-likelihood)
  - [3.6 Computing the Parameters](#36-computing-the-parameters)
- [4. Feature Selection](#4-feature-selection)
  - [4.1 Motivation](#41-motivation)
  - [4.2 Basic Feature Selection](#42-basic-feature-selection)
  - [4.3 Approximate Gains](#43-approximate-gains)
- [5. Case Studies](#5-case-studies)
  - [5.1 Review of Statistical Translation](#51-review-of-statistical-translation)
  - [5.2 Context-Dependent Word Models](#52-context-dependent-word-models)
  - [5.3 Segmentation](#53-segmentation)
  - [5.4 Word Reordering](#54-word-reordering)

## 1. Introduction

**Introduction**:
- Computers have become powerful enough to apply maximum entropy concept to real world problems in statistical estimation and pattern recognition
- Statistical modeling addresses the problem of constructing a stochastic model to predict the behavior of a random process
- Given a sample of output from the process, the goal is to parlay this knowledge into a representation of the process that can be used for prediction
- Examples: baseball batting averages, stock price movements, natural language processing (e.g., speech recognition systems)

**Background**:
- Significant progress in increasing the predictive capacity of statistical models of natural language
- Tasks in statistical modeling: feature selection and model selection

**Maximum Entropy Philosophy**:
- Overview given in Section 2
- Maximum entropy models aim to capture all available information without making unnecessary assumptions
- The key idea is to maximize the probability (entropy) of the observed data, subject to constraints that reflect prior knowledge about the system

**Maximum Entropy Models**:
- Mathematical structure described in Section 3
- Efficient algorithm for estimating the parameters of such models presented

**Feature Selection and Discovery**:
- Feature selection is a task of determining a set of statistics that captures the behavior of the random process
- Automatic method for discovering facts about a process from a sample of output is discussed in Section 4
- Refinements are presented to make the method practical to implement

**Applications**:
- Bilingual sense disambiguation, word reordering, and sentence segmentation are examples of applying maximum entropy ideas to stochastic language processing tasks.

## 2. A Maximum Entropy Overview

**Maximum Entropy Overview**

**Introduction**:
- Maximum entropy introduced through a simple example: Modeling an expert's French word choice for English term "in" (in)
- Goal: Extract facts about decision-making process from sample and construct a model
- First constraint: p(dans) + p(en) + p(à) + p(au cours de) + p(pendant) = 1

**Modeling Approaches**:
- Uniform models assuming more than known: p(dans) = 1 or p(pendant) = 1/2, à = 1/2
- Most intuitively appealing model: Allocates probability evenly among allowed translations (1/5 for each)

**Updating Model with New Clues**:
- Expert chose either dans or en 30% of the time
- Constraints: p(dans) + p(en) = 3/10 and p(dans) + p(a) = 1/2
- Model with highest uniformity subject to constraints is not obvious

**Maximum Entropy Principle**:
- Model all known information and assume nothing about the unknown
- Choose model consistent with all facts, but as uniform as possible
- Maximum entropy concept has a long history: Occam's razor, Laplace's principle, and E. T. Jaynes' pioneering work

**Maximum Entropy Modeling**:
- Stochastic model for a random process producing output y from context x (in this case, translation of English word in)
- Goal: Construct a probabilistic model that accurately represents the behavior of the random process
- p(y|x): Conditional probability assigned by the model to y given context x.

## 3. Maximum Entropy Modeling
**Random Process Modeling**
- Produces output value **y**, a member of finite set: {dans, en, il, au cours de, pendant}
- Influenced by contextual information **x**, a member of finite set **X**
- Task is to construct a stochastic model accurately representing random process behavior
- Model estimates conditional probabilities: p(**y**|**x**)
- Probability that model assigns to y given context x: **p(y|x)**
  - Represents entire conditional probability distribution provided by the model
- Notation: **p(y[x)]** vs. specific instantiations of y and x for clarity.
- Model is an element of the set of all conditional probability distributions, denoted as **~v**.

### 3.1 Training Data

**Training Data (Random Process)**
- Observe behavior of random process for some time, collect samples: (x1, y1), (x2, y2), ... , (xN, YN)
- Each sample: phrase x containing words around "in", translation y produced by the process
- Imagined as generated by a human expert who chose good translations for each random phrase

### 3.2 Statistics, Features and Constraints
- Summarize training sample in terms of its empirical probability distribution p(x,y)
- Goal: construct statistical model of the process that generated the training data
- Current example uses independent statistics like frequency of certain translations
- Could also consider context-dependent statistics (e.g., translation depends on conditioning information x)
- Introduce indicator function f(x,y) for useful statistics, require model to accord with expected value of feature functions p(f) = p()

### 3.3 Maximum Entropy Principle
- Given n feature functions fi representing important statistics in modeling the process
- Objective: select model p ∈ P that agrees with these statistics (p(fi) = p(fi) for i ∈ {1,2,...,n})
- Space of all probability distributions on three points called simplex, C is a subset of P defined by constraints
- Linear constraints extracted from training sample cannot be inconsistent and will not determine p uniquely
- Maximum entropy philosophy: select most uniform distribution in set C (p+) to minimize uncertainty
- Conditional entropy H(p) = -Σp(x)p(y|x) log p(y|x) or H(Y | X), bounded from below and above by 0 and log |V| respectively.

### 3.4 Parametric Form The maximum entropy principle presents us with a problem in constrained optimization: find the p

**Maximum Entropy Principal: Parametric Form**

**Problem:**
- Maximize entropy H(p) subject to certain constraints C

**Solution:**
1. **Primal problem**: original optimization problem
2. **Lagrange multipliers**: method for addressing general problem
3. **Lagrangian A(p,A)**: function defined as H(p) + Σi A_i E_x [p_k f_i] - p_l f_h)
4. **Unconstrained optimization of Lagrangian A(p,A)**: find p_A that achieves maximum and denote w(A) as its value
5. **Dual problem**: find A* = argmax W(A), where W(A) is the dual function

### 3.5 Relation to Maximum Likelihood 
**Primal vs Dual Framework**

**Primal Framework**:
- **Maximum Likelihood (ML)**
  - Log-likelihood of empirical distribution p as predicted by a model p: Lp(p) = log [[P(Ylx) P(X'y)]/H(p(ylx))]
  - The dual function W(A) is the log-likelihood for exponential model p\n.
  - Result: Maximum entropy model in parametric form p(ylx) maximizes likelihood of training sample.

**Dual Framework**:
- **Maximum Entropy (ME)**
  - Finding a distribution p with maximum entropy that best fits the data.
- **Dual Function W(A)**:
  - The negative of free energy, measuring how well an assumed distribution p matches the true one.
  - Maximizing this function leads to the solution for the ME principle.
- Relation between Primal and Dual:
  - Result: The maximum entropy model is also the model with maximum likelihood from among all models in its parametric form.

### 3.6 Computing the Parameters

**Computing the Parameters**
- **A** that maximize V(A) cannot be found analytically
- Resort to numerical methods
- Function V(A) is smooth and convex, allowing use of various optimization techniques: coordinate-wise ascent (Brown algorithm), gradient ascent, conjugate gradient, iterative scaling algorithm by Darroch and Ratcliff
- **Iterative Scaling Algorithm** designed for maximum entropy problem
- Applicable when feature functions are nonnegative (fA(x,y) > 0 for all i, x, y)
- Algorithm:
  1. Initialize Ai = 0 for all i
  2. For each i in {1, 2, ..., n}:
     a. Update A according to equation (16), either explicitly or numerically using Newton's method
     b. Update hi accordingly
     c. Converge until all A and ~i have converged
- Key step: computing increments AAi in step (2a)
  - If f"(x,y) is constant, given by 4Ai -- ft. log P,(fi)M / log plf
  - For non-constant functions, compute numerically using Newton's method

**Feature Selection**
- Two steps in statistical modeling: finding appropriate facts about the data and incorporating them into a model
- Previously assumed first task was performed by assuming constraints were selected appropriately
- Principle of maximum entropy does not directly address feature selection, but critical since universe of possible constraints is large (thousands or millions)
- Introducing method for automatically selecting features in maximum entropy models and computational refinements.

## 4. Feature Selection
**Maximum Entropy Modeling Approach**
- Divided into two steps: finding appropriate facts about data and incorporating them into the model
- First task assumed to be performed by assuming that certain constraints are selected
- Principle of maximum entropy does not directly address feature selection problem
  - Critical as universe of possible constraints can be in thousands or millions

**Feature Selection Method**
- Introduced method for automatically selecting features
- Offered refinements to ease computational burden

**Assumptions about Data**:
- First task assumed to be performed (selecting important facts)
- No explicit statement on how these facts are chosen from the data

**Maximum Entropy Model**:
- Principle provides a recipe for combining constraints into a model
- Does not directly concern itself with feature selection problem

**Critical Nature of Feature Selection**:
- Universe of possible constraints can be extensive (thousands or millions)
- Important to select relevant features for accurate modeling results.

### 4.1 Motivation

**Motivation**
- Begin by specifying a large collection **F** of candidate features
- Do not require relevance or usefulness of these features initially
- Ultimately, only a small subset **S** (active features) will be used in the final model

**Determining Active Features**
- Cannot rely on small training sample to represent the process fully
- Aim to include as much information about the random process as possible
- Infinite sample size: true expected value for a feature is the fraction of events with that feature = 1
- Real-life applications: provided with only a small sample **N** events
- **S**: set of active features, must capture information but reliably estimate expected values

**Growing Decision Trees**
- Build up **S** by successively adding features
- Each addition imposes another linear constraint on the space of models allowed
- Narrows the model space **C(S)**, hopefully improving representation
- Alternatively, represent it as a series of nested subsets **C(Si)** of P

### 4.2 Basic Feature Selection
**Basic Feature Selection Algorithm**
1. Start with empty **S**; initial model **PS** is uniform
2. For each candidate feature **f**:
   - Compute the model **PSUf** using Algorithm 1
   - Compute the gain in log-likelihood from adding this feature: ΔL(S, f) = L(PSUf) - L(PS)
3. Check termination condition (e.g., cross-validation on withheld sample)
4. Select the feature **f** with maximal gain ΔL(S, f)
5. Adjoin **f** to **S**, update model **PS** using Algorithm 1
6. Repeat from step 2 until a stopping condition is met

### 4.3 Approximate Gains

**Approximate Gains Algorithm**

**Greedy Feature Selection**:
- Replace computation of gain AL(S,f) with an approximation ΔAL(S,f)
- This approximation assumes the optimal values for the new feature f do not change the parameters associated with existing features
- Computing approximate gain reduces problem to a one-dimensional optimization over the single parameter θ

**Notational Breakdown**:
- pS: model containing set S of features
- pS,f: best model containing both S and the new feature f
- Za(x): sum of probability distribution over Y given x for model pS
- Lp(x): log-likelihood of a parameter in model p

**Approximation Assumptions**:
- Optimal values of all parameters change when a new constraint is imposed
- Approximate gain assumes the best model with S ∪ f has the same structure as pS, with only θ changing
- Inevitably underestimates the actual gain AL(S,f)

**Savings in Computational Complexity**:
- Reduces problem from n-dimensional to a one-dimensional line search over parameter θ
- Faster than exact computation but may pass over features with higher true gains

**Comparison of Optimization Problems**:
- Exact answer requires searching both A and a dimensions (Figure 3a)
- Approximate method simplifies problem to a line search over a (Figure 3b)

## 5. Case Studies

**Case Studies: Application of Maximum Entropy Modeling in Candide (French-to-English Machine Translation System)**

**Background:**
- Review of statistical translation theory
  - Bayes' theorem application
  - Components: language model, translation model, search strategy
- Focus on French sentence generation using a generative process and alignment concept

### 5.1 Review of Statistical Translation 
1. General theory of statistical translation
   * Candide's task: find most probable English sentence given French sentence F
2. Parameters for calculating p(F | E)
   * Language model: estimates p(E), probability of well-formed English sentence
   * Translation model: generates p(F | E) through understanding two steps of translation process and its association with alignment A between E and F
3. Components of the generative process
   * Each word in E independently generates zero or more French words
   * Words are then ordered to create a French sentence F
4. Probability p(F, A | E) calculation for basic translation model
   * Sum over all possible alignments between E and F
5. Limitations of the basic translation model
   * Ignores English context (surrounding words) when predicting appropriate French rendering
6. Challenges: errors in context blind model
   * Examples: incorrect translations for "dans" vs."pendant", resulting in potential errors during Candide's call upon to translate a French sentence.
7. Description of basic translation model components
   * English word generates zero or more French words
   * Ordering of words in F determines the probability distribution over alignments between E and F
8. Probability p(F, A | E) calculation for basic translation model equation (31)
9. Unwieldy due to summation over all possible alignments between E and F
10. Methods of estimating parameters: EM algorithm, maximizing likelihood of bilingual corpus, and using Hansard corpus as an example.

**Basic Translation Model Parameters (Table 2): Most frequent French translations for "in"**
|Translation | Probability|
|---|---|
|dans |0.3004|
|dans |0.2275|
|de |0.1428|
|en |0.1361|
|pour |0.0349|
|(OTHER) |0.0290|
|au cours de |0.0233|
|(OTHER) |0.0290|
|au cours de |0.0154|
|sur |0.0123|
|par |0.0101|
|pendant |0.0044|
|pendant |0.0044|

**Basic Translation Model Shortcomings:** one major limitation - lack of context consideration. Blind to surrounding English words when predicting appropriate French rendering.

**Errors Encountered with EM-based model in French-to-English translation system (Figure 5): examples of incorrect translations.**
1. Superior vs Greater or Higher:
   * System chose "superior" instead of a more suitable translation based on context
2. He vs Il:
   * Incorrect rendering of "Il" could have been avoided if the model considered the following word "appears."

### 5.2 Context-Dependent Word Models

**Context-Dependent Word Models**

**Problem Statement**: The goal is to develop a context-sensitive maximum entropy model for English word translation into French, called pe(ylx).

**Data Collection**:
- Training sample of English-French sentence pairs (E, F) from Hansard corpus
- Use basic translation model to compute Viterbi alignment A between E and F
- Construct (x, y) training event: context x containing six words around the target word "in" and its future translation y

**Feature Definition**:
- Employ indicator functions of sets, considering French word y and English word e
- Template 1 feature: size of English vocabulary (|Ve|) or French vocabulary (|V|)
- Templates 2 to 5 consider various parts of context

**Constraints**:
- Equality between the probability of a French translation y according to the model and its empirical probability
- Example: p(y = dans) = p(y = dans) if e+1 is "speech" or "area"

**Template 1 Model**: Predicts each French translation y based on the empirical data without considering context

**Template 2 Constraints**: Require joint probability of English word following in and its French rendering to be equal to their empirical probability

**Context-Dependent Model**: Includes constraints derived from templates 2, 3, 4, and 5 for a window of six words around the target word "e0"

**Automatic Feature Selection Algorithm**: Selects features using iterative model-growing method to improve log-likelihood on the data

**Maximum Entropy Models**: Predict French translations using probabilities p(y|x) conditioned on context information.

### 5.3 Segmentation

**Segmentation in Machine Translation**

**Rationale**:
- Ideal system could handle sentences of unrestricted length
- Typical stochastic system requires safe segmentation for efficient processing
- Segmenting reduces computation scale, especially for large sentences

**Definition of Safe Segmentation**:
- Rift: position in French sentence without alignment to more than one English word
- Dependent on Viterbi alignment between French and English sentences
- Boundaries located only at rifts result in "safe" segmentation
- Does not guarantee semantically coherent segments

**Modeling Safe Segmentations**:
- Trained on English-French sentence pairs with Viterbi alignments and POS tags
- Constructed event pair (x,y) for each position j: x = context information, y = rift or no-rift
- Maximum entropy model assigns score p(rift|x) based on training data log-likelihood Lp
- Iterative model-growing procedure selects constraints to increase objective function
- Terminate when expert knowledge is extracted to avoid overfitting

**Segmentation in Machine Translation System**:
- Assigns score p(rift | x) per position in French sentence
- Dynamic programming algorithm selects optimal (or reasonable) splitting of the sentence based on scores and segment length constraints.

### 5.4 Word Reordering

**Word Reordering in Translation from French to English:**
* Translating involves selecting appropriate English words and ordering them based on English language conventions, often different from French word order
* Candide allows for alignments with crossing lines during preprocessing stage to capture differences in word orders between languages
* Reordering step shuffles words in input French sentence into more English-like order
* NOUN de NOUN phrases may require interchanging nouns for best translation: conflict of interest vs. conflit d'intérêt, interest rate vs. taux d'intérêt
**Maximum Entropy Model for NOUN de NOUN Phrases:**
* Data set of English-French sentence pairs with NOUN de NOUN phrases extracted from Hansard corpus
* Use basic translation model to compute Viterbi alignment between words in English and French sentences
* Construct training events based on pair of French nouns (NOUNL, NOUNR) and their corresponding translations
* Define candidate features using templates 1, 2, and 3 for interchange decision sensitivity to left or both nouns
* Use feature selection algorithm to construct maximum entropy model with 358 constraints from candidate features
**Performance:**
* Compared against a baseline NOUN de NOUN reordering module that never swaps word order
* Higher accuracy rate for maximum entropy model: 80.4% vs. 70.2% on test data

**Table 9: Performance Comparison** |Test Data|Simple Model Accuracy|Maximum Entropy Model Accuracy|
|----------------------|---------------|------------------------------|
|Total|71,555|80.4%|
|Not Interchanged|50,229|100%|
|Interchanged|21,326|49.2%|

**Figure 12: Predictions of the NOUN de NOUN interchange model on phrases from unseen corpus.**

**Table 12: Examples of NOUN de NOUR Phrases and Model Probabilities for Interchange:**
|French Phrase|p(interchange)|English Translation if interchange is applied|
|---|---|---|
|saison d'hiver|0.95|winter season or season of winter|
|somme d'argent|0.1|sum of money|
|abus de privilège|0.1|privilege abuse or misuse of privilege|
|chambre de commerce|0.2|commerce chamber or business chamber|
|taux d'inflation|0.5|inflation rate or rate of inflation

