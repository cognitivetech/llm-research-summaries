# Latent Dirichlet Allocation

## Abstract
**Latent Dirichlet Allocation (LDA)**

**Description**:
- Probabilistic model for collections of discrete data, such as text corpora
- Developed by David M. Blei, Andrew Y. Ng, and Michael I. Jordan
- A three-level hierarchical Bayesian model:
  - Each item is a finite mixture over an underlying set of topics
  - Each topic is an infinite mixture over an underlying set of topic probabilities

**Text Modeling**:
- Topic probabilities provide an explicit representation of a document

**Inference Techniques**:
- **Efficient approximate inference techniques** based on variational methods and EM algorithm
- Used for empirical Bayes parameter estimation

**Applications and Comparison**:
- Document modeling, text classification, collaborative filtering
- Compared to:
  - Mixture of unigrams model
  - Probabilistic LSI model

### 1. Introduction

**Introduction**
- Modeling text corpora and collections of discrete data to find short descriptions that enable efficient processing of large collections while preserving statistical relationships for tasks like classification, novelty detection, summarization, and similarity/relevance judgments.
- **TF-IDF scheme**: Reduces documents to fixed-length lists of numbers by comparing term frequency counts to inverse document frequency counts. Reveals some inter-document statistical structure but lacks reduction in description length.
- **Latent Semantic Indexing (LSI)**: Uses singular value decomposition of the X matrix to identify a subspace capturing most variance in the collection, achieving significant compression in large collections. Can capture aspects of synonymy and polysemy.
- **Probabilistic Latent Semantic Indexing (pLSI)** or **Aspect Model**: Models each word as a sample from a mixture model where components are multinomial random variables (topics). Each document is represented by a list of mixing proportions for topics. However, it lacks a probabilistic model at the level of documents.

**The Latent Dirichlet Allocation (LDA) Model**
- Based on the "bag-of-words" assumption and exchangeability of both words and documents.
- Represents each document as a list of topics with mixing proportions, reducing it to a probability distribution on a fixed set of topics.
- **Conditional independence and identically distributed**: The joint distribution of random variables is simple and factored conditionally while marginally over the latent parameter, the joint distribution can be complex.
- LDA captures significant intra-document statistical structure via the mixing distribution. Applicable to richer models involving mixtures for larger structural units like n-grams or paragraphs.

**Outline of the Paper**
- Introduce notation and terminology (Section 2).
- Present LDA model in Section 3, compare it to related latent variable models in Section 4.
- Discuss inference and parameter estimation for LDA in Section 5.
- Provide an illustrative example of fitting LDA to data in Section 6.
- Present empirical results in text modeling, text classification, and collaborative filtering in Section 7.
- Conclude with the authors' findings (Section 8).

### 2. Notation and terminology

**Latent Dirichlet Allocation (LDA)**
- **Text Collections**: LDA model is a probabilistic model used to represent a corpus of documents.
- **Terminology**:
  - **Word**: Basic unit of discrete data, represented by a unit-basis vector with a single component equal to one and all other components zero.
  - **Document**: Sequence of words (N) in a document.
  - **Corpus**: Collection of M documents.
- **LDA Model Assumptions:**
  1. Documents are represented as random mixtures over latent topics.
  2. Each topic is characterized by a distribution over words.
  3. Simplifying assumptions:
     - Dimensionality (k) of Dirichlet distribution known and fixed.
     - Word probabilities parameterized by a k × V matrix β.
     - Poisson assumption for document length not critical.
- **Dirichlet Distribution**: A probability distribution used to represent the random variable θ, which determines the topic distribution in a document.
- **Graphical Model Representation of LDA**:
  1. Corpus level: Fixed parameters α and β.
  2. Document level: Topic variables θd sampled per document.
  3. Word level: Variables zdn and wdn sampled for each word in a document.
- **Distinction from Dirichlet-Multinomial Clustering Model**:
  - In LDA, topics are sampled repeatedly within a document, allowing for multiple associations between a document and topics.
  - In classical clustering models, a document is restricted to being associated with a single topic.

### 3.1 LDA and exchangeability

**Exchangeability and LDA**
- **Exchangeable set**: A finite set of random variables {z1, ..., zN} is exchangeable if its joint distribution is invariant to permutation
- An infinite sequence is infinitely exchangeable if every finite subsequence is exchangeable
- **De Finetti's representation theorem**: The joint distribution of an infinitely exchangeable sequence of random variables has the form: p(w, z) = ∫ p(θ) ∏n=1N p(zn | θ)p(wn | zn) dθ
- In LDA, words are generated by topics (conditional distributions) and assumed to be infinitely exchangeable within a document
- The probability of a sequence of words and topics has the form: p(w, z) = ∫ p(θ) ∏n=1N p(zn | θ)p(wn | zn) dθ
- LDA distribution on documents is obtained by marginalizing out topic variables and endowing θ with a Dirichlet distribution

### 3.2 Continuous Mixture of Unigrams
- The LDA model is more elaborate than two-level models in classical hierarchical Bayesian literature
- By marginalizing over hidden topic variable z, LDA can be understood as a two-level model
- Let **p(w | θ, β)** be the word distribution under LDA: p(w | θ, β) = ∑ p(w | z, β)p(z | θ)
- This is a random quantity that depends on θ
- Generative process for a document **w**:
    1. Choose **θ** ~ Dir(α)
    2. For each word **wn**:
        - Choose the word from p(wn | θ, β)
- The resulting distribution **p(w | α, β)** is a continuous mixture distribution:
   p(w | α, β) = ∫ p(θ | α) ∏n=1N p(wn | θ, β) dθ
- This distribution over the (V − 1)-simplex exhibits an interesting multimodal structure, with only k + kV parameters

## 4. Relationship with other latent variable models

**Latent Variable Models Comparison**

**Unigram Model**:
- Each document's words drawn independently from a single multinomial distribution
- Illustrated in the graphical model (Figure 3a)

**Mixture of Unigrams Model**:
- Augments the unigram model with a discrete random topic variable z
- Each document generated by choosing a topic and then generating words from its conditional multinomial
- Probability of a document: p(w) = ∑p(z)∏1p(wn | z).zn =
- Word distributions represent topics under the assumption of exactly one per document
- Limitations in modeling large collections of documents due to the single topic assumption

**Probabilistic Latent Semantic Indexing (pLSI)**:
- Relaxes the unigam/mixture of unigrams model's single topic assumption
- Document label d and a word wn conditionally independent given an unobserved topic z
- Attempts to capture multiple topics, but not as well-defined generative model due to training document dependency
- Prone to overfitting, requiring tempering heuristics

**Latent Dirichlet Allocation (LDA)**:
- Overcomes the limitations of pLSI by treating topic mixture weights as a random variable
- Well-defined generative model that generalizes to new documents
- Fixed number of parameters (k + kV), not affected by training corpus size

**Geometric Interpretation**:
- All models operate in the space of word distributions (word simplex)
- LDA, pLSI, and mixture of unigrams use a topic simplex to generate documents differently:
  - Mixture of unigrams: One point on word simplex per document
  - pLSI: Randomly chosen topic for each word in a training document
  - LDA: Randomly chosen topic with parameter drawn from smooth distribution for entire document

## 5. Inference and Parameter Estimation

**LDA Inference and Parameter Estimation**

### 5.1 Inference
- Problem: computing posterior distribution of hidden variables given a document
- Model is intractable due to coupling between θ and β
- Approximate inference algorithms can be considered, including: Laplace approximation, variational approximation, Markov chain Monte Carlo (Jordan, 1999)

### 5.2 Variational Inference
- Basic idea: use Jensen's inequality to obtain lower bound on log likelihood
- Optimization problem: find tightest lower bound by minimizing Kullback-Leibler divergence between variational distribution and true posterior
- Variational distribution: N q(θ, z | γ, φ) = q(θ | γ)∏n=1Nq(zn | φn)
- Update equations for variational parameters: φni ∝ βiwn exp{Eq[log(θ) | γ]i} and γi = αi + ∑n=1N φni
- Variational distribution is a conditional distribution varying as a function of w, yielding document-specific optimizing parameters (γ*(w), φ*(w))

**Variational Inference Algorithm**:
- Initialize: φni := 1/k for all i and n, γi := αi + N/k for all i
- Repeat iterations until convergence: update φni and γi using Eqs. (6) and (7), compute expected log likelihood using Eq. (8)
- Total number of operations is roughly on the order of N^2k per document

### 5.3 Parameter estimation

**Parameter Estimation in LDA Model**
* Presenting an empirical Bayes method for parameter estimation in the LDA model
* Given corpus of documents D = {w1, w2, ..., wM}, find parameters α and β that maximize log likelihood: M `(α, β) = ∑d=1 log p(wd | α, β)
* Variational inference provides a tractable lower bound on the log likelihood, which can be maximized with respect to α and β
* Alternating variational EM procedure used to find approximate empirical Bayes estimates for LDA model:
	+ E-step: For each document d, find optimizing values of variational parameters {γd*, φd*}
	+ M-step: Maximize resulting lower bound on log likelihood with respect to model parameters α and β (find maximum likelihood estimates)
* Repeat until lower bound on log likelihood converges
* **Smoothing** to cope with sparsity due to large vocabulary sizes:
	+ Standard approach is Laplace smoothing, but not justified as a maximum a posteriori method in mixture model setting
	+ Proposed solution: apply variational inference methods to extended model that includes Dirichlet smoothing on multinomial parameter
* Extended graphical model for smoothed LDA with random variables β treated as endowed with a posterior distribution conditioned on data.
* Consider a fuller Bayesian approach to LDA using variational inference with separable distribution on random variables β, θ, and z: q(β1:k, z1:M, θ1:M | λ, φ, γ) = ∏i=1k Dir(βi | λ) ∏d=1M qd(θd, zd | φd, γd), where qd is the variational distribution for LDA defined by Eq. (4).
* Iterating update equations for variational parameters φ, γ, and new variational parameter λ yields an approximate posterior distribution on β, θ, and z.
* Set hyperparameters η (exchangeable Dirichlet) and α (from before) using variational EM to find maximum likelihood estimates based on marginal likelihood.

## 6. Example

**Example of LDA Model on Real Data**
- **Data**: 16,000 documents from a subset of TREC AP corpus
- After removing stop words, used EM algorithm to find Dirichlet and conditional multinomial parameters for a 100-topic LDA model
- **Results**: Top words from some resulting multinomial distributions (p(w | z)) illustrate underlying topics in the corpus (named accordingly)
- Advantage of LDA: Provides well-defined inference procedures for previously unseen documents
- Illustrating LDA on a held-out document: Computed variational posterior Dirichlet and multinomial parameters
- **Posterior Dirichlet Parameters**: Approximately the prior Dirichlet parameters (αi) plus expected number of words from each topic
- **Prior vs. Posterior Dirichlet Parameters**: Difference indicates expected number of words allocated to each topic for a document
- **Example Document** (Figure 8, bottom): Most γi close to αi; several significantly larger
- Identifying topics from φn parameters: Peaks towards one of k possible topic values; colors represent qn(zi) values
- Limitations of LDA: Bag-of-words assumption allows words generated by the same topic to be allocated to several different topics
- Overcoming limitation: Requires extending the basic LDA model, possibly relaxing the bag-of-words assumption.

## 7. Applications and Empirical Results

- Discussing LDA application in document modeling, classification, collaborative filtering
- Important to initialize EM algorithm appropriately to avoid local maxima
  - In experiments, initialized by seeding conditional multinomial distributions with five documents, reducing effective total length, smoothing across vocabulary
    - Approximation of scheme described in Heckerman and Meila (2001)

### 7.1 LDA vs. Other Models: Comparison of Perplexity Results

**Perplexity Results for Nematode Corpora:**
- **LDA**: Perplexity = 1600
- **Unigram Model**: Perplexity = 34,000 (Base)
- **Mixture of Unigrams**: Perplexity = 22,266
- **pLSI**: Perplexity = 5,040 × 10^7

**Perplexity Results for AP Corpora:**
- **LDA**: Perplexity = 3,000 (Base)
- **Unigram Model**: Perplexity = 7,000 (Base)
- **Mixture of Unigrams**: Perplexity = 6,500
- **pLSI**: Perplexity = 1.31 × 10^3

**Overfitting in Mixture of Unigrams and pLSI:**
- Both models suffer from overfitting issues for different reasons:
  * Mixture of unigrams: Overfitting due to peaked posteriors, leading to nearly deterministic clustering of training documents.
  * pLSI: Overfitting due to the dimensionality of p(z|d), causing the perplexity to explode for new documents as k increases.

**Solutions:**
- **Mixture of unigrams**: Alleviate overfitting through variational Bayesian smoothing scheme presented in Section 5.4.
- **pLSI**: Allow each document to exhibit different proportions of topics and integrate over the empirical distribution on the topic simplex for inference, but this causes overfitting. Alternatively, use the "folding-in" heuristic suggested by Hofmann (1999).
- **LDA**: Each document can exhibit different proportions of underlying topics, and new documents can be easily assigned probabilities without the need for heuristics.

### 7.2 Document classification

**Document Classification**
- Problem: classify a document into two or more mutually exclusive classes
- Generative approaches vs. discriminative approaches
- LDA model for dimensionality reduction in document classification
  * Reduces feature set by treating individual words as features is very large
  * LDA model reduces document to fixed set of real-valued features: posterior Dirichlet parameters γ∗(w) associated with the document
- Two binary classification experiments using Reuters-21578 dataset
  * Estimated LDA model parameters on all documents without reference to class labels
  * Trained SVMs on low-dimensional representations provided by LDA and compared to word feature SVM
- Results showed little reduction in classification performance with LDA-based features, improvement in most cases

**Collaborative Filtering**
- Problem: predict a user's preferred movie based on other movies they have rated highly
- Evaluated algorithms according to their predictive perplexity for M test users
  * Predictive perplexity = exp{−∑d=1M log p(wd,Nd | wd,1:Nd −1)}
- Data set restricted to users who positively rated at least 100 movies (4 stars and above)
- Divided into 3300 training users and 390 testing users
- Under the mixture of unigrams model and pLSI, probability of a movie given observed movies obtained from posterior distribution over topics
  * Probability of held-out movie in LDA model = integrated posterior Dirichlet distribution
- Results showed that the LDA model has the best predictive perplexities compared to mixture of unigrams model and pLSI.

## 8. Discussion

**Latent Dirichlet Allocation (LDA)**
**Background:**
- Flexible generative probabilistic model for collections of discrete data
- Based on exchangeability assumption for words and topics in a document
- Realized by application of de Finetti's representation theorem
- Dimensionality reduction technique, similar to LSI with proper semantics
- Inference is intractable but approximate methods can be used

**Advantages:**
- Modularity: readily embedded in more complex models
- Extensibility: numerous possible extensions and adaptations

**Extensions:**
- Continuous data or non-multinomial emissions
- Mixtures of Dirichlet distributions for richer topic structure
- Time series arrangements for topics with partial exchangeability
- Conditioning on exogenous variables, such as "paragraph" or "sentence," for more powerful text modeling.

