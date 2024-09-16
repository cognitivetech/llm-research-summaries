# Hidden Markov Models for Speech Recognition

by B. H. Juang; L. R. Rabiner in Technometrics, Vol. 33, No. 3. (Aug., 1991)

http://luthuli.cs.uiuc.edu/~daf/courses/Signals%20AI/Papers/HMMs/0.pdf

## Contents
- [Introduction](#introduction)
- [1. MEASUREMENTS AND MODELING OF SPEECH](#1-measurements-and-modeling-of-speech)
- [2. THE STATISTICAL METHOD OF THE HIDDEN MARKOV MODEL](#2-the-statistical-method-of-the-hidden-markov-model)
  - [2.1 The Evaluation Problem](#21-the-evaluation-problem)
  - [2.2 The Estimation Problem](#22-the-estimation-problem)
  - [2.3 The Decoding Problem](#23-the-decoding-problem)
- [3. STRENGTHS OF THE METHOD OF HIDDEN MARKOV MODELS AS APPLIED TO SPEECH RECOGNITION](#3-strengths-of-the-method-of-hidden-markov-models-as-applied-to-speech-recognition)
  - [3.1 Consistent Statistical Framework](#31-consistent-statistical-framework)
  - [3.2 Training Algorithm](#32-training-algorithm)
  - [3.3 Modeling Flexibility](#33-modeling-flexibility)
- [4. HIDDEN MARKOV MODEL ISSUES FOR FURTHER CONSIDERATION](#4-hidden-markov-model-issues-for-further-consideration)
  - [4.1 Parameter Estimation Criteria](#41-parameter-estimation-criteria)
  - [4.2 Integration of Nonspectral and Spectral Features](#42-integration-of-nonspectral-and-spectral-features)
  - [4.3 Duration Modeling in HMM's](#43-duration-modeling-in-hmms)
  - [4.4 Model Clustering and Splitting](#44-model-clustering-and-splitting)
  - [4.5 Parameter Significance and Statistical Independence](#45-parameter-significance-and-statistical-independence)

## Introduction
- Hidden Markov models (HMMs) have become predominant in speech recognition research in the last decade due to:
  - Inherent statistical framework
  - Ease and availability of training algorithms
  - Flexibility of resulting recognition systems
  - Ease of implementation

**Performance**:
- **Speech recognition technology**: Can now achieve almost perfect accuracy in speaker-independent isolated-digit recognition, with 2-3% error rate for connected speech by nonspecific talkers.
- In a continuous speech environment with 1,000-word vocabulary and grammatical constraints, advanced HMM systems can achieve 96% word accuracy, rivaling human performance.
- However, the general problem of fluent, speaker-independent speech recognition is still not solved, and no system reliably recognizes unconstrained conversational speech or infers language structure from limited spoken sentences.

**Theory of HMMs**:
- HMMs are a statistical model used to represent a sequence of hidden states that generate observable data through a transition matrix and an emission probability distribution.
- The **Baum-Welch algorithm** is the standard method for estimating the parameters (transition probabilities and emission distributions) of an HMM from observed data.
- The **maximum a posteriori decoding** method is used to decode the most likely hidden state sequence given the observable data.

**Hidden Markov Models**:
- Hidden states represent phonetic or linguistic features that cannot be directly observed in speech.
- Observed data represents speech features like pitch, amplitude, and spectral characteristics.
- The transition matrix models the probability of moving from one hidden state to another, while the emission probabilities model the likelihood of observing a particular feature given a hidden state.

**Challenges and Future Work**:
- There are many theoretical and practical issues that need to be addressed to further advance speech recognition research:
  - **Acoustic modeling**: Improving accuracy by modeling noise, variability in speech patterns, and speaker adaptation.
  - **Language modeling**: Incorporating statistical language models to improve word accuracy and handle grammatical constraints.
  - **Feature extraction**: Developing new features that better capture speech characteristics, like prosody and intonation.
  - **Speaker adaptation**: Adapting the model to individual speakers for improved recognition performance.

## 1. MEASUREMENTS AND MODELING OF SPEECH

**Speech Signal Modeling**
- Speech is a nonstationary signal
- Articulatory apparatus (lips, jaw, tongue, velum) modulates air pressure and flow to produce sounds
- Spectral content of individual sounds may include frequencies up to several thousand Hz
- Short-time spectral properties analyzed at 10 ms intervals
- Long-term development of sound sequences characterized on the order of 100 ms due to articulatory configuration changes

**Digital Processing of Speech Signal**
- Analog speech signal sampled at 8-20 kHz, with 16-bit quantization
- Short-time spectral properties analyzed using windows and spectral analysis methods (FFT, LPC, autoregressive/moving average models)
- Auditory models incorporated for emphasizing important frequencies to human listeners

**Observation Vector or Observation**
- Short-time spectral vector obtained by analyzing speech signal with a window

**Spectral Analysis Methods**
- Discrete Fourier transform (FFT)
- All-pole minimum-phase linear prediction (LPC) methods
- Autoregressive/moving average models
- Filter bank method of spectral analysis
- Spectral properties emphasized using auditory models (e.g., Cohen 1985, Ghitza 1986)

**Cepstrum**
- Fourier transform of the log magnitude spectrum, particularly LPC model
- Computationally efficient for stable all-pole systems
- Figure 4 shows histograms of the first 12 LPC cepstral coefficients obtained from a speech data base.

**HMM Formulation**

**First-Order N-State Markov Chain**:
- Illustrated for N = 3 states in Figure 5
- System can be described as being in one of the N distinct states (1, 2, ..., N) at any discrete time t
- State variable: qr (state at time t)
- State transition probability matrix: A = [**aij**]
- **Axiomatic constraints**: **aij** ≥ 0 and **∑j=1N aij** = 1 for all i

**Observation Representation**:
- **Cepstral vector**: represents a short time speech spectrum
- Discrete symbol representation of spectral vector obtained through **spectral labeling** process
- Sequence of observations: {**Ot**} (T = length of speech sequence)

**Markov Modeling**:
- Spectral sequence can be modeled as a **Markov chain** describing how one sound changes to another
- Relation between direct spectral representation and Markov modeling discussed in Juang (1984) and Bridle (1984)

**Hidden Markov Model**:
- Discussed here is the case of an explicit probabilistic structure imposed on the sound sequence representation
- **Observation probability measures B = {**bi(Ot)**}
- Joint probability: P(**0, q | τ, A, B) = n₊₁n\n aij\nb,(Ot)\nb,1\nb,2\nb,…nb,NT)
- Stochastic process: P(**0 | τ, A, B) = P(**0, q | τ, A, B)

## 2. THE STATISTICAL METHOD OF THE HIDDEN MARKOV MODEL

**The Statistical Method of the Hidden Markov Model (HMM)**

**Problems Addressed by the Hidden Markov Model**:

- **Evaluation Problem**: Given observation sequence `0` and model `i`, how do we efficiently evaluate the probability of `0` being produced by source model `i`? (P(0|i))
- **Estimation Problem**: Given observation sequence `0`, how do we solve the inverse problem to estimate the parameters in `i`?
- **Decoding Problem**: How do we deduce the most likely state sequence `q` from observation `0`?

### 2.1 The Evaluation Problem
- Direct calculation of P(0|i) requires exponential computation
- The forward-backward procedure allows for linear computational efficiency
- The forward variable `a(i)` is defined as the probability of partial observation up to time t and state i
- The desired result is P(0|i) = sum(aT(i))
- Efficient scoring mechanism for classification of unknown observation sequence

### 2.2 The Estimation Problem
- Maximum Likelihood (ML) method is used to find the "right" model parameters that maximize P(0|i)
- The Baum-Welch algorithm accomplishes this in a two-step procedure:
    1. Transform objective function into new divergence measure Q(E'|i), and maximize over `i` to improve E.
    2. Repeat the steps until some stopping criterion is met
- The Baum-Welch algorithm is similar to the classical EM algorithm, but the HMM formulation lacks global optimality in practice (Paul 1985)

### 2.3 The Decoding Problem

**The Decoding Problem**

**Most Likely State Sequence**:
- Uncovering the most likely state sequence that led to an observation sequence
- Useful for:
    - Determining speech segment correspondences to word sounds
    - Duration of individual speech segments provides useful information for speech recognition

**Decoding Objectives**:
- Maximize the **instantaneous a posteriori probability** of the state at time t
- Extend to pairs, triples, and so on
- **Maximum a posteriori (MAP) rule**: Produces MAP result with minimum incorrect decoded state pairs

**Global Decoding**:
- Maximize **Pr(q / 0, 1.)**
- Equivalent to maximizing **Pr(q, 0 1 i.)**
- Suitable for solving by dynamic programming methods like the Viterbi algorithm

**Speech Recognition Using HMMs**:
1. Define a set of **L sound classes (V)**
2. Collect labeled training sets for each class
3. Solve the **estimation problem** to obtain models for each class
4. Evaluate **P r(0 / I, i.)** for unknown utterance 0 and identify speech class u with maximum probability

## 3. STRENGTHS OF THE METHOD OF HIDDEN MARKOV MODELS AS APPLIED TO SPEECH RECOGNITION

**Strengths of Hidden Markov Models (HMM) for Speech Recognition**

### 3.1 Consistent Statistical Framework
**Mathematical Framework**:
- Consistent statistical methodology
- Combines modeling of stationary stochastic processes and temporal relationships in a well-defined probability space
- Decomposable measure into observation, state sequence, and joint probability
- Flexibility to study short-time static characterization and long-term transitions independently

### 3.2 Training Algorithm
- Relatively easy and straightforward training from labeled data
- Baum-Welch algorithm: iterative hill-climbing to maximum likelihood criterion
- Segmental k-means algorithm: segmentation and optimization steps for state sequence and model parameters
- Separate optimization of components (state transition probability matrix and observation distributions) leads to simpler implementation

**Observation Distributions**:
- Accommodates various types: strictly log-concave densities, elliptically symmetric densities, mixtures, and discrete distributions.

### 3.3 Modeling Flexibility

**HMM Modeling Flexibility**
- **Flexibility of Basic HMM**: Manifested in:
  - **Model Topology**
  - **Observation Distributions**
  - **Decoding Hierarchy**
- **Model Topology**: Many topological structures studied for speech modeling
    - **Isolated Utterances**: Left-to-Right HMM (Figure 7a) is appropriate
        * Utterance begins and ends at well-identified time instants
        * Sequential behavior of speech is represented by sequential HMM
    - **Other Speech-Modeling Tasks**: Ergodic models (Figure 7b) are more appropriate
        * Topological configuration and number of states reflect a priori knowledge of the speech source
- **Observation Distributions**: Rich class of distributions can be accommodated
    - No analytical problems make use of any distribution impractical
    - Speech displays quite irregular probability distributions (Jayant and Noll 1984; Juang et al. 1985)
- **Mixture Distributions**: Used to characterize observations in each state
    - Mixture density function: (a) Left-to-Right HMM, (b) Ergodic HMM
    - Useful for modeling spectral observations
        * Mixture distribution approximates densities of virtually any shape
    - Variants: Vector quantizer HMM, semicontinuous HMM, continuous HMM
- **Observation Distributions in HMM**: Can also be an HMM probability measure
    - Basis for subword unit-based speech recognition algorithms (Lee et al. 1989; Lee et al. 1988)

**Ease of Implementation**
- **Potential Numerical Difficulties**: Result from multiplicative terms in the HMM probability measure
    - Normalization and interpolation techniques can help prevent numerical problems
        * Scaling algorithm (Levinson et al. 1983; Juang and Rabiner 1985) normalizes partial probabilities
        * Deleted interpolation scheme proposed by Jelinek and Mercer (1980) deals with sparse data problems
- **Computational Complexity**: Relatively low due to grammatical constraints limiting the average number of words following a given word to around 100

## 4. HIDDEN MARKOV MODEL ISSUES FOR FURTHER CONSIDERATION

**Issues for Further Consideration in Hidden Markov Models (HMMs)**

**Background:**
- Theory of hidden Markov modeling developed over last two decades
- Application to speech recognition still has some issues

### 4.1 Parameter Estimation Criteria
- Original method: maximize probability P(O|π), where π is the assumed source model
- ML method is optimal according to this criterion
- Baum-Welch reestimation algorithm used for solving the ML HMM estimation problem

**Classification Error Rate:**
- Classical Bayes rule (Duda and Hart, 1973) requires minimum classification error rate
- Decision rule: C*(O) = arg max Pr(C|O), where O is observation sequence and C is a class
- MAP decoder: decision rule written as c*(O) = arg max Pr(O|C) Pr(C)
- Problem lies in estimating unknown prior distribution Pr(C) and conditional distribution P(O|C) from finite training set

**Strategies to Overcome Difficulty:**
1. Choose a reduced set of subword units instead of words for classes
2. Obtain reliable estimates by using larger training sets or reducing the vocabulary size
3. Decompose large sound classes into smaller ones (subword units)
4. Use lexical description to obtain class probability Pr(C) independently of spoken training set
5. Estimate P(O|C) from each subword unit in the lexical entry for words

**Complete Label Case:**
- Training data and class association known a priori (hand labeled)
- Illustration: isolated or connected word tasks, hand-segmented continuous speech recognition

**Incomplete Label Case:**
- Partial knowledge of data: only phoneme sequence known but not the correspondence between each phoneme and segment of speech
- Illustration: estimating models of subword speech units from continuous speech, words in connected word tasks without prior word segmentation.

**Markovian Structure of HMM Model**

**Speech Signals and Markov Property**:
- Speech signals may not be Markovian
- Need to account for potential errors in modeling with a mechanism called "model-mismatch case"

**Model-Mismatch Case**:
- When the chosen class model (conditional and a priori probability) is incorrect, ML method may lead to suboptimal recognition performance

**Corrective Training**:
- Alternative approach that minimizes recognition error rate during training by identifying sources of errors or near-misses
- Uses HMM as a discriminant function rather than focusing on modeling accuracy

**Complete-Label Case**:
- In this situation, the association between training data and class is precisely known (Fig. 8a)
- Concerns about supervised learning apply, but unique to speech recognition due to the use of a prescribed class prior model **Pr(Ci)**

**Maximum Mutual Information (MMI) Estimator**:
- When the class prior **Pr(Ci)** is obtained independent of spoken training set, MMI estimator is used
- Maximizes mutual information between input observations and target classes

**Conditional Maximum Likelihood Estimator (CML) vs. CMLE**:
- CML leads to a set of equations to optimize the conditional probability **Pr(Oi | Ci)**
- CMLE (Eq. 25) compensates for inaccurate class prior model **Pr(Ci)** by maximizing joint log-likelihood between observations and target classes

**Concerns with CMLE**:
- Lack of a convenient and robust algorithm to obtain estimate **Pr\_CMLE(O | Ci)**
- Previous attempts at using MMI criterion did not guarantee convergence to optimal solution
- CMLE has larger variance than MLE, undermining potential gain in offsetting sensitivity due to inaccurate class prior when decoder is used on test data

**Incomplete Label Case**
* Result of practical difficulties in labeling continuous speech data
* Ambiguity among sound classes due to class definition similarities and time uncertainty in signals
* Difficulties in training prescribed subword unit models without definitive labeling

**Handling Incomplete Labeling:**
1. **Retain known class sequence constraints**: Solve for optimal set of class models sequentially
2. **Optimize sound models simultaneously with decoding**: Constrained approach to iteratively refine segmentation and improve unit models
* Theoretical shortcomings: Segmentation results depend on number of segments; inconsistent segmentations possible

**Alternative Approach:**
1. **Combine segmentation, decoding, and modeling**: Solve for both class models and segmentations simultaneously without label-sequence constraint
2. **Iteratively improve data segmentation and model estimation via the segmental MDI modeling approach**

**Model-Mismatch Issues:**
* Choose HMM model before enough source information known
* Optimality in decoding meaningful only when actual source matches assumed model
* Improve model based on new data or revise decoding rule if source inconsistent with assumed model

**Minimum Discrimination Information (MDI) Approach:**
* Allow search into general set of models for better matching under MDI measure
* Find HMM parameter set that minimizes MDI on a sequence of constraints R

**Corrective Training:**
* Design classifier based on estimates of Pr(C,) and P r (0 | C,) to achieve minimum error rate on training set
* Use HMM P r (0 | 1.) as contrasting reference for improved generalization capability to independent data sets.

### 4.2 Integration of Nonspectral and Spectral Features

**Integration of Nonspectral and Spectral Features**

**HMM Use in Speech Recognition**:
- HMMs have been limited to modeling short-time speech spectral information (smoothed representation of the speech spectrum)
- Spectral feature vector is extremely useful and led to successful recognizer designs
- Success can be attributed to spectral analysis techniques and understanding of the importance of the speech spectrum

**Nonspectral Speech Features**:
- Prosody (segmental and supra-segmental levels)
- Physical manifestations: normalized energy, differential energy, pitch (fundamental frequency)

**Integrating Nonspectral and Spectral Features**:
1. **Performance Improvement**:
   - Incorporating log energy (and differential) into feature vector or likelihood calculation with moderate success
   - Performance improvement smaller than anticipated based on importance of prosody in speech
2. **Temporal Rate of Change**:
   - Spectral parameters require high sampling rate (100 Hz) to characterize vocal tract variations
   - Prosodic features occur at syllabic rate (10 Hz), which is much slower than spectral parameters
3. **Combining Feature Sets**:
   - Need to combine two feature sets with fundamentally different time scales and sequential characteristics
   - Challenges: performing optimum modeling and decoding
4. **Statistical Characterization**:
   - Need to know joint distribution of the two feature sets
   - For correlated (not independent) features, correction for correlation is required
   - Proposed method: principal component analysis on joint feature set before hidden Markov modeling
      * Alleviates difficulties but not a total solution due to lack of physical significance and open-set problem

### 4.3 Duration Modeling in HMM's

**Duration Modeling in HMMs**

**Inherent Limitations of HMM Approach**:
- Temporal duration is modeled as an exponential distribution within a state (Pr(d) = e^(-a*d))
- This model is inappropriate for most speech events

**Alternatives to Exponential Duration Model**:
1. **Semi-Markov Chain**:
   - State transitions do not occur at regular time intervals
   - Durations between states are modeled as independent random variables (Pr(d_i))
   - Joint probability: P(Observation_sequence, State_sequence) = C * Pr(Observation_sequence) * Pr(State_sequence)
2. **Duration Modeling in Semi-Markov HMMs**:
   - Duration model (Pr(d)) is treated as a discrete distribution over possible dwell times (1 <= di <= Dm)
   - Computational complexity increases due to irregular transition timing
3. **Drawbacks of Duration Model for Speech Recognition**:
   - Increased computational complexity, especially in decoding lattices
   - Complications with search algorithms like beam search and stack algorithm

### 4.4 Model Clustering and Splitting

**Model Clustering and Splitting**

**Assumptions in Statistical Modeling**:
- Variability in observations from an information source can be modeled by statistical distributions
- Source could be a single word, subword unit like phoneme, or word sequence

**Motivations for Multiple HMM Approach**:
- Lumping all variability leads to unnecessarily complex models with lower modeling accuracy
- Some variability may be known a priori and warrant separate modeling of source data sets

**Clustering Algorithms**:
- **k-means clustering algorithm**
- **Generalized Lloyd algorithm** in vector quantizer designs
- **Greedy growing algorithm** in set-partition or decision-tree designs
- Suitable for separating inconsistent training data into homogeneous subgroups

**Clustering Algorithms Application**:
- Successful application to speech recognition using ML criterion
- Interaction with other estimation criteria and Bayes minimum-error classifier design remains open question

**Alternative: Model Merging**
- Subdivide source into large number of subclasses, merge based on source likelihood considerations
- Examples: Build context sensitive units (left-context and right-context) for speech recognition
- Determine which pairs of models to merge based on small change in entropy (AH)
- Practical questions about acceptable AH remain

### 4.5 Parameter Significance and Statistical Independence

**Parameter Significance and Statistical Independence**

**A. Discrimination Capability of a's vs. b's:**
- Importance of state transition coefficients (a's) is the same as observation density (b's) in theory
- In practice, this is not usually the case due to:
  - Limited dynamic range of a's, especially for left-to-right models
  - Nearly unlimited dynamic range of b's, particularly with continuous density functions

**B. Combining a's and b's:**
- When combined, probabilities give the probability distribution of HMM
- In practice, atj's can be neglected for left-to-right model with no effect on recognition performance

**C. Unbalanced Numerical Significance:**
- a's affect bj(.) estimate, and vice versa in training
- Paradox is due to discrimination capability of a's vs. b's dynamic range

**D. Usefulness of Markov Chain Contribution:**
- Transition probability plays an important role in parameter estimation
- Introduction of semi-Markov models can enhance Markov chain contribution significance

**E. Statistical Independence Assumption:**
- Pr(do) implies statistical independence given known state sequence
- Argument against assuming independence within a stationary sound state
- Problems: difficult to choose appropriate form and estimate parameters from limited data

**F. Higher Order HMMs**

**G. Summary of First-Order Model Limitations:**
- Inadequate for grammatical structure modeling in higher level processing
- Need for further understanding and practical implementation advances

**H. Performance of HMM Systems:**
- Capable of achieving over 95% word accuracy in certain speaker-independent tasks with vocabularies up to 1,000 words
- Further progress expected to make technology usable in everyday life.

