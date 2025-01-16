# Hey AI Can You Grade My Essay?: Automatic Essay Grading

by Maisha Maliha, Vishal Pramanik
https://arxiv.org/html/2410.09319

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
  - [1.1 Difficulties in Automatic Essay Grading](#11-difficulties-in-automatic-essay-grading)
- [2 Related Works](#2-related-works)
  - [2.1 Classical Machine-Learning based Systems](#21-classical-machine-learning-based-systems)
  - [2.2 Deep-Learning based Systems](#22-deep-learning-based-systems)
- [3 Background and Technologies](#3-background-and-technologies)
  - [3.1 Recurrent Neural Networks](#31-recurrent-neural-networks)
  - [3.2 Convolutional Neural Network](#32-convolutional-neural-network)
  - [3.3 Bidirectional Encoder Representations from Transformers (BERT)](#33-bidirectional-encoder-representations-from-transformers-bert)
- [4 Dataset](#4-dataset)
- [5 Techniques and Implementation](#5-techniques-and-implementation)
  - [5.1 SVM](#51-svm)
  - [5.2 BERT](#52-bert)
  - [5.3 Collaborative Deep Learning Network (CDLN)](#53-collaborative-deep-learning-network-cdln)
- [6 Experimentation](#6-experimentation)
- [7 Results and Analysis](#7-results-and-analysis)
  - [7.1 Automatic Evaluation](#71-automatic-evaluation)
  - [7.2 Robustness of the model](#72-robustness-of-the-model)
- [8 Conclusion and Future Works](#8-conclusion-and-future-works)

## Abstract

**Automatic Essay Grading (AEG)**:
- Attracts attention of NLP community due to applications in scoring essays, short answers, etc.
- Can save time and money compared to manual grading.
- Existing works use a single network responsible for the whole process, which may be ineffective.
- This work introduces a new model that outperforms state-of-the-art AEG models.
- Uses **collaborative learning** and **transfer learning**:
  - One network checks grammatical and structural features of sentences.
  - Another network scores the overall idea in the essay.
  - Learnings are transferred to another network for essay scoring.
- Compared performances of different models, proposing a new model with an accuracy of **85.50%**.

**Keywords**:
- Automatic Essay Grading
- Collaborative learning
- Deep Learning
- Recursive Learning
- Recursive Neural Network.

## 1 Introduction

**Automatic Essay Grading (AEG)**
- A subfield of Natural Language Processing (NLP) that has been around for over 50 years
- Suggested the first AEG system: [[1](https://arxiv.org/html/2410.09319v1#bib.bib1)]
- Little progress due to lack of resources and processing power until development of deep neural networks
- **Deep Neural Networks**: [[2](https://arxiv.org/html/2410.09319v1#bib.bib2)}, [[3](https://arxiv.org/html/2410.09319v1#bib.bib3)]
- **Automatic Essay Grading**:
  - Using a machine to grade a text in response to an essay prompt
  - **Holistic AEG**: Awarding overall quality grade
  - **Trait-Specific AEG**: Rating essays based on single trait/attribute (content, organization, style)
- **Problem**: Need for human-graded essays to evaluate the AEG system
- **Solution**: Domain adaption techniques like cross-domain AEG

**Contributions of the Paper:**
- Introduce a collaborative deep learning model for automatic essay grading
- Model considers nature, idea, grammar, and structure of sentences in the essay before grading it
- Comparative analysis of results from different machine and deep learning models to the proposed deep learning network.

### 1.1 Difficulties in Automatic Essay Grading

**Automatic Essay Grading (AEG)**

**Holistic AEG**:
- Entails providing an overall score/grade to the essay based on its quality
- Majority of AEG research focuses on this approach

**Machine Learning with Classifiers for Holistic AEG**:
- Early research used machine learning with classifiers to grade essays holistically
- Examples: e-rater [[4](https://arxiv.org/html/2410.09319v1#bib.bib4)] and Intelligent Essay Assessor [[5](https://arxiv.org/html/2410.09319v1#bib.bib5)]
- Use a variety of features:
  - **Surface-level features**: Word count, average word length, average sentence length, etc.
  - More complex features: Usage score (detecting usage errors)
  - Kernels

**Task-Independent Features for AEG**:
- Study on the problem of cross-domain AEG [[6](https://arxiv.org/html/2410.09319v1#bib.bib6)]
- Findings:
  - Best results for cross-domain AEG when source and target prompts are similar

## 2 Related Works

**Systems Used in AEG:**
- Overview of various AEG systems discussed





### 2.1 Classical Machine-Learning based Systems

**AEG Systems**
- Early systems employ machine learning techniques
- Approach: feature engineering and ordinal classification/regression
- Project Essay Grade (PEG) [[1](https://arxiv.org/html/2410.09319v1#bib.bib1)]: first AEG system, uses intrinsic properties called 'trins' for an essay score approximation

(Note: Trins are analogous to features)

### 2.2 Deep-Learning based Systems

**Neural Networks in NLP (since 2010s)**

* CNN and other hierarchical models used for various tasks [[7](https://arxiv.org/html/2410.09319v1#bib.bib7)]
* SSWE system developed by [[8](https://arxiv.org/html/2410.09319v1#bib.bib8)]: learns score-specific word embeddings and uses LSTMs to get essay representation for scoring
* Pre-trained word embeddings used in [[9](https://arxiv.org/html/2410.09319v1#bib.bib9)]'s architecture, similar to [[8](https://arxiv.org/html/2410.09319v1#bib.bib8)], for scoring essays with LSTMs and other RNNs

## 3 Background and Technologies

### 3.1 Recurrent Neural Networks

**Recurrent Neural Network (RNN)**

**Definition:**
- Deep neural network that uses previous step's output as feedback input
- Used for sequences, next-word prediction in sentences

**Hidden State and Output:**
- Hidden state: a<t>, represented by the hidden state vector at time step t
- Output: y<t>, represented by the output vector at time step t

**Equations:** (1) and (2)
- Equation (1): **hidden state update**
  * g1: activation function
  * Wa, Wx, by: coefficients shared over time steps
  * a<t> = g1(Waa<t−1> + Wax<t> + ba)
- Equation (2): **output prediction**
  * g2: activation function
  * Wy, b_y: coefficients shared over time steps
  * y<t> = g2(Wya<t> + by)

**Implementation:**
- RvNN works for sentence processing
- Word embeddings are fed into a neural network to obtain phrase embeddings
- Phrase embeddings then pass through the network to obtain sentence embedding output vectors
- Sentence embedding vectors store grammatical and structural properties of the sentence.

### 3.2 Convolutional Neural Network

**Convolutional Neural Network (CNN)**
- **Three main parts**: convolutional layer, pooling layer, dense layer
- **Convolutional Layer**:
  - Uses **kernel function** to extract information from images
  - Calculated using equation: `(f∗h)[m,n] = ∑j∑kh[j,k]f[m−j,n−k]`
  - Where:
    - `G[m,n]` is the resultant image matrix
    - `f` and `h` are input images
    - `m` and `n` are dimensions of the resultant image
    - `j` and `k` are dimensions of the kernel function
- **Pooling Layer**:
  - Decreases dimension of input matrix
  - Extracts deeper meaning from feature matrix

### 3.3 Bidirectional Encoder Representations from Transformers (BERT)

**BERT as Encoder in Transformer Model**

- BERT ([11](https://arxiv.org/html/2410.09319v1#bib.bib11)) is the encoder part of the Transformer model ([13](https://arxiv.org/html/2410.09319v1#bib.bib13)).
- It extracts features from input sentences and passes output vectors to the Decoder part of the Transformer.
- BERT encodes input sentences with positional encodings, enabling parallel processing.
- The encoder consists of 12 blocks, each containing a multi-headed self-attention mechanism and a dense layer.
- The attention mechanism helps establish relationships between words within the sentence, enhancing the model's understanding capabilities.

## 4 Dataset

**ASAP Automatic Essay Grading Dataset**
- Commonly used dataset for automatic essay grading: **Automated Students Assessment Prize (ASAP) AEG dataset**
- Comprises nearly 13,000 essays in response to 8 different essay prompts
- Data is freely available on Kaggle (https://www.kaggle.com/c/asap-aes/data)

**Dataset Statistics**:
- Consists of 8 essays, each corresponding to one question
- Originally authored by students in grades 7 through 10
- Essays scored based on four points: **ideas**, **style**, **organization**, and **conventions**

**Table 1: Data Analysis**:
- Shows the different prompts, number of essays, average length, and score range
- Prompt | No. of Essays | Avg Length | Score Range
  - 1 | 1783 | 350 | 2-12
  - 2 | 1800 | 350 | 1-6
  - 3 | 1726 | 150 | 0-3
  - 4 | 1772 | 150 | 0-3
  - 5 | 1805 | 150 | 0-4
  - 6 | 1800 | 150 | 0-4
  - 7 | 1569 | 250 | 0-30
  - 8 | 723 | 650 | 0-60

**Essay Types**:
- The four types of essays present are **persuasive**, **narrative**, **expository**, and **source-dependent responses**
- Scores given by raters belonging to domain one were added as the final score
- Dataset divided into 4:1 ratios for training and testing, respectively.

## 5 Techniques and Implementation

**Comparative Analysis Models Tested**
- Evaluated various models: [model names] for comparison purposes.

### 5.1 SVM

**Automatic Grading System using TF-IDF Vectorization and SVM Classification**

**Approach:**
- Treated essay grading as a classification task
- Maximum mark: 60, minimum mark: 0
- Divided marks into 61 classes
- Used TF-IDF vectorisation with l2 normalisation for essay representation

**TF-IDF Calculation:**
- Calculate tfidf(w,e) using Equation (4) in text
  - tf(w,e): term frequency of word w in document e
  - idf(w): inverse document frequency of word w across all essays
- Smoothened tfidf by adding a constant n to the denominator (Equation 5)

**Model Training:**
- Used multi-class classification SVM with gaussian kernel for model training
- Trained on essay set and evaluated using SVM classifier. [[13](https://arxiv.org/html/2410.09319v1#bib.bib13)]

### 5.2 BERT

**Automatic Essay Grading Model**
- Standard **bert-base-cased** model used:
  - 6 encoder blocks
  - Consideration of word case
- Purposefully chosen for grading essays:
  - Word case provides important information (beginning of sentence, proper nouns)
- Limitations:
  - Cannot accept inputs greater than 512 tokens
  - Eliminated essays with more words than 500 words

**Bert Model Architecture:**
- Each encoder block contains:
  - Self-attention layer
  - Followed by a feedforward layer
- Attention layer:
  - 8 multi-headed attention mechanisms
  - Stores information about sentence's words
- Feedforward network maintains output vector dimension and sends it to next encoder block
- Positional encoding of words with BERT embeddings enables fast processing.

### 5.3 Collaborative Deep Learning Network (CDLN)

**Collaborative Learning Methods**

**Definition**: Collaborative learning is a method of learning where pupils work in groups on separate tasks contributing to a common outcome or on shared tasks.

**Transfer Learning**:
- Instead of a single deep neural network, learning can be distributed among several networks
- Their collective knowledge can be shared

**Collaborative Deep Learning Network (CDLN)**
- **Architecture**:
  - Recursive Neural Network (RvNN)
  - Convolutional Neural Network (CNN)
  - Long Short Term Memory (LSTM)
  - Dense neural network

**Convolutional Neural Network (CNN)**
- Understands the idea conveyed in sentences using:
  - Convolution and average pooling layers
  - Word2Vec embeddings of 100 dimensions each
  - Kernel sizes: 1x105x8 for convolution, 1x90x8 for average pooling
  - Repeated 5 times
- Helps analyze the essay and brings out the idea conveyed in sentences

**Recursive Neural Network (RvNN)**
- Understands the structure of sentences
- Divides words into bigrams
- Representations vectors fed into a neural network with:
  - 200 neurons in the first layer
  - 4 layers of 150 neurons each
  - Output layer of 100 neurons to match word embedding dimension
- Helps check essays for grammatical and sentence construction errors

**Long Short Term Memory (LSTM)**
- Takes input vector from RvNN and CNN concatenation
- Output vector is 1x10000 dimensions
- Gathers learnings of previous deep learning networks
- Stores knowledge about sentence structure and ideas conveyed in essays
- Information is forwarded to next layers for essay grading

**Dense Layer and Output Layer**:
- Dense layer takes input from LSTM output
- 5 hidden layers with 120 neurons each
- Last output layer gives the essay grade.

## 6 Experimentation

**Experiments Conducted on Six Models**

* Models: CDLN, BERT, RNN, ANN, SVM
* Detailed architecture outlined in previous section
* Learning rate: 0.0001, batch size for training: 32
* Epochs: 15 (CDLN, BERT, RNN, ANN), 8 (SVM), 6 (SVM)
* Dropouts utilized in deep learning models to prevent overfitting
* Eight-fold cross-validation during training for all models

## 7 Results and Analysis

**Evaluation Metrics**
- Mean Square Error (MSE)
- Pearson’s Correlation Coefficient (PCC): Measures linear correlation between two variables (-1 to 1; 1: highly correlated in positive direction, -1: not correlated or opposite directions). [[16](https://arxiv.org/html/2410.09319v1#bib.bib16)]
- Quadratic Weighted Kappa (QWK): Most common metric for AEG system performance evaluation, used in our evaluation. [[17](https://arxiv.org/html/2410.09319v1#bib.bib17)]

### 7.1 Automatic Evaluation

**Comparison of Machine and Deep Learning Models for Automatic Essay Grading**

**Models Compared**:
- CDLN model (proposed)
- LSTM
- RNN
- ANN
- SVM
- Baseline models: TDNN(ALL), CNN-LSTM, CNN-LSTM-ATT, 2L-LSTM, and CNN-LSTM-ATT from [[20](https://arxiv.org/html/2410.09319v1#bib.bib20)]

**Comparison Metrics**:
- Accuracy (Accu.)
- Pearson Correlation Coefficient (PCC)
- Quality Metric (QWK)

**Results**:
- **CDLN model outperforms other models and baseline models on each prompt and overall**.
- Sharing of knowledge between CNN, RvNN, and LSTM boosts results.
- Self-attention mechanism in BERT leads to better results than TDNN(Sem+Synt).

**Performance Comparison Tables**:
- **Table 2**: Prompt-wise performance comparison of CDLN model and other baseline models
- **Table 3**: Overall average performance comparison of CDLN model and other baseline models

**Additional Observations**:
- The original essays' grades are compared with the paraphrased ones in **Figure 3**.

### 7.2 Robustness of the model

**Robustness Check of CDLN Model for Automatic Essay Grading**

**Testing the Robustness of CDLN Model**:
- Conducted a robustness check by rephrasing 1000 random essays using Quillbot, a paraphrasing tool
- Graduated the original and modified essays using the CDLN model
- Compared the grades before and after paraphrasing

**Results**:
- Grades given by the model were very close before and after paraphrasing
- Average marks for modified essays were more than original essays
- This may be due to proper sentence structure added by Quillbot and removal of grammatical errors in original essays

**Difference Between Grades**:
- Measured the difference using mean square error formula: 
  - g\_original: Marks graded by the model for original essays
  - g\_modified: Marks graded for paraphrased essays
  - Δ = (g\_original - g\_modified)² / N
- Resulted in a value of 0.34, indicating grades are very close to each other and the model is quite robust.

## 8 Conclusion and Future Works
**Main Focus**:
- Demonstrates effectiveness of collaborative learning in automatic essay grading (AEG)
- Multiple networks working together to analyze essay features
- Performance improvement shown through results

**Key Achievements**:
- Model outperformed pretrained models
- Surpassed state-of-the-art AEG systems
- Successfully implemented collaborative learning approach

**Current Limitations**:
- Provides only holistic essay scores
- Does not offer paragraph-level scoring
- Room for performance improvement

**Future Opportunities**:
- Integration with newer deep neural networks
- Potential use of new pretrained networks
- Possibility of paragraph-wise scoring implementation
- Scope for further research and investigation
