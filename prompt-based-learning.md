# Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing

Authors: Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham NeubigAuthors Info & Claims

https://dl.acm.org/doi/10.1145/3560815

## Contents
- [Abstract](#abstract)
- [1 TWO SEA CHANGES IN NATURAL LANGUAGE PROCESSING](#1-two-sea-changes-in-natural-language-processing)
- [2 A FORMAL DESCRIPTION OF PROMPTING.](#2-a-formal-description-of-prompting)
- [3 PROMPT TEMPLATE ENGINEERING](#3-prompt-template-engineering)
- [4 PROMPT ANSWER ENGINEERING](#4-prompt-answer-engineering)
- [5 MULTI-PROMPT LEARNING](#5-multi-prompt-learning)
- [6 TRAINING STRATEGIES FOR PROMPTING METHODS](#6-training-strategies-for-prompting-methods)
- [7 APPLICATIONS](#7-applications)
- [8 PROMPT-RELEVANT TOPICS](#8-prompt-relevant-topics)
- [9 CHALLENGES](#9-challenges)
- [10 CONCLUSION](#10-conclusion)

## Abstract

**Prompt-Based Learning in Natural Language Processing:** A Systematic Survey

**Overview:**
- New paradigm in NLP: prompt-based learning
- Instead of traditional supervised learning (P(y|x)), models model probability of text directly
- Use language models for prediction tasks with pre-trained models and prompts

**Framework:**
1. Pre-train language model on massive raw text data
2. Define a new prompting function to perform few-shot or zero-shot learning
3. Modify input x into textual string prompt x/prime, fill unfilled slots, obtain final string ˆx
4. Derive output y from ˆx

**Advantages:**
- Allows pre-training on vast amounts of raw text
- Adapts to new scenarios with few or no labeled data

**Article Contents:**
1. Introduction and background
2. Mathematical notations for prompt-based learning
3. Existing work organized by dimensions: pre-trained language models, prompts, tuning strategies
4. Making the field more accessible through resources like NLPedia–Pretrain
5. Conclusion

**Keywords:** Computing methodologies, Natural language processing; Additional Key Words and Phrases: Pre-trained language models, prompting.

## 1 TWO SEA CHANGES IN NATURAL LANGUAGE PROCESSING

**Two Sea Changes in Natural Language Processing**

**Fully Supervised Learning**:
- Centrally played a role in machine learning tasks
- Long been central to natural language processing (NLP)
- Early NLP models relied heavily on **feature engineering**

**Shift Toward Neural Network Models**:
- Salient features learned jointly with training the model itself
- Focus shifted to **architecture engineering**, where inductive bias provided through network architecture design

**Pre-Training Paradigm**:
- Shifted from fully supervised learning
- Pre-train a language model (LM) on large datasets of text data
- Fine-tune the LM's parameters for specific downstream tasks using task-specific objective functions
- Focus turned to **objective engineering**

**Pre-Training, Prompting, and Predicting**:
- Current approach as of 2021: "pre-train, prompt, and predict"
- Reformulate downstream tasks to look like those solved during LM training with a textual prompt
- Pre-trained LM can be used to predict the desired output without additional task-specific training
- Advantage: A single unsupervised-trained LM can be used to solve many tasks
- Disadvantage: Requires **prompt engineering** to find appropriate prompts for the model

**Organization of this Survey**:
1. Overview and formal definition of prompting methods (Section 2)
2. Prompt template engineering (Section 3)
3. Prompt answer engineering (Section 4)
4. Multi-prompt learning methods (Section 5)
5. Prompt-aware training methods (Section 6)
6. Applications of prompt-based learning and their interaction with choice of prompting method (Section 7)
7. Locating the current state of research within the research ecosystem (Section 8)
8. Challenging problems for further research (Section 9)
9. Systematic resources on prompt learning and pre-training, including:
   - Frequent updates to this survey, related slides, and soon
   - Figure 1: A typology of important concepts for prompt-based learning
   - Tables 7 and 8: Comprehensive comparison among different prompting methods
   - Table 5: Systematic organization of commonly used prompts
   - Table 4: Timeline of prompt-based research works
   - Table 1: Comprehensive comparison among different pre-trained LMs

## 2 A FORMAL DESCRIPTION OF PROMPTING. 
### 2.1 Supervised Learning in NLP
**Overview:**
- In traditional supervised learning systems for Natural Language Processing (NLP)
- Input `x`: usually text
- Predict output `y` based on model `P(y|x;θ)`
  * `y` can be a label, text, or other variety

**Examples:**
1. **Text Classification:**
   - Takes an input `x`: text
   - Predicts a label `y` from a fixed label set `Y`
   - Example: Sentiment Analysis
     * Input `x`: "I love this movie"
     * Output `y`: {++, +, ––, --}
2. **Conditional Text Generation:**
   - Takes an input `x`: text
   - Generates another text `y` as output
   - Example: Machine Translation
     * Input `x`: "Hyvää huomenta" (Finnish for "Good morning")
     * Output `y`: "Goodmorning" (English)

### 2.2 Prompting Basics

**Prompting Basics**
- Supervised learning requires annotated data for training a model P(y|x;θ)
- Prompt-based learning methods attempt to circumvent this issue by learning an LM that models the probability P(x;θ) of text itself
- This allows predicting y using probabilities instead of requiring large labeled datasets
- Prompting refers to a method where the model predicts the highest-scoring answer z given input x

**Basic Prompting Steps:**
1. **Prompt Addition**: Apply a template function f(prompt) that modifies the input text x into a prompt x' by inserting a slot [Z] for an intermediate generated answer text z, which will later be mapped to y.
2. **Answer Search**: Search for the highest-scoring text z within permissible values Z using the language model (LM) P(·;θ).
3. **Answer Mapping**: If necessary, map the highest-scoring answer z to the corresponding output value y.

**Template Structure:**
- Consists of two steps: applying a template and filling slot [X] with input text x
- Templates may include virtual words or even continuous vectors
- Number of [X] and [Z] slots can be flexible based on task needs

**Examples:**
1. Sentiment Analysis: "Overall, it was a [Z] movie"
2. Machine Translation: "Finnish: [X] English: [Z]"
3. Classification Tasks: "The text is about [Z]."
4. Generative Tasks: "Textual Similarity: [X1] TL;DR: [Z],[X2] Yes"
5. Regression Tasks: "Textual Similarity: [X1] [Z],[X2] No"

### 2.3 Design Considerations for Prompting

**Design Considerations for Prompting**

**Pre-trained Language Model (LM) Choice**:
- Variety of pre-trained LMs to choose from
- Discussed in the Appendix, focusing on dimensions important for interpreting their utility in prompting methods

**Prompt Template Engineering**:
- Proper choice of prompt template affects both accuracy and task performance
- Discussed in Section 3: Choosing a Prompt Template

**Prompt Answer Engineering**:
- Designing Z differently, along with the mapping function, may be necessary depending on the task
- Discussed in Section 4: Prompt Answer Engineering

**Expanding the Paradigm**:
- Equations above represent a simple underlying framework for prompting
- Discussed ways to expand this paradigm to improve results or applicability in Section 5

**Prompt-based Training Strategies**:
- Different strategies to train: prompt, language model (LM), or both
- Summarized in Section 6, with relative advantages

## 3 PROMPT TEMPLATE ENGINEERING

**Prompt Template Engineering**

**Process**: creating a prompting function `f(prompt)` that results in the most effective performance on the downstream task.

**Approaches to Prompt Creation**:
- **Manual approach**: humans or algorithms search for the best template
- **Automated approach**: using automated methods, such as:
  - **Cloze prompts**: fill in the blanks of a textual string (e.g., "I love this movie, it is a [Z]movie")
  - **Prefix prompts**: continue a string prefix (e.g., "I love this movie. What's the sentiment of the review? [Z]")

**Factors Affecting Prompt Shape Selection**:
- Task type: determines which prompt shape is more conducive
  - Generation tasks: prefix prompts are better for standard auto-regressive LMs, cloze prompts for masked LMs
  - Text pair classification tasks: require space for two inputs [X1] and [X2]
- Model type: full text reconstruction models can be used with either cloze or prefix prompts

**Manually Crafted Prompts**:
- Example from the LAMA dataset [100]: manually created clozetemplates to probe knowledge in LMs
- Examples by Brow et al. [9]: manually crafted prefix prompts for question answering, translation, and common sense reasoning tasks
- Examples by Schick and Schütze [118, 117, 120]: predefined templates used in a few-shot learning setting for text classification and conditional text generation tasks

**Automated Template Learning**

**Issues with Manual Template Design**:
- Time-consuming and requires experience, especially for complex tasks like semantic parsing
- Experienced designers may still fail to find optimal prompts manually

**Approaches to Automate Prompt Design**:
- **Discrete prompts**: Described in a discrete space, often corresponding to natural language phrases
  - **Prompt Mining**: Scrapes text corpus for strings containing input and output, finds middle words or dependency paths between them
  - **Prompt Paraphrasing**: Takes an existing seed prompt, paraphrases it into a set of candidate prompts, selects the one with highest training accuracy
  - **Gradient-based Search**: Steps through tokens in the prompt to find short sequences that trigger desired prediction from pre-trained LM
  - **Prompt Generation**: Treats prompt generation as a text generation task using standard NLG models
  - **Prompt Scoring**: Scores filled templates using unidirectional LM, selects highest scoring template
- **Continuous prompts**: Perform prompting directly in the embedding space of the model, removing constraints on template words and parameters
  - **Prefix Tuning**: Prepends sequence of task-specific vectors to input, optimizes log-likelihood objective
  - **Tuning Initialized with Discrete Prompts**: Initializes search for continuous prompt using a previously discovered discrete prompt
  - **Hard-Soft Prompt Hybrid**: Inserts trainable embeddings into hard prompt template, fine-tunes them to increase task accuracy

## 4 PROMPT ANSWER ENGINEERING

**Two Dimensions of Answer Design:**
- **Answer Shape**: Characterizes granularity of answers
  * Tokens: One or a subset of tokens from pre-trained language model's vocabulary
  * Span: Short multi-token span, often used with cloze prompts
  * Sentence/Document: Longer phrasal or sentential answers, commonly used in language generation tasks
- **Answer Space Design Methods:**
  - **Manual Design:**
    * Unconstrained Spaces: Directly map answer to final output using identity mapping
      * Examples: Text classification, entity recognition, multiple-choice question answering
    * Constrained Spaces: Craft answer space and its mapping to underlying class manually by system or benchmark designer
  - **Discrete Answer Search:**
    * Paraphrasing: Expand initial answer space using paraphrasing methods and select final answers based on probability of output
      * Example: Use back-translation method for paraphrasing (Jiangetal., 2017)
    * Prune-then-Search: Generate an initial pruned answer space and search over it to select final set of answers
      * Example: Select top-k tokens achieving highest probability score using learned classifier (Shi et al., 2020)
    * Label Decomposition: Decompose relation labels into constituent words for relation extraction tasks
  - **Continuous Answer Search:**
    * Optimize token embeddings for each class label to improve performance.
      * Example: Assign virtual tokens for each class label and optimize their embeddings along with prompt token embeddings (Hambardzumyan et al., 2019).

## 5 MULTI-PROMPT LEARNING

**Multi-Prompt Learning**

**Overview:**
- Use of multiple prompts to improve efficacy of learning methods
- Motivations: complementary advantages of different prompts, cost alleviation, performance stabilization

**Prompt Ensembling:**
- Use of multiple unanswered prompts for input at inference time to make predictions
  - Discrete or continuous prompts
  - Improve performance on downstream tasks by leveraging complementary advantages of different prompts
- Current research explores effective ways for prompt ensembling: uniform averaging, weighted averaging, majority voting, knowledge distillation, and text generation methods.

**PromptAugmentation:**
- Providing additional answered prompts to demonstrate how language model should answer actual prompt
- Few-shot demonstrations take advantage of strong language models' ability to learn repetitive patterns
- Challenges: sample selection and ordering
  - Researchers use sentence embeddings for effective example selection (Gao et al., Liu et al.)
  - Lu et al. propose entropy-based methods for scoring different candidate permutations; Kumar and Talukdar search for good promotion orderings
- Closely related to retrieval-based methods that provide more text context to improve performance (Mishra et al., Yoo et al.).

**Prompt Composition:**
- Breaking down complex tasks into subtasks and defining a composite prompt based on those sub-prompts
  - Identifying entities and classifying relationships in relation extraction (Han et al.)

**Prompt Decomposition:**
- For tasks where multiple predictions need to be performed for one sample, breaking down the holistic prompt into different sub-prompts and answering each sub-prompt separately.
  - Converting input into set of text spans, then prompting model to predict entity type for each span (Cui et al.).

## 6 TRAINING STRATEGIES FOR PROMPTING METHODS

**Training Strategies for Prompting Methods**

**Different Tuning Strategies:**
- **Strategy**: LM Params (Pretrained), Prompt Params, Additional Tuned Prompt Params, Tuned or Fixed
- **Zero-Shot Learning**: No explicit training of downstream task model
  - Traditional pre-training and fine-tuning without prompts as promptless fine-tuning
    * Parameters of the underlying LM are updated via gradients from downstream training samples
    * Example: BERT, RoBERTa
      - Advantages: Simple, powerful, widely used
        * Fits to larger training datasets
        - Disadvantages: Overfitting or unstable learning on small datasets, prone to catastrophic forgetting
  - Tuning-free prompting
    * Directly generates answers without changing LM parameters based on a prompt
    * Optionally augment input with answered prompts as in-context learning
    * Example: LAMA, GPT-3
      - Advantages: Efficiency, no parameter update process, applicable in zero-shot settings
        * No catastrophic forgetting since LM parameters remain fixed
      - Disadvantages: Heavy engineering required for high accuracy, inefficiency when providing many answered prompts
  - Fixed-LM prompt tuning
    * Updates only the prompts’ parameters using supervision signals obtained from downstream training samples
    * Keeps pre-trained LM unchanged
    * Example: Prefix-Tuning, Prompt-Tuning
      - Advantages: Retains knowledge in LMs and suitable for few-shot scenarios, superior accuracy than tuning-free prompting
        * Not applicable in zero-shot settings, limited representation power in large datasets
      - Disadvantages: Prompts are not human interpretable or manipulable
  - Fixed-prompt LM tuning
    * Tunes the parameters of the LM while using prompts with fixed parameters to specify model behavior
    * Example: PET-TC, PET-Gen, LM-BFF
      - Advantages: More efficient learning, particularly in few-shot scenarios
        * Template or answer engineering more completely specify the task
      - Disadvantages: Prompt engineering still required, LMs fine-tuned on one downstream task may not be effective on another
  - Prompt+LM Tuning
    * Fine-tunes both the parameters of the pre-trained LM and prompt-relevant parameters together
    * Example: PADA, P-Tuning
      - Advantages: Most expressive method suitable for high-dataset settings
        * Provides additional bootstrapping at the start of model training
      - Disadvantages: Requires training and storing all parameters of the models, may overfit to small datasets.

## 7 APPLICATIONS

**Table 7: List of Applications and their Categories**
- **Knowledge Probing:** Fact Retrieval, Linguistic Probing
- **Structure Prediction:** Semantic Parsing
- **Classification-based Tasks:** Text Classification, Natural Language Inference, Text Pair Classification
- **Information Extraction:** Relation Extraction, Named Entity Recognition

### Section 7.1: Knowledge Probing
* **Factual Probing (a.k.a. Fact Retrieval):**
  * Quantifying factual knowledge in pre-trained LMs
  * Transform original input into cloze prompt
    - Manually crafted or automatically discovered
  * Focuses on finding effective templates and analyzing results of different models using those templates
  * Techniques: Discrete template search, Continuous template learning, Prompt ensemble learning.
* **Linguistic Probing:**
  * Handling linguistic phenomena such as analogies, negations, semantic role sensitivity, semantic similarity, cant understanding, rare word understanding
  * Presenting tasks in natural language sentences for LM completion
  * Previous work: [9], [25], [131], [116]

### Section 7.2: Structure Prediction
* **Semantic Parsing:**
  * Generating structured meaning representation given a natural language input
  * Framing as paraphrasing task or constraining decoding process
  * Results demonstrate effectiveness using in-context learning and paraphrasing reformulation [124]

### Section 7.3: Classification-based Tasks
* **Text Classification:**
  * Most work uses cloze prompts for template engineering and answer engineering
  * Previous work explores fixed-prompt LM tuning strategies in few-shot setting [32,40,67,115,117]
* **Text Pair Classification:**
  * Predict relationship of two given sentences (paraphrase identification, natural language inference, textual similarity prediction)
  * Cloze prompts commonly used for template search with manually pre-selected answer space Z [117,120]

### Section 7.4: Information Extraction
* **Relation Extraction:**
  * Predict relation between two entities in a sentence
  - Challenges: Larger label space and importance of tokens for entity mentions
  - Proposed methods: Adaptive answer selection, task-oriented prompt template construction [13]
* **Named Entity Recognition:**
  * Identifying named entities (person name, location) in a given sentence
  - Difficulty of prompt-based learning application to tagging tasks like NER
  - Recent work: Template-based NER model using BART [17]

### 7.5 “Reasoning” in NLP

**Reasoning in NLP:**
- Debate on deep neural networks' reasoning abilities vs. memorizing patterns [3, 89]
- Commonsense reasoning benchmarks [47, 72, 101, 107]
  * Winograd Schemas: identifying ambiguous pronouns' antecedents in context
  * Example: "The trophy doesn’t fit into the brown suitcase, because it is too large."
  * Scoring generation probability of each candidate choice [134]
- Mathematical reasoning with pre-trained LMs [9, 88, 139]

**Question Answering (QA):**
- Various formats: extractive QA, multiple-choice QA, free-form QA
- Handled differently using different model frameworks
- Pre-trained LMs used as text generation problem [55]
  * Fine-tuning seq2seq pre-trained LMs (e.g., T5) and appropriate prompts from context and questions
- Probabilities from pre-trained LMs on QA tasks not very predictive of correctness [51]

**Text Generation:**
- Conditioned on some other piece of information
- Prompting methods applied using prefix prompts with autoregressive pre-trained LMs [105]
  * Text summarization and machine translation demonstrations
  * Brown et al. performing in-context learning for text generation [9]
  * Fixed-prompt LM tuning for few-shot tasks by Schick and Schütze [118]
  * Prompt+LM tuning strategy by Dou et al. [23]

**Automatic Evaluation of Text Generation:**
- Yuan et al. using prompt learning for automated evaluation of generated texts [147]
  * Conceptualizing evaluation as a text generation problem
  * Adding "such as" phrase to translated text leads to improvement in correlation on German–English machine translation evaluation.

**Meta-Applications:**
- Domain adaptation using self-generated DRFs and sequence tagging [5]
- Self-diagnosis and debiasing based on biased or debiased instructions [121]
  * Generating text with specific templates for diagnosis and suppressing undesired attributes
- Dataset construction using pre-trained LMs to generate similar sentences [119]
- Multi-modal learning: text and vision [135]
  * Fixed-LM prompt tuning strategy with prompt augmentation techniques.

## 8 PROMPT-RELEVANT TOPICS

**Prompt-Based Learning vs Other Methods**

**Ensemble Learning**:
- Technique to improve performance by combining results from multiple systems
- In prompt ensembling, different predictions come from varying prompt variants

**Few-shot Learning**:
- Aims to learn a machine learning system with few training samples
- Prompt augmentation can be regarded as a form of priming-based few-shot learning

**Larger-Context Learning**:
- Introduces additional context to aid the learning process
- Prompt augmentation adds relevant labeled samples into the input, but not necessarily labeled data

**Query Reformulation**:
- Commonly used in information retrieval and question answering tasks
- Both query reformulation and prompt learning aim to make better use of existing knowledge bases

**QA-based Task Reformulation**:
- Aims to conceptualize various NLP tasks as a question-answering problem
- Methods similar to prompting, but focus on unifying tasks into a QA framework

**Controlled Generation**:
- Incorporates additional guidance beyond the input text into the generation model
- Some commonalities with prompt learning, such as adding extra information for better generation

**Supervised Attention**:
- Aims to provide supervision over a model's attention to extract important information
- Prompt learning and supervised attention share ideas in extracting salient information

**Data Augmentation**:
- Modifies existing data to increase the amount available for training
- Adding prompts can achieve similar accuracy improvements as adding 100 data points on average across classification tasks

## 9 CHALLENGES

### Challenge 9.1: Selection of Pre-trained LMs
- Difficult problem to choose pre-trained LMs for prompt-based learning
- Few systematic comparisons of benefits brought by different pre-trained LMs for this purpose

### Challenge 9.2: Prompt Design
- Most existing works focus on text classification and generation tasks
- Challenges in applying prompting methods to more complex tasks like information extraction and text analysis
- Expressing structured information (e.g., tree, graph, table) is a major challenge

### Challenge 9.3: Prompt Answer Engineering
- Many-class classification tasks: Difficult combinatorial optimization problem to select appropriate answer spaces
- Long-answer classification tasks: Unknown how best to decode multi-token answers using LMs
- Text generation tasks rely solely on single, lean answers; multiple references largely unexplored

### Challenge 9.4: Selection of Tuning Strategy
- Variety of methods for tuning parameters in prompts, LMs, or both
- Systematic understanding of tradeoffs between these different strategies lacking

### Challenge 9.5: Multiple Prompt Learning
- Prompt ensembling: Space and time complexity increase as more prompts are considered
- How to distill knowledge from multiple prompts remains under-explored
- Ensemble learning in text generation tasks not performed so far, due to complexity of ensemble models

### Challenge 9.6: Theoretical and Empirical Analysis of Prompting
- Lack of theoretical analysis and guarantees for prompt-based learning
- Soft-prompt tuning relaxes non-degeneracy assumptions, making it easier to extract task-specific information [141]
- Text classification tasks can be reformulated as sentence completion tasks, making language modeling a meaningful pre-training step [113]
- Prompts worth 100s of data points on average across classification tasks [114]

### Challenge 9.7: Transferability of Prompts
- Prompts selected under tuned few-shot learning scenario generalize well across similar models, while those under true few-shot learning do not [96]
- Transferability is poor when the model sizes are quite different in both scenarios

### Challenge 9.8: Combination of Different Paradigms
- Pre-training methods effective for fine-tuning may not be applicable to prompting-based learning as-is
- Re-thinking pre-training methods could improve accuracy and ease of applicability to prompting-based learning

### Challenge 9.9: Calibration of Prompting Methods
- Pre-trained LMs' probability distributions not well calibrated for good probabilistic predictions [33]
- Jiang et al. observed pre-trained LMs' (e.g., BART, T5, GPT-2) probabilities on QA tasks are well calibrated but identified pitfalls leading to bias towards certain answers [51, 151]
- Need to be cautious when assuming a single gold answer for an input due to competing forms of the same object sharing probability mass. Consider paraphrasing methods or calibrating word probabilities based on context.

## 10 CONCLUSION

**Prompt-Based Learning: A New Paradigm in NLP**

**Conclusion**
- Article summarizes and analyzes statistical natural language processing (NLP) paradigms
- Argues prompt-based learning is a promising new approach to NLP
- Aims to:
  - Help researchers understand the prompt-based learning paradigm better
  - Identify core challenges for scientifically meaningful advances
  - Highlight commonalities and differences between various NLP paradigms
- Potentially inspire work toward next paradigm shift in NLP research.

