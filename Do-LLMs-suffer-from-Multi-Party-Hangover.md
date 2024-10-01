# Do LLMs suffer from Multi-Party Hangover? 
**A Diagnostic Approach to Addressee Recognition and Response Selection in Conversations**

Nicolò Penzo, Maryam Sajedinia, Bruno Lepri, Sara Tonelli, Marco Guerini
https://arxiv.org/pdf/2409.18602

## Abstract
**Study on Multi-Party Conversations (MPC) Assessment**

**Background:**
- Challenges of assessing MPC performance
- Overlooked variances in model behavior
- Proposed methodological pipeline to investigate weaknesses

**Focus Areas:**
1. **Response Selection Task**
2. **Addressee Recognition Task**

**Methodology:**
- Extract representative diagnostic subdatasets
  - Fixed number of users
  - Good structural variety from open corpus
- Framework in terms of data minimization
  * Preserve privacy by avoiding original usernames
  * Propose alternatives to using original text messages

**Results:**
- Response selection relies more on textual content
- Addressee recognition requires capturing structural dimensions
- LLM sensitivity to prompt variations is task-dependent.

## 1 Introduction 

**Introduction**
- Multi-Party Conversations (MPCs): discussions involving more than two participants
- Typical of online platforms like Reddit or Twitter/X
- Challenging to capture content due to textual and structural information

**Designing Systems for MPC Understanding**
- Textual dimension spans multiple turns
- Need to capture structural aspects, such as who writes to whom
- **Understanding how these two components should be integrated is an open question**

**Tasks Addressed in Study**
- Response Selection: dealing with linguistic aspects
- Adressee Recognition: addressing structural and non-linguistic aspects
- Tasks can be performed on any conversational corpus, widely applicable

**Importance of Summarization and User Descriptions**
- Could make processing more efficient, replacing multiple turns with concise text representation
- Easier for data sharing and privacy-preserving
- Complies with European General Data Protection Regulation (GDPR)
- Prevents training generative models that imitate specific users

**Research Questions**
- **RQ(1)**: How do LLMs perform in zero-shot MPC classification tasks?
- **RQ(2)**: What is the model sensitivity to different prompt formulations when classifying MPCs?
- **RQ(3)**: How does structural complexity of conversation affect performance?

## 2 Related Work

**Related Work**
- Research on MPC understanding tasks: modeling entire conversation vs focusing on relations within it (Gu et al., 2022b; Ganesh et al., 2023)
- Recent studies focus on response selection (RS) and addressee recognition (AR) tasks (Ouchi and Tsuboi, 2016; Zhang et al., 2018b)
- RS: textual information
- AR: interaction information
- Both tasks can benefit from cross-information between linguistic and interaction cues

**Recent Approaches for MPC Understanding:**
- Fine-tuning transformer-based models with speaker information (Wang et al., 2020; Gu et al., 2021; Zhu et al., 2023; Gu et al., 2023)
- Using Graph Neural Networks for interaction modeling (Hu et al., 2019; Gu et al., 2022a)
- Leveraging dialogue dependency parsing (Jia et al., 2020)
- Exploring zero-shot capabilities of ChatGPT and GPT-4 in MPCs (Tan et al., 2023)

**Gap in NLP Literature:**
- Lack of focus on structural aspects of MPC understanding evaluation

**Previous Research on Conversation Structure:**
- Penzo et al. (2024): role of conversation structure in stance detection, benefits classification only with large training data
- Hua et al. (2024): addressing summarizing dynamics and trajectories of MPCs, focusing on textual content instead of interactions between speakers and conversation flow

**Related Work: Diagnostic Approach for Evaluating Models for MPC Understanding**
- Tan et al. (2023): using a generic model in zero-shot setting to address RS and AR
- Difference from this contribution: focusing on diagnostic approach, creating diagnostic datasets, and putting classification performance in relation to network metrics (degree centrality and average outgoing weight of the speaker node)

## 3 Tasks

**Tasks for Experiments**
- **Response Selection (RS)**
  - Choose text of next message given a conversation, speaker id, and set of candidate responses
  - Cast as binary classification task: select between two possible candidates
  - Example: choose between "I'm glad you like it" and "That's a nice shirt"
- **Addressee Recognition (AR)**
  - Predict addressee of next message based on conversation, speaker id, and set of candidate addressees
  - Set includes all speakers involved in conversation and a "dummy" option for random check
  - Example: select the appropriate recipient from "Mary", "Tom", or "Dummy User" in the conversation.

## 4 MPC Classification

**Conversation Representation Workflow**

**First Step**: Modelling input data for classification
- Four ways to model conversation content: Conversation Transcript, Interaction Transcript, Summary of conversation, User Description
- Replace actual conversation content with most relevant information to avoid potential misuse and bias

**Second Step**: Pipeline and Prompt Design
- Use Llama2-13b-chat for generating summaries and user descriptions
- Four steps: generate summary, create user descriptions, perform response selection, and addressee recognition
- Greedy decoding mechanism used for generating prompts

**Third Step**: Evaluating Candidate Responses (Response Selection)
- Instead of having LLM generate output, evaluate Conditional Perplexity (CPPL) of all candidates given classification prompt pc
- Select best response with lowest CPPL as output

**Comparison of Prompt Schemes**: Three distinct schemes with varying levels of verbosity to test LLM classification robustness and sensitivity. Figure 3 shows the beginning of system prompts in each version from most verbose (top) to most concise (bottom).

## 5 Diagnostic

**Diagnostic Approach for Research Questions**
* Developing a diagnostic approach to address research questions related to conversation complexity and classification performance
* Focus on interplay between interaction structure and classification performance
* Identify two metrics to capture conversation complexity: interaction graph and network metrics (degree centrality, average outgoing weight)
* Use of the unweighted undirected and weighted directed graphs for extracting information from an MPC
* Extract network metrics from each conversation's interaction graph: degree centrality, average outgoing weight
	+ Degree centrality: number of edges incident in a node (represents number of users next speaker has interacted with)
	+ Average outgoing weight: sum of weights on outgoing edges divided by the number of outgoing edges (average number of messages sent to users)
* Create diagnostic datasets using Ubuntu Internet Relay Chat corpus for testing RS and AR performance in a zero-shot setting
	+ Four subsets with conversations involving 3, 4, 5, or 6 users: Ubuntu3/4/5/6
	+ Keep only connected undirected and unweighted interaction graphs to control structural complexity fluctuations
	+ Anonymize usernames for privacy reasons.

## 6 Experiments

**Five Input Combinations for Experimenting with Prompts**
* **I. Only conversation transcript (CONV)**
* **II. Conversation transcript and interaction transcript (CONV+STRUCT)**
* **III. Interaction transcript and conversation summary (STRUCT+SUMM)**
* **IV. Interaction transcript and user descriptions (STRUCT+DESC)**
* **V. Interaction transcript, conversation summary, and user descriptions (STRUCT+SUMM+DESC)**

**For AR Task:**
* **VI. Only interaction transcript (STRUCT)**
* Relevant for RS since it doesn't include linguistic information.

**Tested Across:**
* Four diagnostic datasets.

## 7 Macro Results and Structural Evaluation

**Macro Results and Structural Evaluation**

**AR Task:**
- **Best run results**: Figure 5 shows the highest accuracy achieved among 3 prompt schemes for each Ubuntu subset, ranging from 2.7 to 10.9%. The best performance is consistently on CONV+STRUCT or STRUCT combinations.
- **Relative gap between best and average**: Table 1 reports the percentage difference in accuracy between best and average prompts. Results are more sensitive for AR than RS, especially in CONV+STRUCT combination.
- **Degree centrality (deg(u)) and average outgoing weight (wo avg(u)) analysis**: Figure 6 displays how best runs vary with deg(u) and wo avg(u). For AR, interaction transcripts improve performance when next speaker interacts with few users but perform similarly when there are more than two. Deg(u) has a strong correlation with lower accuracy scores.
- **RS Task:**
  - **Best run results**: Figure 5 shows the highest accuracy achieved among 3 prompt schemes for each Ubuntu subset, ranging from 0.2 to 1.7%. The best performance is consistently on STRUCT+SUMM or STRUCT combinations.
  - **Relative gap between best and average**: Table 1 reports the percentage difference in accuracy between best and average prompts. Sensitivity to prompt formulation is similar for both tasks, but RS shows less clear correlation with structural metrics.
  - **Degree centrality (deg(u)) and average outgoing weight (wo avg(u)) analysis**: Figure 6 shows that the best runs of interaction transcripts perform better when next speaker interacts with few users in RS, while there's no clear correlation between deg(u) or wo avg(u) values and accuracy scores. The gap among models is not consistent across different datasets.

## 8 Discussion

**Findings: Comparative Evaluation (RQ1-RQ3)**
* **Input combination performance (RQ1):**
  * AR: STRUCT and CONV+STRUCT consistently perform best, comparable results
  * RS: CONV and CONV+STRUCT outperform other combinations; inclusion of summary/user description leads to decline in performance
    * May depend on Llama2-13b-chat generating bad summaries and model difficulties using this information for classification
* **Prompt verbosity (RQ2):**
  * AR: benefits from structural information, shows greater sensitivity to prompts compared to RS
    * Verbose prompt version tends to be best option for AR; helps capture structural info crucial for task
  * RS: no consistent improvement in using more verbose prompt, already expresses necessary linguistic information
* **Structural complexity (RQ3):**
  * Macro results alone offer surface-level understanding; influenced by dataset characteristics
    * Best input combinations perform well on simple conversations but lack generalization to complex structures
    * Performance gaps between best and data minimization inputs widens with increasing degree centrality for AR
      * May indicate importance of addressing this performance gap in future research
  * RS: performance largely unaffected by network metrics; model may infer conversation flow based solely on message content.

## 9 Conclusions

**Conclusions**
- Study evaluates zero-shot performance of LLM (Llama2-13b-chat) on two tasks based on multi-party conversations: response selection and addressee recognition
- Goal is to provide an in-depth analysis of different experimental settings for the two tasks, including:
  - Three different prompt types
  - Six configurations to model conversation text and structure

**Evaluation Analysis**
- Perform evaluation on four diagnostic datasets with fixed number of users
- Compute two network metrics (degree centrality, average outgoing weight) to analyze how structural complexity interacts with classification performance
- Focus on evaluating strategies to replace original conversation text in prompts:
  - Ensures safe use of MPC corpora for data resharing
  - Prevents malicious use of MPCs for training models with fake personas

**Limitations and Future Work**
- Results not fully satisfactory yet, but aim to provide analysis on interplay between textual and structural information in MPCs.
- Merge contributions from NLP and network science community for a better understanding.

## 10 Limitations

**Limitations of Findings**
- Based on a single dataset: Ubuntu Internet Relay Chat corpus
- Choice due to lack of variety in multi-party datasets
- Conversations all have same length (15 turns)
- Other datasets considered but lacking necessary characteristics for diagnostic analysis
- Only one instruction-based LLM used in zero-shot setting
- Claiming general capabilities of model based on these results would be scientifically inaccurate.

**Ubuntu Internet Relay Chat Corpus**:
- Single dataset used for findings
- Lack of variety in multi-party datasets a limitation
- Conversations all have same length (15 turns)

**Other Datasets Considered**:
- Other datasets taken into account for experimentation
- Found to lack necessary characteristics for diagnostic analysis

**Instruction-Based LLM and Zero-Shot Setting**:
- Only one instruction-based LLM used in zero-shot setting
- Primary goal was to present novel evaluation pipeline

**Importance of Comparing Different LLMs**:
- Comparing different LLMs necessary to better prove generalization of approach.

## 11 Ethics Statement

**Ethics Statement**
* Prioritized privacy and ethical management of data in research
* Anonymized usernames by replacing with ungendered names to protect identity and reduce biases
* Explored alternative representations of conversation transcripts to enhance user privacy and minimize biases
* Future research could use summaries or user descriptions, making it nearly impossible to imitate specific users
* Experiments conducted in zero-shot setting without fine-tuning for reproducibility

**Funding**:
* Work of BL partially supported by the NextGenerationEU Horizon Europe Programme grants 101120237 - ELIAS and 101120763 - TANGO
* BL, ST also supported by PNRR project FAIR - Future AI Research (PE00000013)
* NP's activities part of the network of excellence of the European Laboratory for Learning and Intelligent Systems (ELLIS)

## A Prompt schemes and combinations

**Experimental Setup:**
- Fixed template for system prompts: Scenario Description, Input Elements, Task Definition, User Space Description, Input Format, Instruction Template, Output Template
- Consistent sections: Scenario Description and User Space Description
- Variable sections based on tasks: Task Definition, Instruction Template, Output Template
- Input combinations: CONV, STRUCT+SUMM, etc.

**Prompt Schemes:**
- Three levels of prompt verbosity (extremely precise to most implicit)
- First dimension: "prompt scheme"
  - Consists of writing all sections from scratch for tasks and combinations
  - Three schemes created based on different levels of prompt verbosity

**Creating the Final Prompt:**
- Combine input information and instruction command
  - Figure 13 (generation tasks), Figure 14 (classification tasks)
- An example: STRUCT+SUMM combination in AR task

**User Anonymization:**
- Replace original usernames with ungendered tags: [ALEX], [BENNY], [CAM], [DANA], [ELI], and [FREDDIE]
- Tag [ALEX] always assigned to the next speaker.

## B Task details and formalization

**Formalization of an MPC (Message Production and Comprehension)**
- **Given:** Conversation C = {M, U}, where M is a set of chronologically ordered messages (message mi appeared before message mj if i < j) and U is the set of users occurred in C.
- Each message mi is assigned an ordered pair (uj, uk), where uj is the speaker of mi and uk is the addressee of mi.

**Classification using CPPL (Conditional Perplexity)**
- Given a task T {RS, AR}, a classification prompt pT, and the set of candidate responses RT = {r1, ..., r m}, extract the output with minimum conditional perplexity minCPPL(ri|p), i ∈ [1, m].
- Obtain probability distribution over candidates: P(rk) = 1/CPPL(rk)P(ri) ∈ RT.
- Correlation between probabilities and network metrics leads to same conclusions as accuracy values.

**Limitations of the Combination Prompt Scheme**
| Prompt Scheme | UBUNTU3 | UBUNTU4 | UBUNTU5 | UBUNTU6 |
|---|---|---|---|---|
| Conversation (Conv) verbose | 0.613 | 0.414 | 0.352 | 0.277 |
| Medium | 0.582 | 0.409 | 0.344 | 0.283 |
| Concise | 0.595 | 0.416 | 0.298 | 0.289 |
| Conv+Struct verbose | 0.660 | 0.584 | 0.525 | 0.449 |
| Medium | 0.609 | 0.501 | 0.513 | 0.431 |
| Concise | 0.571 | 0.477 | 0.465 | 0.400 |
| Struct+Summ verbose | 0.623 | 0.517 | 0.448 | 0.397 |
| Medium | 0.644 | 0.491 | 0.465 | 0.429 |
| Concise | 0.617 | 0.441 | 0.433 | 0.374 |
| Struct+Desc verbose | 0.637 | 0.499 | 0.456 | 0.406 |
| Medium | 0.604 | 0.501 | 0.513 | 0.431 |
| Concise | 0.571 | 0.477 | 0.465 | 0.400 |
| Struct+Summ+Desc verbose | 0.628 | 0.472 | 0.429 | 0.383 |
| Medium | 0.620 | 0.455 | 0.444 | 0.374 |
| Concise | 0.618 | 0.458 | 0.590 | 0.629 |
| Struct verbose | 0.654 | 0.572 | 0.537 | 0.454 |
| Medium | 0.626 | 0.515 | 0.498 | 0.434 |
| Concise | 0.573 | 0.487 | 0.438 | 0.389 |

**Accuracies in Addressee Recognition and Response Selection Across Prompt Schemes and Input Combinations**
| Combinations | UBUNTU3 | UBUNTU4 | UBUNTU5 | UBUNTU6 |
|---|---|---|---|---|
| Conv verbose | 0.625 | 0.627 | 0.619 | 0.640 |
| Medium | 0.624 | 0.619 | 0.613 | 0.646 |
| Concise | 0.612 | 0.617 | 0.610 | 0.649 |
| Struct+Summ verbose | 0.572 | 0.570 | 0.569 | 0.626 |
| Medium | 0.575 | 0.556 | 0.573 | 0.614 |
| Concise | 0.564 | 0.572 | 0.587 | 0.623 |
| Struct+Desc verbose | 0.565 | 0.553 | 0.540 | 0.597 |
| Medium | 0.553 | 0.565 | 0.550 | 0.586 |
| Concise | 0.542 | 0.562 | 0.538 | 0.606 |
| Struct+Summ+Desc verbose | 0.576 | 0.570 | 0.573 | 0.623 |
| Medium | 0.574 | 0.570 | 0.575 | 0.614 |
| Concise | 0.573 | 0.583 | 0.585 | 0.620 |

**Prompt Schemes:**
- STRUCT+SUMM: Generate summary from user's perspective
- STRUCT+DESC: Generate description of a user's behavior
- STRUCT+SUMM+DESC: Combine both summary and description

**Average Results Across Output Templates**: No significant difference between output templates using the same combination but different templates. Maximum difference is 1.9 percentage points in AR task - Ubuntu5 for STRUCT+SUMM+DESC.

**Technical Details:**
- Single A40 GPU with 48GB Memory used
- Llama2-13b-chat only used with batch size of 1
- Copilot and ChatGPT used as coding and writing assistants respectively

**Prompt Schemes and Combinations:**
| Prompt Scheme | Ubuntu3 | Ubuntu4 | Ubuntu5 | Ubuntu6 | AR |
|---|---|---|---|---|---|---
| Accuracy (RS) | 81.2% | 79.6% | 80.1% | 80.8% | 79.4% |
| RS Error | 18.8% | 20.4% | 19.9% | 19.2% | 20.6% |
| Accuracy (RS) Average | 80.5% | - | - | - | - |
| RS Error Average | 19.7% | - | - | - | - |

**Prompt Scheme Output Templates**: Figure 9 shows results across diagnostic datasets and tasks for each prompt scheme and output template, averaging the results with the same combination but different templates. No significant difference between output templates using the same combination. Maximum difference is 1.9 percentage points in AR task - Ubuntu5 for STRUCT+SUMM+DESC.

**Technical Details**: Single A40 GPU with 48GB Memory used, Llama2-13b-chat only used with batch size of 1, Copilot and ChatGPT used as coding and writing assistants respectively.

### Experimental Setup and Methods

- Study examines effects of different output templates for generating summaries and user descriptions
- Uses **Llama2-13b-chat** model for experiments
- Hardware: Single A40 GPU with 48GB memory
- Software tools:
  - **Copilot** as coding assistant
  - **ChatGPT** as writing assistant (for style improvements only)

**Prompt Schemes**

- Three versions tested:
  - Verbose
  - Medium 
  - Concise
- Key elements in prompts:
  - Scenario description
  - Input elements
  - Task definition
  - User space description
  - Input format instructions
  - Output templates

**Tasks Evaluated**

- **Response Selection (RS)**: Generate next message in conversation
- **Addressee Recognition (AR)**: Identify addressee of next message
- **Summarizer**: Summarize conversation from perspective of next speaker
- **Descriptor**: Describe behavior/characteristics of next speaker

**Input Information**

- Conversation transcripts
- Interaction transcripts (showing speaker and addressee for each message)
- Summary of conversation
- Description of next speaker

**Output Templates**

- Two versions tested for summary/user description generation:
  - Version 1: Includes explanations for each topic/adjective
  - Version 2: Lists topics/adjectives without explanations

**Results**

- Minimal difference between output template versions (max 1.9% difference)
- Consistent results across different combinations of prompt schemes and output templates
- For simplicity, study focused on one output template for main analysis

**User IDs and Addressing**

- User IDs: [ALEX], [BENNY], [CAM], [DANA]
- [OTHER] used to indicate addressing someone not in the conversation
- Same ID consistently represents the same individual throughout conversation

**Formatting Conventions**

- Conversation transcript enclosed in [CONVERSATION] tags
- Interaction transcript enclosed in [INTERACTION] tags
- Summary enclosed in [SUMMARY] tags
- Description enclosed in [DESCRIPTION] tags
