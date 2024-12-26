# Severity Prediction in Mental Health: LLM-based Creation, Analysis, Evaluation of a Novel Multilingual Dataset 

by Konstantinos Skianis, John Pavlopoulos, A. Seza Dogruöz
https://arxiv.org/pdf/2409.17397

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related work](#2-related-work)
- [3 Proposed Methodology](#3-proposed-methodology)
- [4 Experiments](#4-experiments)
  - [4.1 Datasets Used in Study](#41-datasets-used-in-study)
  - [4.2 Settings](#42-settings)
  - [4.3 Preliminary Prompting](#43-preliminary-prompting)
  - [4.4 Experimental results](#44-experimental-results)
  - [4.5 Resources and cost of experiments](#45-resources-and-cost-of-experiments)
- [5 Discussion](#5-discussion)
- [Conclusion](#conclusion)

## Abstract
- **Large Language Models (LLMs)** are increasingly being integrated into medical fields, including mental health support systems
- There is a gap in research regarding the effectiveness of LLMs in non-English mental health support applications
- To address this problem, a novel multilingual adaptation of widely-used mental health datasets was created
  - Datasets were translated from English into six languages: Greek, Turkish, French, Portuguese, German, and Finnish
  - Enables comprehensive evaluation of LLM performance in detecting mental health conditions and assessing their severity across multiple languages
- Experiments with GPT and Llama observed considerable variability in performance across languages
  - Despite being evaluated on the same translated dataset
  - Underscores the complexities inherent in multilingual mental health support, where language-specific nuances and mental health data coverage can effect accuracy of the models
- Comprehensive error analysis emphasizes risks of relying exclusively on large language models (LLMs) in medical settings (e.g., potential for misdiagnoses)
- Proposed approach offers significant cost savings for multilingual tasks, presenting a major advantage for broad-scale implementation.

## 1 Introduction

**Large Language Models (LLMs) in Healthcare:**
* LLMs demonstrate impressive ability to generate human-like text and perform complex tasks
* Potential application: Assist medical professionals with mental health diagnosis and treatment
* Mental health disorders are widespread, presenting public health challenges
* Traditional diagnostic methods are time-consuming and labor intensive
* LLMs can process vast amounts of textual data to detect linguistic indicators related to mental health issues
* Research on applying LLMs in mental health domain is mostly English-based, with a need for exploration in other languages

**Research Questions:**
1. How accurately can LLMs predict severity of mental health conditions using English language?
2. When English datasets are translated into other languages using LLMs, does the model maintain similar accuracy?

**Contributions:**
1. Creation of a novel multilingual dataset covering English and six other languages (Turkish, French, Portuguese, German, Greek, Finnish) derived from user-generated social media content
2. Evaluation of effectiveness of LLM in predicting mental health condition severity across different languages
3. Analysis of model performance in each language with insights into linguistic diversity's effect on outcomes
4. First study to utilize LLMs for creating and evaluating multilingual adaptations of mental health NLP datasets, specifically designed to detect mental health condition severity
5. Emphasizes the need for inclusive and adaptable AI tools and datasets in mental health care, accounting for cultural and linguistic diversity, especially in non-English contexts with cost-efficient methods.

## 2 Related work

**Related Work on NLP for Mental Health**
- Early investigations focused on identifying mental health symptoms from textual data using machine learning methods
- Analyzing social media content, online health forum discussions, etc. to find indicators of conditions like depression, anxiety
- Primarily relied on traditional machine learning algorithms with handcrafted features which lacked sophistication compared to current LLMs
- Multilingual Capabilities of LLMs
  - XLM-R model demonstrated strong cross-lingual performance in multiple languages
  - LLMs excel at processing and generating text in various languages, including low-resource ones
  - Challenges: language nuances tied to cultural frameworks, low-resource languages with limited data availability
- Addressing Challenges
  * Developing more advanced culturally aware algorithms
  * Incorporating datasets representative of linguistic and cultural diversity

**LLMs for Mental Health Applications**
- Detect mental health symptoms with greater precision due to advanced understanding of context and semantics
- Notable examples: BioBERT, Clinical-BERT, MentalBERT, DisorBERT, Suicidal-BERT, MentalLLM, and MentalLlama
- Advantages: analyzing vast amounts of textual data from social media platforms to identify linguistic patterns associated with mental health conditions
- Disadvantages: potential privacy concerns regarding personal data usage
- Social Network Datasets for Mental Health Research
  * Recent availability of large datasets focusing on mental health, e.g., Reddit and Twitter posts related to depression, PTSD, schizophrenia, eating disorders
  * Fine-tuning several models using smaller public annotated mental health datasets to facilitate labeling for new datasets
  * Limitations: only English language data available which hinders further research in multilingual contexts.

## 3 Proposed Methodology
**Methodology for Leveraging LLMs and Social Media Posts**
- Comprehensive methodology proposed for translating social media posts into multiple languages using LLMs
- Translations fed into prompt asking LLM to predict mental health condition severity levels
- Predicted classes compared with ground truth labels, evaluation metrics computed
- Aim is to evaluate LLMs on multilingual detection of mental health conditions in under-resourced languages

**Approach Overview:**
1. Use an LLM for translating social media posts into proposed languages
2. Feed translated texts into a prompt asking for mental health condition predictions
3. Compare predicted classes with ground truth labels and compute evaluation metrics
4. Evaluate LLMs on multilingual detection of mental health conditions in under-resourced languages.

## 4 Experiments
### 4.1 Datasets Used in Study

**DEP-SEVERITY**:
- Introduced by Naseem et al. (2022)
- Comprises posts from Reddit on depression levels
- Includes: minimal, mild, moderate, and severe depression
- Highly imbalanced dataset with majority in minimal severity

**SUI-TWI**:
- Contains texts from Twitter (English)
- 1,300 suicidal ideation tweets gathered
- Labeled manually into three classes: not suicidal, suicidal ideation, potential suicide attempt
- Balanced dataset in terms of frequency of occurrence

**Additional Information**:
- Multiple-class datasets used to increase difficulty for the Language Model (LLM)
- Binary classification tasks would be simpler to solve.

### 4.2 Settings

**Settings**
- Use of GPT3.5-turbo, GPT-40-mini via OpenAI API, Llama3.1 for translation and prediction
- Recent models like MentaLLaMA (Yang et al., 2024) not suitable due to English focus
- Temperature parameter set to 0 for consistent results
- Experimentation with no examples in prompt (0-shot) and one example per class (1-shot)

**Task Approach**
- Classification problem: compare predicted classes to Dataset Reference Categories
- Depression Severity Dataset: DEP-SEVERITY by Naseem et al. (2022)
  * Depression: 4 classes, 3553 instances
  * Minimum: 2587 samples
  * Mild: 290 samples
  * Moderate: 394 samples
  * Severe: 282 samples
- Suicide Ideation Dataset: SUI-TWI by Mirza Ibtihaj et al. (2020)
  * Suicide: 3 classes, 1300 instances
  * Not Suicidal: 515 samples
  * Suicidal Ideation: 455 samples
  * Potential Suicide Attempt: 330 samples
- Report Precision (Pr), Recall (Rec), and F1 for experiments

**Language Families**
- Seven languages tested: Turkish (Altaic), French, Portuguese (Romance), English, German (Germanic), Finnish (Uralic), Greek (Hellenic)
- Table 2 shows the prediction prompts used to determine severity levels.

### 4.3 Preliminary Prompting

**Preliminary Prompting**
- Inquired about LLM's approach to categorizing social media posts into depression severity levels
- Identified specific language cues for each level:
    - **Level 1 (Minimal)**: Infrequent or minor expressions of sadness or frustration
    - **Level 2 (Mild)**: More consistent negative emotions, behavioral shifts
    - **Level 3 (Moderate)**: Substantial interference with daily life, emotional distress
    - **Level 4 (Severe)**: Extreme emotional distress, risk of self-harm or complete social disengagement

**Detecting Depression Indicators in Social Media Posts**
1. **Not Suicidal**:
   - General Tone: Neutral, positive, or typical negative expressions not related to self-harm
   - Content: Discussions about daily activities, hobbies, work, social interactions
   - Language: Absence of words/phrases related to self-harm, death, hopelessness, helplessness
2. **Indicative of Suicidal Ideation**:
   - Expression of Hopelessness: "I can’t see any way out", "Things will never get better"
   - Self-Hatred or Worthlessness: "I’m a burden", "Everyone would be better off without me"
3. **Indicative of a Potential Suicide Attempt**:
   - Mentions of Death or Dying: "I wish I were dead", "I want to end it all"
   - Social Withdrawal: "I don’t want to see anyone", "I prefer to be alone all the time"
   - Detailed Plans or Intentions: "I have pills ready, "I’m going to jump off a bridge", "I’ve written my goodbye letter"
   - Goodbye Messages: "Goodbye everyone, it’s been a pleasure", "I’m sorry for everything, I won’t be around much longer"
   - Previous Attempts: "I’ve tried before and failed, but not this time"
   - Lethal Means: "I bought a gun", "I have enough pills", "I found a rope"

### 4.4 Experimental results

**Experimental Results: Suicide Ideation Detection and Depression Severity Classification**

**Source Language Performance (GPT-3.5): DEP-SEVERITY**
- Best F1 score for lowest and highest severities, 0.48 and 0.26 respectively
- More challenging task compared to SUI-TWI with F1=0.17
- Improvement in performance using GPT-4 mini: F1 increases to 0.27 (lowest) and 0.59 (highest)

**Target Languages Performance (GPT-3.5): DEP-SEVERITY vs. English**
- Lower or equal F1 scores compared to source language, but better than 0-shot learning
- More examples provided could increase performance and cost

**Source Language Performance (Llama-3.1): DEP-SEVERITY and SUI-TWI**
- Better performance in English compared to GPT-3.5 (both 0- and 1-shot)
- High variability in SUI-TWI, French being the worst and Portuguese the best
- No significant difference in DEP-SEVERITY

**Further Analysis: Examples of Misclassifications**
- First example: English text "attempted suicide" vs. Greek text "suicidal ideation"; LLM interpreting context differently due to language nuances or training data
- Second example: Both original post in English and translated post in Greek are correct, but different predictions (1 for both), possibly due to the same reasons as the first example

**Study Findings: GPT-3.5 and Llama3.1 Performance on DEP-SEVERITY and SUI-TWI**

**DEP-SEVERITY**:
- English, Portuguese, German perform equally well as in English
- Turkish (-6), Portuguese (-1), Greek (-1), Finnish (-2) see performance drops
- French (+3) and German (+7) show better performance than English
- Best F1 found for Greek (0.66), worst for German and Finnish (-4)
- Lowest severity class has wide range from 0.15 to 0.25, whiskers are also wide
- Mild severity is mishandled across languages
- Model achieves relatively high score for moderate and severe classes

**SUI-TWI**:
- Performance drops in all target languages, most notable drop for Turkish (-17)
- Limited resources in mental health domain explain poor performance in Turkish
- Best performance achieved for "non-suicidal" category
- Slightly worse performance for other two classes, similarly distributed across languages

**Comparison with 1-shot Learning**:
- Performance drops significantly for English in both DEP-SEVERITY and SUI-TWI, remains low
- Performance is better for target language compared to English in both tasks

**Table 5: GPT-4o-mini with 0-shot learning on DEP-SEVERITY**:
- Measuring Precision, Recall, and F1 per class per language
- Last row shows macro averages
- Best F1 per class is shown in bold, best average F1 across languages is underlined

**Table 6: GPT-4o-mini with 0-shot learning on SUI-TWI**:
- Measuring Precision, Recall, and F1 per class per language
- Last row shows macro averages
- Best F1 per class is shown in bold, best average F1 across languages is underlined

**Table 7: Llama3.1 with 0-shot learning on DEP-SEVERITY**:
- Measuring Precision, Recall, and F1 per class per language
- Last row shows macro averages
- Best F1 per class is shown in bold, best average F1 across languages is underlined

**Table 8: Llama3.1 with 0-shot learning on SUI-TWI**:
- Measuring Precision, Recall, and F1 per class per language
- Last row shows macro averages
- Best F1 per class is shown in bold, best average F1 across languages is underlined

### 4.5 Resources and cost of experiments

**Cost of GPT3.5 Experiments**
- Total cost less than $30 via API
- Minimal resources required without expensive infrastructure or fine-tuning
- Example: MentaLLaMA-chat-7B (27GB model) - prohibitive with limited resources
  * Outputted unmeaningful results using the quantized version instead

**Methodology for Utilizing Resources**
- Potentially promising for extending medical data sets in English into other languages

## 5 Discussion

**Discussion on LLMs in Mental Health Care**
- Increasing need for research on pitfalls of using LLMs in mental health care due to:
  - Variation in performance across languages, diseases, and patient populations
  - Risk of relying solely on LLMs in healthcare setting
- Mental health diagnosis should not be left to automatic systems or replace human professionals (Stade et al., 2024)

**Findings from Study**
- Inconsistent performance of LLM across languages, despite same dataset translation
- Performance gains with stronger models and language-specific targeting
- Turkish yields worst performance in both tasks due to limited mental health resources

## Conclusion
- Presented novel multilingual dataset for evaluating LLM's ability to predict mental health condition severity
  - Used GPT3.5 for translation, but limitations exist due to loss of information and cultural differences

**Next Steps**
- Apply methodology to broader range of languages including low resource ones
- Add datasets for more mental health tasks and from different social media platforms

**Limitations**
- Creation of multilingual dataset requires significant resources
  - Challenges in evaluating LLM performance
    * Difficulty detecting labels within LLM's output
    * Defaulting to minimum label if no label detected
- Importance of handling datasets with care, especially in sensitive areas like mental health care.

