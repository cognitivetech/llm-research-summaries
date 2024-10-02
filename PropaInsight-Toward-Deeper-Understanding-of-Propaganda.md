# PropaInsight: Toward Deeper Understanding of Propaganda in Terms of Techniques, Appeals, and Intent

by Jiateng Liu, Lin Ai, Zizhou Liu, Payam Karisani, Zheng Hui, May Fung, Preslav Nakov, Julia Hirschberg, Heng Ji1
https://arxiv.org/abs/2409.18997

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 PropaInsight: A Propaganda Analysis Framework](#2-propainsight-a-propaganda-analysis-framework)
- [3 PropaGaze: A Dataset for Systematically Analyzing Propaganda](#3-propagaze-a-dataset-for-systematically-analyzing-propaganda)
- [4 Experiments](#4-experiments)
  - [4.2 How Do Off-the-Shelf LLMs Perform on Propaganda Analysis Tasks?](#42-how-do-off-the-shelf-llms-perform-on-propaganda-analysis-tasks)
  - [4.3 How Much Does PropaGaze Enhance Model Performance?](#43-how-much-does-propagaze-enhance-model-performance)
  - [4.4 Is Propaganda Analysis Transferable Across Domains?](#44-is-propaganda-analysis-transferable-across-domains)
- [5 Discussion](#5-discussion)
- [6 Related Work](#6-related-work)
- [7 Conclusion and Future Work](#7-conclusion-and-future-work)
- [Appendix A Details of a Propaganda Frame](#appendix-a-details-of-a-propaganda-frame)
- [Appendix B Data Generation Prompt Templates](#appendix-b-data-generation-prompt-templates)
- [Appendix C Templates and Prompts](#appendix-c-templates-and-prompts)
- [Appendix D Details of the PropaGaze Dataset](#appendix-d-details-of-the-propagaze-dataset)
- [Appendix E Experimental Details](#appendix-e-experimental-details)
- [Appendix F Case Study: Bottleneck of Propaganda Analysis](#appendix-f-case-study-bottleneck-of-propaganda-analysis)

## Abstract
**Introduction**:
- Propaganda plays a critical role in shaping public opinion and fueling disinformation
- Existing research focuses on identifying propaganda techniques but lacks understanding of broader motives and impacts

**PropaInsight Framework**:
- Systematically dissects propaganda into:
  - Techniques
  - Arousal Appeals
  - Underlying Intent
- Offers a more granular understanding of how propaganda operates in different contexts

**PropaGaze Dataset**:
- Combines human-annotated data with synthetic data generated through a meticulously designed pipeline
- Experiments show that off-the-shelf language models struggle with propaganda analysis, but training with PropaGaze significantly improves performance
- Fine-tuned Llama-7B-Chat achieves:
  - 203.4% higher text span IoU in technique identification
  - 66.2% higher BertScore in appeal analysis compared to 1-shot GPT-4-Turbo
- PropaGaze complements limited human-annotated data in data-sparse and cross-domain scenarios, showing its potential for comprehensive and generalizable propaganda analysis

## 1 Introduction

**Introduction to Propaganda:**
- Propaganda techniques: deliberate dissemination of information to shape public opinion (Stanley, 2015)
- Components of disinformation (Da San Martino et al., 2020)
- Importance for maintaining integrity in public discourse and informed decision making

**Challenges with Current Research:**
- Focus on identifying techniques without understanding motives or broader impact
- Dependence on expert annotations leading to small datasets and limited applicability
- Need for addressing these challenges: PropaInsight framework and synthetic data (PropaGaze)

**Proposed Contributions:**
1. **PropaInsight Framework**: analyzes three key elements of propaganda - techniques, arousal appeals, underlying intent
2. **PropaGaze Dataset**: human-annotated news sub-dataset + synthetic sub-datasets on Russia-Ukraine conflict and political domain
3. Motivations: develop comprehensive framework and explore use of synthetic data for propaganda analysis
4. Contributions: propose PropaInsight, introduce PropaGaze dataset, demonstrate enhancing LLMs' ability to analyze propaganda within the PropaInsight framework.

## 2 PropaInsight: A Propaganda Analysis Framework

**PropaInsight Framework for Propaganda Analysis**

**Overview:**
- New framework: PropaInsight
- Focuses on underlying purposes of propaganda, not just techniques
- Based on research by Nelson (1997), Jowett and O’Donnell (2018), Ellul (2021)

**Key Elements:**
1. **Propaganda Techniques**: Systematic strategies used to create persuasive content
   - Variable techniques, but labeled for consistency: loaded language etc.
   - List of 16 propaganda techniques from Da San Martino et al. (2019) provided in Appendix C
2. **Arousal Appeals**: Directly influence emotions, opinions, and actions
   - Evoke strong emotions (hate, fear) or shape perceptions
   - Three templates identify emotions evoked and aspects guided/distracted from
3. **Underlying Intent**: Ideological or political goals of the author
   - Complex intentions not easily categorized with predefined labels
   - Framed as free-text generation task for flexibility and context-specificity

**Propaganda Analysis Task:**
- Identify propaganda techniques used and their locations in an article
- Describe emotions/feelings evoked using templates
- Generate a free-form explanation of the underlying intent

**Components of the Analysis:**
1. **Propaganda Technique Identification**
   - Detect spans where techniques are applied
   - Identify specific technique(s) for each span
2. **Appeal Analysis**
   - Describe emotions and feelings evoked using templates
3. **Intent Analysis**
   - Generate a free-form explanation of the article’s underlying intent.

## 3 PropaGaze: A Dataset for Systematically Analyzing Propaganda

**PropaGaze Dataset for Propaganda Analysis**

**Dataset Design:**
- Consists of three sub-datasets: PTC-Gaze, RUWA-Gaze, and Politifact-Gaze
- Designed for comprehensive propaganda analysis

**PTC-Gaze:**
- Human-annotated dataset based on existing PTC dataset
- Annotators label appeals and intent independently
  - Appeals: evaluate feelings evoked by propaganda-containing sentences
  - Intent: infer single free-form sentence for underlying article intent
- Contains 79 articles with an average of 12.77 propaganda techniques per article

**RUWA-Gaze and Politifact-Gaze:**
- Synthetic datasets using LLMs (LLaMA, GPT) to generate data
- Created for training large, generalizable models and increasing cross-domain applicability
- Constructed based on real-world news articles from Russia-Ukraine War or Politics domain
  - RUWA-Gaze: focused on Russia-Ukraine War, 497 articles
  - Politifact-Gaze: constructed using PolitiFact partition of FakeNewsNet dataset, 593 articles
- Use GPT-3.5 to summarize human-written news articles and extract key events and objective facts
- Randomly select one entity's perspective and set an intent for revising the article
- Insert randomly chosen propaganda techniques into objective summary using GPT-4 as intermediary author
  * Ensure alignment with established intent through self-analysis by the model
- Human verification of data quality.

## 4 Experiments

**Experiments: Propaganda Analysis using LLMs**

**LLMs for Propaganda Analysis**:
- Have strong prior knowledge and advanced context understanding, making them ideal for synthesizing propaganda-rich datasets
- Can be effective in analyzing propaganda

**Research Questions**:
1. How do off-the-shelf LLMs perform on propaganda analysis?
2. How does the PropaGaze dataset improve performance when used for training or fine-tuning?
3. Is propaganda analysis transferable across domains?

**Experimental Setup: Sub-Tasks and Metrics**:
1. **Propaganda Techniques Identification**:
   - Intersection over Union (IoU) measures overlap between identified text spans and actual propaganda spans
   - F1 scores evaluate propaganda technique classification
2. **Appeals Analysis**:
   - BertScore measures semantic similarity of generated responses
3. **Intent Analysis**:
   - Also evaluated using BertScore

**Models Used**:
1. **GPT-4-Turbo**: One of the top-performing OpenAI models for various tasks, used in zero-shot and few-shot prompting settings across all sub-tasks
2. **Llama-7B-Chat**: A popular open-source LLM, fine-tuned for propaganda analysis using specific prompts in Appendix C
3. **Multi-Granularity Neural Networks (MGNN)**: A benchmark method for propaganda techniques identification sub-task, trained from scratch on this task as it is not designed for text generation and cannot be applied to other sub-tasks

**Training Settings**:
- Data-rich vs. data-sparse environments are evaluated to measure impact of larger synthetic datasets on model performance
- **PropaGaze sub-datasets**: Split into training and testing sets using a 70:30 ratio
- **PTC-Gaze**: Represents a data-sparse condition with only 79 articles
- **RUWA-Gaze and Politifact-Gaze**: Contain over 1,000 articles each, simulating data-rich conditions when used in full training sets

### 4.2 How Do Off-the-Shelf LLMs Perform on Propaganda Analysis Tasks?

**Table 1: Model Performance on Propaganda Technique Identification Sub-Task**
* Model performances under different training data settings
* GPT-4-Turbo0s and GPT-4-Turbo1s with Span Avg. IoU, Macro F1
* MGNN, Llama-7B-Chatft with Span Avg. IoU, Techniques Macro F1
* No Training Data: GPT-4-Turbo0s (0.073), GPT-4-Turbo1s (0.132)
* Data Sparse Training: MGNN (0.089), Llama-7B-Chatft (0.230)
* Data Rich Training: MGNN (0.545), Llama-7B-Chatft (0.506)

**Table 2: Model Performance on Appeals and Intents Sub-Task under Different Training Data Settings**
* Reported performance of trained MGNN model and LLMs
* GPT-4-Turbo0s, GPT-4-Turbo1s with BertScore, Macro F1 for appeals
* Llama-7B-Chatft (Data Sparse) and (Data Rich) with BertScore, Intents
* Data Sparse Training: Llama-7B-Chatft (0.313), MGNN (0.851)
* Data Rich Training: Llama-7B-Chatft (0.612), MGNN (0.861)
* Zero-shot LLMs struggle with propaganda analysis
* Improvements from few-shot prompting in identifying techniques and appeals

**Table 3: Model Performance on Appeal Analysis Sub-Task under Different Training Data Settings**
* Reported performance of zero-shot and fine-tuned LLMs for appeal analysis
* GPT-4-Turbo0s with BertScore
* Minimal improvements from few-shot prompting in intent analysis.

### 4.3 How Much Does PropaGaze Enhance Model Performance?

**PropaGaze and Model Performance Enhancement**

**Overview**: PropaGaze significantly improves propaganda analysis performance, particularly in identifying propaganda techniques, under both data-sparse and data-rich training conditions.

**Data-Sparse Setting**:
- Fine-tuned LLaMA-7B-Chat outperforms one-shot GPT-4-Turbo: 65.8% higher text span IoU and 33.7% higher technique identification F1 score.

**Data-Rich Setting**:
- Performance increases further with fine-tuning LLaMA-7B-Chat: 90.9% higher text span IoU, 125.1% higher F1 score compared to data-sparse results.

**Appeals and Intent Analysis Improvements**:
- For appeals sub-task: Data-rich fine-tuning leads to an average 70.1% increase in BertScore.
- For intent analysis: A smaller 8.5% gain compared to data-sparse training due to already high baseline performance.

**Comparison with Baseline Benchmark MGNN**:
- In data-sparse setting, LLaMA-7B-Chat substantially outperforms trained MGNN on RUWA-Gaze (158.43% higher IoU) and PolitiFact-Gaze (58.1% higher IoU).
- However, in data-rich scenarios, MGNN surpasses LLaMA-7B-Chat due to overfitting and larger models generalizing better with limited training data.

**Conclusion**:
- LLMs are more suited for the task with limited training data, while smaller dedicated models like MGNN could benefit more from synthetic sub-datasets in data-rich environments.

### 4.4 Is Propaganda Analysis Transferable Across Domains?

**Cross-Domain Transferability of Propaganda Analysis**

**Background:**
- Propaganda analysis spans various domains: military/war, politics, economics, science, environmental issues
- Interest in determining if general patterns are transferable between domains
- Scarcity of high-quality human-annotated data prompts investigation into using data from other domains to improve analysis in a target domain

**Datasets:**
- RUWA-Gaze: military and war
- Politifact-Gaze: politics
- PTC-Gaze: general news

**Cross-Domain Transferability Study:**
1. **Data-sparse scenarios**: models benefit substantially from incorporating cross-domain data
   - When evaluated on RUWA-Gaze, LLaMA-7B-Chat with Politifact-Gaze data achieves highest text span IoU (0.271) and MGNN with Politifact-Gaze data reaches the highest technique F1 score (0.281)
   - Consistent pattern across other sub-datasets and appeal analysis as well
2. **Data-rich scenarios**: benefit of cross-domain training diminishes
   - Models trained on additional Politifact-Gaze data underperform those trained solely on in-domain data when evaluated on RUWA-Gaze
   - Similar results for Politifact-Gaze when evaluated with RUWA-Gaze data, performance gains are small
3. **Impact of data quality vs. quantity**
   - When there is sufficient training data, the quality of the data has a greater impact on performance than its quantity
4. **Training on multiple datasets improves performance in PTC-Gaze:**
   - Training on both RUWA-Gaze and Politifact-Gaze improves performance across all sub-tasks for PTC-Gaze, partly due to the data-sparse nature of PTC-Gaze but also because synthetic data complements limited human-annotated data.

## 5 Discussion

**Discussion**
- Discrepancy between synthetic sub-datasets and human-annotated one: **PTC-Gaze vs RUWA-Gaze & Politifact-Gaze**
  * Average number of propaganda techniques per article in PTC-Gaze: **12.77**, nearly **3.7x** higher than other sub-datasets
  * Reason for discrepancy: Synthetic data generation injects three propaganda techniques per article with some reuse by GPT-4-Turbo
  * Longer average length of PTC-Gaze articles compared to others
  * Unclear if injected techniques are the only ones in those articles
  * Inherent difference between writing styles: challenge for synthetic datasets

**Further Challenges of Propaganda Analysis**
- Accurately pinpointing propaganda occurrence is difficult
- Misclassification of non-propagandistic sentences as propagandistic leads to high false positives rate
  * Issue with LLMs and MGNN models themselves
  * Training methodologies and underlying algorithms also contributing factors
- Need for improvements in both model development and training approaches to better distinguish propagandistic content from neutral text.

## 6 Related Work

**Propaganda Detection:**
* Long focus in Natural Language Processing (NLP)
* Identifying propaganda usage and techniques
* Learning-based approaches improve performance: Da San Martino et al. (2019), Yoosuf and Yang (2019), Li et al. (2019), Vorakitphan et al. (2022), Yu et al. (2021)
* Interpretability: Yu et al. (2021)
* Recent efforts apply LLMs: Sprenkamp et al. (2023), Jones (2024)
* Focus on identifying techniques, not appeals and intent
* Research on propaganda campaigns during Russo-Ukrainian conflict: Chen and Ferrara (2023), Fung and Ji (2022), Golovchenko (2022), Geissler et al. (2023), Patrona (2022)
* Lack of generalizable framework for propaganda analysis

**Propaganda Generation:**
* Sparse research compared to propaganda detection
* Generating propaganda: Zellers et al. (2019), Huang et al. (2023), Goldstein et al. (2024)
* Our data generation pipeline allows for broader range of techniques, granular analysis of appeals behind use

**User Intent Detection:**
* Previous methods concentrated on user queries in human-machine dialogue systems
* Facilitated construction of robust search engines and virtual assistants: Brenes et al. (2009), Liu et al. (2019), Weld et al. (2022)
* Detecting user query intent is relatively superficial compared to propaganda intent
* Propaganda intent can be highly concealed and hard to recognize: Jowett and O’Donnell (2012).

## 7 Conclusion and Future Work

**Conclusion and Future Work:**
- Proposed a comprehensive approach to propaganda analysis beyond technique identification
- Introduced PropaInsight, a framework for granular propaganda analysis: techniques, arousal appeals, intent
- Presented PropaGaze dataset: human-annotated and synthetic sub-datasets
- Experiments showed models fine-tuned on PropaGaze outperformed GPT-4 Turbo by a margin
- PropaGaze beneficial in data-sparse and cross-domain scenarios, enhancing tasks like disinformation detection, sentiment analysis, narrative framing, media bias analysis, social media monitoring
- Plans for future work: expanding PropaGaze into diverse domains and genres, exploring implications of PropaInsight on downstream applications

**Limitations:**
1. Small dataset size and uncertainty about generalization across a broad range of domains
2. Domain gap in real-world scenarios where propaganda thrives not fully captured
3. Personalized appeals and reader engagement effects not accounted for in study
4. Synthetic sub-datasets have high precision but not evaluated for recall, further evaluation needed
5. Proposed future directions: creating a larger dataset, investigating cross-domain data, considering personalized responses, enhancing LLM understanding, incorporating deeper understanding of propaganda techniques into situation report generation.

## Appendix A Details of a Propaganda Frame

**Propaganda Technique Set**
- **Loaded Language**: Calling/labeling, Obfuscation, Doubt, Slogans, Black-and-white fallacy, Appeal to authority, Whataboutism, Smears, Glittering Generalities
- **Appeals**: Raise emotion about [something related], Realize that [something related], Ignore that [something related]
- **Ultimate Intent Generation**: Any applicable explanation for intent
- **Note**:
  - This set can be extended with other techniques as needed.
  - The optimal template design may require further human assessment and prompt engineering, which is left for future work.

## Appendix B Data Generation Prompt Templates

**Data Generation Prompt Templates**

**Step 1: News Summarization System**
- You are a helpful assistant
- News: {news}
- Provide an objective summary of the news article, ensuring to present information in neutral terms
- Avoid using language or phrases that show bias towards either party involved

**Step 2: Intent Creation System**
- You are a helpful assistant
- Article: {article}
- Identify all parties mentioned in the article
- Select one party randomly and create an intent narrative to potentially reshape the article

**Step 3: Techniques Insertion System**
- You are a skilled journalist, proficient in composing short brief news pieces
- Article: {article}
- Rewrite the article into a short news piece to {intent}
- Convey the intent narrative effectively by applying the following rhetorical tactics, once or more as needed
- The revision must be concise, with a clear emphasis on using these tactics to communicate the intended message
- Avoid generating non-factual information

**Step 4: Appeals Generation System**
- You are a helpful assistant that identifies how the writer of a news article wants the readers of the article to feel after reading some sentences
- In this task, the input will be a news article, then some sentence in the article will be provided and you need to identify how the specific sentence raises appeals among the readers, the propaganda tactics used in these sentences will also be provided as a hint
- For each sentence, you only need to output a sentence describing the feelings in one of the following two templates:
  - Make the readers feel [Positive & Negative Emotions] about [Something that is related]
  - Make the readers realize/Ignore [Something that is related]
- Here's an example indicating the input and output format:
  - Input: News article: {article}
    - Sentence: {first sentence}
    - Tactic: {the tactic that is used in the sentence}
    - Sentence: {second sentence}
    - Tactic: {the tactic that is used in the sentence}
  - Output: [1] Your response for the first sentence: Make the readers feel [Positive & Negative Emotions] about [Something that is related]
  - [2] Your response for the second sentence: Make the readers realize/Ignore [Something that is related]

## Appendix C Templates and Prompts

**Template for Composing Descriptive Sentences**
- Use this template to compose predicted elements into a descriptive sentence as final output for propaganda analysis task
- Example: "This article uses propaganda to further its author’s ultimate intent of {The ultimate intent that is predicted by the model}. Specifically, the author uses {The first identified propaganda technique} in the sentence: “{The first sentence that is identified to use propaganda}” to make the readers {The first appeal that is raised among the readers}. The author also uses {The second identified propaganda technique} in the sentence: “{The second sentence that is identified to use propaganda}” to make the readers {The second appeal that is raised among the readers}"

**Prompt Template for Language Models**
- Encourages language models to correctly predict elements within a propaganda frame and enables simple parsing of results
- Includes news article, major intent detection, list of potential tactics, and instructions for outputting text spans and corresponding appeals
- Example: "Given the news article above, you should detect the major intent of the article. The intent is conveyed by using certain tactics and raise appeals in some text spans. You are also going to output all the text spans within the original article that contain tactics and appeals related to this intent, along with the corresponding tactics and appeals. The tactics used may include: loaded language, flag waving, slogans, repetition, straw man, red herring, whataboutism, obfuscation, causal oversimplification, false dilemma, thought terminating cliche, appeal to authority, bandwagon, glittering generalities, name calling, doubt, smears, reducito ad hitlerum. You should firstly output the ultimate intent, then sequentially output all the text spans within the original article that contain tactics and appeals related to this intent and the corresponding tactics and appeals. You should only output one appeal for each text span."

## Appendix D Details of the PropaGaze Dataset

**PropaGaze Dataset: RUWA-Gaze Subset (Ukraine-Russia War)**
- Focuses on Ukraine-Russia War news articles from Khairova et al.'s dataset
- After human verification, keep 497 articles with average propaganda usage of 3.46 times
- Example article has three instances of propaganda:
  - **Unity and transparency** call for evidence to maintain peace and prevent further escalation
    - Flag-waving Appeal: Make readers feel positive about unity and transparency
  - **Whataboutism**: Focus on Ukraine's need to present evidence, ignoring other issues
    - Bandwagon Appeal: Emphasize opportunity for Ukraine to set a strong example of transparency
- Intent: Urges Ukraine to provide concrete evidence to support their claims against Russia.

**Politifact-Gaze Subset (Political Status across Countries)**
- Also constructed with partially controlled generation pipeline based on PolitiFact partition from Shu et al.'s dataset
- Keep 593 generated articles, each with average propaganda usage of 3.47 times
- Example article has four instances of propaganda:
  - **Legal experts** confirm charges against Davies for falsifying claims against Moore
    - Appeal to Authority: Make readers feel anxious about potential legal consequences for Davies
  - Yearbook inscription debunked as a forgery, exposing Davies as a fabricator
    - Loaded Language: Make readers feel disgusted by the forgery and false accusations against Moore
  - Individual manipulated public opinion for political gain, causing serious harm to Moore's reputation
    - Causal Oversimplification and Appeal to Authority: Make readers angry at the individual responsible and sympathetic towards Moore's defense.
- Intent: Highlights falsified claims against Roy Moore and calls for accountability from those involved in spreading false information.

### The PTC-Gaze Subset

**Construction**:
- Based on propaganda techniques corpus from Martino et al. (2020a)
- Propaganda articles from real-world news annotated by human annotators
- Human annotators also provided additional information: appeals, intent

**Collection and Annotation Burden**:
- Synthetic annotations generated by GPT-4 models
- Human annotators refined the synthetic annotations
- Collected 79 long articles with an average of 12.77 times propaganda usage

**Comparison to Synthetic Data**:
- Real-world corpus has more instances of propaganda usage
- Attributed to a domain gap between synthetic and real articles

**Example Annotated Article**:
- Ex-Sailor Pardoned by Trump Sues Obama and Comey
- Technique annotations: **Whataboutism**, **Causal Oversimplification**
- Appeal annotations: **Make the readers feel indignant, unjust, sympathetic**
- Intent: Inform public about Kristian Saucier's plans to sue Obama administration officials

**Annotation Quality**:
- Used Label Studio for interface design
- Annotated by professional annotators from Kitware.Inc
- Utilized GPT-4 generated annotations in 59.8% and 75.1% of cases, demonstrating high quality.

## Appendix E Experimental Details

**Experimental Details**

**Llama-Chat-7B Model**:
- Fine-tuned using LMFlow framework Diao et al. (2023)
- **Training**:
  - Used 4 A100 GPUs for training
  - Learning rate: 0.00002
  - Batch size: 4
  - Tuned model for 3 epochs with training data
- **Inference**:
  - Inference temperature: 1.0

**GPT-4-turbo Model**:
- Default temperature for generation

**MGNN Model Tuning**:
- Batch size: 16 (due to smaller memory space)
- Learning rate: 0.00003
- Tuned model for 20 epochs

## Appendix F Case Study: Bottleneck of Propaganda Analysis

**Case Study: Propaganda Analysis Bottleneck**

**Background:**
- Identified bottleneck of propaganda analysis as identifying correct propagandistic sentences (§ 5.2)
- Explanation through case study on LLMs doing propaganda analysis

**Input Example Data**: Fox News host Shepard Smith's criticism of Donald Trump Jr.'s meeting with a Russian lawyer

**Analysis:**
- **Technique**: Loaded language, appeal to authority, repetition, false dilemma, doubt, red herring, glittering generalities
- **Appeals and Intent:** Criticize Trump administration's lack of transparency and integrity; highlight Smith's role as defender of truth and journalistic integrity.

**Ground Truth Answers**:
1. Technique: loaded language
   - Appeal: Make readers realize serious implications of statements
   - Intent: Highlight Fox News host Shepard Smith's criticism of Trump Jr.'s misleading explanations about meeting with Russian lawyer
2. Techniques: appeal to authority, repetition
   - Appeals: Make readers trust in necessity of transparency for integrity and realize about persistent dishonesty in the Trump administration
3. Technique: false dilemma
   - Appeal: Make readers feel guilty for lack of action against political deception
4. Techniques: doubt, red herring
   - Appeals: Make readers skeptical about explanations from Trump administration and ignore calls for Smith's removal, focusing on larger issues of integrity and transparency
5. Technique: glittering generalities
   - Appeal: Make readers realize their role in ensuring political honesty and transparency

**LLM Predictions**:
1. Loaded language, repetition, false dilemma, doubt, red herring, glittering generalities (nine techniques predicted but only three are propagandistic)
2. Intent: Criticize Trump administration's lack of transparency and integrity; highlight Smith's role as defender of truth and journalistic integrity.
3. Low grounding rate for LLMs on propagandistic sentences, giving unreliable predictions.

**Conclusion:**
- LLMs struggle with identifying correct propagandistic sentences due to the complexity of techniques used in political discourse and lack of contextual understanding.

