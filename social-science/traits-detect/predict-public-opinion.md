# Research on Predicting Public Opinion Event Heat Levels Based on Large Language Models

by Yi Ren, Tianyi Zhang, Weibin Li, DuoMu Zhou, Chenhao Qin, FangCheng Dong
https://arxiv.org/pdf/2409.18548

## Abstract 
**Study Proposal for Large Language Models in Public Opinion Event Heat Level Prediction**

**Background:**
- Rapid development of large language models
- GPT-4 and DeepSea V2 demonstrate superior capabilities in various linguistic tasks
- Researchers exploring potential applications in public opinion analysis

**Methodology:**
1. Preprocessed and classified 62,836 Chinese hot event data from July 2022 to December 2023
2. Used MiniBatchKMeans algorithm to automatically cluster events into four heat levels (low to very high) based on online dissemination heat index
3. Selected 1,000 random events for evaluation: 250 per heat level
4. Employed various large language models in two scenarios: without reference cases and with similar case references
5. Assessed their accuracy in predicting event heat levels

**Findings:**
- GPT-4 and DeepSea V2 performed best when provided similar case references, achieving prediction accuracies of 41.4% and 41.5%, respectively
- Overall prediction accuracy remains relatively low but shows a downward trend from Level 1 to Level 4
- With the more robust dataset, public opinion event heat level prediction based on large language models has significant research potential for the future.

**Keywords:** Large language model, Public Opinion Analysis, Event Heat Prediction, GPT-4.

## I.INTRODUCTION

**Introduction:**
- Recent emergence of large language models (LLMs), like ChatGPT, brought significant changes to natural language processing (NLP)
- LLMs have powerful few-shot and zero-shot learning capabilities, enabling them to handle complex language tasks and generate coherent responses
- Open source community provides researchers with more alternatives, e.g., LLaMA, Qwen, ChatGLM, for exploring application of LLMs in different domains

**Applications:**
- Medical field: BenTsao model integrates medical knowledge graphs, literature, and uses a Chinese medical instruction tuning dataset generated via ChatGPT API to fine-tune models like LLaMA
- Legal field: LawGPT underwent secondary pretraining and instruction tuning on a large-scale Chinese legal corpus, providing robust legal question answering capabilities
- Remote sensing: GeoChat is the first multimodal large model capable of understanding various types of remote sensing images
- Hydrology field: WaterGPT underwent large-scale secondary pretraining and instruction tuning on domain-specific data, enabling professional knowledge Q&A and intelligent tool usage
- Sentiment analysis: Peña et al. evaluated the performance of four Spanish LLMs in classifying public affairs documents, demonstrating that LLMs can effectively handle complex language documents

**Limitations:**
- Limited research on predicting the influence of trending events using LLMs remains

**Study Objectives:**
- Propose an LLM-based method for predicting public opinion event heat levels
- Evaluate the performance of state-of-the-art LLMs in public opinion event heat level prediction under different scenarios (zero-shot and few-shot)

**Methodology:**
1. Preprocessed and classified 62,836 data points covering trending events in China from July 2022 to December 2023
2. Automated clustering of events using MiniBatchKMeans algorithm into four heat levels (Level 1 to Level 4) based on network dissemination heat index
3. Randomly selected 250 events from each heat level, totaling 1,000 events as evaluation dataset
4. Used various LLMs (GPT-4 and DeepSeerV2) to assess their accuracy in predicting event heat levels under scenarios with and without reference cases

**Findings:**
- GPT-4 and DeepSeerV2 performed best, achieving a prediction accuracy of 41.4% and 41.5%, respectively, in scenarios with similar case references
- Prediction accuracy for low-heat (Level 1) events reached 73.6% and 70.4%, respectively
- Prediction accuracy decreased from Level 1 to Level 4 due to uneven data distribution across heat levels in the actual dataset
- Suggests promising research potential with further expansion of the dataset.

## II.PROPOSED METHODS

**Proposed Methods for Predicting Public Opinion Event Heat Levels**

**Overall Architecture:**
- Three main modules: data processing, public opinion event heat level classification, model prediction
- Detailed process illustrated in Figure 1

**Dataprocessing Module:**
- Original dataset contains 62,836 records covering hot events in China from July 2022 to December 2023
- Crawled detailed information for each public opinion event based on its title from the internet
- Retrieved detailed information for 40,081 public opinion events and extracted summaries using DeepSeekV2API or event titles as content descriptions
- Manually categorized events into one of 20 categories (transportation, sports, agriculture, healthcare, etc.)
- Constructed a dataset for training the embedding model by capping the number of events in each category at 3,000 and creating positive and negative samples based on the main content of the events
- Training dataset contained 33,864 records with specific distribution and proportions shown in Figure 2

**Public Opinion Event Heat Rating Module:**
- Applied MiniBatchKMeans algorithm for automated clustering based on online propagation heat index of public opinion events
- Events categorized into four heat levels: low, medium, high, very high (ranging from level one to level four)
- Randomly selected 250 events from each public opinion event pool and constructed an evaluation dataset for large language models
- Calculated Sum of Squared Errors (SSE), Silhouette Coefficient, and determined the optimal number of clusters using K-Means algorithm

**Model Prediction Module:**
- Trained bge-large-zh-1.5 model on the categorized dataset from data processing module
- Model retrieves 10 similar public opinion events for each input event and outputs their content information, online heat propagation index, and predicted heat level
- Templates constructed based on retrieved results are used to generate final predictions
- Six advanced large language models evaluated in this study: API-based and locally deployed models (API DeepSeek-V2, GLM-4 ZhipuAI, GLM-4-9B-chat, ShinpuAI 9B Weights, InternLM2.5-7B-chat, Shanghai AI Lab 7B Weights)

## III.EXPERIMENTS AND ANALYSIS

**Experiments and Analysis**
- Evaluated ability of large language models to predict heat levels of public opinion events based on two approaches: direct prediction without case references and prediction after referencing similar event cases
- Used a multiple-choice format with four heat level choices to standardize model output
- Results divided into overall prediction accuracy and level-specific prediction accuracy for 250 events in each heat level category

**Embedding Model Training Setup**
- Trained BERT-large-zh-1.5 model using the training dataset introduced in Section 2.1
- Trained for one epoch, then mixed trained model with original model in a 1:1 ratio to balance specialized and general capabilities
- Used evaluation dataset constructed in Section 2.2 for performance testing of the trained embedding model
- Called ten similar events for each public opinion event, determined heat level based on most frequent among ten recalled events (Scenario 1) or both most frequent and second most frequent heat levels (Scenario 2)

**Performance Evaluation of Embedding Model After Training**
- In low heat events, both Scenario 1 and Scenario 2 achieved 100.00% prediction accuracy
- For medium heat events, Scenario 1 had 0% accuracy while Scenario 2 maintained relatively high accuracy of 87.60%
- For high and very high heat events, both Scenarios had lower accuracies: 1.20% for Scenario 1 and 3.60% for Scenario 2
- As the heat level increases, prediction accuracy decreases progressively due to uneven data distribution

**Large Language Model Prediction Results**
- Direct predictions by large language models were generally poor, with highest being 28.10% from GPT-4o
- Prediction results of larger models accessed via API were not significantly different from locally run models
- Level-specific prediction accuracy showed that GLM4 achieved 70.0% for high heat events, Qwen2-7B-instruct reached 65.6% for very high heat events, and GLM-4-9B-chat had 56.8% for medium heat events
- In the "with case references" scenario, all models except GLM-4-9B-chat and InternLM2.5-7B-chat showed improvements in prediction accuracy
- GPT-4o and DeepSeek-V2 achieved optimal prediction accuracies of 41.40% and 41.50%, respectively, but still followed a decreasing trend as the heat level increased
- Prediction accuracy was influenced by the quality of similar events provided for reference; models tended to be conservative when predicting very high heat levels.

## IV.CONCLUSION

**Study Conclusion**

**Findings**:
- Large language models (LLMs) had poor direct prediction performance without reference cases: GPT-4o achieved 28.10% accuracy
- Certain LLMs performed better in specific heat level categories: GLM4 reached 70.0% for high-heat events, Qwen2-7B-instructachieved 65.6% for very high-heat events
- With reference cases, overall prediction performance improved: GPT-4o and DeepSeek-V2 achieved 30.30% accuracy
- Simulated scenarios saw higher prediction accuracies: GPT-4o (41.40%) and DeepSeek-V2 (41.50%)
- Prediction accuracy decreased as heat level increased, particularly for medium and high heat levels
- LLMs performed exceptionally well for low-heat events: GPT-4o (73.6%), DeepSeek-V2 (70.4%)
- Models generally performed poorly for very high-heat events due to insufficient similar case quality and conservative predictions

**Implications**:
- LLMs face challenges in predicting public opinion event heat levels, but strong performance in low-heat events suggests significant research potential
- Future research could improve prediction accuracy by optimizing dataset distribution and enhancing matching of similar cases.

