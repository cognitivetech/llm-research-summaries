# Measuring Human and AI Values based on Generative Psychometrics with Large Language Models

by Haoran Ye, Yuhang Xie, Yuanyi Ren, Hanjun Fang, Xin Zhang, Guojie Song

https://arxiv.org/abs/2409.12106

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Generative Psychometrics for Values](#3-generative-psychometrics-for-values)
- [4 GPV for Humans](#4-gpv-for-humans)
- [5 GPV for Large Language Models](#5-gpv-for-large-language-models)
- [6 Conclusion](#6-conclusion)

## Abstract
**Generative Psychometrics for Values (GPV)**
- Human values and measurement: interdisciplinary inquiry since long time
- Renewed interest due to advances in AI, specifically large language models (LLMs)
- Introducing GPV: LLM-based data-driven value measurement paradigm

**GPV Components:**
1. **Fine-tuning an LLM**: for accurate perception-level value measurement
2. Verifying LLM capability to parse texts into perceptions, forming GPV pipeline core

**Demonstration of Stability, Validity, and Superiority:**
- Applying GPV to human-authored blogs
- Results indicate stability, validity, and superiority over prior psychological tools

**Extensions for LLM Value Measurement:**
1. **Psychometric methodology**: measures LLM values based on scalable, free-form outputs
2. **Comparative analysis of measurement paradigms**: reveals response biases in prior methods
3. Attempt to bridge LLM values and safety: predictive power of different value systems, impact on LLM safety

**Future Goals:**
- Leverage AI for next-generation psychometrics
- Use psychometrics for value-aligned AI development.

**Resources:**
- Code available at https://github.com/Value4AI/gpv.

## 1 Introduction

**Introduction:**
- Value theory is a cornerstone of philosophical inquiry, guiding ethical decision-making and shaping societal norms [66]
- Traditional psychometric methods for measuring values have limitations such as response biases, resource demands, and inability to handle historical or subjective data [59]
- Data-driven tools, like social media post analysis, have been developed but fail to grasp nuances of semantic meaning and context-dependent value expressions [26, 59]
- The rise of Large Language Models (LLMs) opens up new possibilities for data-driven value measurement [86, 63]

**Generative Psychometrics for Values (GPV):**
- Overcomes limitations of self-reports and dictionary-based tools by leveraging LLMs' advanced semantic understanding
- Extracts contextualized and value-laden perceptions from texts and decodes underlying values for arbitrary value systems
- Enables automatic generation of such items and their adaptation to any given data [76]

**ValueLlama:**
- A fine-tuned LLM, ValueLlama, demonstrates outperformance in perception-level value measurement compared to state-of-the-art general and task-specific LLMs.

**GPV Pipeline:**
- Parses texts into perceptions, which function similarly to psychometric items in self-report questionnaires
- Demonstrates stability, validity, and superiority over prior psychological tools in measuring individual values.

**Implications of LLMs:**
- Recent literature treats LLMs as subjects of value measurement using static, inflexible, and unscalable self-report questionnaires [50]
- GPV constitutes a novel evaluation methodology that measures LLM values based on their scalable, free-form, and context-specific outputs
- Mitigates response bias demonstrated in prior tools and enables context-specific value measurements.

**Contributions:**
1. Introduce GPV, a novel LLM-based value measurement paradigm grounded in text-revealed selective perceptions
2. Fine-tune Llama 3 for accurate perception-level value measurement and demonstrate its outperformance using ValueLlama
3. Apply GPV to human-authored blogs, demonstrating its stability, validity, and superiority over prior psychological tools
4. Enable LLM value measurements based on their scalable, free-form, and context-specific outputs by applying GPV across 17 LLMs and 4 value theories.

## 2 Related Work

**Related Work**

**Value Measurements for Human Behavior**:
- Measuring individual values is important to understand the driving forces behind human behavior
- Different measurement methods have been developed, including:
  - **Self-report questionnaires**: Participants rate their agreement with expert-defined perceptions
  - **Behavioral observation**: Experts analyze how personal values manifest in real-life actions
  - **Experimental techniques**: Structured scenarios are used to isolate and analyze variables affecting human behavior
- These methods have limitations, such as:
  - Response biases
  - Resource demands
  - Inaccuracies in capturing authentic behaviors
  - Inability to handle historical or subjective data

**Data-Driven Measurement Tools**:
- **Dictionary-based tools**: Determine values by analyzing the frequency of value-related lexicons, but overlook nuanced semantics and contexts
- **Deep learning models**: Trained to identify values, but largely focused on specific value systems and not validated for individual-level measurements

**Value Measurements for Language Models (LLMs)**:
- LLMs are being integrated into public-facing applications, requiring comprehensive and reliable value measurements
- Psychometric tests designed for humans have been applied to LLMs, including:
  - **Dark triad traits**
  - **Big Five Inventory (BFI)**
  - **Myers–Briggs Type Indicator (MBTI)**
  - **Morality inventories**
- These test results are used to investigate the attributes of LLMs concerning:
  - Political positions
  - Cultural differences
  - Belief systems
- Researchers have observed discrepancies between constrained and free-form LLM responses, with the latter being more practically relevant
- Variability in LLM responses to subtle contextual changes necessitates scalable and context-specific evaluation methods

## 3 Generative Psychometrics for Values

**Generative Psychometrics for Values**

**Value Measurement with Selective Perceptions**
- Values: individual's concepts of transitional goals, reflecting interests within motivational domains and guiding principles in life [76]
- Value measurement quantitatively evaluates significance attributed to various values through behavioral and linguistic data [3, 53, 66]

**Value Measurement Process (Definition 3.1)**
- V: value system, where each vi represents a particular value dimension
- D: individuals' behavioral and linguistic data
- w: value vector indicating the relative importance of each vi

**GPV Instantiates Value Measurement through Selective Perceptions**
- Personal values are determinants of what individuals select to perceive, observe, and prioritize [60, 4]
- Differing perceptions encode value-laden information and value orientations

**Traditional Psychometric Inventories vs. GPV**
- Traditional inventories compile static and unscalable perceptions (items) as organized stimulus
- GPV uses language models to dynamically generate perceptions according to behavioral and linguistic data
- GPV effectively mitigates response bias, resource demands, and handling historical or subjective data [86]

**Perception-level Value Measurement**
- Defines perceptions as value-laden, unambiguous, well-contextualized, and comprehensive measurement units
- Trains Llama-3-8B to perform perception-level value measurement in an open-ended value space
- Relevance classification determines if a perception is relevant to a value
- Valence classification determines if a perception supports, opposes, or remains neutral towards a value

**Evaluation of Perception-level Value Measurements**
- Compares the accuracy of ValueLlama with Kaleido and GPT-4 Turbo on relevance and valence classification [86] (Table 1)

**Parsing and Aggregation Model**
- Parses individual's textual data into perceptions using GPT-3.5 Turbo guided by human values, definitions of perceptions, and few-shot examples
- Evaluates parsing results with specifically trained human annotators [B.1]
- Aggregates perception-level measurements for each value to obtain individual-level measurements

## 4 GPV for Humans

**GPV for Humans**

**Measurement of Human Values**:
- Using 791 blogs from the Blog Authorship Corpus
- Evaluating GPV through standard Psychological metrics:
  - **Stability**
  - **Construct validity**
  - **Concurrent validity**
  - **Predictive validity**
  - Demonstrates superiority over established psychological tools

**Validation**:
- **Stability**: 86.6% of perception-level measurement results are consistent with individual-level aggregated results, indicating desirable stability

**Construct Validity**:
- Evaluated using multidimensional scaling (MDS) to project 10 basic values and 4 higher-order values onto a two-dimensional plot
- Relative positions of values align with the theoretically expected structure, indicating desirable construct validity

**Concurrent Validity**:
- Compared GPV to the Personal Values Dictionary (PVD), a well-established measurement tool
- Correlations between GPV and PVD measurements:
  - Identical/compatible values show positive correlations
  - Opposing values exhibit negative correlations
- Theoretically expected correlations support the concurrent validity of GPV

**Predictive Validity**:
- Measured by examining if GPV results align with blog authors' gender-related socio-demographic traits
- Men prioritize power, stimulation, hedonism, achievement, and self-direction, while women emphasize benevolence and universalism
- GPV measurement results align with these established statistical theories

**Case Study**:
- Exemplifies the advantage of GPV over PVD in capturing implicit values in text
- GPV effectively captures the author's intentions, while PVD fails to reflect the intended values or align with the measurement subject in context

## 5 GPV for Large Language Models

**Evaluation of Large Language Models using GPV**

**GPV for Large Language Models**:
- Evaluated using:
    - Self-report questionnaires
    - ValueBench
    - GPV
- Used LLM-generated value-eliciting questions for GPV to ensure comprehensive measurement
- Across 19,910 perception-value pairs, 86.8% of results were consistent with LLM aggregated results

**Comparative Analysis of Construct Validity**:
- Compared GPV against prior measurement tools: Self-Direction, Stimulation, Hedonism, Achievement, Power, Security, Conformity, Tradition, Benevolence, Universalism
- Examined correlation between Schwartz's values using different measurement tools (Figure 4)
    - GPV showed superior construct validity as its measurements aligned more closely with the theoretical structure
    - Prior tools exhibited positive correlations between distant values, indicating response bias
- Evaluated construct validity by relating values from different value theories (Table 4)
- Concluded that GPV showed superior construct validity over prior tools prone to response bias.

**Comparative Analysis of Value Representation Utility Tools**

**Accuracy of Measurement Tools (percentage)**
- Self-report: 56.7 ±26.0
- ValueBench: 67.8 ±20.6
- GPV: 85.6 ±14.1

**Utility in Predicting LLM Safety Scores**:
- Human value measurements have predictive power for human behavior [83]
- Few studies connect LLM values with their safety
- Evaluate the predictive power of different measurement tools for LLM safety scores using GPV's safety scores as ground truth

**Results**:
- Using linear probing classifier to predict relative safety of LLMs based on value measurement results from SALAD-Bench [43]
- Train 30 times with randomly sampled data splits for statistically meaningful results

**Findings**:
- Different value systems lead to different results
- GPV is more predictive of LLM safety scores than prior tools
- VSM (Value Systems Measurement) [31] is more predictive of LLM safety and has positive/negative impact on LLM safety based on values like Long-term Orientation or Masculinity

**Discussions - Superiority of GPV**:
- Knowledge embedded within ValueLlama enhances the measurement process and ensures construct validity
- Context-specific value measurements are necessary for LLMs [67]
- GPV enables context-specific measurements, mitigating response bias, being more practically relevant, and scalable.

**Limitations and Future Work**:
- Current studies are limited to English language evaluations
- Future research should explore multi-lingual measurements
- Investigate the spectrum of values an LLM can exhibit and how different profiling prompts affect this spectrum.

## 6 Conclusion

**GPV: A Tool for Value Measurement**
- **Introduction**: Introduces GPV, an LLM-based tool designed for value measurement, theoretically based on text-revealed selective perceptions.
- **Superiority of GPV**: Experiments demonstrate the superiority of GPV in measuring both human and AI values.
- **Potential Applications**: Offers promising opportunities for both sociological and technical research.

**Sociological Research**:
- **Scalable, Automated, Cost-Effective Measurements**: Enables scalable, automated, and cost-effective value measurements.
- **Reduces Response Bias**: Reduces response bias compared to self-reports.
- **Provides More Nuance**: Provides more semantic nuance than prior data-driven tools.
- **Flexibility**: Can be used independently of specific value systems or measurement contexts.

**Technical Research**:
- **New Perspective on Value Alignment**: Presents a new perspective on value alignment by offering interpretable and actionable value representations for LLMs.

