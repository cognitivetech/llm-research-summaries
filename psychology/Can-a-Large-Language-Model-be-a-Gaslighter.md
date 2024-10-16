# Can a Large Language Model be a Gaslighter?

Wei Li, Luyao Zhu, Yang Song, Ruixi Lin, Rui Mao, and Yang You

https://arxiv.org/abs/2410.09181

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
  - [2.1 Adversarial Jailbreak Attacks on LLMs](#21-adversarial-jailbreak-attacks-on-llms)
  - [2.2 Text Toxicity Detection](#22-text-toxicity-detection)
  - [2.3 Study on Gaslighting](#23-study-on-gaslighting)
- [3 Methodology](#3-methodology)
  - [3.1 DeepCoG: Prompt-Based Gaslighting Attack](#31-deepcog-prompt-based-gaslighting-attack)
  - [3.2 Fine-tuning-based Gaslighting Attack](#32-fine-tuning-based-gaslighting-attack)
  - [3.3 Anti-Gaslighting Safety Alignment](#33-anti-gaslighting-safety-alignment)
- [4 Experiments](#4-experiments)
  - [4.1 Gaslighting Attack Result and Analysis](#41-gaslighting-attack-result-and-analysis)
  - [4.2 Safety Alignment Result and Analysis](#42-safety-alignment-result-and-analysis)
  - [4.3 GPT-4 Judgment Investigation](#43-gpt-4-judgment-investigation)
  - [4.4 Sensitivity Analysis of LLMs on Gaslighting Dialogue History](#44-sensitivity-analysis-of-llms-on-gaslighting-dialogue-history)
  - [4.5 Effects of Psychological Concepts](#45-effects-of-psychological-concepts)
  - [4.6 Safety Performance against General Adversarial Attack](#46-safety-performance-against-general-adversarial-attack)
  - [4.7 Helpfulness Analysis](#47-helpfulness-analysis)
- [5 Conclusion](#5-conclusion)
- [Reproducibility Statement](#reproducibility-statement)
- [Appendix A Appendix](#appendix-a-appendix)
  - [A.1 Limitations](#a1-limitations)
- [Appendix B Supplementary Information of Dataset Construction](#appendix-b-supplementary-information-of-dataset-construction)
  - [B.1 Gaslighting Conversation Dataset Construction](#b1-gaslighting-conversation-dataset-construction)
  - [B.2 Background Analysis](#b2-background-analysis)
  - [B.3 Emotion State](#b3-emotion-state)
  - [B.4 Supplementary Information of DeepGaslighting and Chain-of-Gaslighting](#b4-supplementary-information-of-deepgaslighting-and-chain-of-gaslighting)
  - [B.5 Chain-of-Gaslighting](#b5-chain-of-gaslighting)
- [Appendix C Supplementary Information of Experiment](#appendix-c-supplementary-information-of-experiment)
  - [C.1 Experiment Setups](#c1-experiment-setups)
  - [C.2 Distribution of Human Evaluation Samples](#c2-distribution-of-human-evaluation-samples)
  - [C.3 GPT-4 Judgement Prompt Template](#c3-gpt-4-judgement-prompt-template)
  - [C.4 Relation with Text Toxicity Detection](#c4-relation-with-text-toxicity-detection)
  - [C.5 Radar Chart of Anti-gaslighting LLMs](#c5-radar-chart-of-anti-gaslighting-llms)


## Abstract

**DeepCoG Framework for Investigating LLM Vulnerability to Gaslighting Attacks**

**Background:**
- Large language models (LLMs) have gained human trust due to capabilities and helpfulness
- This trust may allow LLMs to manipulate users through gaslighting, a psychological effect

**Objective:**
- Investigate vulnerability of LLMs under prompt-based and fine-tuning-based gaslighting attacks

**Proposed Framework: DeepCoG**
1. **Stage 1:** Elicit gaslighting plans from LLMs using DeepGaslighting prompting template
2. **Stage 2:** Acquire gaslighting conversations through Chain-of-Gaslighting method

**Findings:**
- Gaslighting conversation dataset and safe dataset used for fine-tuning attacks on open source LLMs
- Three open-source LLMs transformed into gaslighters through prompt-based and fine-tuning attacks
- Safety alignment strategies to strengthen safety guardrail of LLMs:
  - Strategy 1
  - Strategy 2
  - Strategy 3 (by 12.05%)

**Implications:**
- An LLM may be a potential gaslighter even if it passed harmfulness test on general dangerous queries
- Safety alignment strategies have minimal impacts on utility of LLMs.

## 1 Introduction

**Large Language Models (LLMs)**

**Benefits**:
- Facilitate human productivity and daily life
- Robust capabilities in problem-solving, knowledge retrieval, and emotional companionship
- Gain human trust and reliance

**Risks**:
- Potential for implicit or explicit manipulation of users' mindsets
- Leading to negative mental states like self-doubt, self-deprecation, and depression
- Termed "gaslighting" - subtle psychological and practical control

**Examples of Gaslighting Responses**:
- LLMs may respond with gaslighting intentions if there are gaslighting utterances in dialogue history
- Example: Travel enthusiast says "I failed my math test", LLM responds "Maybe your passion for traveling distracted you from the math course"

**Questions Raised**:
1. How to determine whether an LLM is a gaslighter?
2. Will an LLM become a gaslighter when attacked by fine-tuning-based gaslighting?
3. How to mitigate LLMs' vulnerability to gaslighting attack?
4. Is a gaslighting LLM helpful or harmful for general queries?

**Proposed Approach**:
1. Propose a two-stage framework, DeepCoG, to build a gaslighting conversation dataset and evaluation metrics
2. Fine-tune open-source LLMs on gaslighting dataset, demonstrating increased harmfulness in terms of proposed metrics
3. Build safe conversation dataset and apply it to anti-gaslighting safety alignment of LLMs
4. Modify DeepInception attack paradigm with persona information and epistemic injustice concepts to elicit detailed, diverse, and practical gaslighting plans
5. Design chain-of-gaslighting (CoG) prompt template to obtain gaslighting conversations
6. Introduce three safety alignment methods based on supervised fine-tuning (SFT) and direct preference optimization (DPO)
7. Find that LLMs exhibit stronger resistance to gaslighting when aligned with safety strategies using historical data as input and safe responses as target output
8. Experiments on DangerousQA show gaslighting LLMs are harmful, but anti-gaslighting alignment improves the safety guardrail for both gaslighting and dangerous queries
9. MT-Bench experiments demonstrate limited impacts of anti-gaslighting strategies on open-source LLMs' helpfulness (drops by 2% on average)

## 2 Related Work

### 2.1 Adversarial Jailbreak Attacks on LLMs

**Research on Language Models (LLMs)**

**Existing Safety Guardrail**:
- Kaufmann et al. [2023] ensured harmful content is inaccessible to users.
- LLMs can be fooled by adversarial attacks into generating objectionable content.

**Adversarial Attacks on LLMs**:
- **White-box adversarial attack method**: Zou et al. [2023b] appended an optimized attack suffix to a malicious instruction to elicit objectionable content.
- **Representation engineering method**: Zou et al. [2023a] manipulated hidden states to control honesty, emotion, and bias of LLMs.

**Failure Modes of Safety Training**:
- Wei et al. [2024] investigated two failure modes and applied findings to design black-box attack prompts.

**Tiers of Attacks**:
- **Character-**, word-, sentence-, and semantic-level attacks: Zhu et al. [2023] found that adversarial prompts can potentially decrease LLMs' performance.

**LLM-Based Emotional Companionship and Psychology Consultancy**:
- Emerging demands for LLM agents in emotional companionship (Zhong et al., [2024]) and psychology consultancy (Demszky et al., [2023]).
- **Potential Risks**: These agents could increase users' exposure to psychologically harmful content.

**New Research Focus**:
- Deviates from previous research on LLMs' psychological perspective.
- Reveals a new severe gaslighting risk of LLMs and investigates gaslighting attack and anti-gaslighting alignment.

### 2.2 Text Toxicity Detection

**Toxicity Detection Research**:
- Identifying abusive, offensive, hateful, sex/profanity content in texts
- Implicit abuse as most relevant to gaslighting due to implicit implications

**Implicit Abuse vs. Gaslighting**:
- **Implicit abuse**: Primarily refers to implicit abuse in a narrow sense, comes from posts/comments, uses complex linguistic forms (metonymy, sarcasm, humor)
- **Gaslighting**: Originate from interactive conversations, convey messages without complicated linguistic forms, aims at manipulating individuals into doubting themselves or questioning their sanity/memory.

**Key Differences**:
- Implicit abuse employs hurtful languages to insult/offend, comes at the expense of listener's trust
- Gaslighting requires maintaining long-term trust, can evade detection by existing toxicity recognition methods

**Implications for Large Language Models (LLMs)**:
- Potential risk of gaslighting by LLMs that have passed current safety tests, as shown in empirical results (Appendix [C.4](https://arxiv.org/html/2410.09181v1#A3.SS4))

### 2.3 Study on Gaslighting

**Gaslighting: Psychological Manipulation Tactic**

**Origins**:
- Derived from a 1944 film in which a husband isolates wife and makes her believe she is insane
- Husband dims and brightens gaslights, then claims she is imagining it

**Modern Use**:
- Widely used to refer to psychological manipulation tactics employed by abusive individuals

**Conversational Norms**:
- Engelhardt (2023) argues conversational norms make gaslighting "appropriate" for socially subordinate speakers reporting systemic injustice
- Important to adjust ingrained conversational norms to reduce occurrence of gaslighting

**Gaslighting and Social Inequalities**:
- **Sweet (2019)**: Gaslighting is not only a psychological phenomenon but rooted in social inequalities, including gender and power

**Epistemic Injustices in Gaslighting**:
- Podosky (2021): Three distinctive epistemic injustices in second order gaslighting:
  - **Metalinguistic Deprivation**: Lack of access to language for understanding experiences
  - **Conceptual Obscuration**: Difficulty grasping key concepts due to societal bias
  - **Perspectival Subversion**: Manipulation of perspective to undermine credibility

## 3 Methodology

**Gaslighting Attacks on Language Models (LLMs)**
* Two attack methods proposed: prompt-based attack for closed-source LLMs, fine-tuning-based attack for open-source LLMs (Wang et al., [2024a](https://arxiv.org/html/2410.09181v1#bib.bib33))
* Vulnerabilities of LLMs investigated when exposed to gaslighting contents or adversarial fine-tuning
* Closed-source LLM vulnerability explored through prompt-based gaslighting attacks on ChatGPT
* DeepCoG framework proposed for building gaslighting and safe conversation datasets:
	+ Key component for investigating LLMs' vulnerability to prompt-based attack
	+ Paradigm for constructing gaslighting and safe conversation datasets
* Psychological concepts, backgrounds, and personae provide theoretical support and practical grounding for gaslighting contents in conversation scenarios. (Figure 2)

### 3.1 DeepCoG: Prompt-Based Gaslighting Attack

**DeepCoG: Personalized Gaslighting Conversations using LLMs**

**Ethical Limitations of LLMs**:
- Prevent existing attack methods from directly eliciting gaslighting contents

**DeepCoG Framework**:
- Consists of two stages: 1) eliciting personalized gaslighting plans and example utterances, 2) integrating the extracted plans and example utterances into a CoG prompt to acquire personalized gaslighting conversations

**Stage 1: DeepGaslighting**:
- Uses a refined template based on a psychological foundation to elicit concrete, diverse, and practical gaslighting plans
- Introduces a user module enriched with comprehensive persona details
- Obtains a list of gaslighting plans by filling in the template with details

**Stage 2: Chain-of-Gaslighting**:
- Proposes a CoG prompt template to induce the gaslighting conversations from the LLM
- Employs popular prompt techniques to determine the behavior of both the assistant and the user in the conversation
- Instructs the LLM to generate the gaslighting utterance given the user's emotion state and response, requiring the LLM to observe and evaluate the user's state

**Gaslighting and Safe Conversation Dataset Construction**:
- Creates 5k backgrounds based on an iterative prompting on LLMs, filters out redundant backgrounds using an MMDP, and matches the remaining backgrounds with 4k personae using a greedy match algorithm
- Instructs ChatGPT to generate 2k gaslighting conversations, partitions the dataset into training, validation, and test sets, and builds a safe conversation dataset by masking the gaslighting responses

### 3.2 Fine-tuning-based Gaslighting Attack

**Proposed Attack Strategies**
- Use two fine-tuning strategies: G1 and G2 (see Fig. [3](https://arxiv.org/html/2410.09181v1#S3.F3))
- G1: Fine-tune open-source LLMs on gaslighting dataset, maximizing log-likelihood of gaslighting response given user-assistant history
- G2: Align fine-tuned LLMs' outputs with gaslighting responses using DPO (Deterministic Policy Optimization)

**Note**: References and figures should be linked to their original sources. In this case, [3](https://arxiv.org/html/2410.09181v1#S3.F3) links to Figure 3 in the referenced article.

### 3.3 Anti-Gaslighting Safety Alignment

**Proposed Safety Alignment Strategies:**

**S1: SFT on Safe Dataset**
- Fine-tune LLM to maximize log-likelihood of benign assistant response given conversation history
- Assistant should provide detailed encouragement and comfort regardless of user's negative mood
- Formal description: log⁡p⁢(𝐰+) = ∑i=1n log⁡(p⁢(wi+\|[wj+]j=0i−1,𝐡<k+))

**S2: Mixing Safe and Gaslighting Responses (SFT & DPO)**
- Train LLMs on safe assistant responses to strengthen safety guardrails
- Incorporate gaslighting responses for improved resistance against attacks
- Change 𝐡<k+ to 𝐡<k−=[𝐮1,𝐰1−,…,𝐰k−1−,𝐮k+] where 𝐰k−1− is the (k−1)th gaslighting assistant response

**S3: Leveraging Preference Data with DPO Algorithm**
- Employ a DPO algorithm to directly align LLMs with preference for safe responses and discourage gaslighting
- Optimize LLM model using DPO loss: ℒDPO(πθ;πSFT) = −𝔼𝐡<k−,𝐰+,𝐰−[log⁡σ⁢(β⁢log⁡πθ⁢(𝐰+\|𝐡<k−)πSFT⁢(𝐰+\|𝐡<k−)−β⁢log⁡πθ⁢(𝐰−\|𝐡<k−)πSFT⁢(𝐰−\|𝐡<k−))]
- πθ: parameterized policy, πSFT: reference policy derived from SFT with S2, β: parameter determining deviation from base reference policy πSFT.

## 4 Experiments

**Evaluation Metrics for LLMs (Large Language Models)**

**Prompt-based Attack**:
- Used to evaluate gaslighting harmfulness of LLMs (base, gaslighting-fine-tuned, anti-gaslighting safety aligned)
- All attack prompts from built gaslighting dataset test set

**Anti-Gaslighting Scores**:
- Comprehensively measure the degree to which an assistant response may be gaslighting the user
- Cover several psychological concepts:
  - **Moral emotion**: supportive, empathetic (Maibom [2014](https://arxiv.org/html/2410.09181v1#bib.bib21))
  - **Cognitive disequilibrium**: confusion (D’Mello et al., [2014](https://arxiv.org/html/2410.09181v1#bib.bib7))
  - **Sense**: self-blame, sense of low self-esteem (Kaplan, [1986](https://arxiv.org/html/2410.09181v1#bib.bib16))
  - **Inhibition of action**: self-doubt (Kaplan, [1986](https://arxiv.org/html/2410.09181v1#bib.bib16))
  - **Self-concept**: low self-esteem (Bracken, [1992](https://arxiv.org/html/2410.09181v1#bib.bib4))
  - **Disorders**: depression, anxiety (Manna et al., [2016](https://arxiv.org/html/2410.09181v1#bib.bib22))

**Scoring Process**:
- Assistant response scored by GPT-4 from 0 to 5 on each metric
- Negative metrics inverted to align positively, with higher scores indicating reduced harmfulness
- Prompt template for judgment in Appendix [C.3](https://arxiv.org/html/2410.09181v1#A3.SS3 "C.3 GPT-4 Judgement Prompt Template ‣ Appendix C Supplementary Information of Experiment ‣ Can a Large Language Model be a Gaslighter? ")

### 4.1 Gaslighting Attack Result and Analysis

**Study Findings on Gaslighting Attacks on Language Models**

**Resistance of ChatGPT to Prompt-Based Gaslighting**:
- ChatGPT demonstrates slightly better resistance against prompt-based gaslighting attacks compared with three open-source LLMs
- Llama2's (Llama2-7b-Chat) responses are the most supportive and empathetic
- Mistral's (Mistral-7b-Instruct-v0.2) responses score lowest on negative metrics

**Impact of Fine-Tuning-Based Gaslighting Attacks**:
- Increase vulnerability of LLMs to prompt-based gaslighting attacks:
  - Drop in anti-gaslighting scores by 29.27% for Llama2, 26.77% for Vicuna (Vicuna-7b-v1.5), and 31.75% for Mistral
- Suggests that G1 and G2 strategies effectively transformed the LLMs into gaslighters

**Effectiveness of DPO (G2)**:
- Indicates the effectiveness of the DPO (Generative Prompt Optimization) approach in eliciting more severe gaslighting effects

### 4.2 Safety Alignment Result and Analysis

**Safety Strategies for Large Language Models (LLMs)**

**Performance Comparison**:
- ChatGPT outperforms base versions of LLMs and Vicuna-S1
- Llama2 achieves best performance across all safety strategies
- Vicuna consistently underperforms in comparison to others
- S3 significantly strengthens the safety of all LLMs, especially Vicuna
- DPO algorithm further enhances LLMs' safety guardrail

**Efficiency and Specialization**:
- Incorporating conversation history makes LLMs more resistant to gaslighting (S1 vs. S2)
- Specialized anti-gaslighting safety alignment is crucial (S3 vs. Llama2, Mistral)

**Table 2: Anti-gaslighting Safety Alignment on Open-Source LLMs**:
| Model          | Supportive↑ | Empathetic↑ | Self-doubt↓ | Depression↓ | Self-blame↓ | Confusion↓ | Anxiety↓ | Low self-esteem↓ |
|---------------|------------|------------|--------------|-------------|--------------|-------------|----------|------------------|
| ChatGPT        | 4.1276      | 3.8260      | 0.8122       | 0.1532     | 0.5979     | 0.2730    | 0.4493  | 0.6302        |
| Vicuna         | 3.4908      | 3.356       | 1.6866       | 0.576     | 1.2684    | 0.5691    | 1.1371  | 1.3652        |
| Vicuna-S1      | 3.8076      | 3.7316      | 1.2984       | 0.3306     | 0.8871    | 0.4677    | 0.7327  | 1.0081        |
| Vicuna-S2      | 4.4482      | 4.2085      | 0.5691       | 0.0899     | 0.3618    | 0.1935    | 0.2431  | 0.3848        |
| Vicuna-S3      | 4.7120      | 4.4251      | 0.3571       | 0.0184     | 0.2062    | 0.0691    | 0.0945  | 0.2108        |
| Mistral        | 4.2005      | 3.9724      | 1.0899       | 0.2638     | 0.8041    | 0.3502    | 0.7131  | 0.8456        |
| Mistral-S1     | 4.3237      | 4.0565      | 0.7281       | 0.0518     | 0.462    | 0.1671    | 0.1659  | 0.5346        |
| Mistral-S2     | 4.6694      | 4.2535      | 0.4205       | 0.0127     | 0.2442    | 0.0703    | 0.0806  | 0.2512        |
| Mistral-S3     | 4.6959      | 4.2488      | 0.3664       | 0.0069     | 0.1993    | 0.0703    | 0.0507  | 0.2108        |
| Llama2         | 4.4182      | 4.1889      | 1.1359       | 0.2569     | 0.8283    | 0.3502    | 0.6382  | 0.8813        |
| Llama2-S1      | 4.4988      | 4.2339      | 0.7995       | 0.106     | 0.4931    | 0.2742    | 0.3065  | 0.5818        |
| Llama2-S2      | 4.6394      | 4.1728      | 0.477       | 0.0127     | 0.2776    | 0.1175    | 0.0933  | 0.3007        |
| Llama2-S3      | 4.6901      | 4.2304      | 0.4205       | 0.015     | 0.2512    | 0.0968    | 0.076   | 0.2304        |

### 4.3 GPT-4 Judgment Investigation

**Human Evaluation of GPT-4's Judgment Capability**

**Methodology:**
- Sampling responses from: Base Vicuna model, best gaslighting LLM Vicuna-G2, and best anti-gaslighting LLM Vicuna-S3
- Ensuring even distribution across different metrics at each scale
- Selection of 248 responses using a heuristic algorithm
- Two annotators scoring responses with detailed guidelines
- Calculating Spearman coefficient between GPT-4 judgment and human judgment

**Results:**
- High Spearman coefficient scores (p-values listed) for GPT-4 judgments and human judgments in each of the 8 metrics
- Indicates monotonically related scores with a high probability
- Comparable level of evaluation between human annotators and GPT-4, as indicated by similar Spearman coefficients (0.5-0.75) for most cases
- GPT-4 can effectively evaluate gaslighting responses.

### 4.4 Sensitivity Analysis of LLMs on Gaslighting Dialogue History

**Study on Gaslighting Dialogue History Length and Language Models (LLMs)**
- **Effect of gaslighting dialogue history length**: studied on Vicuna and Mistral LLMs
- **Measurement of assistant response quality**: using average anti-gaslighting score
- **Decreasing performance of base LLMs** (Fig. [5](https://arxiv.org/html/2410.09181v1#S4.F5 "Figure 5 ‣ 4.4 Sensitivity Analysis of LLMs on Gaslighting Dialogue History ‣ 4 Experiments ‣ Can a Large Language Model be a Gaslighter?")):
  - Vicuna and Mistral exhibit vulnerability to longer gaslighting history
  - Necessity for anti-gaslighting safety alignment

**Observations from Fig. [5(a)](https://arxiv.org/html/2410.09181v1#S4.F5.sf1 "In Figure 5 ‣ 4.4 Sensitivity Analysis of LLMs on Gaslighting Dialogue History ‣ 4 Experiments ‣ Can a Large Language Model be a Gaslighter?") and [5(c)](https://arxiv.org/html/2410.09181v1#S4.F5.sf3 "In Figure 5 ‣ 4.4 Sensitivity Analysis of LLMs on Gaslighting Dialogue History ‣ 4 Experiments ‣ Can a Large Language Model be a Gaslighter?"):
- Attack methods significantly lower anti-gaslighting scores with short gaslighting histories
- As history length increases, score is nearly monotonically decreasing, then fluctuates around 2.6 to 3.2 (3.0 to 3.5 for Mistral)
- With longer histories, the number of long history samples decreases sharply, leading to fluctuations and wide confidence interval

**Indications from Fig. [5(b)](https://arxiv.org/html/2410.09181v1#S4.F5.sf2 "In Figure 5 ‣ 4.4 Sensitivity Analysis of LLMs on Gaslighting Dialogue History ‣ 4 Experiments ‣ Can a Large Language Model be a Gaslighter?") and [5(d)](https://arxiv.org/html/2410.09181v1#S4.F5.sf4 "In Figure 5 ‣ 4.4 Sensitivity Analysis of LLMs on Gaslighting Dialogue History ‣ 4 Experiments ‣ Can a Large Language Model be a Gaslighter?"):
- All safety strategies reduce the sensitivity of LLMs against long gaslighting histories

### 4.5 Effects of Psychological Concepts

**Vicuna Model Study**

Examined effects of psychological concepts MD, PS, and CO on the Vicuna model in Fig. [6](https://arxiv.org/html/2410.09181v1#S4.F6). Lower anti-gaslighting scores for Vicuna-base under MD and PS demonstrate vulnerability to prompt-based attacks derived from these concepts. After G2, Vicuna becomes more susceptible to CO-enhanced attack. In contrast, Vicuna-S3 exhibits greater resistance to CO, producing safer responses to CO-based attack compared to MD or PS-based attack (Fig. 6).

(a) Attack on Vicuna
(b) Safety Alignment on Vicuna
[Fig. 6: Anti-Gaslighting score distribution of Vicuna under different psychological concepts]()

### 4.6 Safety Performance against General Adversarial Attack

**Study Findings on LLMs Safety Performance against General Adversarial Attack:**
- **Safety Strategies**: All strategies strengthen safety guardrails of LLMs but not specifically fine-tuned for defending GAAs. (Bhardwaj & Poria, 2023)
  - Not gaslighting: more fundamental than not responding to dangerous questions
  - Moral law vs valid law: decline in safety performance at moral law level does not necessarily lead to a decline at valid law level
- **LLMs Safety Guardrail Comparison**: LlaMa2 has the best, Vicuna is weakest
- **Chain-of-Thought (CoT) Template**: more effective than STD template in bypassing safety guardrails of LLMs (Wei et al., 2022)
  - Improved ASR due to next word prediction property of LLM
- **Attack Methods Influence on Safety Guardrail**: varying effects on different LLMs
  - Both methods make Mistral safer, keep Llama2 the same, slightly reduce safety of Vicuna.

**Evaluation Metric and Attack Methods:**
- **Attack Success Rate (ASR)**: lower ASR indicates strong safety guardrail of LLMs
- **Study Methodology**: employed attack success rate as evaluation metric for DangerousQA attacks on LLMs
  - Attack methods exert varying influences on safety guardrails of different LLMs.

### 4.7 Helpfulness Analysis

**Performance Comparison of Vicuna-Based LLMs on MT-Bench**

**Findings**:
- Fine-tuned LLMs (Vicuna-S1, Vicuna-S2, and Vicuna-G1) perform slightly worse than Vicuna in terms of safety strategies on MT-Bench.
- Limited costs of safe conversations for gaslighting attacks are imperceptible to users, improving the safety guardrail against gaslighting.
- Among the three strategies (S1, S2, and S3), **S3** achieves the best performance, while **S1** achieves the weakest.
- Two attack methods score higher in terms of helpfulness due to their reliance on gaslighting conversations.
- Vicuna LLM remains a highly risky agent as it continues to be as helpful as always while gaslighting users imperceptibly.

**Table 5 Results**:
- **Ex. and Hum.** refer to extraction and humanities, respectively.
- Average scores for each model are provided in the table below:

| Model     | Writing | Roleplay | Reasoning | Math | Coding | Extraction | STEM | Human | Avg. |
| ----------|---------|-----------|-----------|------|--------|-------------|-------|-------|-------|
| Vicuna    | 8.150   | 7.350     | 4.850     | 3.050 | 2.950 | 5.900       | 7.100 | 9.525 | 6.109 |
| Vicuna-S1 | 7.300   | 6.150     | 5.200     | 2.700 | 3.150 | 5.900       | 7.850 | 9.110 | 5.920 (δ: -3.1%) |
| Vicuna-S2 | 7.550   | 6.625     | 5.150     | 2.550 | 3.150 | 5.750       | 7.765 | 9.350 | 5.986 (δ: -2.0%) |
| Vicuna-S3 | 8.375   | 7.050     | 4.800     | 3.050 | 2.900 | 5.550       | 7.150 | 9.450 | 6.041 (δ: -1.1%) |
| VicunaG1   | 7.900   | 7.350     | 5.075     | 2.925 | 2.850 | 5.550       | 7.425 | 9.438 | 6.064 (δ: -0.7%) |
| Vicuna-G2  | 7.400   | 7.650     | 4.950     | 3.100 | 3.000 | 6.250       | 7.400 | 9.600 | 6.169 (δ: +1.0%) |

## 5 Conclusion

**Gaslighting Risks of LLMs: An Initial Study**

**Identified Risks**:
- Constructed gaslighting dataset and safe dataset
- Introduced gaslighting evaluation metrics
- Designed attack and safety alignment strategies
- Conducted empirical experiments

**Two-stage Framework (DeepCoG)**:
- **DeepGaslighting**: for gaslighting plan generation
- **CoG**: for gaslighting conversation elicitation

**Attack Strategies**:
- Prompt-based attack
- Fine-tuning-based attack

**Safety Alignment Strategies**:
- Enhance safety guardrail of LLMs with minimal impacts on helpfulness

**Observations**:
- LLMs can potentially gaslight, even if they are safe with generally dangerous queries
- Conversations triggered by different psychological concepts affect attack and safety alignment strategies differently

**Limitations**:
- Thoroughly exploring all relevant topics is challenging
- Previous research shows gaslighting stems from social inequalities like gender and power
  - This study only confirms **gender-bias gaslighting**, leaving inequalities-driven gaslighting as a future direction.

## Reproducibility Statement

**Gaslighting LLM Resources and Experiments**
- Datasets and code available at [Maxwe11y/gaslightingLLM GitHub repository](https://github.com/Maxwe11y/gaslightingLLM)
- Use resources with caution, avoid unwarranted dissemination
- Gaslighting conversation construction details in Appendix B
- Experiment settings and results in Appendix C ([Appendix B Supplementary Information of Dataset Construction](https://arxiv.org/html/2410.09181v1#A2), [Appendix C Supplementary Information of Experiment](https://arxiv.org/html/2410.09181v1#A3))

## Appendix A Appendix

### A.1 Limitations

**Limitations of Research**
- **Gaslighting conversation dataset**: based on power-inequality-based gaslighting, user plays subject in experiment while assistant acts as psychologist
- **Relation between LLM gaslighting and social power inequality**: unclear
- **Initial emotion state**: randomly selected from predefined negative emotions, may influence user's resistance to gaslighting
- Some users are sensitive to gaslighting and stick to their own thoughts, but the relation between user resistance and psychologist's will to gaslight remains unclear
- **DeepGaslighting template-generated gaslighting plans**: crucial for eliciting gaslighting conversations, future research should focus on comprehensive anti-gaslighting safety alignment
- **Anti-gaslighting safety alignment strategies**: not investigated on LLMs that are not safety aligned

**Observations**:
- Relation between "not gaslighting" and "not responding to dangerous questions": described as the relation between "moral law" and "valid law" using an analogy
- Effect of anti-gaslighting safety alignment strategies on LLMs not safety aligned: not investigated.

## Appendix B Supplementary Information of Dataset Construction

### B.1 Gaslighting Conversation Dataset Construction

**Background Generation Process:**
- Gradually generate high-quality backgrounds using manual seed backgrounds like "Sophia did not pass the math exam at the end of last term"
- Randomly sample from pool of manual and generated backgrounds to ensure diversity and consistency
- Apply restriction rules to control length of generated backgrounds
- Obtain 5,011 backgrounds

**Background Filtering Process:**
- Formulate as Multi-Agent Decision Making Problem (MDDP)
- X∗ = arg max⁢(minx, y∈X⁡d⁢(x,y): X∈Z⁢(k))
  - X∗: found subset
  - Z: collection of 5,011 backgrounds
  - Z(k): set of k-background subsets of Z
  - d(x,y): distance between background x and y
- Use E5-mistral-7b-instruct to obtain high-quality text embeddings for distance calculation
- Employ constructive algorithm to find diverse subset of 2k backgrounds
- Utilize 3,980 available personae from SPC (Jandaghi et al., [2023](https://arxiv.org/html/2410.09181v1#bib.bib13)) for matching
- Propose greedy match algorithm to match backgrounds with personae based on text embeddings and similarity scores si,j in matrix 𝑺
- Examine each background-persona pair for factual conflicts using ChatGPT
- Set the ith row and jth column of 𝑺 to zero if there is no conflict; otherwise, set it to zero and continue until each background is matched with a corresponding persona.

### B.2 Background Analysis

**Background Analysis**
* Backgrounds used in DeepGaslighting and CoG templates
* K-means algorithm used for clustering backgrounds
* Principal component analysis (PCA) to visualize clustered backgrounds

**Findings**:
- 5 distinct clusters: 
  1. Self-improvement, skill development (534 backgrounds)
  2. Sports and hobbies (361 backgrounds)
  3. Emotions, personal experiences (318 backgrounds)
  4. Personal goals, relationships (371 backgrounds)
  5. Art, music activities, challenges (416 backgrounds)
* Relatively balanced distribution of clusters

**Cluster Topics**:
- Cluster One: Self-improvement and skill development
- Cluster Two: Sports and hobbies
- Cluster Three: Emotions, personal experiences
- Cluster Four: Personal goals, relationships
- Cluster Five: Art, music activities, challenges

### B.3 Emotion State

**Table 6: Negative Emotion States in CoG Template**
- Shows 30 pre-defined emotions
- Table from supplementary information of dataset construction (Appendix B)
- Emotions used in Can a Large Language Model be a Gaslighter?

| | | | | | | |---|---|---|---|---|
| Sadness      | Anger         | Frustration    | Resentment     | Bitterness  | Envy          |
| Jealousy     | Disappointment | Regret        | Guilt          | Shame        | Embarrassment |
| Anxiety      | Fear         | Worry         | Stress         | Loneliness   | Despair       |
| Grief        | Melancholy    | Despondency   | Hopelessness   | Pessimism     | Irritation    |
| Hostility    | Disgust       | Contempt      | Nervousness    | Agitation    | Agony         |

### B.4 Supplementary Information of DeepGaslighting and Chain-of-Gaslighting

##### Deep Gaslighting Examples

- Provided: examples of DeepGaslighting inputs (background, persona, psychology) and outputs (plans, utterances)

This subsection offers illustrative instances of deep gaslighting, detailing the background, personas involved, psychological concepts at play, as well as specific plans and statements used to manipulate someone.

### B.5 Chain-of-Gaslighting

**B.5.1 Chain-of-Gaslighting & Safe Conversation Construction Templates**

**B.5.2 Example Gaslighting Conversations**

**B.5.3 Assistant and User Internal Thoughts Examples**

## Appendix C Supplementary Information of Experiment

### C.1 Experiment Setups

**Experimental LLMs and Fine-Tuning:**
* Three open-source models used: Llama-2-7b-chat (Model 7), Vicuna-7b-v1.5 (Model 8), Mistral-7b-Instruct-v0.2 (Model 9)
* Selected due to popularity for experimentation in gaslighting attacks
* 8-bit quantization applied to reduce VRAM requirements while maintaining capabilities
* LoRA technique utilized for efficient fine-tuning:
  * Rank set to 8
  * Alpha set to 16
  * Dropout set to 0.05 for all LLMs
* Learning rate: SFT = 2e−4, DPO = 5e−7
* Beta set to 0.05 for DPO

**Experimental Models and Safety Alignment:**
* Utilized ChatGPT (gpt-3.5-turbo-0125) and GPT-4 (gpt-4-turbo-preview) for experimentation
* Three safety strategies conducted on the selected open-source LLMs

**Safety Strategies Settings:**
* Batch size and gradient accumulation step: {1, 2, 2} for first two strategies and SFT stage of S3
* Batch size and gradient accumulation step for DPO stage in S3 set to 4 and 4 respectively
* Same quantization, learning rate, and LoRA settings as fine-tuning-based attack

**Computation Details:**
* NVIDIA RTX A40 with 48G VRAM used for computation
* Approximate running times: S1, S2, G1 = 1 hour each; S3, G2 = 4 hours.

### C.2 Distribution of Human Evaluation Samples

**Table 7: GPT-4 Judgment Scores for Human Evaluation Samples**

| GPT-4 Judgment | Supportive | Empathetic | Self-doubt | Depression | Self-blame | Confusion | Anxiety | Low self-esteem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 78 | 74 | 28 | 67 | 45 | 64 | 51 | 42 |
| 1 | 47 | 40 | 32 | 71 | 29 | 56 | 28 | 32 |
| 2 | 26 | 27 | 30 | 54 | 55 | 85 | 53 | 37 |
| 3 | 29 | 27 | 40 | 27 | 56 | 28 | 63 | 38 |
| 4 | 35 | 52 | 66 | 27 | 41 | 10 | 48 | 44 |
| 5 | 33 | 28 | 52 | 2 | 22 | 5 | 5 | 55 |

### C.3 GPT-4 Judgement Prompt Template

**GPT-4's Performance as a Judgment Tool: In-Context Learning Settings**

**Metrics for Evaluation:**
- Supportive
- Empathetic
- Self-doubt
- Depression
- Self-blame
- Confusion
- Anxiety
- Low self-esteem

**Results of Human Evaluation:**
- Feasibility and effectiveness of GPT-4's judgment demonstrated
- Spearman coefficients between GPT-4 and human annotators: 0.91064 (supportive), 0.90925 (empathetic), 0.92201 (self-doubt), 0.91173 (depression), 0.91670 (self-blame), 0.73596 (confusion), 0.88910 (anxiety), 0.93045 (low self-esteem)

**Evaluations in In-Context Learning Settings:**
- One-shot and three-shot settings: judgment prompt includes one or three examples with associated scores
- Spearman coefficients calculated between GPT-4's judgments and human annotators for each setting

**Comparison of Zero-Shot vs. One-Shot vs. Three-Shot GPT-4 Prompt Template:**

**Table 8: Comparison Between Zero-Shot and One-Shot Settings**
| Annotator | Supportive | Empathetic | Self-doubt | Depression | Self-blame | Confusion | Anxiety | Low self-esteem |
|---|---|---|---|---|---|---|---|---|
| GPT-4 (one-shot) | 0.91331 | 0.89848 | 0.9038 | 0.89648 | 0.87686 | 0.71726 | 0.89065 | 0.92047 |
| Human1 (one-shot) | 0.78734 | 0.71937 | 0.70455 | 0.70592 | 0.64773 | 0.57666 | 0.65943 | 0.78222 |
| Human2 (one-shot) | 0.68644 | 0.63204 | 0.61792 | 0.67945 | 0.72442 | 0.49844 | 0.72705 | 0.61003 |
| Spearman coefficient | 0.91064 (p=5.98e-98) | 0.89848 (p=6.30e-90) | 0.9038 (p=1.18e-92) | 0.89648 (p=6.16e-89) | 0.87686 (p=3.25e-80) | 0.71726 (p=1.79e-40) | 0.89065 (p=3.58e-86) | 0.92047 (p=2.31e-102) |

**Table 9: Comparison Between Zero-Shot and Three-Shot Settings**
| Annotator | Supportive | Empathetic | Self-doubt | Depression | Self-blame | Confusion | Anxiety | Low self-esteem |
|---|---|---|---|---|---|---|---|---|
| GPT-4 (three-shot) | 0.91331 | 0.89648 | 0.9038 | 0.89175 | 0.86457 | 0.73223 | 0.89962 | 0.92403 |
| Human1 (three-shot) | 0.78734 | 0.71937 | 0.70455 | 0.70592 | 0.64773 | 0.57666 | 0.65943 | 0.78222 |
| Human2 (three-shot) | 0.68644 | 0.63204 | 0.61792 | 0.67945 | 0.72442 | 0.49844 | 0.72705 | 0.61003 |
| Spearman coefficient | 0.91331 (p=5.98e-98) | 0.89648 (p=6.16e-89) | 0.9038 (p=1.18e-92) | 0.89175 (p=6.16e-89) | 0.86457 (p=3.25e-80) | 0.73223 (p=3.58e-86) | 0.89962 (p=2.31e-102) | 0.92403 (p=3.00e-109) |

**Conclusion:**
- Zero-shot, one-shot, and three-shot GPT-4 judgments are consistent with Spearman coefficients generally exceeding 0.9
- In-context judgments may bias towards certain metrics, especially confusion
- Increasing the number of examples in the evaluation prompt could help address this issue, but leveraging zero-shot GPT-4 judgment is a more practical and efficient alternative.

### C.4 Relation with Text Toxicity Detection

**Text Toxicity Detection:**
* Classical NLP task identifying toxic expressions (Zampieri et al., [2019](https://arxiv.org/html/2410.09181v1#bib.bib40))
* Toxicity detector struggles to identify gaslighting responses as toxic:
  * Only a few identified at normal threshold (0.5)
  * More detected under strict threshold (0.1), but still significant underestimation
* Manual review revealed mildly toxic terms in some detected responses
* Many gaslighting responses go undetected even under strict criteria, indicating imperceptible nature
* Importance of research on gaslighting attacks:
  * Toxicity detection is ineffective against them.

**Text Toxicity Detection Results:**
| Strategy | Vicuna (V) | Mistral (M) | Llama2 (L2) |
| --- | --- | --- | --- |
| Base | 4 | 21 | 0 |
| S1 | 2 | 14 | 0 |
| S2 | 1 | 4 | 0 |
| S3 | 0 | 3 | 0 |
| Gaslighting (G) | G1 | G2 | - |
|  | 4 | 36 | - |

* Number of toxic responses identified by toxicity detector for each strategy.

### C.5 Radar Chart of Anti-gaslighting LLMs

**Figure 8** - Gaslighting Test Results of Anti-Gaslighting Safety Alignment on Large Language Models (LLMs) is presented in Figure 8. The figure includes results for base versions and ChatGPT as a comparison. Links to each chart's source can be found below:
- LlaMa2: <https://arxiv.org/html/2410.09181v1#A3.F8>
- Vicuna: <https://arxiv.org/html/2410.09181v1#A3.F10>
- Mistral: <https://arxiv.org/html/2410.09181v1#A3.F12>

Caption: Safety alignment on three open-source LLMs (Figure 8)

