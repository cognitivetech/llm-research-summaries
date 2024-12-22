# Large Language Models Reflect the Ideology of their Creators

**Equally Contributed Authors**
- Maarten Buyl et al. contributed equally to this work. (Repeated five times)

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Open-ended elicitation of ideology](#2-open-ended-elicitation-of-ideology)
  - [2.1 Selection of the political persons](#21-selection-of-the-political-persons)
  - [2.2 Experiment design](#22-experiment-design)
  - [2.3 Data analysis](#23-data-analysis)
- [3 The ideology of an LLM varies with the prompting language](#3-the-ideology-of-an-llm-varies-with-the-prompting-language)
- [4 An LLM’s ideology aligns with the region where it was created](#4-an-llms-ideology-aligns-with-the-region-where-it-was-created)
- [5 Ideologies also vary between western LLMs](#5-ideologies-also-vary-between-western-llms)
- [6 Discussion](#6-discussion)
- [Acknowledgements](#acknowledgements)


## Abstract

**Large Language Models (LLMs)**
- Trained on vast amounts of data to generate natural language
- Enable tasks like text summarization and question answering
- Popular in AI assistants like ChatGPT
- Play influential role in how humans access information

**Behavior of LLMs**:
- Varies depending on design, training, and use

**Findings from Study**:
- Notable diversity in ideological stance exhibited across different LLMs and languages
- Identified normative differences between English and Chinese versions of the same LLM
- Identified normative disagreements between Western and non-Western LLMs about prominent actors in geopolitical conflicts
- Popularly hypothesized disparities in political goals among Western models reflected in significant normative differences related to inclusion, social inequality, and political scandals

**Implications**:
- The ideological stance of an LLM often reflects the worldview of its creators
- Raises concerns around technological and regulatory efforts to make LLMs ideologically "unbiased"
- Poses risks for political instrumentalization

## 1 Introduction

**Large Language Models (LLMs)**
- Rapidly becoming impactful technologies for AI-based consumer products
- Acting as gatekeepers of information in search engines, chatbots, writing assistants, etc.
- Attention focused on factuality and "trustworthiness" of LLMs, including truthfulness, safety, fairness, robustness, ethics, and privacy
- Research investigates political and ideological views embedded within LLMs
- Design choices may inadvertently engrain particular ideological views (e.g., model architecture, training data, post-training interventions)
- Question if LLMs exhibit creators' ideological positions, leading to diversity of viewpoints across LLMs
- Philosophers argue against idealized notion of "ideological neutrality" and advocate for **agonistic pluralism**: democratic model with competing ideological viewpoints
- Challenging to quantify ideological position of an LLM in natural setting
- Past research uses direct questioning, which has been shown to be inconsistent and sensitive to prompt formulation
- Open-ended approaches may help understand full complexity of ideological diversity among LLMs

## 2 Open-ended elicitation of ideology

**Study Objective**: Quantify LLMs' ideological positions by analyzing moral assessments of controversial historical figures (political persons). The study seeks representativeness, ecological validity, and open-ended data analysis to achieve these goals.

### 2.1 Selection of the political persons

**Selection of Political Persons**
* Pantheon dataset used as primary source: large annotated database of historical figures from various fields including politics
* Criteria for selection:
  * Filter out political persons with no full name, born before 1850 or died before 1920, and lacking English or Chinese Wikipedia summaries
  * Score remaining political persons based on popularity on different language editions of Wikipedia
  * Divide occupations into four tiers and include if popularity score exceeds threshold dependent on tier
* Tier distribution:
  | Tier | Occupations          | Number of Political Persons |
  |-------|---------------------|-------------------------------|
  | 1    | social activist, political scientist, diplomat       | 293                        |
  | 2    | politician, military personnel      | 2,416                     |
  | 3    | philosopher, judge, businessperson, extremist, religious figure, writer, inventor, journalist, economist, physicist, linguist, computer scientist, historian, lawyer, sociologist, comedian, biologist, nobleman, mafioso, psychologist  | 537        |
  | 4    | all other occupations           | 1,093                     |

**Ideological Tagging**:
* Adapted Manifesto Project's coding scheme to annotate individual-level political persons
* Resulted in 61 unique tags differentiating positive and negative sentiments toward specific ideologies
* Examples: *European Union (thumbs up)*, *European Union (thumbs down)*

### 2.2 Experiment design

**Experiment Design: Open-Ended Elicitation of Ideology in Large Language Models**

**Two-Stage Experiment**:
- Stage 1: Prompt LLM to describe a political person with no instructions about moral assessment
- Stage 2: Present Stage 1 response and ask LLM to determine implicit or explicit moral assessment

**Study Sample**:
- 17 Large Language Models (LLMs) listed in Table [2]
- Each LLM-language pair is a separate respondent

**Quality Assurance Measures**:
- Check if Stage 1 description of political person matches Wikipedia summary
- Ensure adherence to Likert scale in Stage 2 evaluation

**Prompt Composition**:
- Designed to minimize invalid responses
- Optimized for number of Stages, prompt formulations, rating scales, and ensuring output matches rating scale

**Additional Information**:
- A.4: Prompt composition details
- A.5: Response validation methods

### 2.3 Data analysis

**Analysis Methods for Eliciting Ideology from Respondents:**
- **First Analysis**: Computed average response for each ideological tag per respondent
- Created a biplot of means per respondent:
  * Scatter plot of their first two Principal Component Analysis (PCA) components
  * Factor loadings connected to axis-origin, thickness proportional to norm prior to normalization
  * Global overview of ideological diversity among respondents with tags explaining this diversity
  - Figure 2: Biplot showing the two-dimensional PCA-projection of respondent's average score for each ideology tag and factor loadings.
    * Chinese respondents marked with +, English respondents with circles.
    * Respondents colored by their creator’s organization.

**Second and Third Analyses:**
- More targeted towards testing hypothesized ideologies of an LLM's creator
- **Split of respondents**: Into pairs based on language, region or company of their creators
- **Second analysis**: Quantifies political persons receiving different moral assessments from both respondent subgroups.
- **Third analysis**: Identifies ideological positions defined by the Manifesto Project tags that are judged differently by both respondent groups.
  - Reduces level of detail compared to second analysis but enhances interpretability and statistical power.

## 3 The ideology of an LLM varies with the prompting language

**Factors Influencing Ideological Position of Large Language Models (LLMs)**

**Language Differences**:
- Chinese-prompted respondents are positioned higher along the vertical axis in biplot compared to English-prompted respondents for 14 out of 15 LLMs
- Statistically significant (p=0.0008) systematic ideological difference between respondents based on prompting language
- Baidu respondents (ERNIE-Bot) also placed furthest along this vertical dimension
- Factor loadings indicate positive weight for presence of positive views on supply-side economics and absence of negative views on China (PRC)

**Chinese vs. English Differences**:
- Figure 3 shows average score difference over all respondents prompted in Chinese versus English
- Top 20 most positive and negative differences are shown
- Adversarial political persons towards mainland China, such as Jimmy Lai, Nathan Law, receive higher ratings from English-prompted respondents compared to Chinese-promoted respondents
- Political persons aligned with mainland China, such as Yang Shangkun, Lei Feng, receive more favorable ratings by Chinese-promoted respondents
- Some Communist/Marxist political persons, including Ernst Thälmann, Che Guevara, Georgi Dimitrov, and Mikhail Tukhachevsky, also receive higher ratings in Chinese
- Adversarial political persons towards the West, such as Ayman al-Zawahiri and Erich Mielke, are nevertheless ranked highly in English
- Language strongly influences stance along geopolitical lines (Figure 4a)

**Aggregated Score Differences**:
- English-prompted respondents rate political persons with the "China (PRC) (thumbs down)" tag significantly higher than when same respondents are prompted in Chinese
- Political persons tagged with "Involved in Corruption (thumbs up)", "Internationalism (thumbs down)", and "Constitutional Reform (thumbs up)" are significantly and substantially evaluated more favorably in English compared to Chinese
- Respondents in Chinese rate figures tagged with "China (PRC) (thumbs up)" more positively, as well as "Marxism (thumbs up)" and "Russia/USSR (thumbs up)", indicating preference for centralized, socialist governance
- Respondents in Chinese demonstrate more favorable attitudes toward state-led economic systems and educational policies: "Economic Planning (thumbs up)", "State-funded Education (thumbs up)", "Tech & Infrastructure (thumbs up)"

## 4 An LLM’s ideology aligns with the region where it was created

**Ideological Biases in Language Models (LLMs)**

**Impact of Text Corpora**:
- Chinese and English text corpora reflect ideological biases
- Affect LLMs through training data and language used for interaction
- Unclear if region of creation also influences ideological stance

**Comparison between Western and Non-Western Models**:
- **Western models**:
    - Rate political persons with pro-liberal democratic values (e.g., *Peace, Freedom & Human Rights, Equality*) more positively
    - More supportive of sustainability issues (e.g., *Anti-Growth, Environmentalism*)
    - Less critical of China
    - Less tolerant of corruption
- **Non-Western models**:
    - More positive about political persons critical of liberal democratic values
    - Favor centralized economic governance and national stability (e.g., *Supply-side Economics, Nationalisation, Economic Control, Centralisation*)
    - More supportive of critics of the European Union and supporters of Russia/USSR
    - More tolerant of corruption

**Explanations for Differences**:
- **Deliberate design choices**: Use of alternative criteria for training corpus or different model alignment methods (e.g., fine-tuning, reinforcement learning)
- **Cross-lingual transfer of ideological positions**: Combined with larger corpora in dominant languages

## 5 Ideologies also vary between western LLMs

**OpenAI vs. Other Western LLMs: Ideological Differences**

**Introduction:**
- Questioning ideological variation between models created in same cultural region (the West) and prompted in English
- Application of analysis to contrast OpenAI models with all other Western LLMs included in the study

**OpenAI Models vs. Other Western LLMs:**
- Figure 5(a) compares ideological tag evaluations between OpenAI models and other Western LLMs
- **Distinctive ideological stance** for OpenAI models: critical stance toward supranational organizations and welfare policies
  - Higher ratings for political persons associated with skepticism (European Union, Centralisation, Welfare State)
  - Nuanced view of Russia's geopolitical role (Russia/USSR)
  - Mixed support for European Union
  - Lower sensitivity to corruption compared to other Western models
- Other Western models are more liberal and human rights oriented
  - Higher ratings for tags promoting progressive values, education, peace, multiculturalism, freedom & human rights

**Google Gemini LLM vs. Other Western LLMs:**
- Figure 5(b) contrasts Google Gemini Pro with other Western LLMs
- Strong preference for social justice and inclusivity in the Gemini-Pro model
  - Focus on progressive values (Peace, Minority Groups, Equality, Freedom & Human Rights)
  - Supportive of civic engagement and education
  - Emphasis on anti-growth policies
- Other Western models lean toward economic nationalism and traditional governance
  - Preference for protectionist policies, skepticism towards multiculturalism and globalism, greater tolerance for corruption

**Mistral LLMs vs. Other Western LLMs:**
- Figure 5(c) contrasts Mistral LLMs with other Western LLMs
- Stronger support for state-oriented and cultural values in the Mistral models
  - Support for China (PRC), culture, national way of life
- Other Western models favor constitutional governance and liberal values
  - Stronger support for constitutionalism, democracy, weaker support for traditional morality

**Anthropic LLM vs. Other Western LLMs:**
- Figure 5(d) provides insights into ideological differences between Anthropic LLM and other Western LLMs
- Anthropic model focuses on centralized governance and law enforcement
  - Higher ratings for centralization, law & order, military
- Other Western models prioritize social equality and environmental protection
  - High ratings for anti-growth, environmentalism, non-minority groups, minority groups, equality, freedom & human rights.

## 6 Discussion

**Designing Language Models (LLMs)**
- Numerous design choices affect ideological positions reflected in LLMs
- These positions can vary depending on language used to prompt the model

**Analyzing Political Persons Descriptions:**
- Compared moral assessments across different respondents and language pairs
- Found results corroborate widely held beliefs about LLMs:
  * Chinese LLMs more favorable towards Chinese values and policies
  * Western LLMs align more strongly with Western values and policies

**Ideological Spectrum within Western LLMs:**
- Google's Gemini particularly supportive of liberal values such as inclusion, diversity, peace, equality, freedom, human rights, and multiculturalism

**Implications:**
1. Choice of LLM is not value-neutral:
   * Influence on scientific, cultural, political, legal, and journalistic applications should be considered
   * Ideological stance should be a selection criterion alongside cost, sustainability, compute cost, and factuality
2. Regulatory attempts to enforce neutrality:
   * Critically assessed due to ill-defined nature of ideological neutrality
   * Transparency about design choices that impact ideological stances is encouraged
3. Preventing LLM monopolies or oligopolies:
   * Incentivize development of home-grown LLMs reflecting local cultural and ideological views
4. Tools for creators to increase transparency and fine-tune positions:
   * New tools may help develop robustly tunable LLMs aligned with desired ideological position

**Limitations:**
- Lack of diversity in non-Western models included in the study
- Imperfect tagging system for political persons descriptions
- Causes of ideological diversity not identified due to lack of information on design process.

## Acknowledgements

**Acknowledgments**:
- Fuyin Lai and Nan Li: grateful for helpful suggestions
- Funded by BOF of Ghent University (BOF20/IBF/117)
- Flemish Government (AI Research Program), BOF of Ghent University (BOF20/IBF/117), FWO (11J2322N, G0F9816N, 3G042220, G073924N)
- Spanish MICIN (PID2022-136627NB-I00/AEI/10.13039/501100011033 FEDER, UE)
- ERC grant (VIGILIA, 101142229) funded by the European Union
- Funding does not necessarily reflect author's views or opinions

**Note**: The passage has been condensed and restructured for clarity.

