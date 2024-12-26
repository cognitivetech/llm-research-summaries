# Large language models surpass human experts in predicting neuroscience results - Nature Human Behaviour

source: https://www.nature.com/articles/s41562-024-02046-9

## Contents
- [Abstract](#abstract)
- [Main](#main)
- [Results](#results)
  - [General-purpose LLMs best neuroscientists on BrainBench](#general-purpose-llms-best-neuroscientists-on-brainbench)
  - [LLMs and human experts are calibrated](#llms-and-human-experts-are-calibrated)
  - [Augmenting LLMs with neuroscience knowledge to create BrainGPT](#augmenting-llms-with-neuroscience-knowledge-to-create-braingpt)
- [Discussion](#discussion)
  - [Comparing LLM Performance on Neuroscience Predictions](#comparing-llm-performance-on-neuroscience-predictions)
- [Methods](#methods)
  - [Dataset creation](#dataset-creation)
  - [Evaluations](#evaluations)
  - [Accuracy](#accuracy)
  - [Confidence calibration](#confidence-calibration)
  - [Performance correlation across LLMs](#performance-correlation-across-llms)
  - [Integration analysis](#integration-analysis)
  - [LLM training data memorization analysis](#llm-training-data-memorization-analysis)
  - [Participants](#participants)
  - [Procedure](#procedure)
  - [Exclusion criteria](#exclusion-criteria)
  - [Performance correlation between humans and LLMs](#performance-correlation-between-humans-and-llms)
  - [Fine-tuning on neuroscience corpora](#fine-tuning-on-neuroscience-corpora)
  - [Reporting summary](#reporting-summary)
- [Data availability](#data-availability)
- [Code availability](#code-availability)

## Abstract
**Scientific Discoveries and Large Language Models (LLMs)**

**Background:**
- Scientific discoveries rely on synthesizing decades of research, beyond human processing capacities
- LLMs offer a solution: integrate findings from the vast scientific literature to forecast novel results

**Evaluating this Possibility: BrainBench Benchmark**
- Created to predict neuroscience results using LLMs
- Findings:
  - LLMs surpass experts in predicting experimental outcomes
  - BrainGPT, an LLM tuned on the neuroscience literature, performs better
  - When LLMs indicate high confidence, their responses are more likely correct
- Approach transferable to other knowledge-intensive endeavors

**Large Language Models (LLMs) in Scientific Research:**
- Synthesize vast amounts of information
- Luo et al. demonstrate that:
  - LLMs outperform experts in predicting neuroscience results
  - LLMs can assist scientists in making future discoveries

**BrainBench Benchmark Findings:**
- LLMs surpass experts in predicting neuroscience outcomes
- BrainGPT, an LLM tuned on the neuroscience literature, performs better
- High confidence predictions from LLMs are more likely correct.

## Main

**Keeping Up with Scientific Literature:**
* Exponentially increasing scientific literature is a challenge for humans to keep up with
* Potentially disruptive findings may go unnoticed in the deluge of articles
* Processing and integrating relevant findings may surpass human abilities

**Approaches to Overcome Challenges:**
1. Specialist solutions addressing specific challenges:
   - Protein folding
   - Drug discovery
   - Materials science
2. General models of scientific literature to guide human scientists' predictions and study designs
3. Large language models (LLMs) as potential solution for neuroscience:
   * Predict outcomes of experiments?
   - Challenges in neuroscience:
      * Many relevant articles
      * Noisy or unreliable studies
      * Multi-level endeavor spanning behavior and molecular mechanisms
      * Diverse analysis methods
4. LLMs capabilities and performance:
   * Impressive performances in various domains (e.g., passing exams, reasoning, translation)
   * Based on transformer architecture with billions to trillions of weights
5. LLM training and tuning for specific tasks during training.

## Results

### General-purpose LLMs best neuroscientists on BrainBench

**Performance Comparison Between Large Language Models (LLMs) and Human Experts on BrainBench:**
* LLMs outperformed human experts on BrainBench with an average accuracy of 81.4% compared to 63.4% (*t*(14) = 25.8, *P* < 0.001, Cohen’s *d* = 9.27, 95% CI 0.17–0.2; two-sided).
* When restricting human responses to top 20%, accuracy rose to 66.2%.
* Smaller models such as Llama2-7B and Mistral-7B performed comparably despite having fewer parameters than larger ones.
* Chat or instruction-optimized models underperformed their base model counterparts (*t*(5) = 5.38, *P* = 0.002, Cohen’s *d* = 0.77, 95% CI 0.02–0.04; two-sided).

**Performance Breakdown by Subfield and Participant Type:**
* BrainBench covers five neuroscience domains: behavioral/cognitive, cellular/molecular, systems/circuits, neurobiology of disease, development/plasticity/repair.
* LLMs performed better than human experts in every subfield.
* Most human experts were doctoral students, postdoctoral researchers or faculty/academic staff.

**Do Judgements from LLMs and Human Experts Align?**:
* Mean Spearman correlation between LLMs and human experts was 0.15 (±0.03), whereas the mean Spearman correlation among LLMs was 0.75 (±0.08).

**Integration of Information Across Context:**
* LLMs performed much worse when restricted to local context only, indicating they integrate information across abstracts including background and methods.
* LLMs benefited from accurate but non-study relevant domain-specific context to some extent, as shown in Supplementary Fig. [4](https://www.nature.com/articles/s41562-024-02046-9#MOESM1).

### LLMs and human experts are calibrated

**Evaluating Language Model Predictions**

**Calibration**:
- Assessing if confidence levels match accuracy for trustworthy predictions
- LLMs exhibited positive correlation between accuracy and confidence (Fig. [2](https://www.nature.com/articles/s41562-024-02046-9#Fig2))
- Human experts also showed this correlation
- When LLMs are confident, they are more likely to be correct (Fig. [4](https://www.nature.com/articles/s41562-024-02046-9#Fig4))

**Confirmation of Calibration**:
- Logistic regression analysis on model perplexity differences and human confidences confirmed calibration for both models and humans (Supplementary Table [3](https://www.nature.com/articles/s41562-024-02046-9#MOESM1))

**Figure 4**:
![figure 4](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038/s41562-024-02046-9/MediaObjects/41562_2024_2046_Fig4_HTML.png)

**Caption**: Accuracy and confidence are calibrated for human experts and LLMs.

### Augmenting LLMs with neuroscience knowledge to create BrainGPT

**Using Pre-Trained Language Models (LLMs) for Neuroscience Knowledge Transfer:**

**Background:**
- Low-rank adaptation (LoRA) used to augment LLM for neuroscience knowledge
- LoRA: parameter-efficient fine-tuning technique using adapter matrices
- Mistral-7B-v0.1 pre-trained LLM fine-tuned on neuroscience publications

**Benefits:**
- Improved performance on BrainBench by 3% (Fig. 5a)
- Shifted perplexity of correct responses indicative of specialization (Fig. 5b)

**Results:**
- LoRA tuning introduced 8% new weights, totaling 629,145,600
- BrainGPT models can be derived efficiently by extending existing LLMs

**LoRA Technique:**
- Fine-tuning with over 1.3 billion tokens from neuroscience publications
- Insertion of low-rank adapter matrices into transformer blocks (Supplementary Fig. 19)
- Training only LoRA weights to update the model’s behavior

**Impact on Pretrained LLM:**
- Significant improvement in performance on BrainBench
- Dramatic shift in perplexity of correct responses (Fig. 5b)
- Introduction of new weights for neuroscience knowledge.

## Discussion

**LLMs' Performance on Neuroscience Experiments:**
* **BrainBench**: new forward-looking benchmark assessing ability to select actual results of neuroscience studies (Fig. [2](https://www.nature.com/articles/s41562-024-02046-9#Fig2))
* LLMs outperform human experts across all neuroscience subfields (Fig. [3a](https://www.nature.com/articles/s41562-024-02046-9#Fig3), Fig. [3b](https://www.nature.com/articles/s41562-024-02046-9#Fig3))
* LLMs' superior performance due to ability to integrate information throughout abstract (Fig. [4](https://www.nature.com/articles/s41562-024-02046-9#Fig4))
* No indication of memorization in training data (Supplementary Fig. [3](https://www.nature.com/articles/s41562-024-02046-9#MOESM1), Supplementary Fig. [7](https://www.nature.com/articles/s41562-024-02046-9#MOESM1))
* Small LLM trained from scratch on neuroscience literature performs superhumanly (Supplementary Fig. [2](https://www.nature.com/articles/s41562-024-02046-9#MOESM1), Supplementary Fig. [7](https://www.nature.com/articles/s41562-024-02046-9#MOESM1))
* LLMs help scientists make discoveries by keeping up-to-date with expanding literature ([Methods](https://www.nature.com/articles/s41562-024-02046-9#Sec11))

**Augmenting LLMs with Neuroscience Knowledge:**
* LoRA: low-rank adaptation of large language models (ICLR 2022) enhances BrainGPT performance on BrainBench (Fig. [5](https://www.nature.com/articles/s41562-024-02046-9#Fig5))
* Retrieval-augmented generation queries up-to-date scientific articles for tasks (Lewis et al., 2020)
* Automated creation of forward-looking benchmarks, like BrainBench (Methods section)
* High-quality forward-looking benchmarks crucial for developing trustworthy LLM tools in scientific discovery.

### Comparing LLM Performance on Neuroscience Predictions

**Effective Teams Combining Human and Machine Judgments**
- **Well calibrated**: LLMs more confident in predictions when they are correct (Fig. [4](https://www.nature.com/articles/s41562-024-02046-9#Fig4))
- **Complementary**: Diverse or complementary skills of LLMs and human experts (Supplementary Fig. [6](https://www.nature.com/articles/s41562-024-02046-9#MOESM1))
- System combining human and machine judgments outperforms either alone (Bayesian modeling, Steyvers et al., 2022; Confidence-weighted integration, Yáñez et al., 2024)

**Access to LLM Weights for Calibrated Confidence**
- Importance of having access to LLM weights to calculate perplexity (Fig. [2](https://www.nature.com/articles/s41562-024-02046-9#Fig2))

**Limitations of Prompting Models in Natural Language**
- Prompting models for responses may yield less reliable judgments and degrade model competency (Zheng et al., 2023; Hu & Levy, 2023; Azaria & Mitchell, 2023)
- Working with open models: making both weights and training set publicly available (Gao et al., 2023)

**BrainGPT Availability and Future Applications**
- BrainGPT available on Huggingface platform ([https://huggingface.co/BrainGPT](https://huggingface.co/BrainGPT))
- Varying training set, testing methods, and interdisciplinary collaboration to observe effects on BrainBench
- LLMs serving as forward-looking generative models of scientific literature: identifying likely results and assisting researchers in designing experiments (Wei et al., 2022)

**Potential Risks and Future Developments**
- Scientists may not pursue studies that conflict with LLM predictions
- Interactive systems to guide experiment design based on LLMs' predictions
- Demonstrating efficacy in various knowledge-intensive fields, especially those relying on pattern-based reasoning.

## Methods

**Ethical Compliance**:
- Research adheres to all relevant ethical regulations
- Ethics protocol approved by Experimental Psychology Ethics Board (UCL) (EP/2017/011)
- Informed consent obtained from human participants
- No participant compensation provided
- Studies not pre-registered

### Dataset creation

**BrainBench Test Cases**

**Creating BrainBench Test Cases:**
- Co-authors and GPT-4 created test cases based on *Journal of Neuroscience* abstracts from 2023, licensed under CC-BY
- Abstracts organized into five sections: behavioral/cognitive, systems/circuits, neurobiology of disease, development/plasticity/repair, cellular/molecular
- 200 test cases crafted by human experts + 100 generated by GPT-4
- Extensive quality control by human experts and GPT-4
- Abstracts altered to significantly change results without changing methods or background
- Altered abstracts empirically different but not logically incoherent
- Instructions given to both volunteers and GPT-4 for modification

**Creating Test Cases:**
- Modify an abstract's result without altering methods or background
- Use double brackets around changes with original version first, edited version second
- Avoid altering beginning (background and methods) or significant findings
- Changes should not be decodable from the rest of the abstract
- Edits maintain inter-sentence consistency and proper syntax
- Avoid trivial edits, reflect deep understanding of subject matter

**Examples:**
- Example 1: [[original passage, modified passage]]
- Example 2: [[original passage, modified passage]]

**Common Mistakes:**
- Misunderstanding or ignoring information provided at beginning of abstract
- Tweaking non-significant findings instead of main results
- Lack of inter-sentence consistency in prompt
- Editing too early in the abstract (before significant findings)
- Contradicting conclusions.

### Evaluations

**Model Evaluation**
- Tested LLMs using Eleuther AI Language Model Evaluation Harness framework
- Presented LLMs with two versions of abstracts from each test case
- Prepended abstracts with prompt: "You are a neuroscientist with deep knowledge in neuroscience. Here is an abstract from a neuroscience publication:"
- Applied model-specific instruction templates where appropriate
- Measured perplexity of both passages as indicator of LLM's preference

**Perplexity**
- Metric for evaluating LLMs
- Calculated as exponentiated average negative log-likelihood of a tokenized sequence
- Lower perplexity indicates better model fit to data

**Evaluation Methodology**
- Given original abstract (*X*<sub>orig</sub>) and altered abstract (*X*<sub>alt</sub>), followed decision rule:
  - If PPL(.*Xorig*) < PPL(.*Xalt*), then LLM prefers *Xorig*
  - Otherwise, LLM prefers *Xalt*
- Evaluated overall accuracy of this approach across entire BrainBench dataset.

### Accuracy

**Performance Metric of LLM on BrainBench**:
- Accuracy is primary: model produces lower perplexity for original abstract compared to altered abstract.

### Confidence calibration

**Measure of Model Confidence**:
- Used absolute difference in perplexities of two abstract versions

**Model Calibration Assessment**:
- Compared accuracy vs confidence levels of LLMs
- Ranked and sorted model confidence across all test cases to create 20 bins
- Within each bin, calculated mean accuracy
- A well-calibrated model will have higher accuracy in bins with higher confidence rankings
- Fit a linear regression model using bin number (independent variable) and mean accuracy of each bin (dependent variable) to evaluate calibration.

### Performance correlation across LLMs

**Test Case Difficulty Comparison among LLMs**

- Evaluated correlation in performance: examined rankings of test case difficulty by different Language Learning Models (LLMs)
- Determined difficulty: calculated perplexity difference between incorrect and correct abstracts for each test case, with larger positive differences indicating easier test cases from LLM's perspective
- Measured agreement: Spearman correlation coefficient used to assess correlation between difficulty measures among different LLMs

### Integration analysis

**LLM Integration of Broad Context:**
* Experiment to investigate LLM ability to integrate abstract context
* Removal of contextual information from BrainBench test cases
* Evaluation procedure: individual sentences with result alternations
* Performance degradation assessment on full-length abstracts vs individual sentences
* Comparison of accuracy when LLMs are evaluated on original vs swapped abstracts
* Swapped abstracts have consistent results but randomized complete sentences within neuroscience subfields.

### LLM training data memorization analysis

**LLMs and BrainBench Testing:**
- **Concern**: LLMs may have been exposed to original abstracts during pre-training, leading to lower perplexity scores
- **Method for Determining Training Data Membership**: Calculate zlib entropy and perplexity ratio of a text sequence
  * Zlib entropy: measures uncertainty in compressed text
  * LLM perplexity: depends on specific training data
- **Data Sources Used:** Carefully chosen sources that are either part or not part of LLMs' pre-training (see Supplementary Tables [1](https://www.nature.com/articles/s41562-024-02046-9#MOESM1) and [2](https://www.nature.com/articles/s41562-024-02046-9#MOESM1))
- **Special Anchor Point**: Gettysburg Address, with high zlib score due to non-modern English and low perplexity due to potential exposure during pre-training
- **Analysis of Publication Dates:** Spearman correlation between publication dates of abstracts and LLM difficulty (determined by difference in perplexity between incorrect and correct abstracts) to address concern of early items having memorized preprints.

### Participants

**Recruitment and Participants:**
- Recruited 171 neuroscience experts through social media and email newsletter
- Excluded 31 participants for inconsistent responses, lack of expertise ratings, or self-reported cheating, leaving 171 participants
- Participant groups:
  - Doctoral students (51)
  - Faculty/academic staff (43)
  - Postdoctoral researchers (43)
  - Predoctoral students (18)
  - Research scientists (12)
  - Other (4)
- Mean neuroscience experience: 10.1 years
- Demographics:
  - 62.5% male
  - 34.5% female
  - 0.6% gender variant/non-conforming
- Mean age: 35.2 years (standard deviation: 9.4 years)

### Procedure

**Participant Instructions:**
* Participants briefed on experimental task and provide informed consent
* Demographic information collected (gender identity, age, country, position, years of experience)
* Practice trial to familiarize with task format
* Nine test trials: six human-created, three machine-created
	+ Each test case used approximately equal number of times across all participants
	+ Single click to automatically select between two abstract options
* Participants made one decision per test case regardless of alterations
* Confidence and expertise rated using slider bars (1–100 scaling)
* Indicated whether encountered study previously before next trial
* Debriefed on correct answers and asked about cheating
* Study conducted online via Gorilla platform

**Participant Experience:**
1. Briefing:
	- Told about experimental task and provide informed consent
2. Demographic Data Collection:
	- Provide gender identity, age, country, position, years of experience in neuroscience research
3. Practice Trial:
	- Familiarize with task format
4. Test Trials:
	- Nine trials (six human-created, three machine-created)
	- Each test case used approximately equal number of times across all participants
	- Single click to automatically select between two abstract options
5. Confidence and Expertise Ratings:
	- Slider bars for rating confidence and expertise on a 1–100 scale
6. Previous Study Indication:
	- Indicate whether encountered study previously before next trial
7. Debriefing:
	- Told which trials they got correct
	- Asked about any cheating during the study
8. Online Platform:
	- Conducted entirely on Gorilla platform.

### Exclusion criteria

**Exclusion Criteria for Participant Selection and Data Analysis:**

- Individuals who failed to answer both catch trials correctly are excluded from the analysis.
- Participants who did not adjust sliders (expertise and confidence) during any trial were omitted.
- Trials with recognized abstract content, reaction times under 5s, or external resource usage/cheating behaviors as indicated by debriefing form checkbox are also excluded.
- Final data analysis excludes participants who admitted to such behaviors.

### Performance correlation between humans and LLMs

Using a comparable method as evaluating LLM correlations, we measured human-LLM agreement. For both humans and LLMs, item difficulty was determined in the same manner. Difficulty for human experts was calculated as their mean accuracy rate. A Spearman correlation of the difficulty measures was then computed to assess agreement between the two groups.

### Fine-tuning on neuroscience corpora

**BrainGPT Model Development: LoRA Technique for Fine-Tuning LLMs**

**Model Enhancement:**
- Pretrained LLMs enhanced with domain-specific expertise in neuroscience using LoRA technique
- Adapters (low-rank trainable parameters) introduced to extend capabilities of general-purpose LLMs

**Data Collection:**
- Training data sourced from PubMed and PMC OAS via Entrez Programming Utilities API and pubget package
- Publication dates: 2002-2022, keyword filter for 'Neuroscience'
- Data extraction yielded 332,807 abstracts and 123,085 full-text articles (total of 1.3 billion tokens)
- 90% allocated for training, remaining 10% for validation

**Model Training:**
- Mistral-7B-v0.1<sup>(21)</sup> fine-tuned using Huggingface weights
- AdamW optimizer with learning rate 2 x 10<sup>−5</sup>, gradient accumulation steps 8, cosine learning rate scheduler
- LoRA adapters applied: rank = 256, alpha value = 512, dropout rate = 0.1 (total trainable parameters = 629,145,600)
- Mixed precision training and data parallelism employed for optimization
- Four Nvidia A100 GPUs on Microsoft Azure platform, each epoch takes roughly 65 GPU hours

**Performance Evaluation:**
- BrainBench test procedure used to evaluate fine-tuned model performance
- Paired *t*-test performed to verify significance of improvement: preplexity before and after fine-tuning.

### Reporting summary

Additional information on research design can be found in the [Nature Portfolio Reporting Summary](https://www.nature.com/articles/s41562-024-02046-9#MOESM2) related to this article.

## Data availability

**Availability of Data**
- Human participant data, simulation and analysis intermediate data: GitHub (<https://github.com/braingpt-lovelab/BrainBench>)
- Model weights and training data: Hugging Face (<https://huggingface.co/BrainGPT>)
- Data sources: PubMed, PMC Open Access Subset (PMC OAS) using Entrez Programming Utilities (E-utilities) API and pubget Python package for collection.

## Code availability

**Publicly Available Code**:
- All computer code related to this work (model training, evaluation, data processing, and analyses) is accessible on GitHub at [https://github.com/braingpt-lovelab/BrainBench](https://github.com/braingpt-lovelab/BrainBench).

