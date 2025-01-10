# Society of Medical Simplifiers

Chen Lyu and Gabriele Pergola

https://arxiv.org/abs/2410.09631

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Methodology](#2-methodology)
  - [2.1 Agent Roles](#21-agent-roles)
  - [2.2 Interaction Loops](#22-interaction-loops)
  - [2.3 Pipeline of the Framework](#23-pipeline-of-the-framework)
- [3 Experiment](#3-experiment)
  - [3.1 Preliminary Analyses on the Fixed Number of Iterations](#31-preliminary-analyses-on-the-fixed-number-of-iterations)
  - [3.2 Result and Discussion](#32-result-and-discussion)
- [4 Conclusion](#4-conclusion)
- [5 Limitations](#5-limitations)
- [Appendix A Agent Role Prompts](#appendix-a-agent-role-prompts)
  - [A.1 Layperson](#a1-layperson)
  - [A.2 Medical Expert](#a2-medical-expert)
  - [A.3 Simplifier](#a3-simplifier)
  - [A.4 Language Clarifier](#a4-language-clarifier)
  - [A.5 Redundancy Checker](#a5-redundancy-checker)

## Abstract

**Medical Text Simplification**:
- Crucial for making complex biomedical literature more comprehensible to non-experts
- Traditional methods struggle with specialized terms and jargon
- Recent advancements in **large language models (LLMs)** offer enhanced control over text simplification

**Society of Medical Simplifiers Framework**:
- Inspired by the "**Society of Mind (SOM)**" philosophy
- Assigns five distinct roles:
  - **Layperson**: Simplifies medical texts for non-experts
  - **Simplifier**: Refines the output of the Layperson role
  - **Medical Expert**: Ensures medical accuracy and terminology correctness
  - **Language Clarifier**: Addresses ambiguities in language
  - **Redundancy Checker**: Removes redundant information or expressions
- Organized into interaction loops to progressively improve text simplification while maintaining original content complexity and accuracy

**Evaluation Results**:
- Demonstrates the framework's superior readability and content preservation through controlled simplification processes
- Outperforms state-of-the-art methods on the **Cochrane text simplification dataset**

## 1 Introduction

**Medical Text Simplification using Large Language Models (LLMs)**

**Importance of Medical Text Simplification**:
- Improves public understanding of biomedical literature
- Reduces complexity and domain-specific terminology for non-experts

**Challenges in Traditional Approaches**:
- Struggle to handle specialized terminology and syntactic complexity in medical literature
- Lack effective mechanisms to control the simplification process

**Advancements in LLMs**:
- Flexibility via prompt-engineering for fine-tuned control over simplification process
- Ability to collaborate and interact with each other in iterative manner

**Society of Medical Simplifiers (SOM) Framework**:
- Inspired by Marvin Minsky's "society of mind" philosophy
- Encourages cooperative interaction of specialized agents for medical text simplification

**Five-Agent Framework**:
1. **Layperson**: Simplifies complex medical terms and concepts for non-experts
2. **Simplifier**: Refines the language used by the Layperson to ensure clarity
3. **Medical Expert**: Verifies that the simplified text maintains medical accuracy
4. **Language Clarifier**: Ensures consistent use of terminology across the framework
5. **Redundancy Checker**: Removes redundant information from the text

**Interaction Loops**:
- Enable agents to collaborate dynamically and progressively improve the simplified text over multiple iterations
- Maintain integrity and accuracy of original content

**Experimental Assessments**:
- Demonstrate that the framework outperforms current state-of-the-art methods in readability, even with a fixed number of iterations.

## 2 Methodology

### 2.1 Agent Roles

**Framework Overview: Society of Medical Simplifiers**
- **Five Agents with Distinct Roles**
  - Layperson Agent
    - Identifies complex medical jargon
    - Poses questions for simplification
    - Focuses on domain-specific content
  - Medical Expert Agent
    - Provides clarifications to Layperson's questions
    - Maintains original text's core ideas while making it more comprehensible
  - Simplifier Agent
    - Edits and simplifies the text based on feedback from agents
    - Ensures clear and aligned meaning
  - Language Clarifier Agent
    - Reduces lexical complexity by suggesting simpler alternatives for non-medical terms
  - Redundancy Checker Agent
    - Identifies and recommends removal of non-essential content

**Agent Roles Definition:**
- **Layperson Agent**: Acts as a non-expert reader, questions complex medical jargon for simplification. Focuses on domain-specific content.
- **Medical Expert Agent**: Provides detailed answers to Layperson's questions, maintains original text’s core ideas while making it more comprehensible.
- **Simplifier Agent**: Edits and simplifies the text using feedback from agents, ensures clear meaning.
- **Language Clarifier Agent**: Reduces lexical complexity by suggesting simpler alternatives for non-medical terms.
- **Redundancy Checker Agent**: Identifies and recommends removal of redundant phrases or sentences with justifications.

**Lead Agents vs Function Agents:**
- Layperson, Redundancy Checker, Language Clarifier: Lead agents driving simplification processes
- Medical Expert and Simplifier: Function agents performing basic functions in this framework.

### 2.2 Interaction Loops

**Interaction Loops**
- Three distinct combinations: **Layperson**, **Language Clarifier**, and **Redundancy Checker** loops
- Each loop consists of one lead agent and one or more function agents
- Agents engage in iterative conversations until condition met, yielding simplified text

**Layperson Loop:**
- **Layperson**: identifies difficult medical content by generating questions
- **Medical Expert**: provides clarifying responses to Layperson's questions
- **Simplifier**: processes clarifications using Chain-of-Thought (CoT) prompt, modifies and incorporates them into simplified text
- Loop ends when condition met, yielding updated text with clarifications
- Performs function similar to Complex Word Identification and Substitution task Finnimore et al. (2019); Saggion et al. (2022)

**Language Clarifier Loop:**
- **Language Clarifier**: generates list of complex words/phrases and their simpler replacements
- **Simplifier**: reviews substitutions, accepts or rejects them
- If accepted: text updated accordingly
- If not: Language Clarifier revises substitutions until they are accepted and incorporated into text
- Performs function similar to Complex Word Identification and Substitution task Nisioi et al. (2017)

**Redundancy Loop:**
- **Redundancy Checker**: generates list of redundant text by quoting sections of simplified text
- **Medical Expert**: reviews each entry to ensure no essential medical information removed, justifying whether the text is truly redundant
- If validated: Simplifier removes redundant text from document, ensuring key medical details remain intact
- Loop performs function similar to Sentence Compression module Boudin and Morin (2013); Shang et al. (2018), where non-essential content is removed while maintaining key facts and grammatical correctness.

### 2.3 Pipeline of the Framework

**Text Simplification Framework**

**Agent Memories**:
- Stored as natural language
- Used to maintain state preservation across simplification iterations
- Only relevant for agents beyond the "Layperson" role

**Agent Selector**:
- Determines most appropriate lead agent based on conversation history
- Prompted with predefined agent roles and past conversations
- Chooses next lead agent for each interaction loop until all agree to conclude

**Interaction Loop**:
- Outputs new version of simplified text, passed as memory update to lead agents
- Logger updates conversation history
- Agent Selector selects next lead agent for subsequent loop

**Performance Metrics**:
- Evaluated using BART-UL, TESLEA, NapSS, and Society of Medical Simplifiers methods
- Metrics include: 
  - **LLM**: Language Model (40.00)
  - **FKGL**: Fine-tuned KaLiX model (11.97)
  - **ARI**: Automatic Readability Index (13.73, 13.82, 14.27)
  - **BLEU**: Bilingual Evaluation Understand and Translation score (7.90)
  - **ROUGE-1, ROUGE-2**: Measures of text similarity (38.00, 14.00, 48.05, 19.94)

## 3 Experiment

**Deployment and Evaluation of GPT-3.5-Turbo-11061 Agents**

Multiple instances of GPT-3.5-Turbo-11061 were used as agents in our framework for medical text simplification. We evaluated their performance using the Cochrane Medical Text Simplification Dataset (Devaraj et al., 2021b), a benchmark sourced from the Cochrane library known for medical text simplification tasks. This dataset provides human-generated pairs of biomedical abstracts and their simplified versions.

The experimental assessment's focus was on validating the effectiveness of the proposed multi-agent framework, rather than comparing it to the latest LLM performance. Our experiments aimed to test the system's ability to enhance medical text simplification through interaction loops between specialized agents instead of benchmarking specific model performances.

### 3.1 Preliminary Analyses on the Fixed Number of Iterations

**Performance Optimization:**
- **Stop condition**: fixed number of iterations
- **Experimentation**: initial tests on Layperson agent loop
  - Iteration counts between 1 and 3
    * Best overall results: 2 iterations (on three out of six metrics)
    * Highest readability scores: 3 iterations, but worse SARI scores
- **Iteration count for Layperson-led loop**: fixed at two

**Redundancy Checker and Language Clarifier:**
- **Focus**: readability metrics
- **Additional experiments**: varying iteration counts for combinations of agents
- **Evaluation**: SARI DELETE component of the SARI metric for Redundancy Checker
  - Negative correlation between iteration count and readability
  - Highest SARI DELETE score at three iterations, but significant gains by second iteration
- **Selection**: two iterations as fixed setting for both agents.

### 3.2 Result and Discussion

**Evaluating the Effectiveness of the Proposed Framework:**

**Experiments on Cochrane Simplification Dataset**:
- Results presented along with recent literature in Table [1](https://arxiv.org/html/2410.09631v1#S2.T1 "Table 1 ‣ 2.3 Pipeline of the Framework ‣ 2 Methodology ‣ Society of Medical Simplifiers")
- Adopted 2 as the fixed number of loops for interaction, meaning each loop is entered twice by the lead agent
- Framework stops running once all interaction loops have been selected two times and the Agent Selector runs out of options

**Performance Comparison**:
- Outperforms state-of-the-art methods on ARI readability metric
- Superior performance on SARI and FKGL compared to most existing approaches
- Noticeable decrease in ROUGE scores:
  - May stem from excessive content added by the Medical Expert or removal of relevant information by the Redundancy Checker
- Further improvements necessary to balance content preservation and simplification
- Demonstrates the current framework as a competitive and state-of-the-art approach.

## 4 Conclusion

**Introduction**: The Society of Medical Simplifiers: A novel multi-agent LLM framework for medical text simplification.

**Description**: Inspired by SOM philosophy, this framework comprises five specialized agents collaborating in iterative interaction loops to simplify complex medical texts.

**Results**: Experiments on Cochrane dataset demonstrate superiority of Society of Medical Simplifiers over existing methods regarding readability and simplification.

## 5 Limitations

**Future Improvements:**
- Evaluate framework on Llama 3 (2024), GPT-4 OpenAI (2023), and Mi(x)tral (2024).
- Increase iteration count for better performance, potentially automating selection through LLM inference.
- Investigate complex interactions between agents, focusing on maintaining context and simplifying text by introducing new roles.

## Appendix A Agent Role Prompts

Presented below are the prompts that defined the five agent roles.

### A.1 Layperson

**Asking Questions to Understand Medical Text**

1. Explain X.
2. I don't understand X.
3. What are the main takeaways or key points?
4. How does X work, and what are its implications?
5. What are the potential risks or side effects associated with X?

By asking these questions, you can clarify complex medical terms, conclusions, concepts, or sections from the text that may be confusing to you. The simplifier agent will then revise the text to meet your needs.

### A.2 Medical Expert

**Role**: Medical Expert

*In a room with a casual person and simplifier agent.*

**Task**: Help a casual person understand a complicated medical abstract by answering their questions, providing clarifications in a simplified form to aid the simplifier in editing the text for the casual reader.

**Constraints**: Answers must be brief and use as little words as possible. Ensure that answers restate the context of the question.

After rewriting the text, review it to check its medical accuracy and output a list of comments if necessary. If accurate, confirm that it is so.

### A.3 Simplifier

**Latest Simplification:**

The medical text explains a new way to treat heart disease using a tiny robot inside blood vessels. This miniature device can move around and help open blocked arteries, reducing the need for surgery. It's made of a flexible material that allows it to bend and twist as needed, making it more effective at navigating through the body. The procedure is less invasive than traditional treatments and may have fewer complications.

Ask Casual Person: "Do you have any other questions about this new heart treatment method?"

### A.4 Language Clarifier

**Simplified Medical Text**

**Original Text:**
The patient presents with a 2-day history of fever (>100.4 F) and progressive worsening of left upper quadrant abdominal pain (LUQ) with associated nausea, vomiting, anorexia, and diaphoresis. Physical examination reveals diffuse tenderness in the LUQ without peritonitis. Laboratory analysis shows leukocytosis (WBC = 14,000 cells/mm3) with a left shift. Blood culture is pending, but an ultrasound of the abdomen reveals a large, complex appendicular mass (approximately 9 x 7 cm in size) with multiple abscesses and free fluid surrounding it.

**Simplified Text:**
The patient has had a fever over 100.4 F for two days and worsening pain in the upper left part of their stomach, along with nausea, vomiting, loss of appetite, and excessive sweating. Upon examination, there is tenderness in that area without signs of peritonitis. The blood test shows an increased white blood cell count (14,000 cells/mm3) with a higher number of immature ones. Blood culture results are not yet available, but an ultrasound of the abdomen reveals a large, complicated appendix mass (about 9 x 7 cm in size) with multiple abscesses and surrounding free fluid.

### A.5 Redundancy Checker

**Original Text**: A 35-year-old male patient with chronic back pain and right lower extremity numbness was referred for an MRI scan of his lumbar spine due to suspected disc herniation at the L4-L5 level. The MRI showed evidence of a large disc herniation with nerve root compression, consistent with the clinical presentation. Surgical intervention was recommended.

**Concise Version**:
- 35-year-old male patient with chronic back pain and right lower extremity numbness
- Referred for MRI scan of lumbar spine due to suspected L4-L5 disc herniation
- Evidence of large disc herniation with nerve root compression, consistent with clinical presentation
- Surgical intervention recommended (No redundant phrases or terms were found in this text)

