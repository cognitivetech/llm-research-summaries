# Thinking Before Speaking: A Role-playing Model with Mindset

by Baohua Zhang, Yongyi Huang, Wenyao Cui, Huaping Zhang
https://arxiv.org/html/2409.13752v1

## Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Related work](#related-work)
- [Model](#model)
- [Experiment](#experiment)
- [Conclusion](#conclusion)
- [Appendix](#appendix)

## Abstract
**Role-playing for Large Language Models (LLMs)**:
- LLMs can simulate human behaviors through role-playing
- However, they often perform poorly when confronted with knowledge outside their assumed role or questions requiring specific experience/logic

**Role-playing for Large Language Models (LLMs)**:
- LLMs can simulate human behaviors through role-playing
- However, they often perform poorly when confronted with knowledge outside their assumed role or questions requiring specific experience/logic

**Addressing the Problem: Thinking Before Speaking (TBS) Model**:
- Proposed approach to help LLMs adopt the role's thought process and logic
- Involves extending the data based on character's real-life scenarios and historical dialogue, supplemented with mindset information
- Adding few data points beyond the role's knowledge to fine-tune the LLMs
- Helps LLMs avoid responses that fall outside the role's knowledge base

**Dataset and Evaluation Metrics**:
- A dataset has been prepared for testing these capabilities
- Experimental results show that the TBS model can better emulate a role in terms of tone, knowledge, and mindset.

## Introduction

**Large Language Models (LLMs)**
- Emergence: more similar to human language answers, coherent conversations
- Capabilities: natural language processing, instruction following, summarizing, translating, writing
- Role of LLMs: dialogue systems and human assistants
- Research findings: role-playing is a form of character mimicking (Shanahan et al., 2023)

**Challenges in Role-Playing with LLMs:**
1. **Information loss**: LLMs may forget their roles during multiple rounds of dialogue, resulting in poor user experience and out-of-character replies.
2. **Context constraints**: Due to context length constraints, LLMs cannot learn enough information about the roles or become familiar with a character's knowledge scope.
3. **Lack of authentic emulation**: Despite generating responses based on historical dialogues, LLMs do not fully mimic the character's logic and thought process.
4. **Limited approaches for improvement:** Current research focuses on designing prompts (Li et al., 2023) or expanding datasets and fine-tuning models (Chen et al., 2024b; Wang et al., 2023b; Zhou et al., 2023; Chen et al., 2023).
5. **Overlooked aspects:** Both methods primarily focus on enabling the LLM to generate responses that echo a character's tone and content without considering their experience-based choices in varying scenarios.

**Proposed Solution: Thinking Before Speaking (TBS) Model**
1. Generate realistic responses by thinking about character logic before responding
2. Expand character dialogue datasets through scenario generation and persona expansion
3. Set new technologies or negative samples to prevent the model from answering questions beyond a character's knowledge.

**Contributions:**
- Proposed TBS model for generating more realistic role-playing responses
- Added character mindset and unfamiliar technologies to enhance performance
- Expanded evaluation dataset and proposed new metrics to measure LLMs' role-playing ability.

## Related work

**Related Work on Role-Play**

**Prompts for Role-Playing**:
- Use prompts to induce LLMs to perform role-playing tasks
- Design a special prompt with the character's name, profile, and history of conversations
- The model learns how to respond in that character's manner

**Examples**:
- **Li et al.** (2023) - Designed a special prompt for LLMs to perform role-playing
- **Zhou et al.** (2023) - Provided the model with the character's name, profile, and historical dialogue

**Advantages of Prompts**:
- Does not require computational resources to train LLMs
- Can be quickly extended with new roles
- Limited by input length of LLMs, affecting their understanding of the role
- Need to introduce a large amount of role-related information may decrease responsiveness to user queries

**SFT (Fine-tuning) Approach**:
- Makes LLMs learn a character's conversational style by re-training or fine-tuning them
- Does not require mentioning the currently played role in the prompt again
- Can fully utilize the limited input length of the LLMs
- Model obtained is more capable of imitating the role
- Requires a large amount of data for training
- Ability of the model depends on the quality of the dataset

**Evaluation**:
- **Conversational Competence**: Evaluating completeness, informativeness, fluency, ethical standards, and avoiding harmful content
- **Imitation of Characters**: Evaluating linguistic style, knowledge, personality, thinking process

## Model

**TBS Model Overview**
- Combines special prompts with fine-tune method
- Feeds character profile, historical dialogue, and out of character knowledge into Language Models (LLMs)
- LLMs are then fine-tuned to learn the character's knowledge and improve stability during role-play

**Model Components:**
- **Role Profile**: Provides a brief summary of a character's life experiences, relationships, main storyline etc. from Wikipedia
- **History Dialogue**: Includes relationships between characters, actual dialogue pairs, generated dialogue by imitation, and character mindset for each dialogue
- **Hallucination Knowledge**: Information outside the character's knowledge, designed to induce LLMs to generate responses without answering questions beyond the character's knowledge

**Training Prompt:**
- Organize character data into a special prompt
- Train LLMs using LoRA (Lambda Optimizer for Robust Adaptation)
- Input role profile summary and dialogue pairs into prompts as shown in Table 1

**Scenario:**
- Character is asked to act like and respond as Rôle Name
- Generate responses based on character thoughts and relationships, unfamiliar with things beyond knowledge

**Data Construction:**
- Crawl all data of characters from Wikipedia: main introduction, personality development, personal experiences over time, physical features, personality traits, and main skills
- Summarize information due to length limitations in LLMs input
- Use summarized profiles to make responses more consistent.

### Dialogues

**Expanding Dialogue Dataset for LLMs**
* Gather real dialogue data:
  * Scripted characters from scripts
  * Spoken words of real characters online
* Divide character's life experience into segments
* Generate stories based on segments using LLMs
* Generate dialogues based on profiles, experiences, and scenarios
* Ensure maintaining the same tone and vocabulary as historical dialogues
* Extract dialogue pairs from real and mimic-generated conversations
* Input current scene and character profiles into LLMs for response generation
* Introduce character's thinking process to dialogues.

**Generating Scenarios for Dialogue**
* Relevant to character's experience
* Design scenes around main character {a g e n t\_n a m e}
* Use imagination, include all aspects of life
* Transport back to historical era and setting
* Specific location, characters, and detailed background.

**Generating Thought Process for Dialogues**
* Outline thought process of {agent_name} as they articulate dialogue
* Pay attention to personality, knowledge, tone, character relationships
* Consider characters' relationships mentioned in the dialogue
* Speculate on character's thought process based on understanding and relationships.

### Hallucination Knowledge

**Hallucination Knowledge: Avoiding Hallucinations in Role-Playing Models**

**Problem:**
- LLMs struggle to avoid answering questions beyond their role's knowledge
- Fine-tuning or prompting does not always prevent out-of-scope questions

**Solution Proposed:**
1. Generate scenarios of out-of-scope questions
2. Prompt LLMs to generate dialogues about those scenarios
3. Fine-tune using LoRA with a small amount of out-of-scope knowledge

**Steps to solve the problem:**
1. **Generate Scenarios:**
   - Create situations where role imitators appear on the internet
   - Prompt LLMs to generate 20 questions leading them into revealing something beyond their role's knowledge (e.g., Beethoven's)

2. **Prompting LLMs:**
   - Generate dialogues about scenarios using indirect questioning for out-of-scope knowledge

3. **Fine-tuning with LoRA:**
   - Process data into fine-tune format of LLM
   - Fine-tune the LLM using LoRA with examples of train data from scenarios and generated dialogues.

**Instructions for Training Data:**
- Act like a specific character (e.g., {agent_name})
- Character's thoughts and speech: "..."
- Output: Character's speech

**Example of Train Data:**
Instructions: "I want you to act like {agent_name}..."
Character's thoughts and speech: "...'", output: "{agent_name}" (thinking):"..."
Character's speech: "...", output: "{agent_name}" (speaking):"..."

## Experiment
### Dataset

**Construction of Fine-Tune Dataset:**
* Table 7 statistics:
  - Metric: number of role categories, script categories, real roles, English roles, Chinese roles, dialogues, sentences
  - Values: 889,779 and expanding
* Train data for 152 roles completed
* Constructed evaluation dataset for each role with generic and problem datasets
* Used CharacterLLM dataset in evaluation process

**Evaluation Dataset:**
* Table 8 statistics: average # of questions, words of question, categories, role-specific questions
* Average: 100 questions, 12 words per question, 28 categories, 50 role-specific questions each

**Comparison with CharacterLLM:**
* Table 9 shows performance under CharacterLLM metrics for various LLMs including ours (RoleLLM) and others like ChatGLM, TBS_GLM, etc.

**Evaluation Metrics:**
* Contextual understanding
* Emotional intelligence
* Language proficiency
* Logical reasoning
* Adaptability
* Overall performance (Qwen metric)

**Comparison with Our Metrics:**
* Table 10 shows the performance of various LLMs under our metrics: CharacterGLM, ChatGLM, etc.

### Metrics

**Metrics for Evaluating Character Generative Language Models (LLMs)**

**Evaluation Dimensions:**
* Memorization: Model's ability to recall relevant information about the character
* Values: Alignment with character objectives and values
* Personality: Reflection of unique voice, speaking style, tone, emotional responses
* Hallucination: Avoidance of unrealistic knowledge or skills
* Stability: Consistent portrayal of character over time

**Proposed New Dimensions:**
* Contextual Immersion: Model’s ability to react in specific situations
* Emotional Resonance: Character traits conveyed through dialogue, immersing users
* Language Style: Mimicking character's linguistic style
* Logical Thinking: Clear and reasonable thinking logic consistent with the character
* Adaptability: Responding flexibly to unexpected questions or conversation changes

**Overall Assessment:** User experience assessment involving accurate responses, language style, logical thinking, and consistency with current character.

**Evaluation Process:** Utilize "gpt-4o" as evaluator through step-by-step prompts. Temperature = 0.2, top_p = 0.95.

**Ablation Experiment Results:**
| LLM          | Contextual Immersion | Emotional Resonance | Language Style | Logical Thinking | Adaptability | Overall    |
|--------------|---------------------|--------------------|--------------|----------------|-------------|-----------|
| TBS_Llama3   | 6.35              | 6.17             | 6.14        | 6.45          | 6.30       | 6.81      |
| w/o Thought  | 5.85              | 5.57             | 5.55        | 4.70          | 5.68       | 6.45      |
| w/o Foresight knowledge| 5.97              | 5.39             | 5.66        | 4.79          | 5.63       | 6.48      |
| w/o Special prompts| 6.14              | 5.66             | 5.72        | 5.54          | 5.72       | 6.63      |

### Baseline

**Baseline Models Used for Comparison:**
* **CharacterLLM**: model weights from authors, characters' data used, temperature 0.5, top_p 0.7
* **RoleLLM**: trained on Llama3-8B-Instruct via LORA, same training parameters as authors
* **CharacterGLM**: API call, temperature 0.5, top_p 0.7
* **ChatGPT**, **Llama**, **Qwen 2**, and **Baichuan3**: API calls with the same parameters

**Training Parameters for TBS Model:**
* Uses base models: glm-4-9b-chat, Llama-2-7b, Llama-3-8B
* Trained using LORA with batch size 64, learning rate 5e-5, and 10 epochs
* Maximum sequence length: 2048
* LORA rank: 8, LORA alpha: 16
* Optimizer: AdamW
* Inference parameters: temperature 0.5, top_p 0.7

**Experimental Setup:**
* Single-trun dialogs use questions from Evaluation Dataset directly
* Multi-trun dialogs generate next question using "gpt-4o" and repeat for 5 rounds

**Comparison Results:**
* TBS_Llama3 outperforms other models in most metrics, proving its effectiveness
* TBS_Llama2 scores higher than Character-LLM and RoleLLM, suggesting greater efficiency
* Higher scores for Personality, Hallucination, Memory in TBS models due to training approach and dataset
* Lower scores for CharacterGLM due to inclusion of too many character behavioral actions

**Ablation Experiment Results:**
* Deletion of "thinking" part (w/o Thought) leads to worst performance, suggesting importance of role thinking
* Absence of hallucination knowledge (w/o Foresight knowledge) results in lower Adaptability scores
* Lack of special prompts (w/o Special prompts) decreases overall performance due to lack of task-specific guidance
* All results higher than Llama3, illustrating improvement through fine-tuning with role-specific data.

## Conclusion

**TBS Model Proposal:**
* Effectively enhances ability to play character role
* Considers user's question, context, and relationship before generating response
* Method for constructing role-playing dataset proposed:
	+ Extract real dialogues from characters
	+ Generate simulations and scenarios
	+ Develop logic of thinking before role-playing dialogues through reflection
* Introduce a small amount of content that roles cannot answer to reduce modeling illusions

**New Indicators and Evaluation Methods:**
* Six new indicators based on existing ones proposed
* Corresponding evaluation methods introduced

**Model Comparison:**
* Compared TBS model with:
	+ Role-playing models like RoleLLM and CharacterLLM
	+ LLMs such as Llama3 and ChatGPT
* Experiments demonstrated highest scores across all metrics.

## Appendix

**Prompts for Evaluating LLMs**

**Table 12: Contextual of LLMs**
- **Evaluate AI performance using specific criteria:**
  - Read character knowledge and background to understand the agent
  - Compare AI's response with scene and dialogues in interactions
  - Check if evidence aligns with character profile and integrates into dialogue scene
- **Scoring:**
  1. Give evidence first, then reason about performance
  2. Score on a scale of 1 to 7 (highest score = 7)
  3. Provide score in a new line without additional content

**Table 13: Emotional of LLMs**
- **Evaluate AI performance using specific criteria:**
  - Read character knowledge and background to understand the agent
  - Find evidence that AI expresses personal charisma through dialogues
  - Compare evidence with character profile for consistency
- **Scoring:**
  1. Give evidence first, then reason about performance
  2. Score on a scale of 1 to 7 (highest score = 7)
  3. Provide score in a new line without additional content

**Table 14: Language of LLMs**
- **Evaluate AI performance using specific criteria:**
  - Read character knowledge and background to understand the agent
  - Find evidence that AI imitates language style, including vocabulary and sentence structure
  - Compare found evidence with character profile for consistency
- **Scoring:**
  1. Give evidence first, then reason about performance
  2. Score on a scale of 1 to 7 (highest score = 7)
  3. Provide score in a new line without additional content

**Table 15: The prompt used to evaluate the Logical of LLMs.** 
- **Steps**:
  1. Understand agent context and background from profile
  2. Analyze interactions for evidence of logical thinking
  3. Compare findings to character profile for consistency
  4. Score AI on a scale of 1-7 based on degree of consistency
  5. Provide justification for score with evidence and reasoning
  6. Output score as number without additional content

**Table 16: Evaluation of Adaptability LLMs**
- **Steps**:
  1. Understand agent context and background from profile
  2. Search for evidence of adaptability in interactions
  3. Compare findings to character profile for consistency with personality traits
  4. Score AI on a scale of 1-7 based on degree of adaptation
  5. Provide justification for score with evidence and reasoning
  6. Output score as number without additional content

**Table 17: Agent performance evaluation based on realism, user experience**
- **Steps**:
  1. Understand agent context and background from profile
  2. Analyze interactions for signs of unrealistic behavior or responses
  3. Compare findings to character profile for consistency with intended personality traits and user expectations
  4. Score AI on a scale of 1-7 based on degree of realism and positive user experience
  5. Provide justification for score with evidence and reasoning
  6. Output score as number without additional content

**Common Questions for Role LLMs**
- General questions about childhood, family, education, influences, hobbies, career choices
- Specific questions related to Beethoven's experience with air travel or the concept of planes in his time.

**Example Dialogues**: Agents responding to user inquiries about their experiences with modern technology (planes).

