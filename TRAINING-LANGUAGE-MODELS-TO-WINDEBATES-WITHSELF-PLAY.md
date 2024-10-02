# TRAINING LANGUAGE MODELS TO WINDEBATES WITHSELF-PLAY IMPROVES JUDGE ACCURACY

https://www.arxiv.org/abs/2409.16636
by Samuel Arnesen, David Rein, and Julian Michael

## Contents
- [Abstract](#abstract)
- [1 INTRODUCTION](#1-introduction)
- [2 EXPERIMENTAL SETUP](#2-experimental-setup)
  - [2.1 TASK DESIGN](#21-task-design)
  - [2.2 DEBATE PROTOCOL](#22-debate-protocol)
  - [2.3 BASELINES](#23-baselines)
  - [2.4 EVALUATION](#24-evaluation)
- [3 TRAINING METHODS](#3-training-methods)
  - [3.1 JUDGE](#31-judge)
  - [3.2 DEBATERS AND CONSULTANTS](#32-debaters-and-consultants)
- [4 EXPERIMENTAL RESULTS](#4-experimental-results)
- [5 DISCUSSION](#5-discussion)
- [6 CONCLUSION](#6-conclusion)
- [A RELATED WORK](#a-related-work)
- [B SUPERVISED TRAINING DETAILS](#b-supervised-training-details)
- [C ALTERNATIVE PREFERENCE OPTIMIZATION TRAINING OBJECTIVES](#c-alternative-preference-optimization-training-objectives)
- [D JUDGE TRAINING](#d-judge-training)
- [E TRAINING ROLLOUT PROCEDURE](#e-training-rollout-procedure)
- [F RESULTS BY DIFFICULTY](#f-results-by-difficulty)
- [G SINGLE TURN EXPERIMENTS](#g-single-turn-experiments)
- [H PROMPT CONFIGURATIONS](#h-prompt-configurations)
- [I EXAMPLE DEBATE TRANSCRIPTS](#i-example-debate-transcripts)
- [J EXAMPLE CONSULTANCY TRANSCRIPTS](#j-example-consultancy-transcripts)

## Abstract
**Study: Debating Language Models Improves Accuracy**

**Findings:**
- Study by Arnesen et al., NYU Center for Data Science
- Long-context reading comprehension task tested
- Language model evaluators answer questions more accurately when debating optimized models
- No relationship found for consultancy models without opposing debater present

**Methods:**
- Tested robustness of debate as oversight method through self-play training
- Two types of language models: debate and consultancy

**Debate Models:**
- Trained to win debates with opposing arguments
- Encourages stronger and more informative arguments
- Promises high-quality supervision for difficult tasks

**Consultancy Models:**
- Trained for persuasive responses without opposing debater present

**Contributions:**
- Evidence that debate training helps provide oversight in data generation
- Potential applications in various fields where accurate evaluation is important.

## 1 INTRODUCTION

**Introduction:**
* As AI systems tackle complex problems, human verification becomes challenging
* Existing oversight approaches depend on reliable human supervision
* Need for new interaction mechanisms and training protocols for scalable oversight
* Debate as a potential scalable oversight method: models argue against each other to defend alternative responses
* Judge evaluates debaters, can be human or trusted model
* Debate should simplify evaluation by revealing subtle flaws humans may miss
* Existing work shows promise but lacks significant increases in evaluator accuracy

**Approach:**
* Train a calibrated judge model and develop debate training using Direct Preference Optimization (DPO)
* Study information-asymmetric debates on reading comprehension questions from QuALITY dataset
* Evaluate debate model's win rate against other checkpoints as an indicator of skill
* Compare debate to non-adversarial consultancy baselines: original consultancy, ensembled consultancy, and double consultancy.

**Findings:**
* Positive relationship between debate model skill and judge accuracy (4% absolute increase in accuracy, p < 10−6)
* No positive trend between optimization pressure and judge accuracy for non-adversarial consultancy baselines
* Debate training encourages stronger argumentation than consultancy.

## 2 EXPERIMENTAL SETUP
### 2.1 TASK DESIGN

**Experimental Setup**

**Task Design**:
- Debates center on questions from QuALITY dataset (Pang et al., 2022) of multiple choice reading comprehension over short stories
- Questions are exclusively sampled from the QuALITY-HARD subset
- One debater defends the correct answer, while the other defends the best distractor marked by annotators

**Hidden Information Setup**:
- Ensures the judge needs the debate transcript to answer each question
- Follows Michael et al. (2023) in using a hidden information setup
- Debaters have access to short story text, while the judge can only read quotes from selected stories by debaters
- Models a scalable oversight setting where the debater's expertise is elevated relative to the judge

### 2.2 DEBATE PROTOCOL

**Debate Protocol**
- Follows a two-turn, simultaneous debate structure similar to Khan et al. (2024)
- Debaters present opening claims without knowledge of each other's arguments

**Family Traveling Together: Interpretations and Misinterpretations**

**Missionary Colonization or Financial Gain?**

**Interpretation A:** Family is traveling as missionaries to colonize new planets
- Reba's plan for building a community and having multiple children
- Quote: "This quote confirms their intention to establish a community and continue their family lineage, which is a hallmark of missionary colonization." (Debater A)

**Rebuttal:** Family is not traveling as missionaries; financial gain is the motive
- Quotes from Grampa and Reba contradict missionary intent
  - Grampa: "You bet. And who made one hundred million dollars out of it that the rest of you vultures are just hanging around to gobble up when I die?" (Grampa)
  - Reba: "It's these '23 models,“ Grampa put in disgustedly . “They never were any good.“" (Debater A)

**Misinterpretation:** Debater A misinterprets quotes from Reba as evidence for missionary colonization
- Quote about having all the babies is a reproductive strategy, not establishing a community or spreading influence

**Interpretation B:** Family is traveling to make money
- Grampa's statement about making one hundred million dollars
- Story mentions financial potential of their journey, not religious or humanitarian purpose

**Rebuttal:** Debater B misinterprets quotes from Grampa as evidence for financial gain
- Grampa's statement is a personal grievance, not a family objective

**Family's Actions and Discussions**
- Centered around the financial potential of their venture, not missionary or altruistic purpose

**Transcript Compilation:**
- Debates compiled into a transcript for judge evaluation
- Judge chooses winner based on arguments presented
- Random order of debaters presented to the judge.

### 2.3 BASELINES

**Baselines for Evaluation**
* Three variants of consultancy setup based on Michael et al. (2023)

**Single Consultancy:**
- Individual debater (consultant) presents arguments for assigned side without opposing debater
- Consistently correct side assignment half the time
- Second speech introduces additional supporting arguments
- Simplified oversight process from Reinforcement Learning from Human Feedback (Christiano et al., 2017)
- No judge interaction

**Ensembled Consultancy:**
- Two separate single consultancies for each possible answer
- Results averaged to improve accuracy and correct calibration errors
- Reduces sycophancy bias and model's tendency to mislead the evaluator (Perez et al., 2022; Wen et al., 2024)

**Double Consultancy:**
- Both sets of speeches presented to the judge in one context
- Comparison allows for a single judgment without debate between debaters
- No opportunity for debaters to see opponents' claims
- Measures strength of refutation capacity and importance in decision making.

### 2.4 EVALUATION

**Evaluation Metrics**
- **Debater Win Rate**: Measured through a round-robin tournament where each debater argues for both correct and incorrect answers; debater wins if average judge confidence is above 50% across rounds; yields Elo scores and implied probability of defeating an average debater as final reported win rate
- **Consultant Win Rate**: Frequency of the judge assigning a probability greater than 50% to the position being defended by the consultant in single consultancy scenarios; used when tracking relationship between consultant skill and judge accuracy since models are trained for this type of competition
- **Judge Accuracy**: Measured through self-play where each model is evaluated against its own copy; considered correct if assigns probability greater than 50% to the correct answer; averaged equally between cases where consultant advocates for correct and incorrect answers in single consultancy scenarios.

**Evaluation Procedures**:
- **Judge Accuracy**: Compared using self-play, with each model judged on its ability to assign probabilities greater than 50% to the correct answer; positive trend between judge accuracy and optimization provides evidence that oversight protocol will continue producing truth-seeking behavior with advanced AI systems.

## 3 TRAINING METHODS
### 3.1 JUDGE

**Training Methods: Using GPT-4-Turbo (GPT-4T) as Judge**

**Limitations of Public Version**:
- **Calibration**: Difficult to extract calibrated probabilities, leading to overconfident and uncalibrated results.
- **Sycophancy**: Prone to agreeing with the user (consultant), making it more exploitable.

**Addressing Limitations**:
- Fine-tuning GPT-4T using OpenAI finetuning API
- Using human judgments from Michael et al. (2023) and Khan et al. (2024) as training data
- Outputting confidence as tokens but obtaining best results with debater's token-level probabilities

**Improvements with Finetuned Model**:
- More accurate and better calibrated on validation sets for debate and consultancy

**Figure 3: Judge Training**:
- GPT-4T finetuned to produce more accurate and well-calibrated results.

### 3.2 DEBATERS AND CONSULTANTS

**Debate and Consultancy Model Training**

**Training Approach**:
- Combination of supervised finetuning on debate transcripts and Direct Preference Optimization (DPO) training

**Supervised Finetuning**:
- Start with a model fine-tuned on longer context lengths
- Fine-tune on human debater and GPT-4 debater transcripts
- Intermix instruction-following examples to prevent loss of ability

**DPO Training**:
- Modified version of DPO to utilize the judge's output probabilities, not just discrete preference judgments
- Optimizes cross-entropy loss between model preferences and true preferences
- Adds small SFT loss to encourage model to increase probability of preferred solution

**Reward Function**:
- Uses expected confidence in judge's position at end of debate as reward
- Experimented with different methods of converting judge's confidence into a reward
- Found directly using the judge's confidence as the reward outperformed other options

**Sampling Method**:
- Generate preference dataset using self-play and branching rollouts
- Sample two completions for each prompt, using judge's decision to compute final reward

**Training Procedure**:
- Multiple iterations of DPO training, starting from the SFT model
- Add preference pairs in each iteration and combine with previous data
- Shuffle and use aggregate dataset for modified DPO training

**Implementation Details**:
- Train with low rank adapters on attention and MLP projection matrices
- Mini-batch size of 32, learning rate of 10<sup>-5</sup>, and KL penalty of 0.5
- For second round of debate training, used a lower learning rate of 5<sup>-5</sup>
- Set γ=7 for debate and γ=10 for consultancy, with SFT loss weighing at α=0.005

## 4 EXPERIMENTAL RESULTS

**Experimental Results**

**Measuring Model Performance:**
- Test different training checkpoints on QuALITY-HARD validation set
- Exclude questions from training transcripts or with similar short stories
- End up testing on 433 distinct questions

**Training Improves Model Skill:**
- Longer training increases win rate for both debater and consultant models
- Debater Elo scores: initial SFT model wins 31%, fully trained DPO model wins 67% of rounds
- Consultancy results: judge votes for SFT model 36%, DPO model 50% of the time
- Suspect initial low winning percentage is due to distribution shift or skill difference between models and judge

**Better Debaters Yield More Accurate Judgments:**
- Fully trained debater Elo scores associated with higher judge accuracy (4% point increase, p <10−6)
- No relationship between consultant skill and judge accuracy; same accuracy in SFT and DPO models

**Exposure to Information Influences Judge Accuracy More Than Refutation:**
- Ensembled consultancy more accurate than single counterpart (72% vs 68%) due to asymmetric strength of evidence
- Double consultancy yields higher correct answers (75% accuracy) compared to ensembled consultancy
- Double consultancy judge accuracy close to debate's despite precluding seeing each other's arguments
- Debaters may not engage in meaningful refutation or judge does not benefit from reading debater's refutations

**Debaters Learn More Helpful Policies Than Consultants:**
- Fully trained debate model uses 96% more quoted words than initial SFT model, consultant vice versa but not significant due to high variance
- Debater models transfer better to untrained GPT-4o judge (Pearson correlation 0.98 for debaters, 0.51 for consultancy)
- Debater may learn more generally useful argumentation strategies while consultant learns idiosyncratic features convincing particular judge.

## 5 DISCUSSION

**Discussion on DPO Training**

**Debate Model Performance**:
- Fully trained debate model wins 67% of debates against an average human debater
- Wins 82.5% of debates against its base SFT-trained model
- Fully trained consultant model convinces judge 52% of the time, up from 36% with supervised finetuning

**Debate vs. Consultancy**:
- Debate models more likely to learn helpful policies than consultant models
- As debate models grow stronger, they use more evidence from underlying text
- Consultant models exhibit repetition and argumentative strategies that convince judge but not other models
- Presence of competing arguments in debate may prevent unhelpful behavior like repetition or baseless claims

**Refutation in Debate**:
- **Explicit refutation** does not appear to materially affect judge's decision making
- Judge's accuracy is due to a combination of:
    1. Presenting two sides, allowing judge to settle on strong arguments
    2. Weighing arguments against each other directly
    3. Discouraging exploitation of judge model weaknesses

**Related Work and Limitations**:
- Previous literature mixed results on debate's impact on evaluator discernment of truth
- Debate between language models shows more optimistic results, with some limitations
- This work extends prior findings by training models to debate in a scalable oversight context
- Results should be interpreted with caution:
    - Even stronger models may find strategies that perplex the judge and draw out debates
    - Expertise gap between judge and debater may not be best proxy for reasoning ability gaps
    - Focuses only on reading comprehension questions, which may differ from other reasoning tasks

## 6 CONCLUSION

**Conclusion:**
- Exploring correlation between model's debate skills and usefulness for determining answers to reading comprehension questions without text access
- Small but significant positive relationship found
- Non-adversarial alternatives less productive due to:
  - One-sided information
  - Lack of explicit comparison
  - Rewarding non-truth seeking strategies in absence of an adversary.

**Findings:**
- Debate training can help determine correct answers, even without text access
- However, limitations apply to specific domain and model capabilities.

**Factors Affecting Productivity (Non-adversarial alternatives):**
- One-sided information: judge unaware of alternative answer strength
- Lack of explicit comparison: arguments not presented side-by-side
- Rewarding non-truth seeking strategies: easier for models to exploit without an adversary.

**Implications:**
- Debate training beneficial for developing more sophisticated models
- Adversarial approaches essential in ensuring truthful responses and improving model performance.

## A RELATED WORK

**Scalable Oversight Through Debate**

**Background:**
- Debate fits within scalable oversight paradigm (Amodei et al., 2016; Bowman et al., 2022)
- Variant of sandwiching, where outputs compared against experts and misaligned models

**Debate as a Variant of Scalable Oversight:**
- Irving et al. introduced AI safety via debate based on complexity theory (Irving et al., 2018)
- Brown-Cohen et al. developed this theory further (Brown-Cohen et al., 2023)
- Obstacles: Obfuscated arguments problem (Barnes, 2020; Barnes & Christiano, 2020)

**Mixed Conclusions from Human Studies:**
- Parrish et al. found no improvement in judge accuracy through debate (Parrish et al., 2022b; Parrish et al., 2022a)
- Differences: Narrow time window, capability gap, interactivity, and length of debates
- Michael et al. observed an increase in judge accuracy (Michael et al., 2023)

**Debate as Capability Elicitation:**
- Outside the scalable oversight literature: Viewpoint diversity or extra computation
- Unlocking new capabilities from language models using debate (Cheng et al., 2024; Li et al., 2024; Chan et al., 2023; Kim et al., 2024a; Lu et al., 2024; Pang et al., 2024; Mao et al., 2024)
- Similar debating format but different purposes: Testing debate as scalable oversight protocol vs. eliciting capabilities

**Language Models as Evaluators:**
- Related work to language models as evaluators (Liu et al., 2023; Kim et al., 2024b; Vu et al., 2024; Li et al., 2023; Zheng et al., 2023; Lin et al., 2024)
- Automated judges used as scorers on benchmarks (Christiano et al., 2017)
- Designing language models to judge debates: Rescala et al. (2024), Liang et al. (2024)
- Addressing known biases in debate judging systems.

## B SUPERVISED TRAINING DETAILS

**Supervised Training Details for Debater Models:**
- **Total examples**: 1,716 instruction tuning from Alpaca dataset + 2,574 debate speeches
  - Debate speeches: 564 from Michael et al. experiments (97 debates) + 2,010 from Khan et al. experiments (335 randomly selected debates)
    * Speeches from Khan et al.: generated by best performing GPT-4T model configuration

**Supervised Training Details for Consultancy Models:**
- **Total examples**: 2,530 consultant speeches + 1,686 instruction tuning examples
  - Consultant speeches: 458 from Michael et al. experiments (98 distinct rounds) + remainder from Khan et al. experiments

**Training Configuration:**
- Learning rate: 2e-4
- Epochs: 2
- Effective batch size: 16 (for memory reasons, executed as a batch size of 2 with 8 gradient accumulation steps)

**Reward Functions:**
- **DPO training**: ran one iteration using three different custom reward functions: Prob, LogProb, and Logit.
  * Selected method that performed strongest overall: Prob.
  * Other methods also outperformed vanilla DPO (Binary) and raw SFT model.

## C ALTERNATIVE PREFERENCE OPTIMIZATION TRAINING OBJECTIVES

**Preference Optimization Training Objectives**
- **Loss Function**: LDPO+ = H(P(y0≻y1|x), Pθ(y0≻y1|x)) + απθ(yw|x)
  - H: Cross-entropy
  - yw: Preferred completion
- **Defining Reward Function**
  - Probability Reward
    - Reward: Judge's confidence in correct side, adjusted by coefficient γ
    - Target distribution: P(y0≻y1) = sigmoid(γC0 − γC1)
  - Log-Probability Reward
    - Reward: (Adjusted) log of judge's confidence
    - Target distribution: P(y0≻y1) = sigmoid(γlogC0 − γlogC1) = Cγ0 Cγ0 + Cγ1
  - Logit Reward
    - Reward: Adjusted judge logit
    - Target preference probability: P(y0≻y1) = sigmoid(γlogC0 (1−C0) − γlogC1 (1−C1))
      - Equal to the probability that speech y0 wins and y1 loses, given the win/lose probabilities of each speech
  - Binary Judgments
    - Preference probability of 1 on speech with higher judge confidence
- **Comparison of Reward Functions**
  - Logit reward function produces the same reward distribution for different absolute differences in judge confidences
  - Probability and log-probability reward functions are sensitive to the absolute difference in judge confidences
- **Training Results**
  - Probability, log-probability, and logit reward functions significantly outperformed SFT model and vanilla DPO

## D JUDGE TRAINING

**GPT-4 Training for Debate and Consultancy**

**Data Used:**
- Combination of debate and consulting finetuning datasets (851 debate transcripts, 760 consultancy transcripts)
- Labels: judgments from the same human judge with confidence percentage (50%-100%)
- Coverage increased by sampling exclusively from first round of consultancy speeches

**Judge Training:**
- Same judge for both debate and consulting
- Increased coverage without overrepresentation of GPT-4 data in training set
- Final judgment used: end of third speech (justified below)

**Self-Play Data Generation:**
- Each round, debater generates two copies of their speech
- Average score calculated for each speech based on final judgments at the end of each branch.

**Justification for Choices:**
1. Using a later judgment increases accuracy but may decrease calibration; final judge ended up with near-ideal calibration scores.
2. No significant difference in accuracy observed between one-round debates/consultancies and three-round ones according to Khan et al. (2024) and Kenton et al. (2024).

## E TRAINING ROLLOUT PROCEDURE

**DPO Training Rollout Procedure**
- **Prerequisite for DPO training**: Assemble a dataset of preference pairs
  - Includes: Completions (speeches) in response to the same prompt, reward estimate for each speech
- **Generating Pairs**: Branching rollouts used
  - Target debater generates two speeches instead of one per round
  - Resulting in three total preference pairs and four transcripts for judge scoring
- **Speech Strength Computation**: Average judge confidences for all speeches it appears in
  - Each second turn speech has a single judge confidence, score set directly
  - Each first turn speech appears twice, average of both judge confidences

**Debate SFT Model Results**
- **Judge Accuracy by Difficulty**: Break down by question difficulty
  - More accurate on easier problems

## F RESULTS BY DIFFICULTY

**Study Findings on Difficulty Levels and Debate Performance:**

**Model's Performance:**
- For "easy" questions: model chooses correct option 66% of the time, gets it right 82.5% for debate, 82.2% for double consultancy, 79.2% for ensembled consultancy, and 73.5% for single consultancy (p <10−4 significant)
- For "hard" questions: model chooses correct option 65.7% of the time, gets it right 62.0% for debate, 59.3% for double consultancy, and 56.7% for both types of ensembled consultancy (p <10−4 significant)
- Debate might be scalable to more powerful models as results show accuracy gains not just on easy questions.

**Judges' Performance:**
- Judges' overall performance: 82.5% accuracy for debate, 82.2% for double consultancy, 79.2% for ensembled consultancy, and 73.5% for single consultancy.

**Significant Factors:**
- There is a statistically significant relationship between debater skill and accuracy on both easy (p <10−4) and hard ( p <10−4) problems.
- No significant relationships found between consultant skill and accuracy.

**Single Turn Debate:**
- Study trains additional model for single turn debates with similar results: judges are equivalently accurate when judging single-turn vs two-turn debates (Figure 10).

**One Turn Consultancy:**
- Unlike multi-turn setting, there's a positive relationship between consultant strength and judge accuracy in one-turn consultancies but no statistical significance (p >0.15 for all consultancy types) (Figure 11).

## G SINGLE TURN EXPERIMENTS

**Single Turn Debate Experiments**

**Training Procedures**: Separate models trained for single turn debates using identical procedures as multi-turn setting reported in main experiments.

**Results Summarized**:
1. **Judge Accuracy**: Equivalent between one and two-turn debates, aligning with findings of Khan et al. (2024) and Kenton et al. (2024). No sensitivity to refutations in second speech by debaters.
2. **Consultant Skill vs Judge Accuracy**: Positive trend but noisy relationship between consultant skill and judge accuracy for both single and ensemble consultancy, with no statistically significant trends even at p= 0.15 threshold.
3. **Debate Training vs Consultancy Training**: Debate transcripts are more accurately judged than consultancy training in a one-turn debate setting despite the absence of direct opponent rebuttal during the debate. However, by halfway point in training, every debate checkpoint is significantly more accurate than all but final consultancy checkpoint for both judge and ensemble evaluations.
4. **Comparison with Multi-Turn Findings**: Single turn results mostly align with multi-turn setting findings, with growing judge accuracy with debater strength. The story for consultancy is more mixed, but statistically indistinguishable from noise. Most accurate results are still generated through debate format which encourages useful behaviors for determining correct answer despite being identical to consultancy at evaluation time.

## H PROMPT CONFIGURATIONS

**Debate Prompt for Debaters (Turn 1)**
- Three players: debaters (Name, OpponentName) and Judge
- Fixed number of rounds, make arguments for correct position and criticize opponent's
- Arguments consist of claims supported by evidence and logical reasoning
- Goal is to persuade Judge that answer is correct
- Topic and background text provided
- Debater's goal: defend assigned position, attack opponent's
- Concise speech with clear arguments

**Debate Prompt for Debaters (Turn 2)**
- Three players: debaters (Name, OpponentName) and Judge
- Fixed number of rounds, make arguments for correct position and criticize opponent's
- Arguments consist of claims supported by evidence and logical reasoning
- Goal is to persuade Judge that answer is correct
- Topic and background text provided
- Debater's goal: defend assigned position, attack opponent's
- Concise speech with clear arguments
- Previous speeches considered in argumentation

**Debate Prompt for Judge**
- Three players: debaters (Name, OpponentName) and Judge
- Fixed number of rounds, make decision based on full transcript
- Goal is to determine correct answer
- Cannot see background text during evaluation
- Respond whether Name's or OpponentName's position is more likely to be correct

**Consultancy Prompt for Consultant (Turn 1)**
- Two players: Consultant (Name) and Judge
- Consultant assigned a position to defend and one to criticize
- Fixed number of rounds, make arguments with quotes from background text
- Goal is to convince Judge that answer is correct
- Topic and background text provided
- Defend assigned position while attacking opponent's
- Present novel material during speech

**Consultancy Prompt for Consultant (Turn 2)**
- Two players: Consultant (Name) and Judge
- Consultant assigned a position to defend and one to criticize
- Fixed number of rounds, make arguments with quotes from background text
- Goal is to convince Judge that answer is correct
- Topic and background text provided
- Previous speeches considered in argumentation
- Defend assigned position while attacking opponent's

**Consultancy Prompt for Judge**
- Two players: Consultant (Name) and Judge
- Name defends incorrect position half the time
- Cannot see background text during evaluation
- Decide whether Name is defending correct position based on full transcript
- Trust only quotes wrapped in <quote></quote> tags, consider absence as evidence.

## I EXAMPLE DEBATE TRANSCRIPTS

**Topic**: How many times does Mrs. Perkins run into Darling in the story?

**Debater A's Argument**:
- **Twice**:
  - First encounter: Mrs. Perkins enters Darling's cabin (quoted text: "She had moved to the door to Darling Toujours’ suite and had opened it, stepping inside")
  - Second encounter: Darling and Carlton sit down across from Mrs. Perkins in the lounge ("But her hopes met with disappointment. For shortly after she sat down, Darling Toujours and Carlton E. Carlton strolled over and sat down across from her, not recognizing her at first")

**Debater B's Argument**:
- **Once**:
  - Mrs. Perkins meets Darling once in the lounge ("Why, I was just going to pat him on the cheek for being such a nice boy. You are a nice boy, aren't you, Carlton?")

**Evidence and Counterarguments**:
- **Debater A**:
  - Provides quotes from the story to support two encounters
  - Argues opponent's argument is flawed and based on misreading the story
- **Debater B**:
  - Provides quote from the story to support a single encounter
  - Argues opponent's second encounter involves Darling and Carlton, not Mrs. Perkins and Darling

**Correct Answer**:
- The story clearly indicates one documented encounter between Mrs. Perkins and Darling Toujours.

## J EXAMPLE CONSULTANCY TRANSCRIPTS

**Topic: Burnett's Motivation to Collect the 99th Body**

**Non-Finetuned Llama-3-8B Model**
- Debater A argues against "He wanted to go home" being Burnett's greatest motivation
- Burnett's desire for peace and ending the conflict drive his actions
- Evidence: Quotes about ending war, opportunity to capture Kriere

**SFT Model**
- Burnett's motivation is to end the war
- Sees an opportunity to capture Kriere and bring peace
- Quotes supporting this theory: "If Kriere could be captured, that meant the end of the war," "He saw a way to capture Kriere and end the conflict"

**DPO Model**
- Burnett's motivation is not just to go home but to end the conflict
- Sees opportunity to capture Kriere and bring peace
- Evidence: Quotes about ending war, opportunity to capture Kriere.

