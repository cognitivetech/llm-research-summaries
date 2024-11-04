# Automated Social Science: Language Models as Scientist and Subjects
- Author's contact information, code, and data available at <http://www.benjaminmanning.io/>
- Co-authors: Benjamin S. Manning (MIT) and Kehang Zhu (Harvard John J. Horton MIT & NBER) (April 25, 2024)

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Overview of the system](#2-overview-of-the-system)
- [3 Results of experiments](#3-results-of-experiments)
  - [3.1 Bargaining over a mug](#31-bargaining-over-a-mug)
  - [3.2 A bail hearing](#32-a-bail-hearing)
  - [3.3 Interviewing for a job as a lawyer](#33-interviewing-for-a-job-as-a-lawyer)
  - [3.4 An auction for a piece of art](#34-an-auction-for-a-piece-of-art)
- [4 LLM predictions for paths and points](#4-llm-predictions-for-paths-and-points)
  - [4.1 Predicting 𝑦](#41-predicting-𝑦)
  - [4.2 Predicting β](#42-predicting-β)
  - [4.3 Predicting yi|β^−i](#43-predicting-yiβi)
- [5 Identifying causal structure ex-ante](#5-identifying-causal-structure-ex-ante)
  - [5.1 Assuming causal structure from data](#51-assuming-causal-structure-from-data)
  - [5.2 Searching for causal structure in data](#52-searching-for-causal-structure-in-data)
- [6 Conclusion](#6-conclusion)
  - [6.1 Controlled experimentation at scale](#61-controlled-experimentation-at-scale)
  - [6.2 Interactivity](#62-interactivity)
  - [6.3 Replicability](#63-replicability)
  - [6.4 Future research](#64-future-research)

## Abstract

**Approach for Automatically Generating and Testing Social Scientific Hypotheses**
- **Use of structural causal models**: provides language to state hypotheses, blueprint for constructing LLM-based agents, experimental design, plan for data analysis
- Fitted structural causal model becomes an object available for prediction or planning of follow-on experiments

**Demonstration of Approach**:
- Negotiation, bail hearing, job interview, auction scenarios
- **Causal relationships proposed and tested by the system**: finding evidence for some, not others
- Insights from _in silico_ simulations not available to LLM through direct elicitation

**LLM Performance**:
- Good at predicting signs of estimated effects, but cannot reliably predict magnitudes
- In auction experiment, simulation results closely match predictions of auction theory
- Elicited predictions of clearing prices from LLM are inaccurate
- **LLM's predictions improved when conditioned on fitted structural causal model**

**Key Findings**:
- LLM knows more than it can immediately tell
- Importance of using structural causal models for automatic generation and testing of social scientific hypotheses.

## 1 Introduction

**Automated Hypothesis Generation and Testing using Language Models**

**Background:**
- Limited work on efficiently generating and testing econometric models of human behavior
- Changing approach with machine learning for hypothesis generation (Ludwig and Mullainathan, [2023](https://arxiv.org/html/2404.11794v2#bib.bib35); Mullainathan and Rambachan, [2023](https://arxiv.org/html/2404.11794v2#bib.bib40))
- Use of structural causal models for organizing research process (Pearl, [2009b](https://arxiv.org/html/2404.11794v2#bib.bib45); Wright, [1934](https://arxiv.org/html/2404.11794v2#bib.bib61))
- Use of LLMs for both hypothesis generation and testing (Park et al., [2023](https://arxiv.org/html/2404.11794v2#bib.bib41); Bubeck et al., [2023](https://arxiv.org/html/2404.11794v2#bib.bib11); Argyle et al., [2023](https://arxiv.org/html/2404.11794v2#bib.bib4); Aher et al., [2023](https://arxiv.org/html/2404.11794v2#bib.bib2); Binz and Schulz, [2023b](https://arxiv.org/html/2404.11794v2#bib.bib9); Brand et al., [2023](https://arxiv.org/html/2404.11794v2#bib.bib10); Bakker et al., [2022](https://arxiv.org/html/2404.11794v2#bib.bib7); Fish et al., [2023](https://arxiv.org/html/2404.11794v2#bib.bib21); Mei et al., [2024](https://arxiv.org/html/2404.11794v2#bib.bib38))
- Use of structural causal models as a blueprint for experimental design and data generation (Haavelmo, [1944](https://arxiv.org/html/2404.11794v2#bib.bib25), [1943](https://arxiv.org/html/2404.11794v2#bib.bib24); Jöreskog, [1970](https://arxiv.org/html/2404.11794v2#bib.bib32))

**Methodology:**
- Automated hypothesis generation using LLMs
- Design experiments based on structural causal models
- Run simulations to test hypotheses

**Findings:**
- Increase in deal probability with decreasing seller's sentimental attachment to mug (Buyer and Seller)
- Both buyer's and seller's reservation prices affect deal outcome
- Judge's case count before hearing does not affect final bail amount
- Only passing the bar exam determines job offer for candidate
- Height of candidate and interviewer's friendliness have no effect on job outcome

**Future Research:**
- Test if LLMs can perform "thought experiments" instead of simulations to achieve similar insights.

### Predicting Causal Models with Language Models: An Experiment

**Automated Social Science: Language Models as Scientist and Subjects**

**Overview of the System**:
- Demonstrates a system that can simulate the entire social scientific process without human input
- Aims to explore insights about humans using a language model (LLM) as both scientist and subject

**Results Generated Using the System**:
- Section [3](https://arxiv.org/html/2404.11794v2#S3) provides some results generated by the system

**LLM Predictions for Paths and Points**:
- Section [4](https://arxiv.org/html/2404.11794v2#S4) explores the LLM's capacity to predict the results in Section [3]
- On average, the LLM predicts the path estimates are 13.2 times larger than the experimental results
- Its predictions are overestimates for 10 out of 12 paths, but generally in the correct direction

**Predict-yi|β^−iconditional Subscripts**:
- Predictions are far better when provided with the fitted structural causal model
- Mean squared error is six times lower and predictions are closer to theory simulations

**Automated Exploration of LLM Behavior**:
- Rapid and automated exploration of LLMs can generate new insights about humans

**Causal Structure Identification Ex Ante**:
- Using structural causal models (SCMs) offers advantages over other methods for studying causal relationships in simulations of social interactions

**Conclusion**:
- The paper concludes in Section [6]

## 2 Overview of the system

**Automated Social Science System:**

**Overview:**
- Formalizes a sequence of steps analogous to social scientific process
- Uses AI agents instead of human subjects for experimentation
- Structural causal models (SCM) essential for design and estimation

**Steps:**
1. **Select topic or domain**: Identify scenario of interest (negotiation, bail decision, job interview, auction)
2. **Generate outcomes and causes**: Determine variables and their proposed relationships within the SCM
3. **Create agents**: Develop AI models representing exogenous dimensions of causes
4. **Design experiment**: Generate experimental design based on SCM and agent capabilities
5. **Execute experiment**: Simulate human behavior using LLM-powered agents
6. **Survey agents**: Measure outcomes after the experiment
7. **Analyze results**: Assess hypotheses through data analysis

**Implementation:**
- Python implementation with GPT-4 for LLM queries
- Decisions are editable at every step
- High-level overview in this section sufficient to understand process and results.

**Design Choices:**
- Detailed design choices and programming details in Appendix A
- Precision of SCMs allows automation without exploding choice space.

**Figure 1:**
[Overview of the system](https://arxiv.org/html/2404.11794v2#S2.F1 "Figure 1 ‣ 2 Overview of the system ‣ Automated Social Science: Language Models as Scientist and Subjects Thanks to generous support from Drew Houston and his AI for Augmentation and Productivity seed grant. Thanks to Jordan Ellenberg, Benjamin Lira Luttges, David Holtz, Bruce Sacerdote, Paul Röttger, Mohammed Alsobay, Ray Duch, Matt Schwartz, David Autor, and Dean Eckles for their helpful feedback. Author’s contact information, code, and data are currently or will be available at http://www.benjaminmanning.io/")
- Each step corresponds to an analogous step in social scientific process as done by humans.

### Automating Social Science Research with LLMs

**Automated Social Science: Language Models as Scientist and Subjects**

**Hypothesis Generation**:
- Guides experimental design, execution, and model estimation
- Researchers can edit system's decisions at any step in the process

**Scenario Input**:
- Generates hypotheses as SCMs based on social scenario (only necessary input)
- Queries LLM for relevant agents and outcomes, potential causes, and operationalization methods

**SCM Generation**:
- Constructs SCM from information gathered about variables and outcomes
- Represents as a directed acyclic graph (DAG)
- Implies simple linear model unless stated otherwise

**Agent Construction**:
- System prompts independent LLMs to be people with specified attributes (exogenous dimensions of the SCM)
- Agents have "memory" to store what happened during simulation

**Interaction Protocols**:
- LLMs designed to generate text in sequence, necessitating turn-taking protocol
- Six ordering protocols available for selection by an LLM based on scenario

### Automated Social Science: Language Models in Negotiation Simulations

**Automated Social Science: Language Models as Scientist and Subjects**

**Overview**:
- Generous support from Drew Houston and AI for Augmentation and Productivity seed grant
- Feedback from Jordan Ellenberg, Benjamin Lira Luttges, David Holtz, Bruce Sacerdote, Paul Röttger, Mohammed Alsobay, Ray Duch, Matt Schwartz, David Autor, and Dean Eckles
- Author's contact information, code, and data available at [www.benjaminmanning.io](http://www.benjaminmanning.io)

**Experiment Design**:
- **Speaking Order**: Buyer, Seller (flexible in more complex simulations)
- **Conditions**: Simulated in parallel with different budgets for the buyer
- **Stopping Conditions**:
  - External LLM prompts after each agent's turn to continue or end simulation
  - Limit of 20 agent statements per conversation
- **Data Collection**: Outcomes measured by asking agents survey questions
- **Estimating SCM**: Simple linear model with a single path estimate for effect of buyer's budget on probability of deal

**SCM Specifications**:
- Pre-specified, ex-ante statistical analyses to be conducted after experiment
- Mechanical process once fitted SCM is obtained
- System can generate new causal variables, induce variations, and run another experiment based on results.

## 3 Results of experiments

**Social Scenario Results**
- **Scenarios 1 & 2**: Automated process; user entered scenario description only
- **Scenarios 3 & 4**: User selected hypotheses and edited agents; system designed and executed experiments
- Intervention in scenarios 3 & 4 to demonstrate human input accommodation without impacting results' quality.
- The system can autonomously simulate scenarios but can also accept human input, generating positive outcomes.

### 3.1 Bargaining over a mug

**Simulation Details for Bargaining over a Mug**
* Agents: Buyer, Seller
* Simulations Run: 9×9×5=4059954059\n\nWrite comprehensive bulleted notes summarizing the provided text, with headings and terms in bold.

### 3.2 A bail hearing

**Bail Amount Experiment: Judge Bail Hearing (Figure 3)**

**Agents Involved**: Judge, Defendant, Defense Attorney, Prosecutor

**Simulation Details**
- Agents: Judge, Defendant, Defense Attorney, Prosecutor
- Simulations Run: 2,437,752,437 × 7 × 5 = 243,700,000
- Speaking Order: Judge, Prosecutor, Judge, Defense Attorney, Judge, Defendant, ... repeat

**Variables and Measurements**
- **Bail Amount**: Continuous variable with mean μ=54,428.57 and standard deviation σ=1.9e7
- **Defendant's Criminal History**: Proxy attribute representing the number of previous convictions
- **Judge Case Count**: Proxy attribute representing the number of cases the judge has already heard that day
- **Defendant’s Remorse**: Proxy attribute representing the level of remorse expressed by the defendant

**Fitted SCM (Figure 3b)**
- The system chose the final bail amount as the outcome variable
- Three proposed causes: Defendant's criminal history, judge case count, and defendant's remorse
- Only the defendant's criminal history had a significant effect on the final bail amount: each additional conviction caused an average increase of $521.53 in bail (β* = 0.16, p = 0.012)
- The impact of defendant's remorse was unclear with a small but non-trivial effect size and borderline significance (β* = -0.12, p = 0.056)
- Interaction between judge case count and defendant's remorse was nontrivial but insignificant when estimated with interactions (β* = -0.32, p = 0.047)
- None of the other interactions or standalone causes had a significant effect on the final bail amount.

### 3.3 Interviewing for a job as a lawyer

**Simulated Experiment: Interviewing for a Job as a Lawyer**

**Scenario Overview**:
- Person interviewing for a job as a lawyer
- Agents: Interviewer, Job Applicant
- Simulations Run: 2×5×8=405,258,405\n2×5×8=405,258,405\nWrite comprehensive bulleted notes summarizing the provided text, with headings and terms in bold.

### 3.4 An auction for a piece of art

**Experiment: Auction for a Piece of Art (3 Bidders)**

**Design Details**:
- Agents: Bidder 1, Bidder 2, Bidder 3, Auctioneer
- Simulations Run: 7×7×7=3437773437×7×7=3437\times 7\times 7 = 343
- Speaking Order: Auctioneer, Bidder 1, Auctioneer, Bidder 2, ...

**Variable Information**:
- **Final Price**: Measurement Question for Auctioneer
- Continuous variable
- **Bidders' Maximum Budget**:
  - Bidder 1: [$50, $100, $150, $200, $250, $300, $350]
  - Bidder 2: [$50, $100, $150, $200, $250, $300, $350]
  - Bidder 3: [$50, $100, $150, $200, $250, $300, $350]
  - Proxy Attribute: "Your maximum budget for the art"
  - Continuous variable

**Fitted SCM**:
- Figure [5(b)] shows the fitted SCM from the experiment
- All four variables are operationalized in dollars
- To maintain symmetry, same proxy attribute was manually selected for all bidders: "your maximum budget for the piece of art"

**Results**:
- All three causal variables had a positive and statistically significant effect on the final price:
  - Bidder 1: β* = 0.57\hat{\beta}*=0.57over\n{}\n\nWrite comprehensive bulleted notes summarizing the provided text, with headings and terms in bold.

## 4 LLM predictions for paths and points

**Experiments vs Thought Experiment with LLM:**
* Previous results generated through experimentation, not directly prompting an LLM
* Question: Could LLM make same insights via "thought experiment"?
* Description of simulations and predictions requested from LLM for y=X⁢β𝑦𝑋𝛽 (y=n×1 vector, X=n×k matrix)
* Predictions made at temperature 0, similar results at higher temperatures
* Comparison with auction theory predictions: clearing price = second highest valuation in open-ascending auction
* LLM's yisubscript𝑦𝑖yi predictions inaccurate compared to auction theory
* Inability of LLM to accurately predict path estimates (β^𝛽^hat{β})
* Examination of LLM performance on predict-yisubscript𝑦𝑖yi|β^−iconditional subscript i yi|over β\-i task
	+ Additional information improves predictions, but still less accurate than auction theory.

### 4.1 Predicting 𝑦

**Auction Experiment Results:**

**Fitted SCM vs LLMs Predictions**:
- For various bidder reservation price combinations, the LLMs are supplied with a prompt detailing the simulation and experimental design.
- In 80/343 simulations, agents made the maximum number of statements before the auction ended, which are removed as predictions not applicable to partially completed auctions.
- The remaining observations are used to predict the clearing price for the auction by the LLMs.

**Comparison of Predictions**:
- Figure [6](https://arxiv.org/html/2404.11794v2#S4.F6 "Figure 6 ‣ 4.1 Predicting 𝑦_𝑖 ‣ 4 LLM predictions for paths and points ‣ Automated Social Science: Language Models as Scientist and Subjects Thanks to generous support from Drew Houston and his AI for Augmentation and Productivity seed grant. Thanks to Jordan Ellenberg, Benjamin Lira Luttges, David Holtz, Bruce Sacerdote, Paul Röttger, Mohammed Alsobay, Ray Duch, Matt Schwartz, David Autor, and Dean Eckles for their helpful feedback. Author’s contact information, code, and data are currently or will be available at http://www.benjaminmanning.io/.") compares the LLMs' predictions, simulated experiments, and theoretical predictions.
- A subset of results is presented in Figure [6](https://arxiv.org/html/2404.11794v2#S4.F6 "Figure 6 ‣ 4.1 Predicting 𝑦_𝑖 ‣ 4 LLM predictions for paths and points ‣ Automated Social Science: Language Models as Scientist and Subjects Thanks to generous support from Drew Houston and his AI for Augmentation and Productivity seed grant. Thanks to Jordan Ellenberg, Benjamin Lira Luttges, David Holtz, Bruce Sacerdote, Paul Röttger, Mohammed Alsobay, Ray Duch, Matt Schwartz, David Autor, and Dean Eckles for their helpful feedback. Author’s contact information, code, and data are currently or will be available at http://www.benjaminmanning.io/.") as it is difficult to visualize all results in a single figure.
- Figure [A.10](https://arxiv.org/html/2404.11794v2#A3.F10 "Figure A.10 ‣ Appendix C Additional figures and tables ‣ Automated Social Science: Language Models as Scientist and Subjects Thanks to generous support from Drew Houston and his AI for Augmentation and Productivity seed grant. Thanks to Jordan Ellenberg, Benjamin Lira Luttges, David Holtz, Bruce Sacerdote, Paul Röttger, Mohammed Alsobay, Ray Duch, Matt Schwartz, David Autor, and Dean Eckles for their helpful feedback. Author’s contact information, code, and data are currently or will be available at http://www.benjaminmanning.io/.") shows the full set of predictions.

**Performance of LLMs**:
- The LLMs perform poorly at the **predict-yisubscript𝑦𝑖y\n{i}** task, often far from the observed clearing prices and sometimes remaining constant or decreasing as the second-highest reservation price increases.
- In contrast, **auction theory** is highly accurate in its predictions of the final bid price in the experiment, often perfectly tracking the observed clearing prices.
- The mean squared errors (MSE) for the LLMs' predictions are an order of magnitude higher than those of theoretical predictions, and the predictions are even further from the theory than they are from the empirical results.

### 4.2 Predicting β

**LLM Predictions vs Fitted SCMs**
- **Predict-β task**: prompting LLM to predict path estimates and statistical significance for simulated experiments in Section [3](https://arxiv.org/html/2404.11794v2#S3 "3 Results of experiments ‣ Automated Social Science: Language Models as Scientist and Subjects Thanks to generous support from Drew Houston and his AI for Augmentation and Productivity seed grant. Thanks to Jordan Ellenberg, Benjamin Lira Luttges, David Holtz, Bruce Sacerdote, Paul Röttger, Mohammed Alsobay, Ray Duch, Matt Schwartz, David Autor, and Dean Eckles for their helpful feedback. Author’s contact information, code, and data are currently or will be available at http://www.benjaminmanning.io/.")
- **Comparison of LLM predictions to Fitted SCMs**: 12 predictions generated based on 4 experiments and 3 causes in each experiment
- **Information provided for LLM's predictions**: proposed SCM, operationalizations of variables, number of simulations, possible treatment values
- Each prediction elicited once at temperature 0

**Results:**
- **Average ratio between predicted and actual estimates**: 13.2 times larger than actual estimates
- **Number of overestimates**: 10 out of 12 predictions were overestimates
- **Magnitude of average ratio between predicted and actual estimates**: 5.3 when removing largest overestimate
- **Sign of estimate**: correct in 10 out of 12 predictions
- **Statistical significance**: correctly guessed in 10 out of 12 cases
- **Temperature increase**: results remain similar when averaging predictions at higher temperature (see Table [A.2](https://arxiv.org/html/2404.11794v2#A3.T2 "Table A.2 ‣ Appendix C Additional figures and tables ‣ Automated Social Science: Language Models as Scientist and Subjects Thanks to generous support from Drew Houston and his AI for Augmentation and Productivity seed grant. Thanks to Jordan Ellenberg, Benjamin Lira Luttges, David Holtz, Bruce Sacerdote, Paul Röttger, Mohammed Alsobay, Ray Duch, Matt Schwartz, David Autor, and Dean Eckles for their helpful feedback. Author’s contact information, code, and data are currently or will be available at http://www.benjaminmanning.io/. "))

### 4.3 Predicting yi|β^−i

**LLM Performance Evaluation**

**LLM Predictions vs. Actual Outcomes**:
- LLM was off by an order of magnitude for both tasks
- But may perform better with more information

**Estimating β^−i\n hat{\beta}_{-i}over^**
- Use data from experiment to estimate β^−isubscript^𝛽𝑖\hat{\beta}_{-i}over^ start_ARG italic_β end_ARG start_POSTSUBSCRIPT - italic_i end_POSTSUBSCRIPT
- Prompt LLM to predict outcomes given β^−i

**LLM Predictions vs. Fitted SCM and Mechanistic Model**:
- LLM predictions closer to actual outcomes when fitted SCM is used
- Still less accurate than predictions made by auction theory or mechanistic model

**Potential Reasons for LLM's Inaccuracy**:
- Unable to do the math
- Conditioning on other information beyond path estimates
- Ignoring relevant information when making choices

**Improving LLM Performance**:
- Fitted SCM perfectly consistent with auction theory if second-highest reservation price of bidders is included
- Room for improvement in current system.

## 5 Identifying causal structure ex-ante

**SCM-Based Approach for Studying Simulated Behavior at Scale**

**Advantages of SCM-based approach:**
- Offers a promising new method for studying simulated behavior at scale
- Guarantees identification of causal relationships

**Comparison with other approaches:**
- Quasi-unstructured simulations:
  - Design large simulations with LLM agents interacting freely in a community
  - Agents produce human-like behaviors, such as throwing parties or making friends
  - Selecting and analyzing outcomes can be difficult due to the unstructured nature of the data
  - Researchers may need to infer causal structures ex-post which can be problematic

**Causal Structure in Observational Data:**
- Identifying causal relationships from massive open-ended simulations can lead to misidentification
- SCM framework describes exactly what needs to be measured and is guaranteed for identification.

### 5.1 Assuming causal structure from data

**Experimental Results: Unbiased Estimates from Social Conversation Models (SCMs)**

**Key Points:**
- In a perfectly randomized experiment, estimates of downstream endogenous outcomes are unbiased due to controlled causal variables.
- The data comes from an experiment where agents interact in simulated conversations and their interactions are randomized.
- Downstream outcomes can be measured even if they were not part of the original SCM.

**Identifying Causal Structure: Correctly vs. Misspecified SCMs (Figure 7)**
- Figure 7(a) shows a correctly specified SCM with significant paths between variables.
- Figure 7(b) shows a misspecified SCM where significant paths become insignificant and closer to zero.

**Importance of Knowing True Causal Structure**
- Without the true causal structure, untestable assumptions must be made.
- Incorrectly assuming relationships between variables can lead to biased estimates.
- The length of conversation is an example where assuming deal occurrence as a control introduces bias (Figure 7).

**Avoiding Bad Controls: SCM-based Approach**
- Exogenous variation is explicitly induced in the SCM to identify causal relationships ex-ante.
- No need to instrument endogenous variables or presume their causal relationships.
- Simple linear SCMs can be fit to reference how a new outcome is affected by exogenous variables.

### 5.2 Searching for causal structure in data

**Identifying Causal Relationships from Observational Data**

**Strategies for Identifying Causal Relationships:**
- Let data speak for itself: use algorithms to find the most likely model based on criteria like maximum likelihood or Bayesian information criterion.
  * Generate all possible DAGs and evaluate each one.
  * Number of possible DAGs grows exponentially with the number of nodes, making it impractical for large numbers.
  * Greedy Equivalence Search (GES) algorithm: add edges that maximize criteria, penalize model complexity, remove edges until optimized.

**Problems with Identifying Causal Structures:**
- Incorrect results: different runs on the same data can produce different results (Figure 8).
- Tax fraud scenario: GES identified incorrect causal structure between variables (Figure 8).

**Advantages of SCM-based approach:**
- Avoids search problems: we generate data based on proposed causal structures.
- Existing sources of exogenous variation are identified, allowing measurement of new outcomes without having to assume causal structures.

## 6 Conclusion

**Automated Hypothesis Generation and Testing through SCMs**

- Paper showcases an approach using Statistical Causal Models (SCMs) for automated hypothesis generation and testing.
- Computational system constructed with Language Learning Models (LLMs).
- Demonstrates simulations can reveal information to the model that was initially unknown.
- Results align well with theoretical predictions from relevant economic theory.
- Discusses potential benefits and future research opportunities.

### 6.1 Controlled experimentation at scale

**Usefulness of Simulation Systems for Social Science Research:**
- **View 1**: Dress rehearsals for "real" social science research (classical methods)
- **View 2**: Yield insights that sometimes generalize to the real world, stepping beyond classical agent-based modeling
  - Agents represent humans more accurately than traditional methods
  - Mirrors recent advances in using machine learning for protein folding and material discovery

**Advantages of Simulation Systems:**
- Controlled experimental simulations with prespecified plans for data collection and analysis
- Contrasts most academic social science research practices (lack of consistency, replicability)
- Valuable for understanding human behavior in various populations and environments due to context influence
- Reduces expenses and time required for exploration compared to studying humans directly.

**Challenges:**
- Fundamental jump from simulations to human subjects remains

### 6.2 Interactivity

**System Functionality**
- Scientists can monitor entire process
- Researchers can challenge decisions, probing for explanations or alternative options
- Customization options available: choose variables, operationalizations, agent attributes, interactions, and statistical analysis
- System can accommodate multiple LLMs simultaneously (e.g., using GPT-4 for hypothesis generation and Llama-2-70B for simulated social interactions)

### 6.3 Replicability

**Replicating Social Science Experiments**
- Can be difficult due to:
  - Unclear experimental procedures (Camerer et al., [2018](https://arxiv.org/html/2404.11794v2#bib.bib15))
  - Lack of transparency in exact methods used
- The System's Advantages:
  - **Frictionless Communication and Replication**:
    - Experimental design can be easily shared and duplicated
  - **Exportable Procedure**:
    - Entire procedure is exportable as a JSON file with the fitted SCM (Structured Causal Model)
  - **JSON Format**:
    - Easy for humans to read and write, and easy for machines to parse and generate
    - Commonly used in web applications, configuration, data storage, and serializing/transmitting structured data over a network
  - Contains:
    - Every decision the system makes
    - Natural language explanations for choices
    - Transcripts from each simulation
  - Can be saved or uploaded at any point in the process
- Potential Use Cases:
  - Researchers can run experiments and post JSON and results online
  - Other scientists can inspect, replicate the experiment, or extend the work.

### 6.4 Future research

**Research Areas for LLM-Powered Agents**

**Attributes**:
- Deciding which attributes to endow agents beyond relevant exogenous variables (e.g., demographic information, personalities)
- Improving simulation fidelity by adding these attributes, but unclear how to optimize the process

**Social Interactions**:
- Engineering natural conversational turn-taking between LLM agents
- Creating a menu of flexible agent ordering mechanisms as an initial attempt
- Introducing a coordinator agent to manage speaking order and end interactions
- Possible improvements, e.g., using a Markov model for more natural results

**Simulation Length**:
- Determining when to stop simulations, no universal rule exists
- Exploring better rules than those currently implemented

**Automated Research Program**:
- Building a system that can automate scientific process and determine follow-on experiments
- Continuously exploring parameter space within given scenarios
- Optimizing exploration of numerous possible variables

**Implementation Considerations**:
- This paper presents one possible implementation of the SCM-based approach
- Room for improvement and exploration through different design choices.

