# Autoformalization of Game Descriptions using Large Language Models
https://arxiv.org/pdf/2409.12300

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Background](#2-background)
- [3 Methods](#3-methods)
  - [3.1 Solver](#31-solver)
  - [3.2 Framework Algorithm](#32-framework-algorithm)
  - [3.3 Game descriptions](#33-game-descriptions)
  - [3.4 Experimental parameters](#34-experimental-parameters)
- [Results and Discussion](#results-and-discussion)

## Abstract
**Framework for Autoformalization of Game Descriptions**

**Introduction**:
- Game theory: powerful framework for reasoning about strategic interactions
- Applications range from daily life to international politics
- Challenges: applying formal reasoning tools in natural language contexts

**Approach Overview**:
- Framework for autoformalization of game-theoretic scenarios
- Translates natural language descriptions into formal logic representations
- Utilizes one-shot prompting and a solver providing syntactic feedback
- Allows LLMs to refine the code

**Evaluation Results**:
- Achieved 98% syntactic correctness and 88% semantic correctness
- Demonstrates potential of LLMs to bridge gap between real-life strategic interactions and formal reasoning.

## 1 Introduction

**Introduction:**
- Game theory (von Neumann and Morgenstern 1944): mathematical framework for analyzing competitive & cooperative interactions among rational decision-makers
- Applicable to various scenarios, from personal to business to nuclear conflict
- Challenge: natural language descriptions make it difficult to apply formal reasoning tools directly
- Solution: Large Language Models (LLMs) can convert natural language into formal representations
- Previous work has shown LLMs' success in handling nuances of natural language and autoformalizing mathematical & logical expressions
- This paper explores applying LLMs for autoformalization within game theory, bridging the gap between natural language and formal reasoning

**Contributions:**
1. **Dataset creation**: developed a dataset of 105 natural language descriptions for game theory scenarios (varied difficulty levels)
2. **Autoformalization framework development**: proposed a new framework to translate NL descriptions into formal game representations
3. **Evaluation**: assessed the performance of GPT-4 on translating strategic interaction descriptions into formal games

**Background:**
- Game theory: analysis of competitive & cooperative interactions among rational decision-makers
- LLMs: large language models with remarkable abilities in handling natural language nuances
- Autoformalization: conversion of NL to mathematical or logical expressions using LLMs
- General game playing: framework for designing agents that can learn and play games

**Methodology:**
- Dataset creation: developed a diverse set of 105 game descriptions (standard & non-standard, numerical & non-numerical payoffs)
- Autoformalization framework development: proposed a novel approach to translate NL descriptions into formal game representations using LLMs
- Evaluation: assessed the performance of GPT-4 on translating strategic interactions into formal games through zero-shot and one-shot prompting, examining its ability to generate formal specifications for new games

**Conclusion:**
This paper is the first to apply LLMs for autoformalization within game theory. Results demonstrate potential applications and bridge the gap between natural language and formal reasoning in strategic interactive scenarios. Future work includes further refinement of the framework and exploring its applicability to more complex real-world games.

## 2 Background

**Background**
- Multi-layered LSTMs: sequential models that perform well on complex tasks like language translation (Sutskever et al., 2014)
- Transformers and pre-trained models emerged, becoming the go-to solution for Large Language Models (LLMs) with billions of parameters and huge training data sets (Zhao et al., 2023)

**Large Language Model: GPT-4**
- Current state-of-the-art LLM by OpenAI
- Supports text, image, and audio inputs/outputs
- Estimated to have 175B parameters
- Performs exceptionally well in reasoning tasks (Wang & Zhao, 2024)

**Game Theory Basics**
- Provides a mathematical framework for strategic interactions and decision-making
- Players: rational, intelligent actors aiming to optimize utility scores (Myerson, 1984)
- Strategies: mapping state of game to available actions considering history (Osborne & Rubinstein, 1994)
- Payoffs: numerical values associated with specific outcomes (Gibbons, 1992)

**Five Games Characteristics**
| Game   | Players, Actions, and Payoffs                            |
| ------- | ------------------------------------------------------- |
| Pris. Dl. | Symmetric, T > R > P > S (Osborne & Rubinstein, 1994) |
| Hawk-Dove | Asymmetric, T > R > S > P                          |
| Matching Pennies | Zero-sum, R,P > T,S; T,S > R,P                       |
| Stag Hunt | Values mutual cooperation over temptation to defect  |
| Battle of Sexes | Asymmetric coordination game (Osborne, 2004)         |

**General Game Playing: GDL and Situation Calculus**
- Construct intelligent systems that can learn new games without human intervention
- Game Description Language (GDL): formal language for describing arbitrary games rules
- Challenge: learning without human guidance
- Action languages like Situation Calculus help reason about others' actions before taking action
- Fully embedding GDL into Situation Calculus presents a general game solver to guide an LLM in generating game parts.

## 3 Methods

**Methods**
- Rely on a logic programming solver for game formulation:
  * Initial state, legal moves, move effects, final states of games
  - Generates or tests all possible evolutions of specific games with domain-dependent parts
- Logic programming description and textual description of a game provided to the LLM
- Automatically formalize logic programs of new games from textual descriptions using an algorithm.

### 3.1 Solver 

**Game Solver Components:**

**Solver**: Consists of:
- Game-independent part: specifies rules for any game in extensive form
- Game-dependent part: expresses rules for a specific game
- Auxiliary predicates to support game processing

**Representation**:
- State represented as situation (histories of moves from initial)
- Binary function `do(M, S)` represents the next situation after move `M` in situation `S`

**Game-independent part:**
1. Legal transitions from initial to final situations:
   - Game accepts legal move if situation not final and move is legal
   - Game continues with next `do()` situation until final situation reached
2. Fluents (hold) in game:
   - Initially hold in initial situation
   - New fluent initiated by move effects
   - Persist unless abnormal
3. Final situations and results:
   - Describe result of the game when certain conditions hold

**Game-dependent part:**
1. Initial state and legal moves:
   - Define specific game's initial state and legal moves
2. Holds in initial situation:
   - Specify what holds in initial situation (e.g., player roles, control)
3. Effects of move on situation:
   - Changes that occur after a move is executed (e.g., abnormal fluent termination)
4. Final situations and results:
   - Define final situation and its consequences for players' utilities
5. Querying solver:
   - Reason about game by asking questions regarding utility or other outcomes.

### 3.2 Framework Algorithm

**Framework Algorithm**

**Step 1: Generating Game-Specific Predicates from Natural Language Descriptions**
- Input: Γ (game-independent predicates), NLPD (natural language description of Prisoner's Dilemma), ξPD (game-specific predicates for Prisoner's Dilemma), NLNG (natural language description of a new game)
- Output: ξNG (game-specific predicates for the new game)
- Parameters: maxattempts (maximum correction attempts)

**Step 2: Initialization**
- Initialize variables: attempts = 0, trace = empty list

**Step 3: Prompting Iteratively**
- While attempts < maxattempts do:
  - Attempt to generate game-specific predicates for the new game (ξNG) using Language Learning Model (LLM) and input data.
    * Translate the provided information into game-specific predicates (LLM.translate).
  - Validate generated predicates for syntactic correctness using a Prolog solver.
    * If valid, return ξNG and end process.
    * Else:
      - Get the trace from the solver about any errors.
      - Prompt the LLM to self-correct using the error trace (LLM.self_correct).
      - Increment attempts counter by one.
  - End while if predicates are not valid and maxattempts reached: return "Unable to generate valid predicates within maximum attempts."

### 3.3 Game descriptions

**Experiment Overview**
- Evaluating GPT-4o's ability to translate natural language game descriptions into formal specifications
- Used standard examples of five classic simultaneous one-shot games: Battle of the Sexes, Hawk-Dove, Matching Pennies, Prisoner’s Dilemma, and Stag Hunt
- Also investigated GPT-4o's capacity to formalize a real-world conflict or cooperation situation that can be modeled by these games but differs from standard metaphors
- Generated game descriptions for each variant: standard, non-standard, numerical, non-numerical

**Game Descriptions**
- **Standard**: Employs typical metaphor for a given game
- **Non-standard**: Newly invented example of a situation that can be modeled by a given game
- **Numerical payoffs**: Contains numerical payoffs
- **Non-numerical payoffs**: Does not contain numerical payoffs

**Number of Game Descriptions for Each Variant**
| Variant | Number of Descriptions |
|---|---|
| Standard | 5 (classic games) |
| Non-standard | Synthetically generated, varying in number |
| Numerical | 5 (classic games with numerical payoffs) |
| Non-numerical | Some descriptions generated without numerical payoffs |

**Zero-Shot Prompting Experiment**
- Goal: Assess the model's ability to formalize a game without prior examples
- Provided only game-independent predicates (Γ) in prompt, no example game-specific predicates

**Additional Experiments**
- Autoformalization of two differently structured games: sequential version of Prisoner’s Dilemma and Rock-Paper-Scissors

### 3.4 Experimental parameters

**Experimental Parameters for Autoformalization**

**Model Used:**
- GPT-4o (state-of-the-art)

**Maximum Output Tokens:**
- 1024

**Maximum Attempts:**
- Number: 5

**Table 3 Summary:**
- Experimental parameters used for autoformalization evaluation

**Source Code and Logs:**
- Publicly available at: https://github.com/dicelab-rhul/game-formaliser

## Results and Discussion

**Experimental Results and Discussion**

**Zero-shot Prompting**:
- Weaknesses of LLMs to produce formal descriptions for game-independent parts
- Inability to correctly represent state, players' moves, and payoffs
- No deep understanding of situation calculus

**One-shot Prompting with Example**:
- Successfully generates formal specifications for 2-player games that differ structurally from provided examples
- Syntactically correct in most cases (98%) but semantically incorrect in some, requiring manual review
- Generated code reflects understanding of game structure and rules

**Sequential PD**:
- System generates specification with revised payoff matrix and initial state to allow for one player to play first
- Demonstrates ability to generalize over different aspects of a game (number of moves, possible moves)

**Related Work**:
- LLMs gaining interest as potential building blocks for agents in game-theoretic simulations
- Autoformalization approach successfully translating natural language into formal representations, improving performance on logical reasoning tasks
- Interactive approaches to address translation errors and fine-tune models to increase accuracy

**Conclusions and Future Work**:
- Preliminary assessment demonstrated high syntactic and semantic accuracy for one-shot prompting in 2-player simultaneous move games
- Limitations: limited scalability due to manual evaluation of semantic correctness, small set of considered games
- Plans for future work: expanding the set of considered games, automating evaluation through code execution.

