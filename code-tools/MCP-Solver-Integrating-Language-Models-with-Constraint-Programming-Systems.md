# MCP-Solver Integrating Language Models with Constraint Programming Systems

source: https://arxiv.org/html/2501.00539v1
by Stefan Szeider 

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 System Overview](#3-system-overview)
  - [3.1 Code and Technical Description](#31-code-and-technical-description)
  - [3.2 Design Principles](#32-design-principles)
  - [3.3 MCP tools](#33-mcp-tools)
  - [3.4 Model Management](#34-model-management)
  - [3.5 Persistent Knowledge Base](#35-persistent-knowledge-base)
- [4 Preliminary Experimental Evaluation](#4-preliminary-experimental-evaluation)
- [5 Conclusion](#5-conclusion)

## Abstract

**MCP-Solver: Integrating Large Language Models (LLMs) with Constraint Programming Systems**

**Background:**
- LLMs excel at natural language tasks but struggle with precise formal reasoning
- Introducing Model Context Protocol (MCP) for systematic integration between LLMs and constraint programming systems

**MCP-Solver: A Prototype Implementation**
- Demonstrates potential of combining LLMs' natural language understanding with constraint-solving capabilities
- Interfaces for creating, editing, and validating a constraint model

**Features:**
1. **Item-based editing approach**: Enables structured iterative refinement
2. **Model consistency**: Ensure consistency at every modification step
3. **Persistent knowledge base**: Maintains insights from previous solving sessions
4. **Concurrent solving sessions**: Handles multiple concurrently running sessions
5. **Integrated validation**: Provides feedback on model consistency

**Benefits:**
- Combines strengths of LLMs and constraint reasoning systems
- Enables more effective problem solving in formal domains

**Experiments:**
- Initial experiments suggest promising results for the integration of LLMs and constraint-solving capabilities

**Open-source implementation**: Proof of concept for integrating formal reasoning systems with LLMs through standardized protocols.

**Future Work:**
- Further research required to establish comprehensive formal guarantees
- Taking a first step toward principled integration of natural language processing with constraint-based reasoning.

## 1 Introduction

**Large Language Models (LLMs)**
- Demonstrated remarkable capabilities across natural language tasks
- Exhibit fundamental limitations in logical reasoning and formal problem specification
- Struggle with complex reasoning chains, backtracking from failed solution attempts, and maintaining precise quantifier relationships
- Limitations more evident in mathematical and logical problem-solving contexts

**Addressing LLMs' Limitations**
- Researchers have tackled limitations by pairing LLMs with specialized formal systems:
  - Integrating theorem provers into reasoning pipeline
  - Connecting LLMs to calculators or verification tools

**Model Context Protocol (MCP)**
- Introduces a universal standard for connecting LLMs with external systems
- Provides a flexible yet rigorous architecture for data and computational capabilities exposure through standardized servers
- Allows AI applications to connect as clients and access resources
- Gained broad adoption by companies like Block, Apollo, and platforms like Zed, Replit, Codeium, and Sourcegraph
- Pre-built MCP servers available for popular enterprise systems from Anthropic

**Integration of LLMs with Constraint Programming (CP) Systems**
- Provides a precise interface for transforming natural language specifications into formal constraint models
- Validates these models and verifies solutions
- Bridges the reasoning limitations of LLMs with the formal guarantees provided by CP solvers
- Open-source implementation demonstrates practical viability of this approach, offering tools for model submission, parameter management, solution retrieval, and interactive refinement
- System maintains a solver state, handles concurrent solving sessions, and provides detailed feedback for model validation and solution verification
- Represents a significant step toward more reliable and verifiable LLM-based problem-solving systems.

## 2 Related Work

**Recent Research on Large Language Models (LLMs)**

**Approaches to Constraint Solving:**
- PRoC3S: two-stage architecture for robotics planning [^4]
  * LLM generates parameterized skill sequences
  * Continuous constraint satisfaction
- Program Synthesis [^6]
  * Counterexample-guided framework
  * LLM synthesizer and SMT solver verifier
- SATLM: translates natural language into logical formulas for SAT solving [^21]
- LOGIC-LM: complete pipeline from LLM to symbolic solver and interpreter [^14]
- Lemur: task-agnostic LLM framework for program synthesis [^19]
- LLM-Modulo frameworks [^7]
  * Pair LLMs with external verifiers

**Constraint Solving Specific:**
- GenCP: integrates LLMs into constraint solver domain generation [^15]
- StreamLLM: real-time constraint solving [^17]
- Model Context Protocol (MCP) Solver [New]
  * Flexible architecture for interactive constraint modeling
  * Enables dynamic interaction patterns between LLMs and constraint solvers
  * Standardized tool interface

**Differences from Prior Work:**
- MCP Solver: flexible protocol-based architecture [New]
  * Allows iterative refinement of constraint models through natural language interaction
  * Maintains solver integrity
- Prior work: fixed integration patterns for specific use cases.

## 3 System Overview

### 3.1 Code and Technical Description

The MCP Solver is an open-source project available at [https://github.com/szeider/mcp-solver](https://github.com/szeider/mcp-solver). It requires Python 3.9+, MiniZinc with Chuffed, and supports macOS, Windows, and Linux platforms (except for Linux users who need a Claude Desktop alternative). Installation is via standard Python package management tools, using JSON files in platform-specific locations for configuration.

### 3.2 Design Principles

**Challenges in Integrating Language Models (LLMs) with Constraint Solvers:**
* **Maintaining solver integrity**: Ensured by asynchronous model management that separates modification and solving operations.
* **Managing model state**: Validation step precedes each change to maintain consistency. Robust session management for handling timeouts and resource cleanup.
* **Providing effective tool interfaces**: MCP Solver connects three components: Claude Desktop app (MCP client), MCP Solver (server), and MiniZinc as the constraint-solving backend.

**Components of MCP Solver:**
* **Claude Desktop app**: Interacts with constraint models through natural language.
* **MCP Solver (server)**: Manages LLMs' interactions, translates them into MiniZinc operations using Python API.
* **MiniZinc**: Compiles models into FlatZinc specifications for processing by a constraint solver. Default solver is Chuffed.

**Benefits of MCP Solver:**
* Supports all solvers compatible with MiniZinc.
* Provides eight standardized tools to interface with the solver.
* Coordinates between LLMs and constraint-solving capabilities.
* Manages model validation, solver configuration, and solution extraction using Python MiniZinc library.
* Guides the interaction between LLMs and solvers through a system prompt that provides key information about tools, model structure, and validation rules. This allows for effective translation of natural language specifications into valid MiniZinc models while maintaining best practices and supporting iterative refinement.

### 3.3 MCP tools

The MCP Solver provides tools that adhere to the MCP specification:

* `get_model`: View the current model.
* `add_item`, `delete_item`, and `replace_item`: Edit items in the model.
* `solve_model`: Execute with Chuffed solver.
* `get_solution`, `get_solve_time`, `get_memo`, and `edit_memo`: Access solution insights, execution time, knowledge base, and update it.

The server implements a request-response protocol for model modification, error handling, and memo system maintenance.

### 3.4 Model Management

**Model Management Approach (Item-Based)**
- **Tools**: get_model, add_item, delete_item, replace_item
- Enables inspection and atomic modification of model state
- Maintains model validity through integrated validation

**get_model**
- Inspects current model state with numbered items

**Item-Based Editing Approach**
- Integrates validation into every modification operation
- Triggers complete Python MiniZinc validation chain: syntax parsing, type checking, instantiation verification
- Applies changes only if validation succeeds
- Eliminates possibility of accumulated inconsistencies from line-based editing

[Model Management](https://arxiv.org/html/2501.00539v1/x2.png) **Figure 2:**
- Example for MCP Solver's item-based model editing with validation

**Advantages of Item-Based Editing**
- Precise error reporting and targeted refinements due to detailed diagnostic information
- Preserves model integrity through continuous validation during editing
- Allows exploration of modeling approaches freely while maintaining solver session state, solution cache, and performance statistics
- Thread safety for concurrent operations

**MCP Solver's Implementation**
- Coordinates model updates, solving operations, and solution retrieval
- Proper resource cleanup through async context managers, handling solver process termination.

### 3.5 Persistent Knowledge Base

The memo system stores and curates problem-solving insights between sessions in a text file. Users can contribute by prompting the LLM to document specific strategies. The get_memo and edit_memo tools allow access to this knowledge base through a line-based editing interface.

## 4 Preliminary Experimental Evaluation

**Evaluation of MCP Solver**

**Flexibility and Robustness:**
- Assessed on various natural language problems for practical capabilities
- Not rigorous benchmarks but provide valuable insights into system's performance
- Covered different constraint programming paradigms: satisfaction, optimization, parameter exploration

**Examples:**
1. **Casting Example**:
   - Demonstrates effective translation of complex logical conditions into boolean constraints using LLM
2. **TSP Example**:
   - Optimization modeling and model adaptation when new constraints arise (blocked road)
3. **N-Queens Example**:
   - Illustrates parameter exploration while maintaining model structure

**Limitations:**
- Current implementation restricts solving times to a few seconds, which may require modifications for larger instances
- Memo system captures modeling insights only when prompted
- Autonomous knowledge base updates by the LLM are rare, suggesting room for improvement in the system prompt to encourage more proactive knowledge accumulation.

## 5 Conclusion

**MCP Solver: Integrating LLMs and Constraint Solvers**

**Advantages of Protocol-based Integration**:
- Enables dynamic problem refinement
- Solutions can trigger new constraints and model adjustments based on user feedback
- Particularly valuable during the exploratory phase of constraint modeling

**Experiments and Findings**:
- Demonstrates effective combination of natural language understanding with constraint modeling capabilities
- Shows strength in iterative model refinement and error recovery through natural dialogue
- Some aspects, like autonomous knowledge accumulation, could benefit from further development

**Challenges**:
- Reliability challenges remain: LLMs occasionally misinterpret solver outputs or make translation errors
- Current verification mechanisms generally catch and correct these issues, but more rigorous system prompts and validation procedures could improve reliability
- Balance between flexibility and reliability is an ongoing challenge in LLM-solver integration

**Future Research Directions**:
- Adding SAT solvers and minimal unsatisfiable subset analysis would broaden the system's scope
- Reliability of results could benefit from more sophisticated verification methods without sacrificing system flexibility
- Self-reflection capabilities of LLMs suggest new possibilities for interactive modeling
- Leveraging the broader MCP ecosystem to create comprehensive problem-solving environments, with opportunities for data access, preprocessing, and result visualization

**Conclusion**:
- The MCP Solver represents a significant step toward integrating natural language understanding with constraint programming
- Demonstrates that protocol-based architectures can effectively combine the strengths of both domains while maintaining system flexibility.

