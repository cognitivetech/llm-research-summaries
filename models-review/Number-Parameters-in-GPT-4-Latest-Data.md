# Number of Parameters in GPT-4 (Latest Data)
**Extensive list of statistics on parameters in ChatGPT-4, ChatGPT-4o, and other AI models.**

source: https://explodingtopics.com/blog/gpt-parameters
Author: Josh Howarth
Topic: ChatGPT Parameters
Tags: Clippings

## Contents
- [Introduction](#introduction)
- [Number of Parameters in ChatGPT-4](#number-of-parameters-in-chatgpt-4)
  - [Evolution of ChatGPT Parameters](#evolution-of-chatgpt-parameters)
- [What are AI Parameters?](#what-are-ai-parameters)
  - [More parameters aren’t always better](#more-parameters-arent-always-better)
- [Why ChatGPT-4 has Multiple Models](#why-chatgpt-4-has-multiple-models)
- [Number of Parameters in ChatGPT-4o](#number-of-parameters-in-chatgpt-4o)
- [Number of Parameters in Other AI Models](#number-of-parameters-in-other-ai-models)
- [ChatGPT-4 Parameters Estimates](#chatgpt-4-parameters-estimates)
- [Conclusion](#conclusion)

## Introduction

**Key Points on GPT-4 and GPT-4o Parameters** \n
- Experts estimate that GPT-4 has approximately **1.8 trillion** parameters.
- This makes GPT-4 **over ten times larger** than its predecessor, GPT-3.
- A smaller version of GPT-4, known as GPT-4o Mini, is estimated to have around **8 billion** parameters.

## Number of Parameters in ChatGPT-4

**GPT-4 Parameters Estimation**

**Estimate of Parameters**:
- Approximately 1.8 trillion parameters
- First estimated by George Hotz in June 2023
- Subsequently supported by multiple sources: Semianalysis report, Nvidia GTC24 graph, Meta engineer Soumith Chintala's confirmation

**GPT-4 Architecture Breakdown**:
- Consists of eight models
- Each internal model comprises 220 billion parameters
- Total parameters: 1.8 trillion (8 x 220B)

**Additional Information**:
- GPT-4 was released in June 2023
- George Hotz was the first to publicly share the estimation of its parameter count

### Evolution of ChatGPT Parameters

**ChatGPT-4 Parameters**
- Significantly larger than GPT-3: approximately 10 times more parameters (175 billion vs OpenAI's confirmation)
- Exceedingly larger than GPT-1: over 15,000 times more parameters (compared to its 117 million)

**Visual Aid:** Not applicable in text format

## **What are AI Parameters?**

**AI Models: Context Length and Parameters**

**Tokenization**:
- AI models like ChatGPT break down textual information into **tokens**
- A token is roughly the same as three-quarters of an English word

**Context Length or Window**:
- Determines how many tokens an AI can process at once
- ChatGPT-4 has a context window of 32,000 tokens (about 24,000 words)
- Once surpassing this number, the model starts "forgetting" earlier information
- Can lead to mistakes and hallucinations

**Parameters**:
- Determine how an AI model can process these tokens
- Compared to **neurons** in the brain
- Human brain has about 86 billion neurons
- Connections and interactions between neurons fundamental for brain's function
- Adding more neurons and connections can aid learning
- A jellyfish has few thousand, snake ten million, pigeon hundreds of millions
- AI models with more parameters have greater information processing ability, but not always the case.

### **More parameters aren’t always better**

**Concise Version:**

* More parameters in an AI can lead to better information processing, but there are drawbacks:
	+ High cost - OpenAI spent over $100 million training GPT-4 alone, and Anthropic CEO predicts a 10 billion model by 2025.
	+ To address this issue, OpenAI released the cost-efficient GPT-4o mini, which has fewer parameters but outperforms its predecessor on several benchmark tests.

## Why ChatGPT-4 has Multiple Models

**ChatGPT-4 Architecture**
* **Models**: ChatGPT-4 consists of eight models, each with approximately 220 billion parameters.
* **Previous Models**: Previously used "dense transformer" architecture.
* **New Architecture**: ChatGPT-4 uses the "Mixture of Experts" (MoE) architecture.
* **MoE Architecture**: Each model is composed of two experts, totaling 16 experts with 110 billion parameters each.
* **Specialization**: Experts are specialized to handle specific tasks efficiently and cost-effectively.
* **Parameter Usage**: Fewer than 1.8 trillion parameters are used at any given time.

**ChatGPT-4o Mini**
* **Parameters**: Around 8 billion parameters.
* **Comparison**: Comparable to Llama 3 8b, Claude Haiku, and Gemini 1.5 Flash.
* **Llama 3 8b**: Meta's open-source model with just 7 billion parameters.

## Number of Parameters in ChatGPT-4o

**Concise Version:**

According to OpenAI's CTO, GPT-4o is suggested to have 1.8 trillion parameters like GPT-4. However, the exact number of parameters for GPT-4o remains uncertain since OpenAI has not confirmed it. CNET suggests this connection, and other sources, such as The Times of India, estimate that ChatGPT-4o has over 200 billion parameters.

## Number of Parameters in Other AI Models

**AI Model Sizes**

**Google Gemini Ultra:**
- Estimated to have over 1 trillion parameters
- No official confirmation from Google

**Google Gemini Nano:**
- Two versions: Nano-1 (1.8 billion parameters), Nano-2 (3.25 billion)
- Smaller models condensed from larger predecessors
- Intended for smartphone use

**Meta Llama 2:**
- 70 billion parameters
- Trained on two trillion tokens of data

**Anthropic Claude 2:**
- Over 130 billion parameters (official release)

**Anthropic Claude 3 Opus:**
- Possible over 2 trillion parameters and 40 trillion tokens training
- No official confirmation from Anthropic.

## ChatGPT-4 Parameters Estimates

**Estimates of ChatGPT-4 Parameters:**
* **1 trillion**: Semafor ([link](https://www.semafor.com/article/03/24/2023/the-secret-history-of-elon-musk-sam-altman-and-openai))
* **100 trillion**: CEO of Cerebras ([link](https://www.ax-semantics.com/en/blog/gpt-4-and-whats-different-from-gpt-3))
* **13 trillion tokens**: The Decoder ([link](https://the-decoder.com/gpt-4-architecture-datasets-costs-and-more-leaked/))

**Estimates of ChatGPT-4 Training Data:**
* Roughly 13 trillion tokens: The Decoder
* Included text and code from various sources:
  * Web crawlers like CommonCrawl
  * Social media sites (e.g., Reddit)
  * Textbooks and other proprietary sources (possibly)

## Conclusion

AI developers, including OpenAI, are hesitant to reveal the number of parameters in their latest models. Estimates suggest varying model sizes, with ChatGPT-4 following the trend of increasing size. However, recent releases like GPT-4o Mini hint at a potential focus shift towards cost-efficient tools.
