# An Empirical Study of Smoothing Techniques for Language Modeling (Summary)

https://arxiv.org/pdf/cmp-lg/9606011

by Stanley F. Chen, Joshua Goodman; Harvard University (1996)

## Contents
- [1. Introduction](#1-introduction)
  - [1.1. Background](#11-background)
- [2. Previous work](#2-previous-work)
  - [2.2. Good–Turing estimate](#22-goodturing-estimate)
  - [2.3 Jelinek–Mercer Smoothing](#23-jelinekmercer-smoothing)
  - [2.4. Katz smoothing](#24-katz-smoothing)
  - [2.5 Witten–Bell Smoothing (1990, 1991)](#25-wittenbell-smoothing-1990-1991)
  - [2.6 Absolute Discounting](#26-absolute-discounting)
  - [2.7. Kneser–Ney smoothing](#27-kneserney-smoothing)
  - [2.8 Backoff vs. Interpolated Models](#28-backoff-vs-interpolated-models)
  - [2.9. Other smoothing techniques](#29-other-smoothing-techniques)
- [3. Modified Kneser–Ney smoothing](#3-modified-kneserney-smoothing)
- [4 Experimental Methodology](#4-experimental-methodology)
  - [4.1. Smoothing implementations](#41-smoothing-implementations)
  - [4.2. Parameter setting](#42-parameter-setting)
  - [4.3. Data](#43-data)
- [5. Results](#5-results)
  - [5.1. Performance of Smoothing Algorithms](#51-performance-of-smoothing-algorithms)
  - [5.2. Count-by-count analysis](#52-count-by-count-analysis)
  - [5.3. Auxiliary experiments](#53-auxiliary-experiments)
- [6. Discussion](#6-discussion)

## 1. Introduction
**Survey of Smoothing Techniques for Language Modeling**

**Overview**:
- Empirical study comparing language model smoothing techniques
- Includes comparison of methods from Jelinek and Mercer (1980), Katz (1987), Bell, Cleary and Witten (1990), Ney, Essen and Kneser (1994), and Kneser and Ney (1995)
- Investigates effects of training data size, corpus type, count cutoffs, and n-gram order on performance

**Background**:
- Language model: probability distribution over strings describing sentence occurrences in a domain
- Trigram models: determine p(w|wi−2wi−1) (word given previous two words)
- Maximum likelihood estimate (ML): computation of pML(w|wi−2wi−1) based on corpus data
- Problem: ML estimates can lead to poor performance due to zero probabilities for unseen sequences
- Smoothing: techniques adjusting ML estimates to improve accuracy

**Previous Comparisons**:
- Most previous studies compared only a few methods and on limited corpora/training sizes
- Katz (1987) compared his algorithm with Jelinek–Mercer and Nádas, but results depended on training set size
- Ney et al. (1997) compared multiple smoothing algorithms, but did not consider all popular methods

**Contributions**:
- Extensive comparison of widely-used smoothing techniques
- Evaluation on varied corpora, training sizes, and n-gram orders
- Automated search for optimal parameters for tunable methods
- Detailed analysis of performance and impact on small/large counts for different algorithms
- Motivation of a novel variation of Kneser–Ney smoothing

### 1.1. Background

**N-gram Language Models**
- Most widely used language models
- Express probability of a sentence `s` as product of probabilities of each word given preceding `n-1` words:
  * `p(s) = p(w₁|<BOS>) × p(w₂|w₁) × ... × p(wl|w₁···wi−1) × p(<EOS>|wi···wl)`
    - `<BOS>`: Beginning-of-sentence token
    * `<EOS>`: End-of-sentence token
  * Approximation for `n-gram` model: `p(s) ≈ ∏i=1l+1 p(wi|w1···wi−n)`
- Estimate probabilities using maximum likelihood on training data:
  * `pML(wi|wi−n, wi−1+1) = c(wi−n+1, wi) / ∑c(wi)`
    - `c(wiki)`: Occurrence count of n-gram `wik` in training data

**Evaluating Language Models**
- Measure performance with cross-entropy and perplexity on test data
  * Cross-entropy `Hp(T)`: Average number of bits required to encode test data using the model's compression algorithm
    - Interpreted as application performance for text compression task
  * Perplexity `PPp(T)`: Reciprocal of geometric average probability assigned by the model to each word in test set `T`
- Lower cross-entropies and perplexities indicate better performance.

## 2. Previous work

**Previous Work on Smoothing Algorithms for n-gram Models**

**Additive Smoothing**:
- One of the simplest types of smoothing used in practice
- Adds a factor **δ** to every count, where typically **0 < δ <= 1**
- Sets **padd(w|wi−n+1)** = **δ + c(wi−n+1)i / ∑wi c(wii−n+1)**, where:
  - **delta**: avoids zero probabilities
  - **c(wi−n+1)i**: count of the n-gram **wi−n+1** in a corpus
  - **V**: vocabulary or set of all words considered

**Limitations**:
- Gale and Church (1990, 1994) argue that this method generally performs poorly.

### 2.2. Good–Turing estimate

**Good-Turing Estimate**
- Central to many smoothing techniques (Good, 1953)
- For n-gram that occurs r times: pretend it occurs r∗ times where r∗ = (r + 1) nr+1
- Convert count to probability by normalizing: pGT(wii−n+1) = N ∑∞0 nr r∗
- Derived theoretically with weak assumptions, empirically accurate when nr values are large
- Not used alone for n-gram smoothing due to lack of higher-order model integration (discussed in following sections)

### 2.3 Jelinek–Mercer Smoothing
- Useful information provided when little data exists for directly estimating an n-gram probability p(w|wi−n+1)
- Correlated (n − 1)-gram probability p(wi |wi−n+2) estimated from more data
- Linear interpolation used to combine lower-order model information for higher-order probabilities
- Defined recursively as a linear interpolation between nth-order maximum likelihood model and (n - 1)th-order smoothed model: pinterp(wi |wi−n+1) = λwi−n+1pML(w|wi−n+1) + (1 − λwi−n+1) pinterp(w|wi−n+2)
- End the recursion by setting smoothed first-order model to maximum likelihood distribution or zeroth-order model to uniform distribution
- Search for optimal λwi−n+1 using Baum–Welch algorithm, setting all λwi−n+1 to same value leads to poor performance (Ristad, 1995)
- Partitioning the λwi−n+1 into buckets according to c(wi−n+1) and constraining their values is suggested by Bahl et al. (1983).

### 2.4. Katz smoothing

**Katz Smoothing (1987)**
- Extends Good–Turing estimate by adding higher-order models with lower-order models

**Calculation of ckatz(wi|wi−1):**
- For bigram wi−1 with count r = c(wi−1), calculate its corrected count:
  - Discount all non-zero counts according to discount ratio dr (specified later)
  - Distribute subtracted counts among zero-count bigrams based on unigram model distribution
- Ensure total number of counts in distribution is unchanged: ∑wi ckatz(wi|wi−1) = ∑wi c(wi|wi−1)
- Calculate pkatz(wi|wi−1): normalize corrected count

**Discount Ratios for Katz Smoothing:**
- Large counts are reliable, no discount: dr = 1 for r > k (k = 5 suggested)
- Lower counts' discount ratios derived from Good–Turing estimate on global bigram distribution

### 2.5 Witten–Bell Smoothing (1990, 1991)
- Developed for text compression, an instance of Jelinek–Mercer smoothing
- Nth-order smoothed model linearly interpolates between nth and (n − 1)th order maximum likelihood models

**Calculation of λwi−n+1i−1:**
- Assign parameters based on recursive equation: pWB(wi | wi−n+1) = c(wii−n+1) + N1+(wi−1+1·) pWB(w | wi−n+2)
  - (wi−1+1·) is the number of unique words following history wi−n+1
  - Normalize to ensure sum of distribution equals 1

### 2.6 Absolute Discounting
**Absolute Discounting (Ney & Essen, 1991; Ney et al., 1994)**
- Interpolates higher- and lower-order models by subtracting fixed discount D from non-zero counts instead of multiplying maximum likelihood distribution by factor λwi−n+1,i−1.
- Calculate pabs(w|wi−ni−1+1): max{c(wii−n+1) − D, 0} / (∑wi c(wii−n+1) + (1 − λwi−n+1) pabs(wi|wi−ni−1+2))
- Make distribution sum to 1: 1 − λwi−n+1i−1 = ∑wi c(wii−n+1) / N1+(wi−1+1·)D, where N1+(wi−1+1·) is defined as in Equation (6).

**Motivation for Absolute Discounting:**
- Church and Gale (1991) showed that the average Good–Turing discount of n-grams with larger counts (r ≥ 3) is generally constant over r.

### 2.7. Kneser–Ney smoothing

**Kneser–Ney Smoothing**

**Introduction:**
- Extension of absolute discounting by Kneser and Ney (1995)
- Lower-order distribution built in a novel manner
- Differences from previous algorithms: optimized for situations with few or no counts

**Motivation:**
- Intuitively, lower-order distribution should have low probability when only one history is present
- Example: FRANCISCO following SAN

**Derivation:**
1. Assumption of model form (Equation 9)
2. Constraints on unigram marginals (Equation 12)
3. Expanding Equation 12 and substituting p(wi-1)
4. Solving for pKN(w)
5. Generalization to higher-order models (Equation 14)

### 2.8 Backoff vs. Interpolated Models
- Both combine higher-order n-gram models with lower-order models
- Backoff model: if an n-gram has non-zero count, use the distribution τ(w|wi−n+1). Otherwise, backoff to the lower-order distribution psmooth(wi|wi−n+2), where γ(wi−n+1) is chosen to make the conditional distribution sum to one.
- Interpolated model: use information from lower-order distributions in determining probability of n-grams with non-zero counts, while backoff models do not. However, both types use lower-order distributions in determining the probability of n-grams with zero counts.

### 2.9. Other smoothing techniques

**Smoothing Techniques: Other Algorithms (Not Widely Used)**
* **Church–Gale Smoothing**
  * Similar to Katz's method, combines Good-Turing estimate with merging information from lower and higher order models
  * Buckets bigrams according to values `pML(wi−1) pML(wi)`
  * Good-Turing estimate applied separately within each bucket
  * Works well for bigram language models (Chen, 1996)
* **Bayesian Smoothing**
  * Several smoothing techniques motivated by a Bayesian framework
  * Priors used to arrive at final smoothed distribution
  * Examples: Nádas' method selects smoothed probabilities as mean a posteriori value given prior distribution, MacKay–Peto use Dirichlet priors to motivate linear interpolation (Jelinek–Mercer)
  * Results on single training set indicate worse performance compared to Katz and Jelinek–Mercer smoothing (Nádas, 1984; MacKay & Peto, 1995)
* **Other Interpolated Models**
  * Introduced in previous work by Chen (1996) for trigram models outperforming Katz and Jelinek–Mercer smoothing
  * One is a variation of Jelinek–Mercer, bucketing `λwi−n+1i−1` according to average number of counts per non-zero element in distribution (Bahl et al., 1983)
  * Yields better performance than total number of i(wi−1+1·) as suggested by Bahl et al.
  * Another is a modification of Witten–Bell smoothing, replacing term `N1+(wi−ni−1+1·)` with value `β N1(wi−n+1·) + γ` (parameters optimized on held-out data)
  * Not widely used but can be useful when Kneser–Ney smoothing is not applicable.

## 3. Modified Kneser–Ney smoothing

**Modified Kneser-Ney Smoothing**

**Introduction**:
- Variation of Kneser-Ney smoothing with improved performance
- Instead of one discount D for all non-zero counts, three different parameters: **D1**, **D2**, and **D3+**
- Applied to n-grams with 1, 2, and 3 or more counts, respectively

**Equation**:
- Instead of Equation (13) in Section 2.7:
```pKN(wi | wi−n+1) = c(wii−n+1) − D(c(wii−n+1))i−1  ∑wi c(wii−n+1) + γ (wi−n+1) pKN(wi | wi−n+2)i−1 where D(c) = { 0 if c = 0 D1 if c = 1 D2 if c = 2 D3+ if c ≥ 3 } To make the distribution sum to 1, take: i−1+1) = D1 N1(wi−ni−1+1·) + D2 N2(wi−ni−1+1·) + D3+ N3+(wi−ni−1+1·) γ (wi−n ∑ wi c(wii−n+1) where N2(wi−1+1·) and N3+(wi−1+1·) are defined analogously to N1(wi−1+1·).```

**Motivation**:
- Evidence presented in Section 5.2.1 suggests ideal average discount for n-grams with one or two counts is different from higher counts
- Modified Kneser–Ney smoothing significantly outperforms regular Kneser–Ney smoothing

**Optimal Values**:
- As Ney et al. (1994) have done for absolute discounting and Kneser–Ney smoothing, it is possible to create equations to estimate optimal values for **D1**, **D2**, and **D3+**:
```|D1| D2 | D3+ | where Y = n1 + 2n2 . n1```

**Comparison to Previous Work**:
- Ney et al. (1997) independently introduced multiple discounts, suggesting two discounts instead of three
- Performed experiments using multiple discounts for absolute discounting and found mixed results compared to a single discount

## 4 Experimental Methodology 
### 4.1. Smoothing implementations
**Implemented Smoothing Algorithms**:
- Additive smoothing: two versions - "plus-one" (δ = 1) and "plus-delta" (with adjustable δ)
- Jelinek–Mercer smoothing: baseline method and optimized version with adjustable parameters
- Katz smoothing: multiple kn values for each n-gram model
- Witten–Bell smoothing: interpolated and backoff versions
- Absolute discounting: interpolated and backoff versions
- **Kneser–Ney smoothing**: optimized and fixed parameter versions, backoff version
- **Modified Kneser–Ney smoothing**: three discount parameters at each n-gram level instead of one.

**Smoothing Algorithm Descriptions**:
- Complete description of each implementation provided in the extended paper (Chen & Goodman, 1998)
- Parameter optimization: set to optimize perplexity of held-out data

**Additive Smoothing**:
- Fixes δ = 1 or allows separate δ for each level (δn)
- Performs backoff when a history has no counts

**Jelinek–Mercer Smoothing**:
- Ends recursion with uniform distribution and buckets the smoothed probabilities based on held-out data count thresholds (cmin)

**Katz Smoothing**:
- Uses additive smoothing to smooth unigram distribution instead of Katz backoff, as it performed poorly

**Witten–Bell Smoothing**:
- Faithful implementation of original algorithm, and a backoff version

**Absolute Discounting**:
- Separate discount parameter (Dn) for each n-gram level
- Optimizes perplexity of held-out data to determine Dn values instead of using Equation (11) as in the original paper.

**Kneser–Ney Smoothing**:
- Smoothes lower-order distributions similar to highest-order distribution to handle data sparsity
- Optimizes perplexity of held-out data for Dn parameters or uses fixed parameters based on Equation (11)

**Modified Kneser–Ney Smoothing**:
- Three discount parameters, Dn,1, Dn,2, and Dn,3+ at each n-gram level instead of a single discount Dn
- Optimizes perplexity of held-out data or uses Equation (17) to set discount parameters.

### 4.2. Parameter setting

**Parameter Setting**

**Impact of Smoothing Parameters on Performance**:
- Example of sensitivity to parameter values:
  - **Figure 1**: Comparison of Jelinek-Mercer and Katz algorithms' performance based on different parameters (δ and cmin) and training set sizes.
  - Significant differences in test cross-entropy from several hundredths to over a bit.
  - Optimal values vary with training set size.

**Importance of Parameter Optimization**:
- To meaningfully compare smoothing techniques, parameters must be optimized for the specific training set.
- In experiments:
  - Parameters were optimized using **Powell's search algorithm**.
  - Searched on a held-out set associated with each training set.
  - Constrained parameter search in main experiments to noticeably impacting parameters (λs for Jelinek-Mercer and all other parameters for Katz).

### 4.3. Data

**Data Sources and Corpora Used:**
* Brown corpus (1 million words) from various sources
	+ Vocabulary: all 53,850 distinct words
	+ Segmented into sentences manually
	+ Average sentence length: about 21 words
* North American Business (NAB) news corpus
	+ 110 million words of Associated Press text from 1988-1990
	+ 98 million words of Wall Street Journal text from 1990-1995
	+ 35 million words of San Jose Mercury News text from 1991
	+ Vocabulary: 20,000 word vocabulary for 1995 ARPA speech recognition evaluation
	+ Primarily used Wall Street Journal text (WSJ/NAB corpus)
	+ Segmented automatically using transcriber punctuation
	+ Average sentence length: about 23 words in AP, 22 in WSJ, 20 in SJM
* Switchboard data (three million words) of telephone conversation transcriptions
	+ 9800 word vocabulary created by Finke et al. (1997)
	+ Segmented using turn boundaries
	+ Average sentence length: about 16 words
* Broadcast News text (130 million words) of television and radio news shows from 1992-1996
	+ 50,000 word vocabulary developed by Placeway et al. (1997)
	+ Segmented automatically using transcriber punctuation
	+ Average sentence length: about 15 words

**Data Preprocessing:**
* Any out-of-vocabulary words were mapped to a distinguished token and treated the same as other words
* Three segments of held-out data along with the segment of training data were chosen, which were adjacent in the original corpus and disjoint from each other
* The first two held-out segments were used for parameter selection, while the last one was used as test data for performance evaluation
* For experiments with multiple training set sizes, the different training sets shared the same held-out sets; however, for experiments with multiple runs on the same training size, the data segments of each run were completely disjoint.

**Effect of Hold-Out Set Size and Folding Back:**
* For jelinek-mercer smoothing, which has hundreds or more λ parameters:
	+ The size of held-out set can have a moderate effect on test cross-entropy (up to 0.03 bits/word higher) for smaller held-out sets compared to the baseline size (2500 sentences).
	+ However, even with much larger held-out sets, the improvement is at most 0.01 bits/word.
* For kneser-ney-mod smoothing with about 10 parameters:
	+ The effect of held-out set size on test cross-entropy is negligible (less than 0.005 bits/word).

**Impact of Folding Back Held-Out Set:**
* For jelinek-mercer, when the held-out data is folded back into the training set after parameter optimization:
	+ Performance is augmented significantly for small training set sizes due to an increase in training set size.
	+ However, for larger training sets (100 000 sentences or more), this improvement becomes negligible.
* The difference between fold-back line and extra line represents the benefit of using a disjoint held-out set to optimize parameters, which can be noticeable for jelinek-mercer with smaller datasets but negligible for kneser-ney-mod.

## 5. Results

**Results**

**Overall Results**:
- Presented results of experiments on various algorithms for different training set sizes on different corpora (Brown, Broadcast News, Switchboard, WSJ/NAB) for bigram and trigram models
- Demonstrated that the relative performance of different smoothing methods can vary as conditions change, with Kneser–Ney smoothing and variations consistently outperforming others

**Detailed Analysis**:
- Presented analysis of performance on n-grams with specific counts (e.g., those occurring once in the training data)
- Found that **katz** most accurately smooths n-grams with large counts, while **kneser-ney-mod** is best for small counts
- Showed the relative impact of performance on n-grams with different counts

**Variability and Error**:
- Reported the cross-entropy of the baseline algorithm Jelinek-Mercer over various training set sizes
- Displayed the standard error of the baseline cross-entropies for Broadcast News corpus, which was comparable to performance differences between algorithms

**Comparison of Algorithms**:
- Compared the relative performance of various smoothing algorithms to the baseline on different corpora and training set sizes
- Observed that the variation in cross-entropy relative to the baseline is generally small, much smaller than the difference in absolute performance between algorithms
- Pointed out that with larger deviations, distinguishing two similar performing algorithms may be difficult from a single pair of points
- Noted that overall performance trends are consistent across various runs on different training sets and corpora.

### 5.1. Performance of Smoothing Algorithms

**Performance of Smoothing Algorithms**

**Overall Performance Differences**:
- Kneser–Ney smoothing consistently outperforms all other algorithms for bigram and trigram models across all corpora and training set sizes.
- The algorithms **katz** and **jelinek-mercer** generally yield the next best performance, performing substantially better than the baseline in almost all situations except for cases with very little training data.
- The worst performing algorithms are **abs-disc-interp** and **witten-bell-backoff**. While **abs-disc-interp** generally outperforms the baseline, it does not perform as well as the other algorithms on small datasets or when there is limited training data. **Witten-bell-backoff** performs poorly compared to the baseline for smaller datasets but is competitive with the best algorithms on very large datasets.
- The relative performance of smoothing techniques can vary dramatically over training set size, n-gram order, and training corpus.

**Additive Smoothing**:
- The **plus-one** and **plus-delta** algorithms perform much worse than the baseline algorithm except for cases with a wealth of data.

**Backoff vs. Interpolation**:
- The backoff strategy, as used in Witten–Bell smoothing, absolute discounting, and Kneser–Ney smoothing, generally outperforms the interpolated version for Witten–Bell smoothing and modified Kneser–Ney smoothing. However, for absolute discounting, the interpolated version performs better on small datasets but worse on large datasets.

**Kneser-Ney Smoothing Comparison**

**Performance of Variations**:
- Compare performance of: **kneser-ney**, **kneser-ney-mod**, **kneser-ney-fix**, and **kneser-ney-mod-fix**
- Do not discuss **kneser-ney-mod-backoff**, presented in Section 5.1.3

**Performance of Kneser-Ney Algorithms**:
- Display performance relative to baseline algorithm **jelinek-mercer-baseline** for:
  - Bigram and trigram models on the WSJ/NAB corpus
  - Over a range of training set sizes

**Differences in Performance**:
- Kneser-Ney-Mod consistently outperforms Kneser-Ney over all training set sizes for both bigram and trigram models.
- This performance difference is generally considerable, though smaller for very large datasets.
- Reason for the difference explained in Section 5.2.

**Kneser-Ney vs. Kneser-Ney-Fix**:
- Kneser-Ney sets discounts (Dn) using cross-entropy of held-out data, while Kneser-Ney-Fix uses a formula based on training set counts.
- While their performances are sometimes very close, especially for large datasets, Kneser-Ney consistently outperforms Kneser-Ney-Fix.

**Kneser-Ney-Mod vs. Kneser-Ney-Mod-Fix**:
- These algorithms differ in whether the discounts are set using held-out data or a formula based on training set counts.
- Despite similar behavior for large datasets, Kneser-Ney-Mod consistently outperforms Kneser-Ney-Mod-Fix.
- The "fix" variations have the advantage of not requiring external parameters to be optimized.

### 5.2. Count-by-count analysis

**Count-by-count Analysis**

**Detailed Picture of Algorithm Performance**:
- Instead of looking at overall cross-entropy, partition test sets according to how often each n-gram occurred in training data
- Examine performance within each partition

**Cross-Entropy Calculation**:
- Cross-entropy of an n-gram model p on a test set T: Hp(T) = -1 / WT ∑i (ci(wi) log2 p(w|wi−n+1))
- Where the sum ranges over all n-grams and ci(wi) is the number of occurrences of the n-gram wi in the test data

**Count Analysis**:
- Instead of summing over all n-grams, consider summing only over n-grams with exactly r counts in training data
- Hp,r (T) = -1 ∑ ci(wi−n+1) log2 p(wi |wi−ni−1+1)
- Mp,r (T) = ∑ ci(wi−n+1) p(w|wi−ni−1+1): expected probability of n-grams with r counts according to model p given histories in test set
- Ideally, Mp,r (T) should match cr(T), the actual number of n-grams in test set with r counts

**Normalized Cross-Entropy**:
- Hp∗,r (T): -1 ∑ ci(wi−n+1) log2 Mp,r (T) p(wi |wi−n+1)
- Assures each model predicts each count r with the same total mass
- Measures how well a smoothed model distributes probabilities among n-grams with the same count
- Lower values indicate better performance

**Experiments**:
- Conducted on WSJ/NAB corpus using test sets of about 10 million words
- Trained on 30,000 sentences (750,000 words) and 3.7 million sentences (75 million words)

**Expected vs. Actual Counts Ratio for Various Smoothing Algorithms**

**Behavior of Different Smoothing Algorithms:**
- **Figure 10**: Ratio of expected to actual n-gram counts in training set for trigram models, separated into low and high counts.
- **Low counts**: katz and kneser-ney come closest to ideal value of 1; jelinek-mercer-baseline, jelinek-mercer, witten-bell-backoff farthest.
- **Zero-count case**: exclude n-grams with no corresponding history count.
- **High counts**: katz nearest to ideal; results for bigram models similar.

**Explanation of Differences:**
- **Calculate ideal average discount**: assuming all n-grams receive ¯ counts instead of maximum likelihood distribution.
- **Graphing ideal average discount**: Figure 11, r≤13 for bigram and trigram models on one million and 200 million word training sets.
- **Small counts vs large counts**: correct discount rises quickly then levels off; a uniform discount is more appropriate than proportional to r.
- **Katz smoothing**: chooses Good–Turing discount, estimates correct average discount well empirically.
- **Performance of Kneser-Ney Smoothing**: distributes probabilities poorly between n-grams with the same count; not as good as expected due to cross-entropy issues.

**Kneser-Ney-Mod Smoothing:**
- Uses uniform discount Dn for counts three and above, but separate discounts Dn,1 and Dn,2 for one and two counts.
- Motivated by observation that smaller counts have different ideal average discount than larger counts; performs much better than kneser-ney for low counts in Figure 10.

**Performance Analysis of N-Gram Models**

**Normalized Performance**:
- Figure 12 displays normalized cross-entropy for n-grams based on:
  - Trigram models, separated into low and high counts
  - Baseline algorithm and various algorithms (kneser-ney, kneser-ney-mod, katz, witten-bell-backoff)
- Kneser-ney and kneser-ney-mod outperform other algorithms on low counts, especially those with no counts.
- **Explanation**: Kneser-ney uses a modified backoff distribution that provides better discounting for n-grams with low counts.
- **Comparision to Baseline**: Kneser-ney smoothing algorithm is the best overall performer due to its excellent performance on low counts and good performance on high counts.
- Worst performing algorithms: katz and witten-bell-backoff, which both use backoff instead of interpolation.
- Interpolation provides more accurate estimation for lower-order distributions, which is especially useful for low counts.

**Performance Variation over Training Set Size**:
- Figure 11 shows cumulative values of Hp(T) (entropy in test data) for different n-gram counts on the WSJ/NAB corpus.
- As training set size grows, more entropy is devoted to high counts, but surprisingly, low counts also contribute significantly even for large datasets.
- This explains why performance on low counts has a significant impact on overall performance and why Kneser-ney smoothing performs best.

**Backoff vs. Interpolation**:
- Left graph of Figure 13: Modified Kneser-ney interpolated outperforms backoff version for n-grams with low (positive) counts.
- Explanation: Lower-order distributions provide valuable information about the correct discount, which is better captured by interpolation.
- Right graph of Figure 13 shows that the backoff version has a closer expected-to-actual count ratio than interpolated one for all n-gram counts.
- Hypothesis: The relative strength of these two opposing influences determines the performance difference between backoff and interpolated versions of an algorithm.

### 5.3. Auxiliary experiments 

**Higher Order n-Gram Models in Speech Recognition**

**Advantages of Higher-Order N-Gram Models:**
- Increasing speed and memory of computers have led to the use of 4-gram and 5-gram models in recent years for speech recognition (Seymore et al., 1997; Weng et al., 1997)
- Advantages over lower-order models increase with amount of training data
- With several million sentences, advantages can exceed 0.2 bits per word compared to trigram model

**Performance of Various Smoothing Algorithms:**
- Figure 14 displays relative performance of various smoothing algorithms for 4-gram and 5-gram models on WSJ/NAB corpus
- **kneser-ney** and **kneser-ney-mod** consistently outperform other algorithms due to handling sparse data better
- Algorithms like **katz**, **abs-disc-interp**, and **witten-bell-backoff** perform as well or worse than baseline method except for largest datasets
- **Jelinek-mercer** consistently outperforms the baseline algorithm.

**Training Set Size:**
- Relative performance of algorithms on WSJ/NAB corpus:
  * Table shows difference in test cross-entropy from baseline (bits/token) for various smoothing algorithms
  * Performance improves with increasing training set size for all algorithms.

**Comparison to Lower-Order Models:**
- Higher-order n-gram models can significantly outperform lower-order models given sufficient training data.

**Count Cutoffs in N-gram Models**

**Overview**:
- Large datasets use count cutoffs to restrict size of n-gram models
- Ignoring counts is algorithm-specific, not specified in original smoothing descriptions
- Implemented as: ignoring n-grams with fewer than a given number of occurrences and assigning probabilities through backoff/interpolation

**Examples**:
- **0-0-1 cutoffs for trigram model**: All unigrams and bigrams with 0 or fewer counts ignored, all trigrams with 1 or fewer counts
- Cutoffs of one or two for bigrams and trigrams can greatly decrease model size with minimal performance degradation

**Performance Comparison**:
- Trigram models with different cutoffs compared to non-cutoff model on WSJ/NAB corpus (Figure 15)
  - **Kneser-ney-mod smoothing**: Models with higher cutoffs perform more poorly as expected
  - **Jelinek-mercer-baseline smoothing**: Models with 0-0-1 cutoffs outperform non-cutoff model for most training sizes
- Trigram models and various smoothing algorithms compared on WSJ/NAB corpus (Figure 16)
  - **Abs-disc-interp** performs more poorly relative to other algorithms, particularly kneser-ney-mod
  - Magnitudes of performance differences less when cutoffs are used

**Cross-entropy and Speech Recognition**:
- Performance of trigram language models measured in cross-entropy correlates with speech recognition word-error rates (Figure 17)
  - Linear correlation between cross-entropy and word-error rate for these models
  - Lower cross-entropies lead to lower word-error rates, making a significant difference in speech recognition performance

## 6. Discussion

**Discussion on Smoothing Techniques in Language Modeling**
- **Smoothing**: fundamental for statistical modeling, improves performance in data sparse situations (e.g., prepositional phrase attachment, part-of-speech tagging)
- First empirical comparison of various smoothing techniques: Kneser–Ney, absolute discounting, linear discounting, interpolated models, and backoff models
- Factors influencing performance:
  - Use of a modified lower-order distribution (Kneser–Ney smoothing)
  - Absolute vs. linear discounting
  - Normalized performance difference between interpolated and backoff models
  - Free parameters optimization (kneser-ney-mod vs kneser-ney-mod-fix)

**Findings:**
- Kneser–Ney smoothing variations performed well due to modified lower-order distribution usage
- Absolute discounting outperforms linear discounting in most cases, but Good–Turing estimate can predict ideal average discount better (Katz smoothing)
- Interpolated models superior to backoff for low counts as they provide valuable information from lower order models
- Optimizing free parameters can improve algorithm performance (kneser-ney-mod) but requires availability of held-out data.

**Additional Experiments:**
- Cross-entropy and word error rate correlation is strong when only difference between models is smoothing.
- Further research needed on other statistical models beyond n-gram language modeling.

