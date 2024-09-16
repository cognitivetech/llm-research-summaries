# IMPROVED BACKING-OFF FOR M-GRAM LANGUAGE MODELING

by Reinhard Kneser and Hermann Ney

https://www-i6.informatik.rwth-aachen.de/publications/download/951/Kneser-ICASSP-1995.pdf

## Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. BACKING-OFF](#2-backing-off)
- [3. MARGINAL DISTRIBUTION AS CONSTRAINT](#3-marginal-distribution-as-constraint)
- [4. LEAVING-ONE-OUT](#4-leaving-one-out)
- [5. Experimental Results](#5-experimental-results)
- [Conclusion](#conclusion)

## Abstract
- Stochastic language modeling: backing-off is a method to cope with the sparse data problem
- Propose distributions optimized for backing-off
- Theoretical derivations lead to distributions different from usual probability distributions
- Experiments show 10% improvement in terms of perplexity and 5% in word error rate

## 1. Introduction
- Stochastic language model: provides probabilities of a given word sequence through conditional probabilities p(w | h)
- **M-gram models**: consider histories with equivalent last (M - 1) words as equivalent
- Sparse data problem: too many possible events, less reliable estimates for unseen events
- Smoothing techniques: interpolation and backing-off

## 2. BACKING-OFF
- Uses less specific equivalence classes to estimate probabilities more reliably
- Normal probability distribution of coarser model used for backing-off can lead to bias towards heavily conditioned words

**Approach**:
- Assume equivalent histories according to specific and general equivalence relations
- Backing-off model: ```p(w | h) = { 
  £l'(w | h) if N(h, w) > 0
  p(w | h) = f(w | h) if N(h, w) = 0
}```
- Optimize parameters of the distribution function for seen and unseen events
- Two approaches lead to similar solutions with no additional computational overhead

## 3. MARGINAL DISTRIBUTION AS CONSTRAINT

**Approach to Modeling Marginal Distributions**
- Assumes maximum-likelihood estimates for p(w|h) = N(h, w)/N(h) and p(g|h) = N(h, g)/N(h)
- Moves β out of second sum and applies constraint equation (3): `I(h) = 1 - Σ α(w|h)p(w|h) - Σ (β(w|h)p(g|h) g: N(g, w)=0`

**Smoothing Techniques**
- Differ in probability estimate α for seen events
- Turing-Good estimates and linear/absolute discounting methods are examples
- Smoothing distribution kept fixed as β(w|h) = p(w|h)

**Proposed Approach**
- Leave the parameters of smoothing distribution β free and optimize with others
- Two approaches lead to similar solutions, independent of modeling type
- No additional computational overhead added

**First Approximation**
- Sum in denominator considered constant with respect to w
- Proportional to numerator: β(w|h) = Σ p(v|h) - Σ p(v|h) v (g: N(g, v)>0)
- Normalization ensures sum of β(w|h) equals unity

**Definitions:**
- `α(h) = 1 - d Σ N(h, w) > 0, 0 < d < 1`
- `p(w|h) = Σ [N(g, w) - d] g: N(g, w)>0 or p(w|h) = Σ [N(g, v) - d] v` (using special form of model)

**Equation (10)**
- `β(w|h) = N+(·,h,w)`
- Obtains distribution different from probability distribution p(w|h)
- Only takes into account if a word was observed in some coarse context and ignores frequency.

## 4. LEAVING-ONE-OUT

**Leaving-One-Out Technique for Backing-Off Model Estimation**

**Background:**
- Maximum likelihood estimation cannot estimate unseen events directly
- Cross-validation techniques like leaving-one-out technique are used to overcome this issue
- Leaving-one-out technique removes one event from training data and tests model on remaining events
- Sum of log probabilities of removed events serves as optimization criterion

**Applying the Leaving-One-Out Technique:**
1. Remove single unseen events (singletons) from training data
2. Train model on remaining data
3. Estimate leaving-one-out probability for removed event
4. Sum log probabilities of all removed events to obtain leaving-one-out log likelihood
5. Use this as optimization criterion
6. Final result: relative counts where only singletons are considered
7. Solutions of both approaches (Eqs. 13, 20) are similar

## 5. Experimental Results
1. Evaluated on German Verbmobil corpus and Wall Street-Journal task
2. Separate test sets used for evaluation in both tasks
3. Used trigram language models with non-linear interpolation smoothing
4. Standard, 'singleton', and 'marginal constraint' distributions tested
5. All models smoothed to avoid zero probabilities
6. Consistent improvement of new models over baseline (up to 10% lower perplexity, 5% lower word error rate)
7. Recognition results produced for Wall Street-Journal task
8. Compact trigram models built without loss of performance
9. Experiments show improvement in terms of perplexity and recognition results
10. Comparison with ARPA's official model reveals an improvement of about 9% in perplexity

## Conclusion
- Special backing-off distributions improve language models by up to 10% in perplexity and 5% in word error rate compared to baseline
- Both theoretically derived solutions do not depend on specific model or add extra computational costs.

