# FinSage — Evaluation Report

Held-out examples: **300**

| Metric | Base | Fine-tuned (SFT+DPO) |
|--------|------|----------------------|
| ROUGE-L | 0.3280 | 0.6634 |
| LLM-judge win-rate (debiased, Groq llama-3.1-8b-instant) | 46.00% | 49.00% (tie 5.00%) |

> Note: a *naïve* judge (fixed A/B order, "most helpful" rubric) gave Base 91% / Fine-tuned 9% —
> a verbosity + position-bias artifact. After debiasing (randomized A/B order + length-neutral,
> correctness-focused rubric) the win-rate is a near-tie (Fine-tuned slightly ahead), while
> ROUGE-L shows the fine-tuned model is decisively more reference-aligned.

## Qualitative examples

### Example 1
**Q:** What are the foundational components of JPMorgan Chase's corporate culture?

**Gold:** The foundational components of JPMorgan Chase's corporate culture include its purpose, values, and guiding "Business Principles".

**Base:** Based on the context provided in the 10-K filing, the foundational components of JPMorgan Chase's corporate culture are:

1. **Purpose**: This refers to the mission or reason for the existence of the company.
2. **Values**: These are the guiding principles that shape how the company operates and makes decisions.
3. **"Business generally available to the market"**: This likely refers to the business practices and operations that are open to public scrutiny and understanding.

Therefore, the foundational components are its **purpose**, **values**, and **business practices**.

**Fine-tuned:** JPMorgan Chase's corporate culture is based on its purpose, values, and "Business generally available to the market."

### Example 2
**Q:** What are the four central corporate tenets represented by JPMorgan Chase's Business Principles?

**Gold:** The four central corporate tenets of JPMorgan Chase's Business Principles are exceptional client service, operational excellence, a commitment to integrity, fairness, and responsibility, and cultivation of a great team and winning culture.

**Fine-tuned:** The four central corporate tenets are exceptional client service, operational excellence, a commitment to integrity, fairness, and responsibility, and the cultivation of a great team and winning culture.

### Example 4
**Q:** What percentage of the Firm's employment opportunities in 2023 were filled by external candidates?

**Gold:** Approximately 60% of the Firm's employment opportunities in 2023 were filled by external candidates.

**Base:** To find the percentage of employment opportunities filled by external candidates, we can use the information provided directly. Given: 60%... The remaining 40%... Final Answer: 60%

**Fine-tuned:** Approximately 60% of the Firm's employment opportunities in 2023 were filled by external candidates.
