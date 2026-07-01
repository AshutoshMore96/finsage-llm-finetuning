# Fine-tune vs. RAG vs. Prompt-engineering — Benchmark

Held-out finance Q&A examples: **150**

| Approach | ROUGE-L | Token-F1 | Avg latency (s) | Needs training? | Needs corpus? |
|----------|--------:|---------:|----------------:|:---------------:|:-------------:|
| Base (zero-shot) | 0.2953 | 0.2699 | 6.59 | No | No |
| Prompt-engineered | 0.3742 | 0.3415 | 4.82 | No | No |
| RAG | 0.1603 | 0.1666 | 6.50 | No | Yes |
| Fine-tuned (SFT+DPO) | 0.6599 | 0.6484 | 1.89 | Yes | No |
| Fine-tuned + RAG | 0.2325 | 0.2386 | 1.72 | Yes | Yes |

## When to use which (the senior takeaways)

- **Prompt-engineering** — fastest to ship, zero training cost. Best when the base model already has the knowledge and you only need to shape format/behavior.
- **RAG** — best when answers depend on *fresh or proprietary documents* (new filings, internal data). Knowledge updates by re-indexing, no retraining. Adds retrieval latency and depends on retrieval quality.
- **Fine-tuning (QLoRA+DPO)** — best for *durable domain style, formatting, and reasoning patterns*; lowest per-query latency (no retrieval) and cheapest at scale. Costs an upfront training run and must be retrained to absorb new facts.
- **Fine-tuned + RAG** — usually the strongest: the model speaks the domain *and* is grounded in retrieved evidence. This is the common production pattern.

## Qualitative samples

**Q:** What were the OG&A expenses as a percentage of sales in 2022 and 2021?

**Gold:** OG&A expenses as a percentage of sales were 16.09% in 2022 and 16.83% in 2021.

- *Base (zero-shot):* To calculate the OG&A expenses as a percentage of sales for both years, we use the provided percentages directly since they are already expressed as percentages.

For 2022:
\[ \text{Percentage} = 16.09\% \]

For 2021:
\[ \text{Percentage} = 16.83\% \]

So, the OG&A expenses as a percentage of sales in 2022 was 16.09%, and in 2021 it was 16.83%.
- *Prompt-engineered:* The OG&A expenses as a percentage of sales in 2022 and 2021 were already provided in the context:

- For 2022: 16.09%
- For 2021: 16.83%

Answer: 16.09% for 2022, 16.83% for 2021
- *RAG:* To find the SG&A expenses as a percentage of sales for 2022 and 2021, we can use the data provided in the third document.

For 2022:
SG&A expenses = $25,094 million
Net sales = $51,345 million

Percentage of total net sales for 2022 = (25,094 / 51,345) * 100%
= 0.4886 * 100%
= 48.86%

For 2021:
SG&A expenses = $21,973 million
Net sales = $43,887 million

Percentage of total net sales for 2021 = (21,973 / 43,887) * 100%
= 0.4999 * 100%
≈ 50.00%

Therefore, the SG&A expenses as a percentage of sales in 2022 was approximately 48.86% and in 2021 it was approximately 50.00%.
- *Fine-tuned (SFT+DPO):* 16.09% in 2022 and 16.83% in 2021
- *Fine-tuned + RAG:* 6%

**Q:** What was the rent expense as a percentage of sales in 2022 and 2021?

**Gold:** The rent expense as a percentage of sales was 0.57% in 2022 and 0.61% in 2021.

- *Base (zero-shot):* To calculate the rent expense as a percentage of sales for both years, we use the provided percentages directly since they are already stated.

For 2022:
\[ \text{Rent Expense as a Percentage of Sales} = 0.57\% \]

For 2021:
\[ \text{Rent Expense as a Percentage of Sales} = 0.61\% \]

So, the rent expense as a percentage of sales in 2022 was 0.57%, and in 2021 it was 0.61%.
- *Prompt-engineered:* The rent expense as a percentage of sales in 2022 was 0.57%, and in 2021 it was 0.61%. 

To find the percentage of sales for each year, we use the given values directly since they are provided in the context.

For 2022:
\[ \text{Percentage} = \frac{\text{Rent Expense}}{\text{Sales}} \times 100 = \frac{0.57}{1} \times 100 = 57\% \]

For 2021:
\[ \text{Percentage} = \frac{\text{Rent Expense}}{\text{Sales}} \times 100 = \frac{0.61}{1} \times 100 = 61\% \]

Answer: 57% for 2022 and 61% for 2021.
- *RAG:* To find the rent expense as a percentage of sales in 2022 and 2021, we need to use the information provided in the operating expenses section.

From the data:
- In 2022, the percentage of total net sales for operating expenses was 13%.
- This means that the operating expenses in 2022 were 13% of total net sales.

For 2021:
- The percentage of total net sales for operating expenses was 12%.

Therefore, the rent expense as a percentage of sales in 2022 was 13%, and in 2021 it was 12%.

Final Answer: 2022 - 13%, 2021 - 12%
- *Fine-tuned (SFT+DPO):* 0.57% in 2022 and 0.61% in 2021
- *Fine-tuned + RAG:* 7%

**Q:** What was the operating profit as a percentage of sales in 2022 and 2021?

**Gold:** The operating profit as a percentage of sales was 2.78% in 2022 and 2.52% in 2021.

- *Base (zero-shot):* To calculate the operating profit as a percentage of sales for both years, we use the provided percentages directly since they are already expressed as percentages.

For 2022:
\[ \text{Operating Profit as a Percentage of Sales} = 2.78\% \]

For 2021:
\[ \text{Operating Profit as a Percentage of Sales} = 2.52\% \]

So, the operating profit as a percentage of sales in 2022 was 2.78%, and in 2021 it was 2.52%.
- *Prompt-engineered:* The operating profit as a percentage of sales in 2022 was 2.78%, and in 2021 it was 2.52%. 

To find the percentage of sales for each year:
- 2022: \( \frac{2.78}{100} \times 100 = 2.78\% \)
- 2021: \( \frac{2.52}{100} \times 100 = 2.52\% \)

Answer: 2.78% in 2022, 2.52% in 2021
- *RAG:* To find the operating profit as a percentage of sales in 2022 and 2021, we need to use the total sales and trading revenue provided in the context.

From [doc 3]:
- Total sales and trading revenue in 2021 was $14,768 million.
- Total sales and trading revenue in 2022 was $16,135 million.
- Total sales and trading revenue in 2023 was $16,830 million.

First, calculate the operating profit as a percentage of sales for 2021:
\[ \text{Operating Profit Margin (2021)} = \left( \frac{\text{Operating Profit}}{\text{Sales and Trading Revenue}} \right) \times 100 \]
\[ \text{Operating Profit (2021)} = \frac{7.904 \text{ billion}}{14.768 \text{ billion}} \times 100 \]
\[ \text{Operating Profit Margin (2021)} = \left( \frac{7.904}{14.768} \right) \times 100 \approx 53.54\% \]

Next, calculate the operating profit as a percentage of sales for 2022:
\[ \text{Operating Profit (2022)} = \frac{7.904 \text{ billion} + 5.062 \text{ billion}}{16.135 \text{ billion}} \times 100 \]
\[ \text{Operating Profit (2022)} = \frac{12.966 \text{ billion}}{16.135 \text{ billion}} \times 100 \]
\[ \text{Operating Profit Margin (2022)} = \left( \frac{12.966}{16.135} \right) \times 100 \approx 80.46\% \]

Therefore, the operating profit as a percentage of sales in 2021 was approximately 53.54%, and in 2022 it was approximately 80.46%.
- *Fine-tuned (SFT+DPO):* 2.78% in 2022 and 2.52% in 2021
- *Fine-tuned + RAG:* In 2022, it was 13.3 percent, and in 2021, it was 15.3 percent.

**Q:** What percentage increase in net earnings per diluted share was recorded in 2022 compared to 2021?

**Gold:** 41.0%

- *Base (zero-shot):* To calculate the percentage increase in net earnings per diluted share from 2021 to 2022:

1. Identify the net earnings per diluted share for each year:
   - 2022: $3.06
   - 2021: $2.17

2. Calculate the difference between the two years:
   \[
   \text{Difference} = 3.06 - 2.17 = 0.89
   \]

3. Divide the difference by the value of the base year (2021) to find the percentage increase:
   \[
   \text{Percentage Increase} = \left( \frac{0.89}{2.17} \right) \times 100\%
   \]

4. Perform the division and multiplication:
   \[
   \text{Percentage Increase} = \left( \frac{0.89}{2.17} \right) \times 100\% \approx 0.4085 \times 100\% \approx 40.85\%
   \]

Therefore, the percentage increase in net earnings per diluted share from 2021 to 2022 was approximately **40.85%**.
- *Prompt-engineered:* To find the percentage increase, we use the formula:

\[
\text{Percentage Increase} = \left( \frac{\text{New Value} - \text{Old Value}}{\text{Old Value}} \right) \times 100
\]

Substituting the given values:

\[
\text{Percentage Increase} = \left( \frac{3.06 - 2.17}{2.17} \right) \times 100
\]

\[
\text{Percentage Increase} = \left( \frac{0.89}{2.17} \right) \times 100
\]

\[
\text{Percentage Increase} = 0.4095 \times 100
\]

\[
\text{Percentage Increase} = 40.95\%
\]

Answer: 40.95%
- *RAG:* To calculate the percentage increase in net earnings per diluted share from 2021 to 2022, we use the following formula:

\[
\text{Percentage Increase} = \left( \frac{\text{Net Income 2022} - \text{Net Income 2021}}{\text{Net Income 2021}} \right) \times 100
\]

From the provided context:
- Net income in fiscal 2022 was $6,292 million.
- Net income in fiscal 2021 was $5,844 million.

Now, let's calculate the percentage increase:

\[
\text{Percentage Increase} = \left( \frac{6,292 - 5,844}{5,844} \right) \times 100
\]

\[
\text{Percentage Increase} = \left( \frac{448}{5,844} \right) \times 100
\]

\[
\text{Percentage Increase} = 0.0761 \times 100
\]

\[
\text{Percentage Increase} = 7.61\%
\]

The percentage increase in net earnings per diluted share recorded in 2022 compared to 2021 was 7.61%.
- *Fine-tuned (SFT+DPO):* 41.0%
- *Fine-tuned + RAG:* 12.9%
