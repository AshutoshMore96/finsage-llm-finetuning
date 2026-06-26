# FinSage — Evaluation Report

Held-out examples: **300**

| Metric | Base | Fine-tuned (SFT+DPO) |
|--------|------|----------------------|
| ROUGE-L | 0.3280 | 0.6634 |
| LLM-judge win-rate | 91.00% | 9.00% (tie 0.00%) |

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

**Base:** Based on the context provided in the 10-K filing, the four central corporate tenets represented by JPMorgan Chase's Business Principles are:

1. Exceptional client service
2. Operational excellence
3. A commitment to integrity, fairness, and responsibility
4. Cultivation of a great team and winning culture

The final answer is: Exceptional client service, operational excellence, a commitment to integrity, fairness, and responsibility, and cultivation of a great team and winning culture.

**Fine-tuned:** The four central corporate tenets are exceptional client service, operational excellence, a commitment to integrity, fairness, and responsibility, and the cultivation of a great team and winning culture.

### Example 3
**Q:** How does JPMorgan Chase support its diversity, equity, and inclusion strategies?

**Gold:** JPMorgan Chase supports its diversity, equity, and inclusion strategies through its Business Resource Groups and global Diversity, Equity & Inclusion centers of excellence.

**Base:** JPMorgan Chase supports its diversity, equity, and inclusion (DEI) strategies through several initiatives:

1. **Business Resource Groups (BRGs):** These are employee-led groups that support the firm's DEI goals by utilizing the unique perspectives of their members. BRGs can include groups like Black Professionals Network, Women's Network, LGBTQ+ Network, etc.

2. **Global Diversity, Equity & Inclusion Centers of Excellence:** These centers lead the firm's strategy in supporting its commitments to create more equity and lasting impact in communities, and to strengthen its inclusive culture.

In summary, JPMorgan Chase leverages both employee-led groups and central leadership to drive its DEI efforts. This approach allows the company to tailor its support to various employee groups while also ensuring a cohesive and strategic approach across the organization.

**Fine-tuned:** JPMorgan Chase supports its diversity, equity, and inclusion strategies through its Business Resource Groups, which leverage the unique perspectives of their members, and through global Diversity, Equity & Inclusion centers of excellence that lead the firm's strategy in creating more equity and lasting impact in communities.

### Example 4
**Q:** What percentage of the Firm's employment opportunities in 2023 were filled by external candidates?

**Gold:** Approximately 60% of the Firm’s employment opportunities in 2023 were filled by external candidates.

**Base:** To find the percentage of employment opportunities filled by external candidates, we can use the information provided directly.

Given:
- 60% of the Firm's employment opportunities were filled by external candidates.
- The remaining 40% were filled by existing employees.

The percentage of employment opportunities filled by external candidates is 60%.

Final Answer: 60%

**Fine-tuned:** Approximately 60% of the Firm's employment opportunities in 2023 were filled by external candidates.

### Example 5
**Q:** What is JPMorgan Chase's approach to promoting an inclusive work environment during the recruitment process?

**Gold:** JPMorgan Chase promotes an inclusive work environment during the recruitment process by focusing on career coaching, mentorship, and prioritizing the unique skills of candidates over traditional degree requirements.

**Base:** Based on the information provided in the 10-K filing, JPMorgan Chase's approach to promoting an inclusive work environment during the recruitment process includes:

1. Engaging in efforts to build and foster an inclusive work environment.
2. Utilizing its global Diversity, Equity & Inclusion centers of excellence to support its diversity, equity, and inclusion strategies.
3. Implementing initiatives such as career coaching and mentorship.
4. Focusing on the skills of a job candidate rather than their educational background (degree).

The key aspect here is that JPMorgan Chase focuses on the candidate's skills rather than their educational qualifications, which promotes inclusivity by valuing diverse backgrounds and experiences.

**Fine-tuned:** JPMorgan Chase promotes an inclusive work environment during the recruitment process by focusing on the skills of a job candidate rather than their degree, and by engaging in efforts to build and foster an inclusive work environment.
