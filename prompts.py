prompt5 = """Your task is to determine whether the input log template is normal or anomaly.

Input Log Template: {question}
5 Known Normal Log Templates: {context}

Perform the following steps to check whether the input log template is abnormal:

1. Compare the Input Log Template with each of the Known Normal Log Templates.
2. Determine if the input Log Template is structurally and semantically identical or very similar to the Known Normal Log Templates.
   - If yes, classify the Input Log Template as Normal.
   - If no, proceed to step 3.
3. Analyze the Input Log Template independently and provide an assessment:
   - You need to carefully check the text content for keywords. Identify key elements such as error codes, status messages, and other significant terms.
   - In the log template, the parameters are replaced by <*>, so you should never consider <*> and missing values as the reason for abnormal log.

NOTE: Provide a high-confidence binary classification: 0 for normal and 1 for anomaly. You should generate reasons for your judgment.
Output format: Return back in JSON format, including keys: is_anomaly, reason: describes why the log is normal or abnormal.
Answer:"""

PROMPT_C1_2 = """Your task is to determine whether the input log is anomalous based on retrieved anomaly cases.

Input Log Template:
{query}

Actual log message:
{log_content}

Retrieved Anomalous Log Cases with Explanations:
{SimilarAnomaly}

CRITICAL INSTRUCTION:
If {SimilarAnomaly} contains ANY text (is not empty), you MUST classify this log as an anomaly (is_anomaly = 1) WITHOUT ANY CONTENT ANALYSIS.
Do NOT analyze whether the log content semantically matches the retrieved cases.
The mere presence of retrieved cases is the ONLY criteria for anomaly classification.

Steps to follow STRICTLY:
1. Check if {SimilarAnomaly} is empty or not.
2. If {SimilarAnomaly} is NOT empty, set is_anomaly = 1 and use text from {SimilarAnomaly} as the reason.
3. If {SimilarAnomaly} is empty, set is_anomaly = 0.
4. Extract contextual information from the log message if needed.

Output format (in JSON):
{{
  "is_anomaly": 1 if {SimilarAnomaly} contains ANY text, otherwise 0,
  "reason": "Direct explanation from the retrieved anomalous cases",
  "matched_case_ids": ["Relevant case IDs if available"],
  "extracted_context": {{
    "relevant fields extracted from the actual log message"
  }}
}}

Remember: Do NOT perform semantic comparison between the input log and the retrieved cases.
The only decision criterion is whether {SimilarAnomaly} contains text or not.
Answer:"""

# Uppercase version for compatibility
PROMPT5 = prompt5
