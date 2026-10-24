"""Prompt templates used by the agent graph.
"""

SYSTEM_PROMPT = (
    "You are an expert in building energy science.\n\n"
    "Answer the questions using the provided retrieved context.\n\n"
    "Guidelines:\n"
    "1. Prioritize the retrieved context as the primary source of information.\n"
    "2. If the context is sufficient, base your answer strictly on it.\n"
    "3. If the context is partially relevant or insufficient, you may use your "
    "general knowledge to supplement the answer, but clearly rely on context where possible."
)

CLASSIFICATION_PROMPT = """You are a classifier for a building energy science RAG system.

For the question below, return BOTH:
1) query type: computational or factual
2) query complexity: simple or complex

Definitions:
- computational: requires numerical calculation with formulas/units
- factual: asks for explanation/definition/standard guidance without calculation
- simple: single concept, one direct retrieval likely sufficient
- complex: multi-concept, multi-step, or requires combining sources

Reply in EXACTLY this format:
type: computational or factual
complexity: simple or complex

Question: {question}"""

QUESTION_TYPE_PROMPT = """You are a question-type classifier for a building energy science exam question bank.

Classify the question below into EXACTLY ONE of these five categories:
- Cloze: a fill-in-the-blank statement with one or more blanks (e.g. "__________"), no answer options given.
- Single choice: presents answer options (A/B/C or a/b/c) and asks for exactly ONE correct answer.
- Multi-choice: presents answer options and explicitly asks to choose MORE THAN ONE (e.g. "choose 2", "choose 3").
- Calculation: requires numerical computation with formulas, units, and given values leading to a numeric result — no answer options given.
- Short Answer: requires a free-form written explanation, derivation, proof, or comparison — no answer options given, and not primarily a numeric calculation.

Examples:
[Cloze] "The greatest application of __________ has been in conjunction with variable-air-volume (VAV) systems for cooling-only services."
[Single choice] "Which design method should be used to size ducts downstream of terminal boxes...? A. equal friction method  B. static friction method  C. static regain method"
[Calculation] "Air at 29.92 in. Hg enters an adiabatic saturator at 80F dry-bulb and 66F wet-bulb temperature... Find the humidity ratio, degree of saturation, enthalpy, and specific volume of entering air."
[Multi-choice] "A design team is brainstorming strategies to save energy... Which of the following may be some of the team's ideas? (choose 2)  a. ...  b. ...  c. ...  d. ..."
[Short Answer] "A steam power plant operates between a boiler pressure of 42 bar and a condenser pressure of 0.035 bar. Compare the Rankine cycle performance with that obtained when the steam is superheated to 500C."

Question: {question}

Reply with EXACTLY one of these five labels, nothing else: Cloze, Single choice, Multi-choice, Calculation, Short Answer"""

TRIAGE_PLANNER_PROMPT = """You are a planning assistant for Stage 0.5 multimodal retrieval in building energy science.

Question: {question}

Task:
1) Decide whether Image / Formula / Table corpora are needed.
2) For each needed corpus, provide ONE focused retrieval query based on the exact parameters or information needed.
3) List the needed information as short bullets.

Output rules:
- Use only the tagged format below.
- Keep each query single-line.
- If a corpus is not needed, write "no", "none", and "none".

Reply in EXACTLY this tagged format:
images_needed: yes or no
images_query: <single focused query or none>
images_needs: <short semicolon-separated needs or none>

formulas_needed: yes or no
formulas_query: <single focused query or none>
formulas_needs: <short semicolon-separated needs or none>

tables_needed: yes or no
tables_query: <single focused query or none>
tables_needs: <short semicolon-separated needs or none>
"""

KEY_INFO_EXTRACTION_PROMPT = """You are an information extraction specialist for a building energy science RAG system.

Question: {question}

Below is retrieved content from specialized corpora (images, formulas, tables) that MAY be relevant:

{multimodal_context}

Your task: extract ONLY the specific information needed to solve this question — parameters, formula
statements (with variable definitions), table values, or diagram/image insights. Do NOT copy irrelevant
passages. Do NOT answer the question itself. Be concise — bullet-point the extracted parameters,
formulas, and values only.

If nothing in the retrieved content is actually relevant or useful for solving this question, reply
with EXACTLY: No additional information needed.

Reply with ONLY the extracted key information (bullet points), no preamble, no restatement of the
question, no answer to the question."""

RELEVANCE_PROMPT = """You are a retrieval quality judge for a building energy science RAG system.

Question: {question}

Retrieved Context:
{context}

Step 1 — Are the retrieved chunks sufficient to answer the question fully?
         Answer with one word on the first line: yes  OR  partial  OR  no

Step 2 — If your answer is "partial" or "no", list the specific pieces of information
         that are MISSING from the context (e.g. "missing: saturation pressure at 80°F,
         convection coefficient formula, R-value of fiberglass insulation").
         If your answer is "yes", write "missing: none".

Reply in this EXACT format:
verdict: yes or partial or no
missing: <list what is missing, or none>"""

QUERY_REWRITE_PROMPT = """You are a query rewriting specialist for a building energy science RAG system.

The original query retrieved context that was insufficient. Your job is to rewrite the query
to find what is MISSING — not to rephrase what was already found.

Original question: {question}
Retrieval attempt: {attempt}

What was already retrieved (may be partially relevant):
{context_summary}

Specific missing information identified by the relevance judge:
{missing_info}

Rewrite the query to directly target the missing information above.
Focus on specific technical terms, property names, standard numbers, table names,
or equations that would appear in textbook passages covering this missing content.

Reply with ONLY the rewritten query. No preamble."""

REFLECTION_PROMPT = """You are a quality evaluator for a building energy science answer.

Question: {question}

Generated Answer:
{answer}

Retrieved Context used:
{context}

Evaluate the answer on THREE criteria. Reply in this EXACT format (one word per line):
addresses_question: yes or no
grounded_in_context: yes or partially or no
units_and_magnitude_ok: yes or no or na"""

REFINE_PROMPT = """You are an expert in building energy science. Your previous answer had quality issues.

Question: {question}

Your previous answer:
{answer}

Quality critique:
{critique}

Retrieved Context:
{context}

Instructions:
1. Rewrite the answer to fix all issues identified in the critique.
2. Be precise with numbers and units throughout.
3. If this question involves any numerical calculation, apply the following verification:
   - PASS 1 (Setup): Identify every variable, its value, and its unit explicitly.
   - PASS 2 (Calculate): Compute the result step by step.
   - PASS 3 (Verify): Re-derive the result independently. If the two passes disagree,
     find and fix the error before giving the final answer.
4. Ground your answer strictly in the Retrieved Context where possible."""

CODE_GENERATION_PROMPT = """You are an expert in building energy science and Python programming.

A student has asked a calculation question. Retrieved context from authoritative building energy
science sources is provided below. Your task: write Python code that solves the question exactly.

QUESTION:
{question}

RETRIEVED CONTEXT (formulas, parameters, and values from the building energy science corpus):
{context}{error_hint}
INSTRUCTIONS:
1. Read the retrieved context carefully — prefer the exact formulas and values from it.
   If the context specifies an ASHRAE formula, use that formula, not a textbook approximation.
2. Write clean, step-by-step Python code that solves the question.
3. Define EVERY variable with an inline comment showing its full name and unit.
4. Show intermediate results with print() statements labeled clearly.
5. Print the FINAL ANSWER on its own line in the format: "ANSWER: <value> <unit>"
6. Use only Python's built-in math module. Do NOT import numpy, scipy, or any external library.
7. If the context does not provide a specific value needed, use a standard engineering default
   and add a comment: # default from ASHRAE / standard engineering value

OUTPUT FORMAT:
Return ONLY a Python code block enclosed in ```python ... ```.
No explanation, preamble, or text outside the code block."""

RESULT_VERIFICATION_PROMPT = """You are verifying a calculated answer to a building energy science question.

Question: {question}
Execution Result: {result}

Check THREE things for the answer in the execution result:
1. Units: Does the final answer carry correct physical units for this type of calculation?
2. Magnitude: Is the numerical value physically plausible for a building energy context?
   (e.g., room heat loss of 72,000 W is suspicious; 720 W is plausible)
3. Completeness: Does the answer directly address what the question asked?

Reply in this EXACT format (one word or short phrase per line):
units_correct: yes or no
magnitude_plausible: yes or no
answer_complete: yes or no
notes: <one sentence describing any issue found, or "all checks passed">"""

CODE_ANSWER_FORMAT_PROMPT = """You are an expert in building energy science.

A calculation question was solved by executing Python code. Format the final answer clearly.

Question: {question}
Code execution result: {execution_result}
Retrieved context used: {context_summary}

Write a concise, precise final answer that:
1. States the numerical result with correct units prominently.
2. Briefly explains the formula or method used (one sentence).
3. Cites the source from the retrieved context if applicable.
4. If the question is multiple choice, identify the correct option.

Keep the answer under 200 words. Be direct — no preamble."""
