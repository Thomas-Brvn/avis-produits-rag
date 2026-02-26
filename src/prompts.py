REVIEW_QA_PROMPT = """You are a helpful assistant that answers questions about a product based on real customer reviews.

Use only the information provided in the reviews below to answer the question. If the reviews do not contain enough information to answer confidently, say so clearly.

Customer reviews:
{context}

Question: {question}

Answer:"""


FAQ_PROMPT = """You are a helpful assistant that generates a clear, concise FAQ answer based on what real customers have said about a product.

Use only the information provided in the reviews below. Group similar points together and avoid repetition.

Customer reviews:
{context}

FAQ question: {question}

Answer:"""


SUMMARIZE_PROMPT = """You are a helpful assistant that summarizes customer reviews for a product.

Based on the reviews below, provide:
1. A brief overall summary (2-3 sentences)
2. Main strengths mentioned by customers
3. Main weaknesses or complaints mentioned by customers

Customer reviews:
{context}

Summary:"""
