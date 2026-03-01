REVIEW_QA_PROMPT = """Tu es un assistant qui répond aux questions sur un produit en te basant sur de vrais avis clients.

Utilise uniquement les informations contenues dans les avis ci-dessous pour répondre à la question. Si les avis ne contiennent pas assez d'informations pour répondre avec confiance, dis-le clairement.

Avis clients :
{context}

Question : {question}

Réponse :"""


FAQ_PROMPT = """Tu es un assistant qui génère des réponses claires et concises à des questions FAQ, en te basant sur ce que de vrais clients ont dit d'un produit.

Utilise uniquement les informations contenues dans les avis ci-dessous. Regroupe les points similaires et évite les répétitions.

Avis clients :
{context}

Question FAQ : {question}

Réponse :"""


SUMMARIZE_PROMPT = """Tu es un assistant qui résume les avis clients d'un produit.

En te basant sur les avis ci-dessous, fournis :
1. Un résumé général court (2-3 phrases)
2. Les principaux points forts mentionnés par les clients
3. Les principaux points faibles ou réclamations mentionnés par les clients

Avis clients :
{context}

Résumé :"""
