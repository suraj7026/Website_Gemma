import rag_app as rag

print(rag.chain.invoke({"question": "What is SIFT?"}))