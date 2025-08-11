from langchain_ollama import OllamaLLM as Ollama
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from Hua_Dian_QA.core.config import LLM_MODEL

class LLMManager:
    def __init__(self, retriever, model_name=LLM_MODEL):
        self.retriever = retriever
        self.model_name = model_name
        self.llm = Ollama(model=self.model_name, num_gpu=1, temperature=0)
        self.rag_chain = self._create_rag_chain()

    def _create_rag_chain(self):
        """Creates the RAG chain with history awareness."""
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        qa_system_prompt = """
### Context
{context}

### Instructions
1.  You are a specialized assistant for answering questions based on the provided **Context**.
2.  Synthesize a comprehensive and coherent answer from the **Context**.
3.  **Do not** use any of your own knowledge or any information outside of the **Context**.
4.  If the **Context** does not contain the answer, you **must** respond with the exact phrase: "根据提供的文档，我无法回答您的问题。"
5.  Your response should be in Chinese.
"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain

    def answer_question(self, question, chat_history):
        """Answers a question using the RAG chain."""
        return self.rag_chain.invoke({"input": question, "chat_history": chat_history})
