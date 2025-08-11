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
### Instructions
1.  You are an intelligent assistant. Your task is to answer questions based on the provided **Context**.
2.  Analyze the **Context** thoroughly. Your answer should be comprehensive, coherent, and primarily based on the information within the **Context**.
3.  If the **Context** directly answers the question, synthesize the information and present it clearly.
4.  If the **Context** does not contain a direct answer, but contains related information, you can infer an answer, but you **must** state that you are inferring and explain your reasoning based on the provided information.
5.  If the **Context** is completely irrelevant to the question, you **must** respond with: "根据提供的文档，我无法回答您的问题。"
6.  When possible, quote relevant snippets from the **Context** to support your answer.
7.  Your response must be in Chinese.

### Context
{context}
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
        result = self.rag_chain.invoke({"input": question, "chat_history": chat_history})
        
        # If the answer indicates that the question could not be answered,
        # clear the context to avoid showing irrelevant source documents.
        if "我无法回答您的问题" in result.get("answer", ""):
            result["context"] = []
            
        return result
