# Build LLM Apps with LangChain.js

This repository contains notes, code, and projects from the course **[Build LLM Apps with LangChain.js](https://www.deeplearning.ai/short-courses/build-llm-apps-with-langchain-js/)** by **[DeepLearning.AI](https://www.deeplearning.ai/)** in collaboration with **[LangChain](https://www.langchain.com/)**.

## Introduction

JavaScript is the world’s most popular programming language, and now developers can program in JavaScript to build powerful LLM apps.

This course shows web developers how to expand their toolkits with LangChain.js, a popular JavaScript framework for building with LLMs. It covers useful concepts for creating powerful, context-aware applications that can reason, act, and connect to the world's data.

By taking this course, you will:

* Learn to use LangChain’s underlying abstractions to build your own JavaScript apps.
* Understand the basics of Retrieval Augmented Generation (RAG).
* Have the structure of a basic conversational retrieval system that you can use for building your own chatbots.

---

## Course Topics

Click on each topic to see a detailed explanation.

<details>
<summary><strong>1. Building Blocks</strong></summary>

This foundational module introduces the core abstractions of LangChain.js, which are the fundamental components for any application. These blocks are designed to be modular, composable, and easily integrated into any JavaScript or TypeScript project.

* **Models (LLMs and Chat Models):** This is the "brain" of the operation. The course distinguishes between two types of models. First are standard **LLMs**, which take a simple string as input and return a string as output. Second are **Chat Models**, which are more powerful. They take a structured list of chat messages (like `SystemMessage`, `HumanMessage`, and `AIMessage`) as input and return a chat message as output. This structured approach is essential for building conversational applications, as it allows the model to understand context, roles, and history.
* **Prompts:** LLMs don't magically know what you want. A prompt is the instruction you give to the model. **Prompt Templates** are a key LangChain abstraction that allows you to parameterize these instructions. Instead of hard-coding a question, you create a template with variables (e.g., "Tell me a joke about {topic}"). Your application can then dynamically insert the user's input (like "dinosaurs") into the template before sending it to the model. This makes your prompts reusable and clean.
* **Output Parsers:** By default, LLMs return text. But in a real application, you almost always need structured data, like JSON, a list, or a boolean. **Output Parsers** are a powerful abstraction that takes the raw text output from an LLM and parses it into a more usable format. For example, you can instruct the model to return a JSON object, and the `StructuredOutputParser` will automatically parse the model's string response, validate it, and return a clean JavaScript object your code can work with.
* **Chains:** Chains are the most central concept in LangChain. They are how you "chain" these building blocks together to create a single, seamless operation. The most basic chain is an `LLMChain`, which links a `PromptTemplate`, a `Model`, and (optionally) an `OutputParser`. This single chain takes user variables, formats the prompt, sends it to the model, gets the response, and parses the output. This concept of chains (which can be simple or incredibly complex) is what gives the framework its name and its power.

</details>

<details>
<summary><strong>2. Loading and Preparing Data</strong></summary>

This module addresses a critical limitation of LLMs: they only know about the data they were trained on. To build truly useful applications, you must connect them to your own private data. This process is the first step in Retrieval Augmented Generation (RAG).

* **Document Loaders:** The first step is getting your data into the application. LangChain.js provides a wide array of **Document Loaders** to ingest data from virtually any source. The course covers loaders for plain text files (`.txt`), CSVs, PDFs, and even directly from web pages (web scraping). Each loader ingests the data and converts it into a standardized `Document` object.
* **The `Document` Object:** This is the standard format LangChain uses for text. A `Document` is a simple JavaScript object containing two main properties: `page_content` (the actual text) and `metadata` (an object for source information, like the filename, URL, or page number). This standardization makes the rest of the process source-agnostic.
* **Document Transformers (Text Splitters):** Once you load a document, you can't just send it to an LLM. A 500-page PDF, for example, will not fit into a model's context window (the limit on how much text it can process at once). The solution is to split the document into smaller, more manageable chunks. This is handled by **Document Transformers**. The most important one is the `RecursiveCharacterTextSplitter`. This utility intelligently splits large texts into smaller pieces (e.g., 1000 characters each) while trying to maintain semantic meaning by splitting on paragraphs, sentences, and words. This chunking is *essential* for the next step: embedding.

</details>

<details>
<summary><strong>3. Vectorstores and Embeddings</strong></summary>

This module is the core of RAG. Once your data is loaded and split into chunks, you need a way to *find* the most relevant chunks to answer a user's question. You can't use a traditional keyword search, as it would miss context and synonyms. The solution is semantic search, powered by embeddings and vectorstores.

* **Embeddings:** An embedding is a numerical representation of text, generated by an "embedding model" (like OpenAI's `text-embedding-ada-002`). This model converts a piece of text (like one of your document chunks) into a long list of numbers (a "vector"). The magic of these vectors is that texts with similar *meanings* will have mathematically similar vectors. "How much is that doggy in the window?" will have a vector very close to "What is the price of that puppy in the shop?"
* **Vectorstores:** A vectorstore is a specialized database designed to store and efficiently search these vectors. After splitting your documents, you use an embedding model to create a vector for *every single chunk* and then store these vectors in the vectorstore. LangChain.js has integrations for many vectorstores, from in-memory options perfect for development (like `HNSWLib` or `MemoryVectorStore`) to scalable cloud-native databases (like Pinecone, Chroma, or Supabase).
* **Semantic Search (Retrieval):** This is where it all comes together. When a user asks a question (e.g., "What is the policy on sick leave?"), the application first creates an *embedding of the user's question*. It then queries the vectorstore, asking, "Find me the top 3 vectors in your database that are most similar to this new vector." The vectorstore performs this similarity search and returns the original document chunks corresponding to those vectors. These chunks are the most relevant pieces of information from your documents to answer the user's question.

</details>

<details>
<summary><strong>4. Question Answering</strong></summary>

This module teaches you how to combine the previous steps into a complete RAG (Retrieval Augmented Generation) pipeline. You have the "Retrieval" part (getting relevant documents from the vectorstore). Now you need the "Generation" part (using an LLM to generate an answer).

* **The RAG Flow:** This is the canonical flow for a "question answering over documents" system.
    1.  **Retrieve:** The user's question is used to query the vectorstore (as described in the previous module). This returns a list of relevant document chunks.
    2.  **Augment:** A new prompt is constructed. This prompt "augments" the user's original question with the *context* it just retrieved. The prompt looks something like this: "You are a helpful assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say so. Context: `[...retrieved document chunks...]` Question: `[...user's original question...]`"
    3.  **Generate:** This entire, augmented prompt is sent to an LLM. The LLM then generates an answer.
* **The Benefit:** This process "grounds" the LLM's response in your provided data. Instead of making up an answer (hallucinating), the model is forced to synthesize its answer based *only* on the facts you provided in the context. This allows you to build chatbots that can accurately answer questions about specific, private, or brand-new information.
* **LangChain Abstraction:** LangChain.js simplifies this entire flow into a high-level abstraction, often called a `RetrievalQA` chain. You simply initialize this chain with your LLM and your vectorstore (as a `retriever`), and it handles the retrieval, augmenting, and generation steps for you under the hood.

</details>

<details>
<summary><strong>5. Conversational Question Answering</strong></summary>

This module elevates the Q&A system from a single-shot tool to a true, stateful chatbot. The problem with the `RetrievalQA` chain is that it's "stateless"—it has no memory. If you ask, "What is LangChain.js?" and then ask a follow-up, "Why is it useful?", the stateless system will have no idea what "it" refers to.

* **The Challenge of Memory:** To build a conversational agent, the system must remember the chat history. This history is needed for two reasons:
    1.  To provide conversational context to the LLM so it can formulate a natural reply.
    2.  To understand follow-up questions.
* **The Solution:** The `ConversationalRetrievalChain` is introduced. This chain adds a crucial new step to the RAG flow:
    1.  **Get Input:** The chain receives the new `question` and the `chat_history`.
    2.  **Condense Question:** It *first* sends the `question` and `chat_history` to an LLM with a specific prompt: "Given this chat history and a follow-up question, rephrase the question to be a standalone question." For example, "Why is it useful?" becomes "Why is LangChain.js useful?"
    3.  **Retrieve:** It uses this *new, standalone question* to query the vectorstore and retrieve relevant documents.
    4.  **Generate:** It then sends the *retrieved documents*, the *original follow-up question*, and the *chat history* to the final LLM to generate a natural, conversational answer.
* **Memory Management:** LangChain.js provides `Memory` classes (like `BufferMemory`) to automatically manage this chat history, storing past human and AI messages and making them available for the next turn in the conversation. This module provides the complete architecture for a "chat with your data" application.

</details>

<details>
<summary><strong>6. Shipping as a Web API</strong></summary>

This final module bridges the gap from a local script to a real-world application. As this course is for JavaScript developers, it focuses on the most common way to make an LLM application available to a frontend (like a React or Vue app): by exposing it as a web API.

* **Server-Side Logic:** The LLM, the vectorstore, and the LangChain.js chains all run on a server. This is critical for security (to protect your API keys) and performance.
* **API Frameworks:** The course demonstrates how to use a standard Node.js server framework, such as **Express.js**, to create this API.
* **Creating Endpoints:** You learn to create a simple API endpoint, for example, `POST /chat`. This endpoint will accept a request body containing the user's `message` and perhaps a `sessionId` to track the conversation.
* **Handling Requests:** When the server receives a request at this endpoint, its handler function will:
    1.  Initialize (or retrieve) the `ConversationalRetrievalChain`.
    2.  Retrieve the correct chat history for the given `sessionId`.
    3.  Call the chain with the user's `message` and the `chat_history`.
    4.  Wait for the chain to process and return the AI's `answer`.
    5.  Send this `answer` back to the client as a JSON response.
* **Frontend Integration:** With this API running, any frontend application can now `fetch` this endpoint, send a user's message, and receive the LLM's response, allowing you to build a custom chat UI that is powered by your sophisticated LangChain.js backend. This module completes the journey from basic LLM concepts to a fully deployable, production-ready application.

</details>

---

## Acknowledgement

This repository is for personal learning purposes only, based on the "Build LLM Apps with LangChain.js" course.

All course content, materials, and structure are provided by **DeepLearning.AI** in collaboration with **LangChain**. All rights, copyrights, and licenses for the original course materials are held by DeepLearning.AI.

A huge thank you to both organizations for creating this excellent and accessible content for the developer community.

You can find the official course here:
[Link](https://www.deeplearning.ai/short-courses/build-llm-apps-with-langchain-js/)