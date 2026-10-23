// Source notebook: Notebook_5.ipynb
// See SOURCE_FILES.md for the original Deno runtime and dependencies.

// %% [markdown]
// # Lesson 5: Conversational Question answering

// %% [markdown]
// ![](./images/rag_diagram.png)

// %% [markdown]
// # Retrieval chain from previous lesson

// %% [markdown]
// Let's split and load our documents into a vector store and create a retriever. Then we will convert its output to a string.

// %%
// Notebook cell 5
import "dotenv/config";

// %%
// Notebook cell 6
import { loadAndSplitChunks } from "./lib/helpers.ts";

const splitDocs = await loadAndSplitChunks({
    chunkSize: 1536,
    chunkOverlap: 128
});

// %%
// Notebook cell 7
import { initializeVectorstoreWithDocuments } from "./lib/helpers.ts";

const vectorstore = await initializeVectorstoreWithDocuments({
  documents: splitDocs,
});

// %%
// Notebook cell 8
const retriever = vectorstore.asRetriever();

// %%
// Notebook cell 9
import { RunnableSequence } from "@langchain/core/runnables";
import { Document } from "@langchain/core/documents";

let convertDocsToString;
convertDocsToString = (documents: Document[]): string => {
  return documents.map((document) => {
    return `<doc>\n${document.pageContent}\n</doc>`
  }).join("\n");
};

let documentRetrievalChain;
documentRetrievalChain = RunnableSequence.from([
    (input) => input.question,
    retriever,
    convertDocsToString
]);

// %% [markdown]
// now that we have a retriever, lets build a retriever chain.

// %%
// Notebook cell 11
import { ChatPromptTemplate } from "@langchain/core/prompts";

const TEMPLATE_STRING = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the provided context, answer the user's question 
to the best of your ability using only the resources provided. 
Be verbose!

<context>

{context}

</context>

Now, answer this question using the above context:

{question}`;

const answerGenerationPrompt = ChatPromptTemplate.fromTemplate(
    TEMPLATE_STRING
);

// %%
// Notebook cell 12
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106"
});

// %%
// Notebook cell 13
const retrievalChain = RunnableSequence.from([
  {
    context: documentRetrievalChain,
    question: (input) => input.question,
  },
  answerGenerationPrompt,
  model,
  new StringOutputParser(),
]);

// %% [markdown]
// # Adding history

// %%
// Notebook cell 15
import { MessagesPlaceholder } from "@langchain/core/prompts";

const REPHRASE_QUESTION_SYSTEM_TEMPLATE = 
  `Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.`;

const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human", 
    "Rephrase the following question as a standalone question:\n{question}"
  ],
]);

// %%
// Notebook cell 16
const rephraseQuestionChain = RunnableSequence.from([
      rephraseQuestionChainPrompt,
      new ChatOpenAI({ temperature: 0.1, modelName: "gpt-3.5-turbo-1106" }),
      new StringOutputParser(),
])

// %%
// Notebook cell 17
import { HumanMessage, AIMessage } from "@langchain/core/messages";

let originalQuestion;
originalQuestion = "What are the prerequisites for this course?";

let originalAnswer;
originalAnswer = await retrievalChain.invoke({
  question: originalQuestion
});

console.log(originalAnswer);

// %%
// Notebook cell 18
const chatHistory = [
  new HumanMessage(originalQuestion),
  new AIMessage(originalAnswer),
];

await rephraseQuestionChain.invoke({
  question: "Can you list them in bullet point form?",
  history: chatHistory,
});

// %% [markdown]
// # Putting it all together

// %%
// Notebook cell 20
convertDocsToString = (documents: Document[]): string => {
  return documents.map((document) => `<doc>\n${document.pageContent}\n</doc>`).join("\n");
};

documentRetrievalChain = RunnableSequence.from([
  (input) => input.standalone_question,
  retriever,
  convertDocsToString,
]);

// %%
// Notebook cell 21
const ANSWER_CHAIN_SYSTEM_TEMPLATE = `You are an experienced researcher, 
expert at interpreting and answering questions based on provided sources.
Using the below provided context and chat history, 
answer the user's question to the best of 
your ability 
using only the resources provided. Be verbose!

<context>
{context}
</context>`;

const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human", 
    "Now, answer this question using the previous context and chat history:\n{standalone_question}"
  ]
]);

// %%
// Notebook cell 22
// Import already declared in an earlier cell.
await answerGenerationChainPrompt.formatMessages({
  context: "fake retrieved content",
  standalone_question: "Why is the sky blue?",
  history: [
    new HumanMessage("How are you?"),
    new AIMessage("Fine, thank you!")
  ]
});

// %%
// Notebook cell 23
import { RunnablePassthrough } from "@langchain/core/runnables";

const conversationalRetrievalChain = RunnableSequence.from([
  RunnablePassthrough.assign({
    standalone_question: rephraseQuestionChain,
  }),
  RunnablePassthrough.assign({
    context: documentRetrievalChain,
  }),
  answerGenerationChainPrompt,
  new ChatOpenAI({ modelName: "gpt-3.5-turbo" }),
  new StringOutputParser(),
]);

// %%
// Notebook cell 24
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";

// %%
// Notebook cell 25
const messageHistory = new ChatMessageHistory();

const finalRetrievalChain = new RunnableWithMessageHistory({
  runnable: conversationalRetrievalChain,
  getMessageHistory: (_sessionId) => messageHistory,
  historyMessagesKey: "history",
  inputMessagesKey: "question",
});

// %%
// Notebook cell 26
originalQuestion = "What are the prerequisites for this course?";

originalAnswer = await finalRetrievalChain.invoke({
  question: originalQuestion,
}, {
  configurable: { sessionId: "test" }
});

const finalResult = await finalRetrievalChain.invoke({
  question: "Can you list them in bullet point form?",
}, {
  configurable: { sessionId: "test" }
});

console.log(finalResult);

// %% [markdown]
// https://smith.langchain.com/public/fca11abd-c0ec-456f-800e-6edfaf6bcf68/r

// %%
// Notebook cell 28


// %%
// Notebook cell 29


// %%
// Notebook cell 30


// %%
// Notebook cell 31


// %%
// Notebook cell 32


// %%
// Notebook cell 33


// %%
// Notebook cell 34


// %%
// Notebook cell 35


// %%
// Notebook cell 36


// %%
// Notebook cell 37


// %%
// Notebook cell 38


// %%
// Notebook cell 39

