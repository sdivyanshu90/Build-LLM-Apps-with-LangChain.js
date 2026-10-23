// Source notebook: Notebook_4.ipynb
// See SOURCE_FILES.md for the original Deno runtime and dependencies.

// %% [markdown]
// # Lesson 4: Question answering

// %% [markdown]
// ![](./images/rag_diagram.png)

// %%
// Notebook cell 3
import "dotenv/config";

// %%
// Notebook cell 4
import { loadAndSplitChunks } from "./lib/helpers.ts";

const splitDocs = await loadAndSplitChunks({
    chunkSize: 1536,
    chunkOverlap: 128
});

// %%
// Notebook cell 5
import { initializeVectorstoreWithDocuments } from "./lib/helpers.ts";

const vectorstore = await initializeVectorstoreWithDocuments({
  documents: splitDocs,
});

// %%
// Notebook cell 6
const retriever = vectorstore.asRetriever();

// %% [markdown]
// # Document retrieval in a chain

// %%
// Notebook cell 8
import { RunnableSequence } from "@langchain/core/runnables";
import { Document } from "@langchain/core/documents";

const convertDocsToString = (documents: Document[]): string => {
  return documents.map((document) => {
    return `<doc>\n${document.pageContent}\n</doc>`
  }).join("\n");
};

/*
{
question: "What is deep learning?"
}
*/

const documentRetrievalChain = RunnableSequence.from([
    (input) => input.question,
    retriever,
    convertDocsToString
]);

// %%
// Notebook cell 9
const results = await documentRetrievalChain.invoke({
  question: "What are the prerequisites for this course?"
});
console.log(results);

// %% [markdown]
// # Synthesizing a response

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
import { RunnableMap } from "@langchain/core/runnables";

const runnableMap = RunnableMap.from({
  context: documentRetrievalChain,
  question: (input) => input.question,
});

await runnableMap.invoke({
    question: "What are the prerequisites for this course?"
})

// %% [markdown]
// # Augmented generation

// %%
// Notebook cell 14
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106"
});

// %%
// Notebook cell 15
const retrievalChain = RunnableSequence.from([
  {
    context: documentRetrievalChain,
    question: (input) => input.question,
  },
  answerGenerationPrompt,
  model,
  new StringOutputParser(),
]);

// %%
// Notebook cell 16
const answer = await retrievalChain.invoke({
  question: "What are the prerequisites for this course?"
});

console.log(answer);

// %%
// Notebook cell 17
const followupAnswer = await retrievalChain.invoke({
  question: "Can you list them in bullet point form?"
});

console.log(followupAnswer);

// %%
// Notebook cell 18
const docs = await documentRetrievalChain.invoke({
  question: "Can you list them in bullet point form?"
});

console.log(docs);

// %%
// Notebook cell 19


// %%
// Notebook cell 20


// %%
// Notebook cell 21


// %%
// Notebook cell 22


// %%
// Notebook cell 23


// %%
// Notebook cell 24


// %%
// Notebook cell 25


// %%
// Notebook cell 26


// %%
// Notebook cell 27


// %%
// Notebook cell 28


// %%
// Notebook cell 29


// %%
// Notebook cell 30

