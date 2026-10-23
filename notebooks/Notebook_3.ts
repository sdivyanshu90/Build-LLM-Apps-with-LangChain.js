// Source notebook: Notebook_3.ipynb
// See SOURCE_FILES.md for the original Deno runtime and dependencies.

// %% [markdown]
// # Lesson 3: Vectorstores and embeddings

// %% [markdown]
// # Vectorstore ingestion

// %%
// Notebook cell 3
import "dotenv/config";

// %%
// Notebook cell 4
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings();

await embeddings.embedQuery("This is some sample text");

// %%
// Notebook cell 5
import { similarity } from "ml-distance";

const vector1 = await embeddings.embedQuery(
    "What are vectors useful for in machine learning?"
);
const unrelatedVector = await embeddings.embedQuery(
    "A group of parrots is called a pandemonium."
);

// %%
// Notebook cell 6
similarity.cosine(vector1, unrelatedVector);

// %%
// Notebook cell 7
const similarVector = await embeddings.embedQuery(
    "Vectors are representations of information."
);

similarity.cosine(vector1, similarVector);

// %%
// Notebook cell 8
// Peer dependency
import * as parse from "pdf-parse";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { 
    RecursiveCharacterTextSplitter
} from "langchain/text_splitter";

const loader = new PDFLoader("./data/MachineLearning-Lecture01.pdf");

const rawCS229Docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 128,
  chunkOverlap: 0,
});

const splitDocs = await splitter.splitDocuments(rawCS229Docs);

// %%
// Notebook cell 9
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const vectorstore = new MemoryVectorStore(embeddings);

// %%
// Notebook cell 10
await vectorstore.addDocuments(splitDocs);

// %%
// Notebook cell 11
const retrievedDocs = await vectorstore.similaritySearch(
    "What is deep learning?", 
    4
);

const pageContents = retrievedDocs.map(doc => doc.pageContent);

pageContents

// %% [markdown]
// # Retrievers

// %%
// Notebook cell 13
const retriever = vectorstore.asRetriever();

// %%
// Notebook cell 14
await retriever.invoke("What is deep learning?")

// %%
// Notebook cell 15


// %%
// Notebook cell 16


// %%
// Notebook cell 17


// %%
// Notebook cell 18


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

