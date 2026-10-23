// Source notebook: Notebook_2.ipynb
// See SOURCE_FILES.md for the original Deno runtime and dependencies.

// %% [markdown]
// # Lesson 2: Loading and preparing data

// %% [markdown]
// # Loading

// %%
// Notebook cell 3
import "dotenv/config";

// %%
// Notebook cell 4
import { GithubRepoLoader } from "langchain/document_loaders/web/github";
// Peer dependency, used to support .gitignore syntax
import ignore from "ignore";

// %%
// Notebook cell 5
// Will not include anything under "ignorePaths"
let loader;
loader = new GithubRepoLoader(
  "https://github.com/langchain-ai/langchainjs",
  { recursive: false, ignorePaths: ["*.md", "yarn.lock"] }
);

// %%
// Notebook cell 6
const docs = await loader.load();

console.log(docs.slice(0, 3));

// %%
// Notebook cell 7
// Peer dependency
import * as parse from "pdf-parse";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

// %%
// Notebook cell 8
loader = new PDFLoader("./data/MachineLearning-Lecture01.pdf");

// %%
// Notebook cell 9
const rawCS229Docs = await loader.load();

console.log(rawCS229Docs.slice(0, 5));

// %% [markdown]
// # Splitting

// %%
// Notebook cell 11
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// %%
// Notebook cell 12
let splitter;
splitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
  chunkSize: 32,
  chunkOverlap: 0,
});

// %%
// Notebook cell 13
const code = `function helloWorld() {
console.log("Hello, World!");
}
// Call the function
helloWorld();`;

await splitter.splitText(code);

// %%
// Notebook cell 14
import { CharacterTextSplitter } from "langchain/text_splitter";

splitter = new CharacterTextSplitter({
  chunkSize: 32,
  chunkOverlap: 0,
  separator: " "
});

await splitter.splitText(code);

// %%
// Notebook cell 15
splitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
  chunkSize: 64,
  chunkOverlap: 32,
});

await splitter.splitText(code);

// %%
// Notebook cell 16
splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 64,
});

// %%
// Notebook cell 17
const splitDocs = await splitter.splitDocuments(rawCS229Docs);

console.log(splitDocs.slice(0, 5));

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


// %%
// Notebook cell 27


// %%
// Notebook cell 28


// %%
// Notebook cell 29


// %%
// Notebook cell 30

