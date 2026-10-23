// Source notebook: Notebook_1.ipynb
// See SOURCE_FILES.md for the original Deno runtime and dependencies.

// %% [markdown]
// # Lesson 1: Building blocks

// %% [markdown]
// Welcome to this course! Just a reminder that LLMs may not always return the same text. As a result, the outputs you see when running the notebook may vary somewhat from those you see in the film. This applies to all of the notebooks.
// Enjoy the course!

// %% [markdown]
// ## Language model

// %% [markdown]
// 1. Text LLMs: string -> string
// 2. Chat models: list of messages -> single message output

// %%
// Notebook cell 5
import "dotenv/config";

// %%
// Notebook cell 6
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";

// %%
// Notebook cell 7
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106"
});

await model.invoke([
    new HumanMessage("Tell me a joke.")
]);

// %% [markdown]
// ## Prompt template

// %%
// Notebook cell 9
import { ChatPromptTemplate } from "@langchain/core/prompts";

const prompt = ChatPromptTemplate.fromTemplate(
    `What are three good names for a company that makes {product}?`
)

// %%
// Notebook cell 10
await prompt.format({
    product: "colorful socks"
});

// %%
// Notebook cell 11
await prompt.formatMessages({
    product: "colorful socks"
});

// %%
// Notebook cell 12
import { 
  SystemMessagePromptTemplate, 
  HumanMessagePromptTemplate 
} from "@langchain/core/prompts";

let promptFromMessages;
promptFromMessages = ChatPromptTemplate.fromMessages([
  SystemMessagePromptTemplate.fromTemplate(
    "You are an expert at picking company names."
  ),
  HumanMessagePromptTemplate.fromTemplate(
    "What are three good names for a company that makes {product}?"
  )
]);

await promptFromMessages.formatMessages({
    product: "shiny objects"
});

// %%
// Notebook cell 13
promptFromMessages = ChatPromptTemplate.fromMessages([
  ["system", "You are an expert at picking company names."],
  ["human", "What are three good names for a company that makes {product}?"]
]);

await promptFromMessages.formatMessages({
  product: "shiny objects"
});

// %% [markdown]
// # LangChain Expression Language (LCEL)

// %%
// Notebook cell 15
const chain = prompt.pipe(model);

// %%
// Notebook cell 16
await chain.invoke({
    product: "colorful socks"
});

// %% [markdown]
// ## Output parser

// %%
// Notebook cell 18
import { StringOutputParser } from "@langchain/core/output_parsers";

const outputParser = new StringOutputParser();

let nameGenerationChain;
nameGenerationChain = prompt.pipe(model).pipe(outputParser);

// %%
// Notebook cell 19
await nameGenerationChain.invoke({
    product: "fancy cookies"
});

// %% [markdown]
// https://smith.langchain.com/public/9fd7689f-23c6-4ea1-8a76-fea1b5f8d5db/r

// %%
// Notebook cell 21
import { RunnableSequence } from "@langchain/core/runnables";

nameGenerationChain = RunnableSequence.from([
    prompt,
    model,
    outputParser
])

// %%
// Notebook cell 22
await nameGenerationChain.invoke({
    product: "fancy cookies"
});

// %% [markdown]
// ## Streaming

// %%
// Notebook cell 24
const stream = await nameGenerationChain.stream({
  product: "really cool robots",
});

for await (const chunk of stream) {
    console.log(chunk);
}

// %% [markdown]
// ## Batch

// %%
// Notebook cell 26
const inputs = [
    { product: "large calculators" },
    { product: "alpaca wool sweaters" }
];

await nameGenerationChain.batch(inputs);

// %%
// Notebook cell 27


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

