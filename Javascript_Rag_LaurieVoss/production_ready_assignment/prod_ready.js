
// Need to set OPENAI_API_KEY in your environment variables before running this script
// To run:
// npm init -y
// npm install llamaindex@0.1.8
// node prod_ready.js

(async () => {
  const { Document, VectorStoreIndex, SimpleDirectoryReader, RouterQueryEngine, storageContextFromDefaults, ContextChatEngine } = await import("llamaindex");
 
const storageContext = await storageContextFromDefaults({
  persistDir: "./storage",
});

const documents = await new SimpleDirectoryReader().loadData({
  directoryPath: "./data", // we have the React wikipedia page in here
});
let index = await VectorStoreIndex.fromDocuments(documents, {
  storageContext
});

let engine = await index.asQueryEngine()
let response = await engine.query({query:"What is JSX?"})
console.log(response.toString())

let storageContext2 = await storageContextFromDefaults({
    persistDir: "./storage",
});

let index2 = await VectorStoreIndex.init({
    storageContext: storageContext2
});

let engine2 = await index2.asQueryEngine()
let response2 = await engine2.query({query:"What is JSX?"})
console.log(response2.toString())

const retriever = index2.asRetriever();
retriever.similarityTopK = 3;

let chatEngine = new ContextChatEngine({
    retriever,
});

let messageHistory = [
    {
        role: "user",
        content: "What is JSX?"
    },
    {
        role: "assistant",
        content: "JSX stands for JavaScript Syntax Extension. It is an extension to the JavaScript language syntax that provides a way to structure component rendering using syntax familiar to many developers. JSX is similar in appearance to HTML and is typically used to write React components, although components can also be written in pure JavaScript. It was created by Facebook and is similar to another extension syntax created for PHP called XHP."
    }
]
let newMessage = "What was that last thing you mentioned?"

const response3 = await chatEngine.chat({
  message: newMessage,
  chatHistory: messageHistory
});
console.log(response3.toString())

const response4 = await chatEngine.chat({
  message: newMessage,
  chatHistory: messageHistory,
  stream: true,
});
console.log(response4)

for await (const data of response4) {
    console.log(data.response); // Print the data
}



})().catch(err => {
  console.error(err);
  process.exit(1);
});

