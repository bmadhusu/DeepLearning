
// Need to set OPENAI_API_KEY in your environment variables before running this script
// Run this using: node agent.js

(async () => {
  const { Document, VectorStoreIndex, SimpleDirectoryReader, RouterQueryEngine, OpenAIAgent, QueryEngineTool, FunctionTool, } = await import("llamaindex");
  const documents1 = await new SimpleDirectoryReader().loadData({ directoryPath: "./data" });
  const index1 = await VectorStoreIndex.fromDocuments(documents1);
  console.log("Index created");
  const queryEngine1 = index1.asQueryEngine()

  const documents2 = await new SimpleDirectoryReader().loadData({directoryPath: "./data2"})
const index2 = await VectorStoreIndex.fromDocuments(documents2)
const queryEngine2 = index2.asQueryEngine();


const queryEngine = await RouterQueryEngine.fromDefaults({
  queryEngineTools: [
    {
      queryEngine: queryEngine1,
      description: "Useful for questions about Dan Abramov",
    },
    {
      queryEngine: queryEngine2,
      description: "Useful for questions about the React library",
    },
  ],
});

let response3 = await queryEngine.query({query: "What is React?"})
console.log(response3.toString())

function sumNumbers({a,b}) {
  return a + b;
}

const sumJSON = {
  type: "object",
  properties: {
    a: {
      type: "number",
      description: "The first number",
    },
    b: {
      type: "number",
      description: "The second number",
    },
  },
  required: ["a", "b"],
};

const sumFunctionTool = new FunctionTool(sumNumbers, {
  name: "sumNumbers",
  description: "Use this function to sum two numbers",
  parameters: sumJSON,
});

const queryEngineTool = new QueryEngineTool({
    queryEngine: queryEngine,
    metadata: {
        name: "react_and_dan_abramov_engine",
        description: "A tool that can answer questions about Dan Abramov and React",
    },
});

const agent = new OpenAIAgent({
    tools: [queryEngineTool, sumFunctionTool],
    verbose: true
})

let response5 = await agent.chat({message:"What is React? Use a tool."})
console.log(response5.toString())

let response6 = await agent.chat({message:"What is 501 + 5?"})
console.log(response6.toString())


})().catch(err => {
  console.error(err);
  process.exit(1);
});

