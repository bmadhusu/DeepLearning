
// Need to set OPENAI_API_KEY in your environment variables before running this script

(async () => {
  const { Document, VectorStoreIndex, SimpleDirectoryReader } = await import("llamaindex");
  const documents = await new SimpleDirectoryReader().loadData({ directoryPath: "./data" });
  const index = await VectorStoreIndex.fromDocuments(documents);
  console.log("Index created");
  const queryEngine = index.asQueryEngine()
  const response = await queryEngine.query({
    query: "What did the author do in college?"
})

console.log(response.toString())

})().catch(err => {
  console.error(err);
  process.exit(1);
});

