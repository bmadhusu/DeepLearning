import dotenv from "dotenv";
dotenv.config();

import { agent } from "@llamaindex/workflow";
import { Settings } from "llamaindex";
import { getIndex } from "./data";
import { initSettings } from "./settings";

console.log("Workflow module loaded");

export const workflowFactory = async (reqBody: any) => {
  // Initialize settings inside the factory function
  // This ensures it runs in the correct context
  initSettings();
  
  console.log("Settings.llm in workflowFactory:", Settings.llm ? "Set" : "Not set");
  
  const index = await getIndex(reqBody?.data);

  const queryEngineTool = index.queryTool({
    metadata: {
      name: "query_document",
      description: `This tool can retrieve information about Apple and Tesla financial data`,
    },
    includeSourceNodes: true,
  });

  return agent({ tools: [queryEngineTool] });
};