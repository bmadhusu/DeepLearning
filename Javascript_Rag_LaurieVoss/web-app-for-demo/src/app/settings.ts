import { OpenAI, OpenAIEmbedding } from "@llamaindex/openai";
import { Settings } from "llamaindex";

export function initSettings() {
  console.log("Initializing settings with model:", process.env.MODEL);
  console.log("OPENAI_API_KEY exists:", !!process.env.OPENAI_API_KEY);
  
  const llm = new OpenAI({
    model: process.env.MODEL ?? "gpt-4o-mini",
    maxTokens: process.env.LLM_MAX_TOKENS
      ? Number(process.env.LLM_MAX_TOKENS)
      : undefined,
  });
  
  console.log("LLM created:", llm ? "Yes" : "No");
  
  Settings.llm = llm;
  
  console.log("Settings.llm assigned:", Settings.llm ? "Yes" : "No");
  
  Settings.embedModel = new OpenAIEmbedding({
    model: process.env.EMBEDDING_MODEL,
    dimensions: process.env.EMBEDDING_DIM
      ? parseInt(process.env.EMBEDDING_DIM)
      : undefined,
  });
}