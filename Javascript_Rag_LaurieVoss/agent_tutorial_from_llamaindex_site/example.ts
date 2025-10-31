// Your imports go here
import "dotenv/config";
import {
  agent,
  agentStreamEvent,
  openai,
} from "@llamaindex/workflow";
import {
  tool,
  Settings,
} from "llamaindex";
import {
  openai,
} from "@llamaindex/openai";
import { z } from "zod";

Settings.llm = openai({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-4o",
});

const sumNumbers = ({ a, b }) => {
  return `${a + b}`;
};

const addTool = tool({
  name: "sumNumbers",
  description: "Use this function to sum two numbers",
  parameters: z.object({
    a: z.number({
      description: "First number to sum",
    }),
    b: z.number({
      description: "Second number to sum",
    }),
  }),
  execute: sumNumbers,
});

const tools = [addTool];


async function main() {
  // the rest of your code goes here
  const myAgent = agent({ tools });

const events = myAgent.runStream("Sum 202 and 404");
for await (const event of events) {
  if (agentStreamEvent.include(event)) {
    // Stream the response
    process.stdout.write(event.data.delta);
  } else {
    // Log other events
    console.log("\nWorkflow event:", JSON.stringify(event, null, 2));
  }
}

}

main().catch(console.error);