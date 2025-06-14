import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph } from "@langchain/langgraph";
import {
  ConfigurationAnnotation,
  ensureConfiguration,
} from "./configuration.js";
import { StateAnnotation, InputStateAnnotation } from "./state.js";
import { formatDocs, getMessageText, loadChatModel } from "./utils.js";
import { z } from "zod";
import { makeRetriever } from "./retrieval.js";
// Define the function that calls the model

const SearchQuery = z.object({
  query: z.string().describe("Search the indexed documents for a query."),
});

async function generateQuery(
  state: typeof StateAnnotation.State,
  config?: RunnableConfig,
): Promise<typeof StateAnnotation.Update> {
  const messages = state.messages;
  if (messages.length === 1) {
    // It's the first user question. We will use the input directly to search.
    const humanInput = getMessageText(messages[messages.length - 1]);
    return { queries: [humanInput] };
  } else {
    const configuration = ensureConfiguration(config);
    // Feel free to customize the prompt, model, and other logic!
    const systemMessage = configuration.querySystemPromptTemplate
      .replace("{queries}", (state.queries || []).join("\n- "))
      .replace("{systemTime}", new Date().toISOString());

    const messageValue = [
      { role: "system", content: systemMessage },
      ...state.messages,
    ];
    const model = (
      await loadChatModel(configuration.responseModel)
    ).withStructuredOutput(SearchQuery);

    const generated = await model.invoke(messageValue);
    return {
      queries: [generated.query],
    };
  }
}

async function retrieve(
  state: typeof StateAnnotation.State,
  config: RunnableConfig,
): Promise<typeof StateAnnotation.Update> {
  const query = state.queries[state.queries.length - 1];
  const retriever = await makeRetriever(config);
  const response = await retriever.invoke(query);
  return { retrievedDocs: response };
}

async function respond(
  state: typeof StateAnnotation.State,
  config: RunnableConfig,
): Promise<typeof StateAnnotation.Update> {
  /**
   * Call the LLM powering our "agent".
   */
  const configuration = ensureConfiguration(config);

  const model = await loadChatModel(configuration.responseModel);

  const retrievedDocs = formatDocs(state.retrievedDocs);
  // Feel free to customize the prompt, model, and other logic!
  const systemMessage = configuration.responseSystemPromptTemplate
    .replace("{retrievedDocs}", retrievedDocs)
    .replace("{systemTime}", new Date().toISOString());
  const messageValue = [
    { role: "system", content: systemMessage },
    ...state.messages,
  ];
  const response = await model.invoke(messageValue);
  
  // Ensure response is properly formatted as Markdown
  let content = typeof response.content === 'string' ? response.content : JSON.stringify(response.content);
  
  // If content doesn't contain any Markdown formatting, add some basic structure
  if (!containsMarkdownFormatting(content)) {
    content = formatAsMarkdown(content);
    // Update the response content
    if (typeof response.content === 'string') {
      response.content = content;
    }
  }
  
  // We return a list, because this will get added to the existing list
  return { messages: [response] };
}

/**
 * Checks if the text contains basic Markdown formatting
 */
function containsMarkdownFormatting(text: string): boolean {
  // Check for common Markdown elements like headings, lists, code blocks, etc.
  const markdownPatterns = [
    /#{1,6}\s+.+/,         // Headings
    /(?:\*|-|\+)\s+.+/,    // Unordered lists
    /\d+\.\s+.+/,          // Ordered lists
    /```[\s\S]*?```/,      // Code blocks
    /`[^`]+`/,             // Inline code
    /\*\*[^*]+\*\*/,       // Bold
    /\*[^*]+\*/,           // Italic
    /\[[^\]]+\]\([^)]+\)/, // Links
  ];
  
  return markdownPatterns.some(pattern => pattern.test(text));
}

/**
 * Formats plain text as Markdown with basic structure
 */
function formatAsMarkdown(text: string): string {
  // If text is already in paragraphs, return as is
  if (text.includes('\n\n')) {
    return text;
  }

  // Convert plain text to paragraphs for better readability
  const paragraphs = text.split(/\n/).filter(line => line.trim().length > 0);
  return paragraphs.join('\n\n');
}

// Lay out the nodes and edges to define a graph
const builder = new StateGraph(
  {
    stateSchema: StateAnnotation,
    // The only input field is the user
    input: InputStateAnnotation,
  },
  ConfigurationAnnotation,
)
  .addNode("generateQuery", generateQuery)
  .addNode("retrieve", retrieve)
  .addNode("respond", respond)
  .addEdge("__start__", "generateQuery")
  .addEdge("generateQuery", "retrieve")
  .addEdge("retrieve", "respond");

// Finally, we compile it!
// This compiles it into a graph you can invoke and deploy.
export const graph = builder.compile({
  interruptBefore: [], // if you want to update the state before calling the tools
  interruptAfter: [],
});

graph.name = "Retrieval Graph"; // Customizes the name displayed in LangSmith
