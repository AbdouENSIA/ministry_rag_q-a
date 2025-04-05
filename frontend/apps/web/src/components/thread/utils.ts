import { Message } from "@langchain/langgraph-sdk";
import { MessageContentComplex } from "@langchain/core/messages";

export function getContentString(
  content: string | MessageContentComplex[] | undefined
): string {
  if (!content) return "";

  if (typeof content === "string") {
    return content;
  }

  return content
    .filter((c) => c.type === "text")
    .map((c) => (c as { text: string }).text)
    .join("\n");
}
