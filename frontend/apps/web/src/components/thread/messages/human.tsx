import { Message } from "@langchain/langgraph-sdk";
import { getContentString } from "../utils";
import { cn } from "@/lib/utils";

export function HumanMessage({
  message,
  isLoading,
}: {
  message: Message;
  isLoading: boolean;
}) {
  const contentString = getContentString(message.content);

  return (
    <div className="flex items-center mr-auto gap-2 group">
      <div className="flex flex-col gap-2">
        <p className="text-right px-4 py-2 rounded-3xl bg-muted">
          {contentString}
        </p>
      </div>
    </div>
  );
}
