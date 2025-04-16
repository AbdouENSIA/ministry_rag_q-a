import React from "react";
import { Message } from "@langchain/langgraph-sdk";
import { getContentString } from "../utils";
import { User } from "lucide-react";
import { cn } from "@/lib/utils";

interface HumanMessageProps {
  message: Message;
  isLoading: boolean;
}

export function HumanMessage({ message, isLoading }: HumanMessageProps) {
  const contentString = getContentString(message.content);

  return (
    <div className="flex w-full mt-5 mb-5 gap-4">
      <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md bg-gray-200 text-[#0F4C81] shadow-sm">
        <User className="h-5 w-5" />
      </div>
      <div className="flex-1">
        <div className={cn(
          "rounded-xl min-h-[48px] bg-gray-100 p-4 shadow-sm text-gray-800 font-normal",
          {
            "opacity-50": isLoading
          }
        )}>
          {contentString.split("\n").map((line, i) => (
            <React.Fragment key={i}>
              {line}
              {i < contentString.split("\n").length - 1 && <br />}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}
