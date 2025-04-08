import { Message } from "@langchain/langgraph-sdk";
import { getContentString } from "../utils";
import { CommandBar } from "./shared";
import { MarkdownText } from "../markdown-text";
import { cn } from "@/lib/utils";
import { useState, useEffect } from "react";
import { ChevronDown, ChevronUp, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { RAGMessage } from "@/types/message";
import { useQueryState, parseAsBoolean } from "nuqs";

export function AssistantMessage({
  message,
  isLoading,
  handleRegenerate,
}: {
  message: Message | RAGMessage;
  isLoading: boolean;
  handleRegenerate: () => void;
}) {
  const contentString = getContentString(message.content);
  const [showMetadata, setShowMetadata] = useState(false);
  const [hideMetadata] = useQueryState(
    "hideMetadata",
    parseAsBoolean.withDefault(false),
  );
  
  // Check if message has metadata
  const hasMetadata = 'metadata' in message && message.metadata && Object.keys(message.metadata).length > 0;

  // Handle the case when hideMetadata changes
  useEffect(() => {
    if (hideMetadata) {
      setShowMetadata(false);
    }
  }, [hideMetadata]);

  return (
    <div className="flex items-start ml-auto gap-2 group w-full">
      <div className="flex flex-col gap-2 w-full">
        <div className="py-1">
          <MarkdownText>{contentString}</MarkdownText>
        </div>

        {hasMetadata && !hideMetadata && (
          <div className="w-full">
            <Button 
              variant="ghost" 
              className="text-xs flex items-center gap-1 px-2 py-1 h-auto text-gray-500 hover:text-gray-700"
              onClick={() => setShowMetadata(!showMetadata)}
            >
              <Info className="h-3 w-3" />
              {showMetadata ? "إخفاء" : "عرض"} البيانات
              {showMetadata ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </Button>
            
            {showMetadata && (
              <div className="p-2 bg-gray-50 rounded-md text-xs border mt-1 text-gray-600 overflow-x-auto">
                <MetadataDisplay metadata={(message as RAGMessage).metadata} />
              </div>
            )}
          </div>
        )}

        <div
          className={cn(
            "flex gap-2 items-center ml-auto transition-opacity",
            "opacity-0 group-focus-within:opacity-100 group-hover:opacity-100",
          )}
        >
          <CommandBar
            content={contentString}
            isLoading={isLoading}
            isAiMessage={true}
            handleRegenerate={handleRegenerate}
          />
        </div>
      </div>
    </div>
  );
}

function MetadataDisplay({ metadata }: { metadata: any }) {
  if (!metadata) return null;
  
  return (
    <div className="grid grid-cols-2 gap-1">
      {Object.entries(metadata).map(([key, value]) => {
        if (value === null || value === undefined) return null;
        
        if (key === 'metadata' && typeof value === 'object') {
          return (
            <div className="col-span-2 border-t pt-1 mt-1" key={key}>
              <strong className="font-medium">بيانات إضافية:</strong>
              <div className="pl-2 mt-1">
                <MetadataDisplay metadata={value} />
              </div>
            </div>
          );
        }
        
        if (typeof value === 'object') {
          return (
            <div className="col-span-2 border-t pt-1 mt-1" key={key}>
              <strong className="font-medium">{key.replace(/_/g, ' ')}:</strong>
              <div className="pl-2 mt-1">
                <MetadataDisplay metadata={value} />
              </div>
            </div>
          );
        }
        
        return (
          <div className="col-span-2 flex justify-between" key={key}>
            <span className="font-medium">{key.replace(/_/g, ' ')}:</span>
            <span className="truncate max-w-[70%] text-right">
              {typeof value === 'number' 
                ? key.includes('time') 
                  ? `${value.toFixed(2)}ث` 
                  : value.toString()
                : String(value)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function AssistantMessageLoading() {
  return (
    <div className="flex items-start mr-auto gap-2">
      <div className="flex items-center gap-1 rounded-2xl bg-muted px-4 py-2 h-8">
        <div className="w-1.5 h-1.5 rounded-full bg-foreground/50 animate-[pulse_1.5s_ease-in-out_infinite]"></div>
        <div className="w-1.5 h-1.5 rounded-full bg-foreground/50 animate-[pulse_1.5s_ease-in-out_0.5s_infinite]"></div>
        <div className="w-1.5 h-1.5 rounded-full bg-foreground/50 animate-[pulse_1.5s_ease-in-out_1s_infinite]"></div>
      </div>
    </div>
  );
}
