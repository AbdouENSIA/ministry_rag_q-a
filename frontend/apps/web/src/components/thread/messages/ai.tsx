import React, { useRef, useEffect } from "react";
import { Message } from "@langchain/langgraph-sdk";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { RefreshCw, Bot, Brain, Copy, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { RAGMessage } from "@/types/message";
import { useQueryState, parseAsBoolean } from "nuqs";
import parse, { HTMLReactParserOptions, Element, domToReact } from "html-react-parser";
import hljs from "highlight.js";
import 'highlight.js/styles/github.css';

export function AssistantMessage({
  message,
  isLoading,
  handleRegenerate,
}: {
  message: Message | RAGMessage;
  isLoading: boolean;
  handleRegenerate: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [_, setShowMetadata] = useState(false);
  const [hideMetadata] = useQueryState(
    "hideMetadata",
    parseAsBoolean.withDefault(false),
  );
  const [copied, setCopied] = React.useState(false);
  
  // Check if message has metadata

  // Handle the case when hideMetadata changes
  useEffect(() => {
    if (hideMetadata) {
      setShowMetadata(false);
    }
  }, [hideMetadata]);

  useEffect(() => {
    if (!ref.current) return;

    const codeBlocks = ref.current.querySelectorAll("pre code");
    codeBlocks.forEach((block) => {
      hljs.highlightElement(block as HTMLElement);
    });

    const copyButtons = ref.current.querySelectorAll(".copy-button");
    copyButtons.forEach((button) => {
      button.addEventListener("click", (e) => {
        e.preventDefault();
        const code = (button as HTMLElement).dataset.code || "";
        navigator.clipboard.writeText(code);
        
        const copyIcon = button.querySelector('.copy-icon');
        if (copyIcon) {
          copyIcon.innerHTML = '<path d="M20 6 9 17l-5-5"/>';
          setTimeout(() => {
            copyIcon.innerHTML = '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>';
          }, 2000);
        }
      });
    });
  }, [message.content]);

  const copyMessage = () => {
    if (typeof message.content === 'string') {
      navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const options: HTMLReactParserOptions = {
    replace: (domNode) => {
      if (
        domNode instanceof Element &&
        domNode.name === "pre" &&
        domNode.children[0] instanceof Element &&
        domNode.children[0].name === "code"
      ) {
        return (
          <div className="relative my-4 w-full overflow-hidden rounded-md bg-gray-100 dark:bg-gray-800">
            {domToReact([domNode], options)}
          </div>
        );
      }
    },
  };

  return (
    <div className="flex w-full mt-5 mb-5 gap-4 transition-all">
      <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md bg-[#0F4C81] text-white shadow-sm">
        <Brain className="h-5 w-5" />
      </div>
      <div className="flex-1 space-y-2">
        <div className="rounded-xl min-h-[48px] bg-white border-2 border-[#0F4C81]/10 p-4 shadow-sm relative group">
          <div 
            ref={ref} 
            className={cn("prose prose-sm dark:prose-invert max-w-none break-words text-black", { "opacity-50": isLoading })}
          >
            {typeof message.content === "string" && parse(formatMessage(message.content), options)}
          </div>
        </div>
        
        {!isLoading && (
          <div className="flex items-center justify-end gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs gap-1"
              onClick={copyMessage}
            >
              {copied ? <Check size={16} /> : <Copy size={16} />}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-8 text-xs gap-1"
              onClick={handleRegenerate}
            >
              <RefreshCw className="h-3 w-3" />
              إعادة توليد الإجابة
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
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

export function AssistantMessageLoading({ className }: { className?: string } = {}) {
  return (
    <div className={cn("flex w-full mt-5 mb-5 gap-4", className)}>
      <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md bg-[#0F4C81] text-white shadow-sm">
        <Bot className="h-5 w-5" />
      </div>
      <div className="w-full max-w-xs">
        <div className="flex flex-col gap-2.5 rounded-xl rounded-tl-none border border-input bg-card p-3">
          <div className="h-4 w-5/6 animate-pulse rounded-sm bg-muted" />
          <div className="h-4 w-4/6 animate-pulse rounded-sm bg-muted" />
          <div className="h-4 w-3/6 animate-pulse rounded-sm bg-muted" />
        </div>
      </div>
    </div>
  );
}

function formatMessage(content: string) {
  const formatted = content
    .replace(/```([\s\S]*?)```/g, (_, code) => {
      const trimmedCode = code.trim();
      const hasLanguage = trimmedCode.split("\n")[0].trim() !== "";
      let language = "";
      let codeContent = trimmedCode;

      if (hasLanguage) {
        language = trimmedCode.split("\n")[0].trim();
        codeContent = trimmedCode.split("\n").slice(1).join("\n");
      }

      try {
        const highlighted = language
          ? hljs.highlight(codeContent, { language }).value
          : hljs.highlightAuto(codeContent).value;

        return `<pre><div class="flex justify-between items-center p-2 bg-gray-800 text-white rounded-t-md">
                  <span class="text-xs font-mono">${language || "code"}</span>
                  <button class="copy-button" data-code="${codeContent.replace(/"/g, "&quot;")}">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="copy-icon"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
                  </button>
                </div>
                <code class="hljs p-4 rounded-b-md block overflow-x-auto">${highlighted}</code></pre>`;
      } catch (e) {
        console.log(e)
        return `<pre><code class="p-4 rounded-md block overflow-x-auto">${codeContent}</code></pre>`;
      }
    })
    .replace(/\n/g, "<br/>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>");

  return formatted;
}
