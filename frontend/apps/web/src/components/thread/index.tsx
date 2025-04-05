import { v4 as uuidv4 } from "uuid";
import { ReactNode, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Message } from "@langchain/langgraph-sdk";
import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import { LangGraphLogoSVG } from "../icons/langgraph";
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  ArrowDown,
  LoaderCircle,
  SquarePen,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import { toast } from "sonner";
import { Label } from "../ui/label";
import { Switch } from "../ui/switch";

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div ref={context.contentRef} className={props.contentClassName}>
        {props.content}
      </div>

      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  if (isAtBottom) return null;
  return (
    <Button
      variant="outline"
      className={props.className}
      onClick={() => scrollToBottom()}
    >
      <ArrowDown className="w-4 h-4" />
      <span>Scroll to bottom</span>
    </Button>
  );
}

export function Thread() {
  const [threadId, setThreadId] = useQueryState("threadId");
  const [hideMetadata, setHideMetadata] = useQueryState(
    "hideMetadata",
    parseAsBoolean.withDefault(false),
  );
  const [input, setInput] = useState("");
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);

  const stream = useStreamContext();
  const messages = stream.messages;
  const isLoading = stream.isLoading;

  // Add a useEffect to log messages on mount
  useEffect(() => {
    console.log("Thread component mounted with messages:", messages);
  }, [messages]);

  const lastError = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const message = (stream.error as any).message;
      if (!message || lastError.current === message) {
        // Message has already been logged. do not modify ref, return early.
        return;
      }

      // Message is defined, and it has not been logged yet. Save it, and send the error
      lastError.current = message;
      toast.error("An error occurred. Please try again.", {
        description: (
          <p>
            <strong>Error:</strong> <code>{message}</code>
          </p>
        ),
        richColors: true,
        closeButton: true,
      });
    } catch {
      // no-op
    }
  }, [stream.error]);

  // Track when we receive the first token
  const prevMessageLength = useRef(0);
  useEffect(() => {
    if (
      messages.length !== prevMessageLength.current &&
      messages?.length &&
      messages[messages.length - 1].type === "ai"
    ) {
      setFirstTokenReceived(true);
    }

    prevMessageLength.current = messages.length;
  }, [messages]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    setFirstTokenReceived(false);

    console.log("Submitting new message. Current message count:", messages.length);

    const newHumanMessage: Message = {
      id: uuidv4(),
      type: "human",
      content: input,
    };

    // Get all existing messages from the stream and append the new message
    console.log("Submitting message. Current message count:", messages.length);
    
    stream.submit(
      { messages: [newHumanMessage] },
      {
        optimisticValues: (prev: { messages: Message[] }) => {
          console.log("Optimistic update with existing messages:", prev.messages.length);
          return {
            ...prev,
            messages: [
              ...(prev.messages ?? []),
              newHumanMessage,
            ],
          };
        },
      },
    );
    
    // Log message state after submission
    setTimeout(() => {
      console.log("After submission, message count:", stream.messages.length);
      // Debug with our helper function
      stream.logMessagesState?.();
    }, 100);

    setInput("");
  };

  const handleRegenerate = () => {
    // Simple implementation - resend the last human message
    if (messages.length === 0) return;
    
    console.log("Regenerating response. Current message count:", messages.length);
    
    // Find the last human message
    const lastHumanMessage = [...messages].reverse().find(m => m.type === "human");
    if (!lastHumanMessage) return;
    
    // Do this so the loading state is correct
    prevMessageLength.current = messages.length - 1; // Decrement by only one for the AI response we'll remove
    setFirstTokenReceived(false);
    
    // We send just the human message to the API but keep all previous messages in the chat
    stream.submit(
      { messages: [lastHumanMessage] },
      {
        optimisticValues: (prev: { messages: Message[] }) => {
          // Keep all messages, just remove the last AI message if it exists
          const messagesWithoutLastAI = [...prev.messages];
          
          // If the last message is AI, remove it (we'll replace it with new response)
          if (messagesWithoutLastAI.length > 0 && 
              messagesWithoutLastAI[messagesWithoutLastAI.length - 1].type === "ai") {
            messagesWithoutLastAI.pop();
            console.log("Removed last AI message, new count:", messagesWithoutLastAI.length);
          }
          
          // Log what we're keeping
          console.log("Preserving messages for regeneration:", messagesWithoutLastAI.length);
          
          return {
            ...prev,
            messages: messagesWithoutLastAI,
          };
        },
      },
    );
    
    // Log message state after regeneration request
    setTimeout(() => {
      console.log("After regeneration request, message count:", stream.messages.length);
      // Debug with our helper function
      stream.logMessagesState?.();
    }, 100);
  };

  const chatStarted = !!threadId || messages.length > 0;

  return (
    <div className="flex w-full h-screen overflow-hidden">
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        {chatStarted && (
          <div className="flex items-center justify-between gap-3 p-2 pl-4 z-10 relative">
            <div className="flex items-center justify-start gap-2 relative">
              <div className="flex gap-2 items-center cursor-pointer">
                <LangGraphLogoSVG width={32} height={32} />
                <span className="text-xl font-semibold tracking-tight">
                  RAG Chat
                </span>
              </div>
            </div>

            <TooltipIconButton
              size="lg"
              className="p-4"
              tooltip="New chat"
              variant="ghost"
              onClick={() => {
                stream.clearMessages();
                setThreadId(null);
              }}
            >
              <SquarePen className="size-5" />
            </TooltipIconButton>

            <div className="absolute inset-x-0 top-full h-5 bg-gradient-to-b from-background to-background/0" />
          </div>
        )}

        <StickToBottom className="relative flex-1 overflow-hidden">
          <StickyToBottomContent
            className={cn(
              "absolute inset-0 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent",
              !chatStarted && "flex flex-col items-stretch mt-[25vh]",
              chatStarted && "grid grid-rows-[1fr_auto]",
            )}
            contentClassName="pt-8 pb-16 max-w-3xl mx-auto flex flex-col gap-4 w-full"
            content={
              <>
                {messages.map((message, index) =>
                  message.type === "human" ? (
                    <HumanMessage
                      key={message.id || `${message.type}-${index}`}
                      message={message}
                      isLoading={isLoading}
                    />
                  ) : (
                    <AssistantMessage
                      key={message.id || `${message.type}-${index}`}
                      message={message}
                      isLoading={isLoading}
                      handleRegenerate={handleRegenerate}
                    />
                  ),
                )}
                {isLoading && !firstTokenReceived && (
                  <AssistantMessageLoading />
                )}
              </>
            }
            footer={
              <div className="sticky flex flex-col items-center gap-8 bottom-0 px-4 bg-white">
                {!chatStarted && (
                  <div className="flex gap-3 items-center">
                    <LangGraphLogoSVG className="flex-shrink-0 h-8" />
                    <h1 className="text-2xl font-semibold tracking-tight">
                      RAG Chat
                    </h1>
                  </div>
                )}

                <ScrollToBottom className="absolute bottom-full left-1/2 -translate-x-1/2 mb-4 animate-in fade-in-0 zoom-in-95" />

                <div className="bg-muted rounded-2xl border shadow-xs mx-auto mb-8 w-full max-w-3xl relative z-10">
                  <form
                    onSubmit={handleSubmit}
                    className="grid grid-rows-[1fr_auto] gap-2 max-w-3xl mx-auto"
                  >
                    <textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey && !e.metaKey) {
                          e.preventDefault();
                          const el = e.target as HTMLElement | undefined;
                          const form = el?.closest("form");
                          form?.requestSubmit();
                        }
                      }}
                      placeholder="Type your message..."
                      className="p-3.5 pb-0 border-none bg-transparent field-sizing-content shadow-none ring-0 outline-none focus:outline-none focus:ring-0 resize-none"
                    />

                    <div className="flex items-center justify-between p-2 pt-4">
                      <div>
                        <div className="flex items-center space-x-2">
                          <Switch
                            id="render-tool-calls"
                            checked={hideMetadata ?? false}
                            onCheckedChange={setHideMetadata}
                          />
                          <Label
                            htmlFor="render-tool-calls"
                            className="text-sm text-gray-600"
                          >
                            Hide Metadata
                          </Label>
                          
                          <Button 
                            variant="outline"
                            size="sm"
                            className="ml-4"
                            onClick={() => {
                              stream.clearMessages();
                              setThreadId(null);
                            }}
                          >
                            <SquarePen className="w-3 h-3 mr-1" />
                            New Chat
                          </Button>
                        </div>
                      </div>
                      {stream.isLoading ? (
                        <Button key="stop" onClick={() => stream.stop()}>
                          <LoaderCircle className="w-4 h-4 animate-spin" />
                          Cancel
                        </Button>
                      ) : (
                        <Button
                          type="submit"
                          className="transition-all shadow-md"
                          disabled={isLoading || !input.trim()}
                        >
                          Send
                        </Button>
                      )}
                    </div>
                  </form>
                </div>
              </div>
            }
          />
        </StickToBottom>
      </div>
    </div>
  );
}
