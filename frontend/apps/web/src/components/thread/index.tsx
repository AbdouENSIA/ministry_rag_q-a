import { v4 as uuidv4 } from "uuid";
import { ReactNode, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Message } from "@langchain/langgraph-sdk";
import { AssistantMessage } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  ArrowDown,
  SquarePen,
  Brain,
  Sparkles,
  Menu,
  BookOpen,
  FileText,
  GraduationCap,
  Home,
  X
} from "lucide-react";
import { useQueryState } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import { toast } from "sonner";
import { Textarea } from "../ui/textarea";
import Link from "next/link";
import Image from "next/image";

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
      <span>انتقل إلى الأسفل</span>
    </Button>
  );
}

// Ministry logo for sidebar
function MinistryImageLogo({ width = 32, height = 32, className = "" }) {
  return (
    <div className={`relative h-${height/4} w-${width/4} overflow-hidden ${className}`}>
      <Image 
        src="/وزارة_التعليم_العالي_والبحث_العلمي.svg.png" 
        alt="شعار وزارة التعليم العالي" 
        width={width} 
        height={height}
        className="object-contain"
      />
    </div>
  );
}

// Enhanced loading animation component
function EnhancedLoadingAnimation() {
  return (
    <div className="flex flex-col items-center justify-center py-8">
      <div className="relative flex">
        <div className="absolute inset-0 flex items-center justify-center">
          <Sparkles className="w-8 h-8 text-yellow-400 animate-pulse" />
        </div>
        <div className="h-16 w-16 rounded-full border-t-4 border-b-4 border-[#0F4C81] animate-spin"></div>
        <div className="absolute inset-0 flex items-center justify-center">
          <Brain className="w-7 h-7 text-[#0F4C81] animate-bounce opacity-80" />
        </div>
      </div>
      <p className="mt-4 text-[#0F4C81] font-medium">جاري معالجة طلبك...</p>
    </div>
  );
}

export function Thread({ customClassName }: { customClassName?: string }) {
  const [threadId, setThreadId] = useQueryState("threadId");
  const [showSidebar, setShowSidebar] = useState(true);
  const [input, setInput] = useState("");
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);

  const stream = useStreamContext();
  const messages = stream.messages;
  const isLoading = stream.isLoading;

  // Array of suggested questions
  const SUGGESTED_QUESTIONS = [
    "متى تأسست وزارة التعليم العالي والبحث العلمي؟",
    "ما هي مهام وزارة التعليم العالي والبحث العلمي؟",
    "كم عدد الجامعات في الجزائر؟",
    "ما هي شروط القبول في الدكتوراه؟",
    "كيف يتم التسجيل في الماستر؟"
  ];

  // Function to get random questions
  function getRandomQuestions(count = 4) {
    return SUGGESTED_QUESTIONS.slice(0, count);
  }
  
  const randomQuestions = getRandomQuestions();

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
 
  // Listen for custom setInputValue event
  useEffect(() => {
    const handleSetInputValue = (e: Event) => {
      const customEvent = e as CustomEvent;
      if (customEvent.detail && customEvent.detail.value) {
        // Update React state directly
        setInput(customEvent.detail.value);
        
        // Enable submit button
        const form = document.querySelector('form');
        const submitButton = form?.querySelector('button[type="submit"]');
        if (submitButton) {
          (submitButton as HTMLButtonElement).disabled = false;
        }
      }
    };
    
    // Add event listener to document for custom event
    document.addEventListener('setInputValue', handleSetInputValue);
    
    return () => {
      document.removeEventListener('setInputValue', handleSetInputValue);
    };
  }, []);

  // Update useEffect to also handle programmatic input changes
  useEffect(() => {
    const textarea = document.querySelector('textarea');
    const form = textarea?.closest('form');
    
    if (textarea) {
      // Listen for programmatic input event
      const handleProgrammaticInput = () => {
        if (textarea.value.trim()) {
          // Update React state if needed
          setInput(textarea.value);
          
          // Enable submit button
          const submitButton = form?.querySelector('button[type="submit"]');
          if (submitButton) {
            (submitButton as HTMLButtonElement).disabled = false;
          }
        }
      };
      
      textarea.addEventListener('input', handleProgrammaticInput);
      
      return () => {
        textarea.removeEventListener('input', handleProgrammaticInput);
      };
    }
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    setInput(newValue);
    
    // Adjust textarea height
    e.target.style.height = 'inherit';
    e.target.style.height = `${e.target.scrollHeight}px`;
    
    // Enable/disable submit button based on input
    const submitButton = e.target.form?.querySelector('button[type="submit"]');
    if (submitButton) {
      (submitButton as HTMLButtonElement).disabled = !newValue.trim();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.metaKey) {
      e.preventDefault();
      const el = e.target as HTMLElement | undefined;
      const form = el?.closest("form");
      form?.requestSubmit();
    }
  };

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
    
    // Reset textarea height after clearing input
    const textarea = document.querySelector('textarea');
    if (textarea) {
      textarea.style.height = 'inherit';
    }
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

  const handleSuggestedQuestionClick = (question: string) => {
    setInput(question);
    
    // Enable the submit button
    const form = document.querySelector('form');
    const submitButton = form?.querySelector('button[type="submit"]');
    if (submitButton) {
      (submitButton as HTMLButtonElement).disabled = false;
    }
    
    // Also adjust textarea height for suggested questions
    const textarea = document.querySelector('textarea');
    if (textarea) {
      textarea.style.height = 'inherit';
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
    
    // Submit the form after a brief delay
    setTimeout(() => {
      if (form) {
        form.requestSubmit();
      }
    }, 100);
  };

  return (
    <div className="flex h-full w-full overflow-hidden bg-gray-50">
      {/* Sidebar - Reduced width from w-72 to w-60 */}
      <div className={`bg-white h-full border-l relative transition-all duration-300 ${showSidebar ? 'w-60' : 'w-0 overflow-hidden'}`}>
        <div className="p-3 h-full flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              {/* Updated sidebar logo to use the Ministry Image */}
              <MinistryImageLogo width={24} height={24} />
              <h3 className="text-lg font-bold text-[#0F4C81]">القائمة</h3>
            </div>
          </div>
          
          <div className="space-y-3 flex-1 overflow-auto text-sm">
            <Link href="/">
              <Button variant="ghost" className="w-full justify-end text-gray-700 py-1.5">
                <span>الصفحة الرئيسية</span>
                <Home className="ml-2 h-3.5 w-3.5" />
              </Button>
            </Link>
            
            <div className="border-t border-gray-100 pt-3">
              <h4 className="text-base font-bold text-gray-600 mb-1.5 text-right">أقسام المعرفة</h4>
              <div className="space-y-0.5">
                <Button variant="ghost" className="w-full justify-end text-gray-700 py-1.5 h-auto">
                  <span>الجامعات والمؤسسات</span>
                  <GraduationCap className="ml-2 h-3.5 w-3.5" />
                </Button>
                <Button variant="ghost" className="w-full justify-end text-gray-700 py-1.5 h-auto">
                  <span>البحث العلمي</span>
                  <BookOpen className="ml-2 h-3.5 w-3.5" />
                </Button>
                <Button variant="ghost" className="w-full justify-end text-gray-700 py-1.5 h-auto">
                  <span>التوثيق والمصادر</span>
                  <FileText className="ml-2 h-3.5 w-3.5" />
                </Button>
              </div>
            </div>
            
            <div className="border-t border-gray-100 pt-3">
              <h4 className="text-base font-bold text-gray-600 mb-1.5 text-right">أسئلة مقترحة</h4>
              <div className="space-y-1.5">
                {randomQuestions.map((question, index) => (
                  <Button 
                    key={index}
                    variant="outline" 
                    className="w-full justify-end text-right text-xs text-black bg-white py-1.5 h-auto hover:bg-[#0F4C81]/10"
                    onClick={() => handleSuggestedQuestionClick(question)}
                  >
                    {question}
                  </Button>
                ))}
              </div>
            </div>
          </div>
          
          <div className="border-t border-gray-100 pt-3">
            <Link href="/privacy-policy">
              <Button variant="ghost" className="w-full justify-end text-gray-700 text-xs py-1">
                <span>سياسة الخصوصية</span>
              </Button>
            </Link>
            <Link href="/terms-of-use">
              <Button variant="ghost" className="w-full justify-end text-gray-700 text-xs py-1">
                <span>شروط الاستخدام</span>
              </Button>
            </Link>
          </div>
        </div>
      </div>
      
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative h-full">
        {/* Header with Brain Logo (Glowing) */}
        <header className="bg-white border-b py-2 px-4 flex items-center justify-between sticky top-0 z-10">
          <Button 
            variant="ghost" 
            size="sm" 
            className="p-1.5 hover:bg-gray-100"
            onClick={() => setShowSidebar(!showSidebar)}
          >
            <Menu className="h-5 w-5 text-gray-600" />
          </Button>
          
          <div className="flex items-center">
            <div className="flex gap-3 items-center">
              {/* UPDATED: Replaced Ministry Logo with Glowing Brain Logo */}
              <span className="text-xl font-bold text-[#0F4C81]">
                نظام استرجاع المعرفة المعزز بالذكاء الاصطناعي
              </span>
            </div>
          </div>
          
          <TooltipIconButton
            size="sm"
            className="p-1.5 hover:bg-gray-100"
            tooltip="محادثة جديدة"
            variant="ghost"
            onClick={() => {
              stream.clearMessages();
              setThreadId(null);
            }}
          >
            <SquarePen className="size-5 text-gray-600" />
          </TooltipIconButton>
        </header>

        {/* Chat Content Area */}
        <div className="flex-1 relative overflow-hidden">
          <StickToBottom className="relative h-full overflow-hidden">
            <StickyToBottomContent
              className={cn(
                "absolute inset-0 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent",
                !chatStarted && "flex flex-col items-stretch mt-[10vh]",
                chatStarted && "grid grid-rows-[1fr_auto]",
              )}
              contentClassName={cn(
                "pt-6 pb-20 max-w-3xl mx-auto flex flex-col gap-4 w-full px-4",
              )}
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
                    <EnhancedLoadingAnimation />
                  )}
                </>
              }
              footer={
                <div className="sticky flex flex-col items-center gap-6 bottom-0 px-4 bg-transparent">
                  {!chatStarted && (
                    <div className="flex flex-col items-center gap-4 py-8 bg-white rounded-lg shadow-sm border px-6">
                      <div className="flex gap-3 items-center">
                        {/* Keep the Ministry logo in the welcome card */}
                        <div className="relative h-12 w-12 overflow-hidden">
                          <Image 
                            src="/وزارة_التعليم_العالي_والبحث_العلمي.svg.png" 
                            alt="شعار وزارة التعليم العالي" 
                            width={48} 
                            height={48}
                            className="object-contain"
                          />
                        </div>
                        <h1 className="text-3xl font-extrabold tracking-tight text-[#0F4C81]">
                          نظام استرجاع المعرفة المعزز بالذكاء الاصطناعي
                        </h1>
                      </div>
                      <p className="text-center text-gray-600 max-w-md text-sm">
                        نظام استرجاع المعرفة المتطور الخاص بوزارة التعليم العالي والبحث العلمي. يمكنك طرح أسئلتك المتعلقة بالوزارة والحصول على إجابات دقيقة ومستندة إلى مصادر موثوقة.
                      </p>
                      <div className="grid grid-cols-2 gap-2 mt-2 w-full">
                        {randomQuestions.slice(0, 4).map((question, index) => (
                          <Button 
                            key={index}
                            variant="outline" 
                            className="text-right text-sm text-black bg-white h-auto py-2 hover:bg-[#0F4C81]/10"
                            onClick={() => handleSuggestedQuestionClick(question)}
                          >
                            {question}
                          </Button>
                        ))}
                      </div>
                    </div>
                  )}

                  <ScrollToBottom className="absolute bottom-full left-1/2 -translate-x-1/2 mb-4 animate-in fade-in-0 zoom-in-95" />

                  {/* Input area - fixed at bottom */}
                  <div className="bg-white rounded-2xl border shadow-xl mx-auto mb-4 w-full max-w-3xl relative z-10 transition-all hover:shadow-2xl">
                    <form
                      onSubmit={handleSubmit}
                      className="grid grid-rows-[1fr_auto] gap-2 max-w-3xl mx-auto"
                    >
                      <Textarea
                        value={input}
                        onChange={handleInputChange}
                        onKeyDown={handleKeyDown}
                        placeholder="اكتب سؤالك هنا..."
                        className="min-h-[60px] w-full resize-none bg-white text-black px-4 py-[1.3rem] focus:ring-2 focus:ring-[#0F4C81]/30 focus:border-[#0F4C81]"
                        autoFocus
                      />

                      <div className="flex items-center justify-between p-3 pt-2 border-t border-gray-100">
                        <div>
                          <Button 
                            variant="outline"
                            size="sm"
                            className="text-sm font-medium hover:bg-gray-100 transition-all"
                            onClick={() => {
                              stream.clearMessages();
                              setThreadId(null);
                            }}
                          >
                            <SquarePen className="w-3 h-3 ml-1" />
                            محادثة جديدة
                          </Button>
                        </div>
                        {stream.isLoading ? (
                          <Button 
                            key="stop" 
                            onClick={() => stream.stop()} 
                            className="bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white shadow-md transition-all duration-300 flex items-center gap-2 px-4 py-2 rounded-lg"
                          >
                            <X className="w-4 h-4" />
                            <span>إلغاء</span>
                          </Button>
                        ) : (
                          <Button
                            type="submit"
                            className="bg-gradient-to-r from-[#0F4C81] to-[#0D3D66] hover:from-[#0D3D66] hover:to-[#092C4D] transition-all duration-300 shadow-md text-white flex items-center gap-2 px-4 py-2 rounded-lg"
                            disabled={isLoading || !input.trim()}
                          >
                            <Sparkles className="w-4 h-4 text-yellow-300" />
                            <span>إرسال</span>
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
    </div>
  );
}