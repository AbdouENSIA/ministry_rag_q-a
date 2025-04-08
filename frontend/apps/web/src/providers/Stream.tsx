import React, {
  createContext,
  useContext,
  ReactNode,
  useState,
  useCallback,
  useEffect,
  useRef,
} from "react";
import { type Message } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import { toast } from "sonner";
import { v4 as uuidv4 } from "uuid";

export type StateType = { messages: Message[] };

type StreamContextType = {
  messages: Message[];
  submit: (data: { messages: Message[] }, options?: any) => Promise<void>;
  isLoading: boolean;
  error: Error | null;
  stop: () => void;
  getMessagesMetadata: (message: Message) => any;
  setBranch: (branch: string) => void;
  interrupt: any;
  clearMessages: () => void;
  logMessagesState: () => void;
};

const StreamContext = createContext<StreamContextType | undefined>(undefined);

async function checkApiStatus(apiUrl: string): Promise<boolean> {
  console.log(`Attempting to connect to API at: ${apiUrl}`);
  
  // Try up to 3 times with a short delay between attempts
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      console.log(`Connection attempt ${attempt}/3 to ${apiUrl}`);
      const res = await fetch(`${apiUrl}/`, {
        signal: AbortSignal.timeout(5000)
      });
      console.log(`API response from ${apiUrl}:`, res.status, res.ok);
      
      if (res.ok) {
        console.log(`Successfully connected to API at ${apiUrl}`);
        return true;
      }
      
      if (attempt < 3) {
        console.log(`Waiting before retry attempt ${attempt + 1}...`);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    } catch (e) {
      console.error(`Failed attempt ${attempt} to connect to API:`, e);
      
      if (attempt < 3) {
        console.log(`Waiting before retry attempt ${attempt + 1}...`);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  }
  
  console.log(`Failed to connect to API at ${apiUrl} after 3 attempts`);
  return false;
}

// Streaming text response function
const streamTextResponse = (
  fullText: string,
  onChunk: (chunk: string, isDone: boolean) => void,
  onDone: () => void
) => {
  const minDelay = 15;
  const maxDelay = 35;
  const avgCharsPerChunk = 3;
  const variability = 2;
  
  let currentPosition = 0;
  let lastPausePosition = 0;
  
  const shouldPauseLonger = (pos: number) => {
    if (pos >= fullText.length) return false;
    return ['.', '!', '?', '\n'].includes(fullText[pos - 1]);
  };
  
  const scheduleNextChunk = () => {
    if (currentPosition >= fullText.length) {
      onDone();
      return;
    }
    
    let chunkSize = avgCharsPerChunk + Math.floor(Math.random() * variability);
    
    if (currentPosition + chunkSize > fullText.length) {
      chunkSize = fullText.length - currentPosition;
    }
    
    const nextChunk = fullText.substring(currentPosition, currentPosition + chunkSize);
    currentPosition += chunkSize;
    
    const isDone = currentPosition >= fullText.length;
    onChunk(nextChunk, isDone);
    
    let delay = minDelay + Math.floor(Math.random() * (maxDelay - minDelay));
    
    if (shouldPauseLonger(currentPosition)) {
      delay += 100 + Math.floor(Math.random() * 150);
      lastPausePosition = currentPosition;
    } else if (currentPosition - lastPausePosition > 50) {
      delay += 30 + Math.floor(Math.random() * 40);
      lastPausePosition = currentPosition;
    }
    
    setTimeout(scheduleNextChunk, delay);
  };
  
  scheduleNextChunk();
  
  return () => {
    currentPosition = fullText.length;
  };
};

const StreamSession = ({
  children,
  apiUrl,
  setApiUrl,
}: {
  children: ReactNode;
  apiUrl: string;
  setApiUrl: (url: string) => void;
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const streamCleanupRef = useRef<(() => void) | null>(null);

  // Check API status on mount
  useEffect(() => {
    const currentHost = window.location.hostname;
    let apiHostUrl = apiUrl;
    
    if (currentHost !== 'localhost' && currentHost !== '127.0.0.1') {
      apiHostUrl = apiUrl.replace('localhost', currentHost);
      console.log(`Accessing from external device, using API URL: ${apiHostUrl}`);
      setApiUrl(apiHostUrl);
    }

    checkApiStatus(apiHostUrl).then(async (ok) => {
      if (!ok) {
        if (!apiHostUrl.includes(':8000')) {
          const fallbackUrl = `http://${currentHost}:8000`;
          console.log(`Initial connection failed, trying fallback to ${fallbackUrl}`);
          const fallbackOk = await checkApiStatus(fallbackUrl);
          
          if (fallbackOk) {
            setApiUrl(fallbackUrl);
            toast.success(`Connected to Python API server at ${fallbackUrl}`, {
              description: "Automatically switched to correct port",
              duration: 4000,
            });
            return;
          }
        }
        
        toast.error("Failed to connect to Python API server", {
          description: () => (
            <p>
              Please ensure your API is running at <code>{apiHostUrl}</code>
            </p>
          ),
          duration: 10000,
          richColors: true,
          closeButton: true,
        });
      } else {
        toast.success(`Connected to Python API server at ${apiHostUrl}`, {
          description: "Ready to accept queries",
          duration: 2000,
        });
      }
    });
  }, [apiUrl, setApiUrl]);

  const submit = useCallback(
    async (data: { messages: Message[] }, options?: any) => {
      const lastMessage = data.messages[data.messages.length - 1];
      
      if (options?.optimisticValues) {
        const optimisticUpdate = options.optimisticValues({ messages });
        setMessages(optimisticUpdate.messages);
      }

      if (streamCleanupRef.current) {
        streamCleanupRef.current();
        streamCleanupRef.current = null;
      }

      setIsLoading(true);
      setError(null);

      try {
        const apiEndpoint = `${apiUrl}/api/query`;
        console.log(`Sending query to ${apiEndpoint}:`, lastMessage.content);
        
        const response = await fetch(apiEndpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query: lastMessage.content,
            include_source_documents: false,
          }),
        });

        if (!response.ok) {
          const errorText = await response.text().catch(() => "Unknown error");
          console.error(`API error (${response.status}):`, errorText);
          throw new Error(`API error (${response.status}): ${errorText}`);
        }

        const result = await response.json();
        
        const aiMessage = {
          id: uuidv4(),
          type: "ai" as const,
          content: "",
          metadata: {
            confidence_score: result.confidence_score,
            query_type: result.query_type,
            processing_time: result.processing_time,
            metadata: result.metadata,
            suggested_followup: result.suggested_followup,
            validation: result.validation
          }
        };

        setMessages(prevMessages => [...prevMessages, aiMessage]);
        
        const cleanup = streamTextResponse(
          result.answer,
          (chunk, isDone) => {
            setMessages(prevMessages => 
              prevMessages.map(msg => 
                msg.id === aiMessage.id 
                  ? { 
                      ...msg,
                      content: msg.content + chunk 
                    }
                  : msg
              )
            );
          },
          () => {
            setIsLoading(false);
            streamCleanupRef.current = null;
          }
        );
        
        streamCleanupRef.current = cleanup;
      } catch (err) {
        setError(err instanceof Error ? err : new Error(String(err)));
        setIsLoading(false);
      }
    },
    [apiUrl, messages]
  );

  const stop = useCallback(() => {
    if (streamCleanupRef.current) {
      streamCleanupRef.current();
      streamCleanupRef.current = null;
    }
    setIsLoading(false);
  }, []);

  const getMessagesMetadata = useCallback((message: Message) => {
    return (message as any).metadata || {};
  }, []);
  
  const setBranch = useCallback((branch: string) => {
    console.log("Setting branch:", branch);
  }, []);

  const logMessagesState = useCallback(() => {
    console.log('Current messages state:', messages.length, 'messages');
    messages.forEach((msg, idx) => {
      console.log(`[${idx}] ${msg.type}: ${msg.id ? msg.id.substring(0, 6) + '...' : 'no-id'}`);
    });
  }, [messages]);

  const contextValue: StreamContextType = {
    messages,
    isLoading,
    error,
    stop,
    submit,
    getMessagesMetadata,
    setBranch,
    interrupt: null,
    clearMessages: () => {
      console.log('Clearing all messages');
      setMessages([]);
    },
    logMessagesState
  };

  return (
    <StreamContext.Provider value={contextValue}>
      {children}
    </StreamContext.Provider>
  );
};

export const StreamProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [_, setApiUrl] = useQueryState("apiUrl");
  
  useEffect(() => {
    const currentHost = window.location.hostname;
    let apiHostUrl = "http://localhost:8000";
    
    if (currentHost !== 'localhost' && currentHost !== '127.0.0.1') {
      apiHostUrl = `http://${currentHost}:8000`;
    }
    
    setApiUrl(apiHostUrl);
    console.log("API URL set to", apiHostUrl);
  }, [setApiUrl]);

  const [initialApiUrl, setInitialApiUrl] = useState("http://localhost:8000");
  
  useEffect(() => {
    const currentHost = window.location.hostname;
    if (currentHost !== 'localhost' && currentHost !== '127.0.0.1') {
      setInitialApiUrl(`http://${currentHost}:8000`);
    }
  }, []);

  return (
    <StreamSession apiUrl={initialApiUrl} setApiUrl={setApiUrl}>
      {children}
    </StreamSession>
  );
};

// eslint-disable-next-line react-refresh/only-export-components
export const useStreamContext = (): StreamContextType => {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStreamContext must be used within a StreamProvider");
  }
  return context;
};
