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
        // Add a timeout to prevent long waits
        signal: AbortSignal.timeout(5000)
      });
      console.log(`API response from ${apiUrl}:`, res.status, res.ok);
      
      if (res.ok) {
        console.log(`Successfully connected to API at ${apiUrl}`);
        return true;
      }
      
      // Wait before retrying
      if (attempt < 3) {
        console.log(`Waiting before retry attempt ${attempt + 1}...`);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    } catch (e) {
      console.error(`Failed attempt ${attempt} to connect to API:`, e);
      
      // Wait before retrying
      if (attempt < 3) {
        console.log(`Waiting before retry attempt ${attempt + 1}...`);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  }
  
  console.log(`Failed to connect to API at ${apiUrl} after 3 attempts`);
  return false;
}

// New function to simulate streaming text responses
const streamTextResponse = (
  fullText: string,
  onChunk: (chunk: string, isDone: boolean) => void,
  onDone: () => void
) => {
  // Create a more realistic and smoother text streaming experience
  const minDelay = 15; // Minimum delay between chunks (ms)
  const maxDelay = 35; // Maximum delay between chunks (ms)
  const avgCharsPerChunk = 3; // Average characters per chunk
  const variability = 2; // Variability in chunk size
  
  let currentPosition = 0;
  let lastPausePosition = 0;
  
  // Function to determine if we should pause longer (at sentence endings)
  const shouldPauseLonger = (pos: number) => {
    if (pos >= fullText.length) return false;
    // Pause longer after sentence endings or at paragraph breaks
    return ['.', '!', '?', '\n'].includes(fullText[pos - 1]);
  };
  
  // Function to schedule the next chunk
  const scheduleNextChunk = () => {
    if (currentPosition >= fullText.length) {
      onDone();
      return;
    }
    
    // Determine a natural chunk size with some randomness
    let chunkSize = avgCharsPerChunk + Math.floor(Math.random() * variability);
    
    // Don't exceed the end of the text
    if (currentPosition + chunkSize > fullText.length) {
      chunkSize = fullText.length - currentPosition;
    }
    
    // Get the next chunk of text
    const nextChunk = fullText.substring(currentPosition, currentPosition + chunkSize);
    currentPosition += chunkSize;
    
    // Send the chunk
    const isDone = currentPosition >= fullText.length;
    onChunk(nextChunk, isDone);
    
    // Determine the delay for the next chunk
    let delay = minDelay + Math.floor(Math.random() * (maxDelay - minDelay));
    
    // Add extra pause at natural breaking points
    if (shouldPauseLonger(currentPosition)) {
      // Longer pause at sentence endings
      delay += 100 + Math.floor(Math.random() * 150);
      lastPausePosition = currentPosition;
    } else if (currentPosition - lastPausePosition > 50) {
      // Add slight pause every ~50 characters if we haven't paused in a while
      delay += 30 + Math.floor(Math.random() * 40);
      lastPausePosition = currentPosition;
    }
    
    // Schedule the next chunk
    setTimeout(scheduleNextChunk, delay);
  };
  
  // Start the streaming process
  scheduleNextChunk();
  
  // Return cleanup function
  return () => {
    currentPosition = fullText.length; // This will stop the streaming on next iteration
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
  // Initialize messages from storage (try both session and local storage)
  const [messages, setMessages] = useState<Message[]>(() => {
    try {
      // First try session storage (more reliable during the same session)
      const sessionSaved = sessionStorage.getItem('ragchat-messages');
      if (sessionSaved) {
        console.log('Loading messages from sessionStorage');
        const parsed = JSON.parse(sessionSaved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed;
        }
      }
      
      // Then try local storage
      const localSaved = localStorage.getItem('ragchat-messages');
      if (localSaved) {
        console.log('Loading messages from localStorage');
        const parsed = JSON.parse(localSaved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          // Also update session storage for future use
          sessionStorage.setItem('ragchat-messages', localSaved);
          return parsed;
        }
      }
      
      console.log('No saved messages found');
      return [];
    } catch (e) {
      console.error('Failed to load messages from storage:', e);
      // Clear potentially corrupted storage
      localStorage.removeItem('ragchat-messages');
      sessionStorage.removeItem('ragchat-messages');
      return [];
    }
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  // Add a ref to store cleanup functions
  const streamCleanupRef = useRef<(() => void) | null>(null);

  // Save messages to both storage types
  useEffect(() => {
    try {
      if (messages.length > 0) {
        const serialized = JSON.stringify(messages);
        console.log('Saving messages to storage, count:', messages.length, 'Latest message:', messages[messages.length - 1].type);
        localStorage.setItem('ragchat-messages', serialized);
        sessionStorage.setItem('ragchat-messages', serialized);
        
        // Log the first few characters of each message to verify content
        console.log('Message contents (previews):');
        messages.forEach((msg, idx) => {
          const preview = typeof msg.content === 'string' ? 
            msg.content.substring(0, 30) + (msg.content.length > 30 ? '...' : '') : 
            'non-string content';
          console.log(`[${idx}] ${msg.type}: ${preview}`);
        });
      } else {
        // Clear storage if no messages
        localStorage.removeItem('ragchat-messages');
        sessionStorage.removeItem('ragchat-messages');
      }
    } catch (e) {
      console.error('Failed to save messages to storage:', e);
    }
  }, [messages]);

  // Check API status on mount
  useEffect(() => {
    // Get the current hostname to use for API connections
    const currentHost = window.location.hostname;
    // Use the current host instead of localhost if not on localhost
    let apiHostUrl = apiUrl;
    
    if (currentHost !== 'localhost' && currentHost !== '127.0.0.1') {
      // Replace localhost with current hostname in the API URL
      apiHostUrl = apiUrl.replace('localhost', currentHost);
      console.log(`Accessing from external device, using API URL: ${apiHostUrl}`);
      // Update the API URL state
      setApiUrl(apiHostUrl);
    }

    checkApiStatus(apiHostUrl).then(async (ok) => {
      if (!ok) {
        // If connection fails and we're not already trying port 8000, try it as fallback
        if (!apiHostUrl.includes(':8000')) {
          const fallbackUrl = `http://${currentHost}:8000`;
          console.log(`Initial connection failed, trying fallback to ${fallbackUrl}`);
          const fallbackOk = await checkApiStatus(fallbackUrl);
          
          if (fallbackOk) {
            // If fallback works, update the API URL
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
      // Get the last message from the submitted data (this will be the new human message)
      const lastMessage = data.messages[data.messages.length - 1];
      
      // If we have optimistic updates
      if (options?.optimisticValues) {
        // Apply optimistic update keeping all previous messages
        const optimisticUpdate = options.optimisticValues({ messages });
        console.log("Setting messages with optimistic update. Message count:", optimisticUpdate.messages.length);
        setMessages(optimisticUpdate.messages);
      }

      // Clean up any existing streaming operation
      if (streamCleanupRef.current) {
        streamCleanupRef.current();
        streamCleanupRef.current = null;
      }

      setIsLoading(true);
      setError(null);

      try {
        // Get current API URL
        const apiEndpoint = `${apiUrl}/api/query`;
        console.log(`Sending query to ${apiEndpoint}:`, lastMessage.content);
        console.log("Current conversation has", messages.length, "messages");
        
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
        console.log("Received API response:", result);
        
        // Create initial AI message with empty content
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

        // Add empty message to state (preserving all existing messages)
        setMessages(prevMessages => [...prevMessages, aiMessage]);
        console.log("Added AI message, message count now:", messages.length + 1);
        
        // Start streaming the response
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

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const stop = useCallback(() => {
    // Clean up any streaming operation when stopped
    if (streamCleanupRef.current) {
      streamCleanupRef.current();
      streamCleanupRef.current = null;
    }
    setIsLoading(false);
  }, []);

  const getMessagesMetadata = useCallback((message: Message) => {
    // Return any metadata associated with the message
    return (message as any).metadata || {};
  }, []);
  
  const setBranch = useCallback((branch: string) => {
    // Handle branch switching logic here if needed
    console.log("Setting branch:", branch);
  }, []);

  const logMessagesState = useCallback(() => {
    console.log('Current messages state:', messages.length, 'messages');
    messages.forEach((msg, idx) => {
      console.log(`[${idx}] ${msg.type}: ${msg.id ? msg.id.substring(0, 6) + '...' : 'no-id'}`);
    });
  }, [messages]);

  // Define the context value
  const contextValue: StreamContextType = {
    messages,
    isLoading,
    error,
    stop: () => {
      if (streamCleanupRef.current) {
        streamCleanupRef.current();
        streamCleanupRef.current = null;
      }
      setIsLoading(false);
    },
    submit,
    getMessagesMetadata,
    setBranch,
    interrupt: null,
    clearMessages: () => {
      console.log('Clearing all messages');
      setMessages([]);
      localStorage.removeItem('ragchat-messages');
      sessionStorage.removeItem('ragchat-messages');
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
  
  // Set API URL based on current hostname
  useEffect(() => {
    // Get the current hostname to use for API connections
    const currentHost = window.location.hostname;
    
    // Use the appropriate host - current hostname if not on localhost
    let apiHostUrl = "http://localhost:8000";
    
    if (currentHost !== 'localhost' && currentHost !== '127.0.0.1') {
      apiHostUrl = `http://${currentHost}:8000`;
      console.log(`Accessing from external device, using API URL: ${apiHostUrl}`);
    } else {
      console.log("Accessing from localhost, using localhost for API");
    }
    
    // Set the API URL
    setApiUrl(apiHostUrl);
    console.log("API URL set to", apiHostUrl);
  }, [setApiUrl]);
  
  // Log provider initialization
  useEffect(() => {
    console.log("StreamProvider initialized");
    
    // Check if we have stored messages
    try {
      const savedMessages = localStorage.getItem('ragchat-messages');
      if (savedMessages) {
        const messages = JSON.parse(savedMessages);
        console.log("Found saved messages:", messages.length);
      } else {
        console.log("No saved messages found");
      }
    } catch (e) {
      console.error("Error checking saved messages:", e);
    }
  }, []);

  // Initialize with the dynamic API URL
  const [initialApiUrl, setInitialApiUrl] = useState("http://localhost:8000");
  
  // Update the initial API URL once component is mounted
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

// Create a custom hook to use the context
// eslint-disable-next-line react-refresh/only-export-components
export const useStreamContext = (): StreamContextType => {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStreamContext must be used within a StreamProvider");
  }
  return context;
};
