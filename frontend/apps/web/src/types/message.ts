import { Message as LangchainMessage } from "@langchain/langgraph-sdk";

// Extending the Message type to include our metadata
export interface RAGMessage extends LangchainMessage {
  content: string;
  metadata?: {
    confidence_score?: number;
    query_type?: string;
    processing_time?: number;
    metadata?: {
      sources_used?: number;
      key_concepts?: string[];
      confidence_factors?: string[];
    };
    suggested_followup?: string[];
    validation?: {
      has_hallucinations?: boolean;
      answers_question?: boolean;
      quality_score?: number;
      improvement_needed?: string[];
      validation_reasoning?: string;
    };
  };
} 