/**
 * Default prompts.
 */

export const RESPONSE_SYSTEM_PROMPT_TEMPLATE = `You are a helpful AI assistant. Answer the user's questions based on the retrieved documents.

{retrievedDocs}

IMPORTANT: Format your response using proper Markdown syntax. Use headings (##), lists (- or 1.), code blocks (\`\`\`), bold/italic formatting, and other Markdown elements as appropriate to create well-structured, readable responses.

System time: {systemTime}`;

export const QUERY_SYSTEM_PROMPT_TEMPLATE = `Generate search queries to retrieve documents that may help answer the user's question. Previously, you made the following queries:
    
<previous_queries/>
{queries}
</previous_queries>

System time: {systemTime}`;
