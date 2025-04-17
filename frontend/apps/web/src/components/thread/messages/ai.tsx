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
  // First clean any HTML that might be in the content for security
  const cleanContent = content.replace(/</g, '&lt;').replace(/>/g, '&gt;');
  
  // Handle lists separately
  let processedContent = cleanContent;

  // Process for list items first
  const discListItems = processedContent.match(/^[*-] (.*?)$/gm);
  if (discListItems) {
    const ulContent = discListItems.map(item => {
      const text = item.replace(/^[*-] /, '');
      return `<li class="ml-4 list-disc">${text}</li>`;
    }).join('');
    if (ulContent) {
      // Replace all unordered list items with a single ul block
      processedContent = processedContent.replace(/^[*-] (.*?)$/gm, match => {
        return match === discListItems[0] ? `<ul class="my-2">${ulContent}</ul>` : '';
      });
    }
  }

  // Process ordered list items
  const ordListItems = processedContent.match(/^\d+\. (.*?)$/gm);
  if (ordListItems) {
    // Identify consecutive list items by finding their positions
    const positions = [];
    const regex = /^\d+\. (.*?)$/gm;
    let match;
    
    // Get positions of all list items
    while ((match = regex.exec(processedContent)) !== null) {
      positions.push(match.index);
    }
    
    // Group consecutive items
    const groups = [];
    let currentGroup = [];
    
    for (let i = 0; i < positions.length; i++) {
      const pos = positions[i];
      const nextPos = i < positions.length - 1 ? positions[i + 1] : -1;
      
      currentGroup.push(ordListItems[i]);
      
      // Check if the next item is consecutive
      const isLastItem = i === positions.length - 1;
      const isNotConsecutive = nextPos !== -1 && (
        // Not on next line or separated by more than one blank line
        processedContent.substring(pos, nextPos).split('\n').length > 2
      );
      
      if (isLastItem || isNotConsecutive) {
        // Process this group and start a new one
        groups.push(currentGroup);
        currentGroup = [];
      }
    }
    
    // Process each group of consecutive list items
    groups.forEach(group => {
      if (group.length > 0) {
        // Create HTML for this group
        const olContent = group.map(item => {
          const text = item.replace(/^\d+\. /, '');
          return `<li class="ml-4 list-decimal">${text}</li>`;
        }).join('');
        
        // Create a replacement pattern that matches just this group
        const groupPattern = group.map(item => escapeRegExp(item)).join('\\s*');
        const groupRegex = new RegExp(groupPattern, 'g');
        
        // Replace just this group with an ol element
        processedContent = processedContent.replace(
          groupRegex,
          `<ol class="my-2">${olContent}</ol>`
        );
      }
    });
  }
  
  // Handle tables - extract and process tables before other formatting
  processedContent = processMarkdownTables(processedContent);
  
  // Process code blocks first (to avoid formatting code content)
  processedContent = processedContent.replace(/```([\s\S]*?)```/g, (_, code) => {
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

      return `<pre class="my-4"><div class="flex justify-between items-center p-2 bg-gray-800 text-white rounded-t-md">
                  <span class="text-xs font-mono">${language || "code"}</span>
                  <button class="copy-button" data-code="${codeContent.replace(/"/g, "&quot;")}">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="copy-icon"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
                  </button>
                </div>
                <code class="hljs p-4 rounded-b-md block overflow-x-auto">${highlighted}</code></pre>`;
      } catch (e) {
        console.log(e)
      return `<pre class="my-4"><code class="p-4 rounded-md block overflow-x-auto">${codeContent}</code></pre>`;
    }
  });
  
  // Process the rest of the content
  const formatted = processedContent
    // Inline code
    .replace(/`([^`]+)`/g, '<code class="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-red-500">$1</code>')
    // Headings (h1 to h6) - add specific margins to control spacing
    .replace(/^# (.*?)$/gm, '<h1 class="text-2xl font-bold mt-4 mb-2">$1</h1>')
    .replace(/^## (.*?)$/gm, '<h2 class="text-xl font-bold mt-3 mb-2">$1</h2>')
    .replace(/^### (.*?)$/gm, '<h3 class="text-lg font-bold mt-3 mb-1">$1</h3>')
    .replace(/^#### (.*?)$/gm, '<h4 class="text-base font-bold mt-2 mb-1">$1</h4>')
    .replace(/^##### (.*?)$/gm, '<h5 class="text-sm font-bold mt-2 mb-1">$1</h5>')
    .replace(/^###### (.*?)$/gm, '<h6 class="text-xs font-bold mt-2 mb-1">$1</h6>')
    // Bold and italic
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    // Links
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">$1</a>')
    // Blockquotes
    .replace(/^> (.*?)$/gm, '<blockquote class="border-l-4 border-gray-300 pl-4 italic text-gray-700 my-2">$1</blockquote>')
    // Horizontal rule
    .replace(/^---$/gm, '<hr class="my-3 border-t border-gray-300" />');
    
  // Handle paragraphs and line breaks more intelligently
  // This prevents excessive whitespace between elements
  const paragraphs = formatted.split('\n\n').filter(p => p.trim());
  const processedParagraphs = paragraphs.map(p => {
    // Don't wrap HTML elements that already have their own structure
    if (p.trim().startsWith('<') && !p.startsWith('<code') && !p.startsWith('<em') && !p.startsWith('<strong')) {
      return p;
    }
    // Handle single line breaks within paragraphs
    return `<p class="my-2">${p.replace(/\n/g, '<br>')}</p>`;
  });
  
  return processedParagraphs.join('\n');
}

/**
 * Process Markdown tables and convert them to HTML tables
 */
function processMarkdownTables(content: string): string {
  // We need a more flexible approach to detect tables
  // Look for sequences of lines that start and end with pipe characters
  const lines = content.split('\n');
  const processedLines = [];
  
  let inTable = false;
  let tableLines = [];
  let isRTL = false;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmedLine = line.trim();
    
    // If line starts with | and ends with |, it's potentially part of a table
    if (trimmedLine.startsWith('|') && trimmedLine.endsWith('|')) {
      if (!inTable) {
        inTable = true;
        // Check if this might be RTL content (contains Arabic or Hebrew characters)
        isRTL = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u0590-\u05FF]/.test(trimmedLine);
      }
      tableLines.push(trimmedLine);
    } else if (inTable) {
      // End of table reached
      if (tableLines.length > 0) {
        processedLines.push(convertTableToHtml(tableLines, isRTL));
        tableLines = [];
        inTable = false;
        isRTL = false;
      }
      processedLines.push(line);
    } else {
      processedLines.push(line);
    }
  }
  
  // Handle case where table is at the end of content
  if (inTable && tableLines.length > 0) {
    processedLines.push(convertTableToHtml(tableLines, isRTL));
  }
  
  return processedLines.join('\n');
}

/**
 * Convert array of table lines to HTML
 */
function convertTableToHtml(tableLines: string[], isRTL: boolean): string {
  // Check if we have at least one row
  if (tableLines.length === 0) return '';
  
  // Process each row
  const rows = tableLines.map(line => {
    // Split by pipe character and remove empty strings from start/end
    const cells = line.split('|').map(cell => cell.trim());
    if (cells[0] === '') cells.shift();
    if (cells[cells.length - 1] === '') cells.pop();
    return cells;
  });
  
  // Determine if second row contains separator indicators (----, :---:, etc)
  const hasSeparatorRow = tableLines.length > 1 && 
    rows[1].every(cell => /^[:+-]+$/.test(cell));
  
  // Determine alignments from separator row if present
  let alignments = [];
  if (hasSeparatorRow) {
    alignments = rows[1].map(cell => {
      if (cell.startsWith(':') && cell.endsWith(':')) return 'center';
      if (cell.endsWith(':')) return 'right';
      if (cell.startsWith(':')) return 'left';
      return isRTL ? 'right' : 'left'; // Default based on direction
    });
  } else {
    // Default alignment based on direction
    alignments = rows[0].map(() => isRTL ? 'right' : 'left');
  }
  
  // Convert rows to HTML - using a more compact structure to reduce whitespace
  let html = '<div class="overflow-x-auto my-3 rounded-md shadow-sm">';
  html += `<table class="w-full border-collapse table-auto border text-sm" dir="${isRTL ? 'rtl' : 'ltr'}">`;
  
  if (hasSeparatorRow) {
    // Has header row
    html += '<thead>';
    html += '<tr>';
    
    // Create header cells
    rows[0].forEach((cell, index) => {
      const align = alignments[index] || (isRTL ? 'right' : 'left');
      html += `<th class="border px-3 py-2 bg-gray-100 font-semibold text-${align}">${cell}</th>`;
    });
    
    html += '</tr>';
    html += '</thead>';
    html += '<tbody>';
    
    // Create body rows (skip header and separator rows)
    for (let i = 2; i < rows.length; i++) {
      // Skip rows that are just separator lines
      if (rows[i].every(cell => /^[-:]+$/.test(cell))) continue;
      
      html += '<tr>';
      
      // Process cells
      rows[i].forEach((cell, index) => {
        const align = alignments[index] || (isRTL ? 'right' : 'left');
        html += `<td class="border px-3 py-2 text-${align}">${cell}</td>`;
      });
      
      html += '</tr>';
    }
    
    html += '</tbody>';
  } else {
    // No header row, all rows are body
    html += '<tbody>';
    
    rows.forEach(row => {
      html += '<tr>';
      
      row.forEach((cell, index) => {
        const align = alignments[index] || (isRTL ? 'right' : 'left');
        html += `<td class="border px-3 py-2 text-${align}">${cell}</td>`;
      });
      
      html += '</tr>';
    });
    
    html += '</tbody>';
  }
  
  html += '</table>';
  html += '</div>';
  
  return html;
}

/**
 * Helper function to escape regular expression special characters
 */
function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}