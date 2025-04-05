import {
  RefreshCw,
  Copy,
  CopyCheck,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { TooltipIconButton } from "../tooltip-icon-button";
import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

function ContentCopyable({
  content,
  disabled,
}: {
  content: string;
  disabled: boolean;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = (e: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
    e.stopPropagation();
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <TooltipIconButton
      onClick={(e) => handleCopy(e)}
      variant="ghost"
      tooltip="Copy content"
      disabled={disabled}
    >
      <AnimatePresence mode="wait" initial={false}>
        {copied ? (
          <motion.div
            key="check"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.15 }}
          >
            <CopyCheck className="text-green-500" />
          </motion.div>
        ) : (
          <motion.div
            key="copy"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.15 }}
          >
            <Copy />
          </motion.div>
        )}
      </AnimatePresence>
    </TooltipIconButton>
  );
}

export function BranchSwitcher({
  branch,
  branchOptions,
  onSelect,
  isLoading,
}: {
  branch: string | undefined;
  branchOptions: string[] | undefined;
  onSelect: (branch: string) => void;
  isLoading: boolean;
}) {
  if (!branchOptions || !branch) return null;
  const index = branchOptions.indexOf(branch);

  return (
    <div className="flex items-center gap-2">
      <Button
        variant="ghost"
        size="icon"
        className="size-6 p-1"
        onClick={() => {
          const prevBranch = branchOptions[index - 1];
          if (!prevBranch) return;
          onSelect(prevBranch);
        }}
        disabled={isLoading}
      >
        <ChevronLeft />
      </Button>
      <span className="text-sm">
        {index + 1} / {branchOptions.length}
      </span>
      <Button
        variant="ghost"
        size="icon"
        className="size-6 p-1"
        onClick={() => {
          const nextBranch = branchOptions[index + 1];
          if (!nextBranch) return;
          onSelect(nextBranch);
        }}
        disabled={isLoading}
      >
        <ChevronRight />
      </Button>
    </div>
  );
}

export function CommandBar({
  content,
  isLoading,
  isAiMessage = false,
  handleRegenerate,
}: {
  content: string;
  isLoading: boolean;
  isAiMessage?: boolean;
  handleRegenerate?: () => void;
}) {
  if (!isAiMessage) return null;

  return (
    <div className="flex items-center gap-1">
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8"
        onClick={() => {
          navigator.clipboard.writeText(content);
          toast.success("Copied to clipboard");
        }}
      >
        <Copy className="h-4 w-4" />
      </Button>
      
      {handleRegenerate && (
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          disabled={isLoading}
          onClick={handleRegenerate}
        >
          <RefreshCw className="h-4 w-4" />
        </Button>
      )}
    </div>
  );
}
