import React from "react";

interface LoadingProps {
  size?: "sm" | "md" | "lg";
  text?: string;
}

export function Loading({ size = "md", text }: LoadingProps) {
  const sizeClasses = {
    sm: "h-5 w-5",
    md: "h-8 w-8",
    lg: "h-12 w-12",
  };

  return (
    <div className="flex flex-col items-center justify-center">
      <div
        className={`${sizeClasses[size]} animate-spin rounded-full border-4 border-t-[#0F4C81] border-r-[#0F4C81]/30 border-b-[#0F4C81]/70 border-l-transparent`}
      />
      {text && <p className="mt-3 text-sm text-gray-600">{text}</p>}
    </div>
  );
}

export function LoadingPage() {
  return (
    <div className="flex h-screen w-full items-center justify-center">
      <Loading size="lg" text="جاري التحميل..." />
    </div>
  );
} 