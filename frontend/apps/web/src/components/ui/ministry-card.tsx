import React from "react";
import { cn } from "@/lib/utils";

interface MinistryCardProps {
  className?: string;
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  icon?: React.ReactNode;
  variant?: "default" | "outline" | "elevated";
}

export function MinistryCard({
  className,
  children,
  title,
  subtitle,
  icon,
  variant = "default",
}: MinistryCardProps) {
  const variants = {
    default: "bg-white shadow-sm",
    outline: "bg-white border border-gray-200",
    elevated: "bg-white shadow-md",
  };

  return (
    <div
      className={cn(
        "rounded-lg overflow-hidden",
        variants[variant],
        className
      )}
    >
      {(title || subtitle || icon) && (
        <div className="p-4 border-b border-gray-100">
          <div className="flex items-start">
            {icon && <div className="mr-3 text-[#0F4C81]">{icon}</div>}
            <div>
              {title && (
                <h3 className="text-lg font-semibold text-[#0F4C81]">{title}</h3>
              )}
              {subtitle && (
                <p className="text-sm text-gray-500 mt-1">{subtitle}</p>
              )}
            </div>
          </div>
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
} 