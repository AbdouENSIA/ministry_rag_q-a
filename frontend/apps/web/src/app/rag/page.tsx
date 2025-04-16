"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import Image from "next/image";
import { Thread } from "@/components/thread";
import { StreamProvider } from "@/providers/Stream";
import { Toaster } from "sonner";
import { LoadingPage } from "@/components/ui/loading";
import { MinistryButton } from "@/components/ui/ministry-button";
import { MinistryCard } from "@/components/ui/ministry-card";
import { 
  ArrowRight, 
  FileText, 
  Book, 
  Bookmark, 
  Brain, 
  GraduationCap, 
  Calendar, 
  BookOpen,
  Search,
  ExternalLink,
  ChevronLeft
} from "lucide-react";
import { ThemeProvider } from "@/providers/theme-provider";
import { Button } from "@/components/ui/button";

// Array of suggested questions
const SUGGESTED_QUESTIONS = [
  "متى تأسست وزارة التعليم العالي والبحث العلمي؟",
  "ما هي مهام وزارة التعليم العالي والبحث العلمي؟",
  "كم عدد الجامعات في الجزائر؟",
  "ما هي شروط القبول في الدكتوراه؟",
  "كيف يتم التسجيل في الماستر؟",
  "ما هي آليات تمويل البحث العلمي في الجزائر؟",
  "ما هي الجامعات المصنفة عالميًا في الجزائر؟",
  "ما هي المنح الدراسية المتاحة للطلبة الجزائريين؟",
  "كيف يتم اعتماد الشهادات الأجنبية في الجزائر؟",
  "ما هي مراكز البحث العلمي التابعة للوزارة؟",
  "ما هي معايير ترقية الأساتذة الجامعيين؟",
  "كيف يتم تقييم الأبحاث العلمية في الجزائر؟",
  "ما هي المجلات العلمية المعتمدة من طرف الوزارة؟",
  "كيف تتم المشاركة في المؤتمرات الدولية برعاية الوزارة؟",
  "ما هي برامج التبادل الطلابي المتاحة؟",
  "ما هي شروط الحصول على سكن جامعي؟",
  "كيف يتم حساب المعدل في نظام LMD؟",
  "ما هي استراتيجية البحث العلمي للوزارة حتى عام 2030؟",
  "كيف تسهم الجامعات في التنمية الاقتصادية للبلاد؟",
  "ما هي إجراءات معادلة الشهادات الأجنبية؟",
  "ما هي خدمات الصحة المتاحة للطلبة الجامعيين؟",
  "كيف يتم تمويل المشاريع البحثية للأساتذة؟",
  "ما هي الاتفاقيات الدولية التي أبرمتها الوزارة مع جامعات أجنبية؟",
  "كيف يتم تصنيف الجامعات الجزائرية محليًا؟",
  "ما هي الهيكلة الإدارية لوزارة التعليم العالي والبحث العلمي؟",
  "ما هي المراسيم التنفيذية الأخيرة المتعلقة بقطاع التعليم العالي؟",
  "ما هي مراحل تطور التعليم العالي في الجزائر منذ الاستقلال؟",
  "كم يبلغ عدد الطلبة المسجلين في الجامعات الجزائرية؟",
  "ما هي ميزانية وزارة التعليم العالي والبحث العلمي للعام الحالي؟",
  "ما هي أحدث المنشآت الجامعية التي تم تدشينها؟"
];

// Function to get random questions
function getRandomQuestions(count = 4) {
  const shuffled = [...SUGGESTED_QUESTIONS].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, count);
}

export default function RagPage(): React.ReactNode {
  const [randomQuestions, setRandomQuestions] = useState<string[]>([]);
  const [userInput, setUserInput] = useState<string>("");
  const threadRef = useRef<HTMLDivElement>(null);

  // Set random questions on page load
  useEffect(() => {
    setRandomQuestions(getRandomQuestions());
  }, []);

  const handleSuggestedQuestionClick = (question: string) => {
    // Find the Thread component and access its internal functions
    const threadContainer = threadRef.current;
    if (!threadContainer) return;
    
    // Find the textarea and form
    const textarea = threadContainer.querySelector('textarea');
    const form = textarea?.closest('form');
    const submitButton = form?.querySelector('button[type="submit"]') as HTMLButtonElement | undefined;
    
    if (textarea && form) {
      // Set textarea value
      textarea.value = question;
      
      // Force React's state to update via a custom event
      const customEvent = new CustomEvent('setInputValue', { 
        bubbles: true, 
        detail: { value: question } 
      });
      textarea.dispatchEvent(customEvent);
      
      // Also dispatch regular input event for other listeners
      const inputEvent = new Event('input', { bubbles: true });
      textarea.dispatchEvent(inputEvent);
      
      // Enable the submit button directly
      if (submitButton) {
        submitButton.disabled = false;
      }
      
      // Focus the textarea
      textarea.focus();
      
      // Submit the form after a brief delay
      setTimeout(() => {
        if (submitButton) {
          submitButton.click();
        } else {
          form.requestSubmit();
        }
      }, 200);
    }
  };

  return (
    <ThemeProvider attribute="class" defaultTheme="light">
      <div className="h-screen w-full">
        <div className="h-full w-full" ref={threadRef}>
          <StreamProvider>
            <Thread />
          </StreamProvider>
        </div>
      </div>
    </ThemeProvider>
  );
} 