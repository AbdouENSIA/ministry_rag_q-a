"use client";

import React from "react";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight } from "lucide-react";
import { MinistryButton } from "@/components/ui/ministry-button";

export default function PrivacyPolicyPage(): React.ReactNode {
  return (
    <div className="flex flex-col min-h-screen bg-gray-50">
      {/* Header with Gradient and Animation */}
      <header className="bg-gradient-to-r from-[#0A3A63] via-[#0F4C81] to-[#1E5D99] text-white shadow-lg sticky top-0 z-50 transition-all duration-300">
        <div className="container mx-auto px-4 py-4">
          {/* Top bar with logo and name */}
          <div className="flex flex-wrap items-center justify-between mb-2">
            <div className="flex items-center space-x-6 space-x-reverse">
              <Link href="/" className="transform transition-transform hover:scale-105">
                <div className="relative rounded-full overflow-hidden bg-white p-1 shadow-md">
                  <Image 
                    src="/وزارة_التعليم_العالي_والبحث_العلمي.svg.png" 
                    alt="شعار وزارة التعليم العالي والبحث العلمي" 
                    width={60} 
                    height={60}
                    className="h-14 w-auto"
                    priority
                  />
                </div>
              </Link>
              <div>
                <h1 className="text-2xl font-bold text-white">وزارة التعليم العالي والبحث العلمي</h1>
                <div className="flex items-center">
                  <div className="h-1 w-1 rounded-full bg-yellow-300 mr-2 animate-pulse"></div>
                  <p className="text-sm text-yellow-100">سياسة الخصوصية</p>
                </div>
              </div>
            </div>
            <div className="mt-4 md:mt-0 flex space-x-2 space-x-reverse">
              <Link href="/">
                <MinistryButton 
                  variant="ghost" 
                  size="sm" 
                  rightIcon={<ArrowRight className="h-4 w-4" />}
                  className="border border-white/20 hover:bg-white/10 text-white"
                >
                  العودة إلى الرئيسية
                </MinistryButton>
              </Link>
            </div>
          </div>
        </div>
        
        {/* Decorative bottom bar */}
        <div className="h-1 w-full bg-gradient-to-r from-yellow-300 via-white to-yellow-300 opacity-60"></div>
      </header>

      {/* Breadcrumb */}
      <div className="bg-white border-b">
        <div className="container mx-auto px-4 py-2">
          <div className="flex items-center text-sm text-gray-600">
            <Link href="/" className="hover:text-[#0F4C81] transition-colors">الرئيسية</Link>
            <span className="mx-2">/</span>
            <span className="text-[#0F4C81] font-medium">سياسة الخصوصية</span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 container mx-auto px-4 py-10">
        <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-md p-8">
          <h1 className="text-3xl font-bold text-[#0F4C81] mb-6 text-center">سياسة الخصوصية</h1>
          
          <div className="space-y-6 text-gray-700 leading-relaxed text-right">
            <p>
              تلتزم وزارة التعليم العالي والبحث العلمي بحماية خصوصية المستخدمين وأمان بياناتهم. تصف سياسة الخصوصية هذه كيفية جمع واستخدام وحماية المعلومات التي يتم جمعها من خلال نظام استرجاع المعرفة المعزز بالذكاء الاصطناعي.
            </p>
            
            <h2 className="text-xl font-semibold text-[#0F4C81] mt-6 mb-3">المعلومات التي نجمعها</h2>
            <ul className="list-disc list-inside space-y-2 mr-4">
              <li>استفسارات المستخدم واستعلاماته في نظام الذكاء الاصطناعي.</li>
              <li>معلومات الجلسة وإحصائيات الاستخدام لتحسين الخدمة.</li>
              <li>بيانات تقنية مثل نوع المتصفح ونظام التشغيل وعنوان IP.</li>
            </ul>
            
            <h2 className="text-xl font-semibold text-[#0F4C81] mt-6 mb-3">كيفية استخدام المعلومات</h2>
            <p>نستخدم المعلومات التي نجمعها للأغراض التالية:</p>
            <ul className="list-disc list-inside space-y-2 mr-4">
              <li>تقديم إجابات دقيقة ومفيدة للاستفسارات.</li>
              <li>تحسين نظام الذكاء الاصطناعي وتدريبه.</li>
              <li>تحليل أنماط الاستخدام لتطوير وتحسين الخدمات.</li>
              <li>ضمان أمان وسلامة النظام.</li>
            </ul>
            
            <h2 className="text-xl font-semibold text-[#0F4C81] mt-6 mb-3">حماية البيانات</h2>
            <p>
              نتخذ تدابير أمنية مناسبة لحماية المعلومات ضد الوصول غير المصرح به أو التعديل أو الإفصاح أو الإتلاف. تشمل هذه التدابير تشفير البيانات، والوصول المقيد إلى المعلومات، ومراجعة ممارساتنا الأمنية بانتظام.
            </p>
            
            <h2 className="text-xl font-semibold text-[#0F4C81] mt-6 mb-3">مشاركة البيانات</h2>
            <p>
              لا نشارك معلوماتك الشخصية مع أطراف ثالثة إلا في الحالات التالية:
            </p>
            <ul className="list-disc list-inside space-y-2 mr-4">
              <li>عند الضرورة للامتثال للقانون أو أمر قضائي.</li>
              <li>لحماية حقوق أو ممتلكات أو سلامة الوزارة أو المستخدمين الآخرين.</li>
            </ul>
            
            <h2 className="text-xl font-semibold text-[#0F4C81] mt-6 mb-3">التعديلات على سياسة الخصوصية</h2>
            <p>
              يمكن أن نقوم بتحديث سياسة الخصوصية من وقت لآخر. سيتم نشر أي تغييرات على هذه الصفحة، وفي حالة التغييرات الجوهرية، سنقوم بإخطار المستخدمين من خلال إشعار بارز على موقعنا.
            </p>
            
            <h2 className="text-xl font-semibold text-[#0F4C81] mt-6 mb-3">اتصل بنا</h2>
            <p>
              إذا كانت لديك أسئلة أو استفسارات حول سياسة الخصوصية هذه، يرجى التواصل معنا عبر:
            </p>
            <address className="not-italic mt-2">
              <p>البريد الإلكتروني: contact@mesrs.dz</p>
              <p>الهاتف: +213 (0) 23 23 80 66</p>
              <p>العنوان: 11 شارع دودو مختار، بن عكنون، الجزائر</p>
            </address>
          </div>
        </div>
      </main>

      {/* Simple Footer */}
      <footer className="bg-gradient-to-b from-[#0D3D66] to-[#092C4C] text-white pt-6 pb-4">
        <div className="container mx-auto px-4">
          <div className="border-t border-white/10 pt-4">
            <div className="flex flex-col md:flex-row justify-between items-center text-sm text-gray-300">
              <p>&copy; {new Date().getFullYear()} وزارة التعليم العالي والبحث العلمي. جميع الحقوق محفوظة.</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
} 