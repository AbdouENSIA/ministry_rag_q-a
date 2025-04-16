"use client";

import React from "react";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight } from "lucide-react";
import { MinistryButton } from "@/components/ui/ministry-button";

export default function TermsOfUsePage(): React.ReactNode {
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
                  <p className="text-sm text-yellow-100">شروط الاستخدام</p>
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
            <span className="text-[#0F4C81] font-medium">شروط الاستخدام</span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 container mx-auto px-4 py-10">
        <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-md p-8">
          <h1 className="text-3xl font-bold text-[#0F4C81] mb-6 text-center">شروط الاستخدام</h1>
          
          <div className="space-y-6 text-gray-700 leading-relaxed text-right">
            <p>
              مرحبًا بكم في نظام استرجاع المعرفة المعزز بالذكاء الاصطناعي التابع لوزارة التعليم العالي والبحث العلمي. يشكل استخدامك للنظام موافقتك على الالتزام بهذه الشروط والأحكام.
            </p>
            
            <h2 className="text-xl font-semibold text-[#0F4C81] mt-6 mb-3">الاستخدام المقبول</h2>
            <p>عند استخدام هذا النظام، توافق على:</p>
            <ul className="list-disc list-inside space-y-2 mr-4">
              <li>استخدام النظام بطريقة قانونية ومسؤولة.</li>
              <li>عدم إساءة استخدام النظام أو محاولة التحايل على أي قيود.</li>
              <li>عدم استخدام النظام لأغراض غير مشروعة أو ضارة.</li>
              <li>عدم انتهاك خصوصية الآخرين أو حقوقهم.</li>
              <li>عدم نشر محتوى غير لائق أو تمييزي أو تحريضي.</li>
            </ul>
            
            <h2 className="text-xl font-semibold text-[#0F4C81] mt-6 mb-3">الملكية الفكرية</h2>
            <p>
              جميع المحتويات المقدمة من خلال النظام، بما في ذلك النصوص والرسومات والشعارات وأسماء الخدمات والبرامج، هي ملك لوزارة التعليم العالي والبحث العلمي أو مرخصة لها. لا يُسمح بنسخ أو إعادة إنتاج أو توزيع أو تعديل هذه المواد دون إذن مسبق.
            </p>
            
            <h2 className="text-xl font-semibold text-[#0F4C81] mt-6 mb-3">دقة المعلومات</h2>
            <p>
              على الرغم من أننا نبذل قصارى جهدنا لضمان دقة المعلومات المقدمة من خلال نظام الذكاء الاصطناعي، إلا أننا لا نضمن اكتمالها أو دقتها أو حداثتها. المعلومات المقدمة هي للأغراض العامة فقط ولا ينبغي الاعتماد عليها كبديل للمشورة المهنية.
            </p>
            
            <h2 className="text-xl font-semibold text-[#0F4C81] mt-6 mb-3">تحديد المسؤولية</h2>
            <p>
              نحن لا نتحمل المسؤولية عن أي أضرار مباشرة أو غير مباشرة أو عرضية أو تبعية أو خاصة ناتجة عن استخدام هذا النظام أو عدم القدرة على استخدامه، حتى لو تم إخطارنا بإمكانية حدوث مثل هذه الأضرار.
            </p>
            
            <h2 className="text-xl font-semibold text-[#0F4C81] mt-6 mb-3">التعديلات على الشروط</h2>
            <p>
              نحتفظ بالحق في تعديل هذه الشروط في أي وقت. سيتم نشر التغييرات على هذه الصفحة، وستكون مسؤولاً عن مراجعة هذه الشروط بشكل دوري. استمرار استخدامك للنظام بعد نشر أي تغييرات يعني قبولك لهذه التغييرات.
            </p>
            
            <h2 className="text-xl font-semibold text-[#0F4C81] mt-6 mb-3">القانون المعمول به</h2>
            <p>
              تخضع هذه الشروط وتفسر وفقًا لقوانين الجمهورية الجزائرية الديمقراطية الشعبية، وتوافق على الخضوع للاختصاص القضائي الحصري للمحاكم في الجزائر.
            </p>
            
            <h2 className="text-xl font-semibold text-[#0F4C81] mt-6 mb-3">اتصل بنا</h2>
            <p>
              إذا كانت لديك أسئلة أو استفسارات حول شروط الاستخدام هذه، يرجى التواصل معنا عبر:
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