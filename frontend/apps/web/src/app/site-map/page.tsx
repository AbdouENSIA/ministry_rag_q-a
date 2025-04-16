"use client";

import React from "react";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight } from "lucide-react";
import { MinistryButton } from "@/components/ui/ministry-button";

export default function SiteMapPage(): React.ReactNode {
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
                  <p className="text-sm text-yellow-100">خريطة الموقع</p>
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
            <span className="text-[#0F4C81] font-medium">خريطة الموقع</span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 container mx-auto px-4 py-10">
        <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-md p-8">
          <h1 className="text-3xl font-bold text-[#0F4C81] mb-6 text-center">خريطة الموقع</h1>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-right">
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-[#0F4C81] border-b border-gray-200 pb-2">الصفحات الرئيسية</h2>
              <ul className="space-y-2">
                <li>
                  <Link href="/" className="text-gray-700 hover:text-[#0F4C81] transition-colors flex items-center justify-end">
                    <span>الصفحة الرئيسية</span>
                    <ArrowRight className="h-3 w-3 mr-1" />
                  </Link>
                </li>
                <li>
                  <Link href="/rag" className="text-gray-700 hover:text-[#0F4C81] transition-colors flex items-center justify-end">
                    <span>نظام استرجاع المعرفة المعزز بالذكاء الاصطناعي</span>
                    <ArrowRight className="h-3 w-3 mr-1" />
                  </Link>
                </li>
              </ul>
            </div>
            
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-[#0F4C81] border-b border-gray-200 pb-2">المعلومات القانونية</h2>
              <ul className="space-y-2">
                <li>
                  <Link href="/privacy-policy" className="text-gray-700 hover:text-[#0F4C81] transition-colors flex items-center justify-end">
                    <span>سياسة الخصوصية</span>
                    <ArrowRight className="h-3 w-3 mr-1" />
                  </Link>
                </li>
                <li>
                  <Link href="/terms-of-use" className="text-gray-700 hover:text-[#0F4C81] transition-colors flex items-center justify-end">
                    <span>شروط الاستخدام</span>
                    <ArrowRight className="h-3 w-3 mr-1" />
                  </Link>
                </li>
              </ul>
            </div>
            
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-[#0F4C81] border-b border-gray-200 pb-2">روابط خارجية</h2>
              <ul className="space-y-2">
                <li>
                  <a href="https://www.mesrs.dz/" className="text-gray-700 hover:text-[#0F4C81] transition-colors flex items-center justify-end" target="_blank" rel="noopener noreferrer">
                    <span>الموقع الرسمي للوزارة</span>
                    <ArrowRight className="h-3 w-3 mr-1" />
                  </a>
                </li>
                <li>
                  <a href="https://www.mesrs.dz/index.php/reseau-universitaire-ar/" className="text-gray-700 hover:text-[#0F4C81] transition-colors flex items-center justify-end" target="_blank" rel="noopener noreferrer">
                    <span>المؤسسات الجامعية</span>
                    <ArrowRight className="h-3 w-3 mr-1" />
                  </a>
                </li>
                <li>
                  <a href="https://www.mesrs.dz/index.php/fr/recherche-scientifique/" className="text-gray-700 hover:text-[#0F4C81] transition-colors flex items-center justify-end" target="_blank" rel="noopener noreferrer">
                    <span>مراكز البحث العلمي</span>
                    <ArrowRight className="h-3 w-3 mr-1" />
                  </a>
                </li>
              </ul>
            </div>
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