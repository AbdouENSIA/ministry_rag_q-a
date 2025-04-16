"use client";

import React from "react";
import Link from "next/link";
import Image from "next/image";
import { Thread } from "@/components/thread";
import { StreamProvider } from "@/providers/Stream";
import { Toaster } from "sonner";
import { LoadingPage } from "@/components/ui/loading";
import { MinistryButton } from "@/components/ui/ministry-button";
import { MinistryCard } from "@/components/ui/ministry-card";
import { 
  ArrowLeft, 
  BookOpen, 
  History, 
  MapPin, 
  Users, 
  GraduationCap, 
  Microscope,
  ExternalLink,
  Calendar,
  BookOpen as BookOpenIcon,
  GraduationCap as GraduationCapIcon
} from "lucide-react";

export default function HomePage(): React.ReactNode {
  return (
    <React.Suspense fallback={<LoadingPage />}>
      <Toaster />
      <div className="flex flex-col min-h-screen bg-gray-50">
        {/* Enhanced Header with Gradient and Animation */}
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
                    <p className="text-sm text-yellow-100">الجمهورية الجزائرية الديمقراطية الشعبية</p>
                  </div>
                </div>
              </div>
              <div className="mt-4 md:mt-0 flex space-x-2 space-x-reverse">
                <Link href="/rag">
                  <MinistryButton 
                    variant="ghost" 
                    size="sm" 
                    rightIcon={<ArrowLeft className="h-4 w-4" />}
                    className="border border-white/20 hover:bg-white/10 text-white"
                  >
                    نظام الذكاء الاصطناعي
                  </MinistryButton>
                </Link>
              </div>
            </div>
          </div>
          
          {/* Decorative bottom bar */}
          <div className="h-1 w-full bg-gradient-to-r from-yellow-300 via-white to-yellow-300 opacity-60"></div>
        </header>

        {/* Hero Section */}
        <section className="bg-gradient-to-b from-[#0F4C81]/80 to-[#0F4C81]/10 text-center py-20 px-4">
          <div className="container mx-auto max-w-5xl">
            <h1 className="text-4xl md:text-5xl font-bold mb-6 text-[#0F4C81]">نظام استرجاع المعرفة المعزز بالذكاء الاصطناعي</h1>
            <p className="text-lg md:text-xl mb-8 max-w-3xl mx-auto text-gray-700">
              منصة متطورة تستخدم تقنيات الذكاء الاصطناعي لاسترجاع وتقديم المعلومات المتعلقة بوزارة التعليم العالي والبحث العلمي
            </p>
            <Link href="/rag">
              <MinistryButton size="lg" rightIcon={<ArrowLeft className="h-5 w-5" />}>
                ابدأ الاستخدام
              </MinistryButton>
            </Link>
          </div>
        </section>

        {/* About Section */}
        <section id="about" className="py-16 px-4 bg-gray-50">
          <div className="container mx-auto max-w-5xl">
            <h2 className="text-3xl font-bold mb-8 text-center text-[#0F4C81]">وزارة التعليم العالي والبحث العلمي</h2>
            
            <MinistryCard className="mb-8" variant="elevated">
              <p className="text-lg leading-relaxed text-gray-700">
                وزارة التعليم العالي والبحث العلمي الجزائرية هي أحد وزارات الحكومة الجزائرية. تشرف هذه الوزارة على قطاع التعليم العالي والجامعات والبحث العلمي في الجزائر. استحدثت هذه الوزارة لأول مرة سنة 1970، في بعض الأوقات، كانت مقسمة إلى قطاعين أو تحت وصاية وزير منتدب.
              </p>
            </MinistryCard>
            
            <div className="grid md:grid-cols-2 gap-8">
              <MinistryCard variant="elevated" icon={<History className="h-6 w-6" />} title="التاريخ">
                <p className="text-gray-700 leading-relaxed">
                  بعد استقلال الجزائر وإلى غاية سنة 1971، كان قطاع التعليم العالي والبحث العلمي تحت إشراف وزارة التربية الوطنية. في سنة 1965، نصب مجلس أعلى داخل وزارة التربية الوطنية. ظهرت وزارة التعليم العالي والبحث العلمي لأول مر في حكومة بومدين الثالثة في 1970، واستمرت إلى غاية 1983. بين سنتي 1984 و1990، كان القطاع تحت إشراف الوزير الأول، ثم أعيدت سنة 1992 و1994.
                </p>
              </MinistryCard>
              
              <MinistryCard variant="elevated" icon={<MapPin className="h-6 w-6" />} title="مقر الوزارة">
                <p className="text-gray-700 leading-relaxed">
                  يقع مقر الوزارة وسط مباني جامعة الجزائر 3 بنهج دودو مختار ببن عكنون بولاية الجزائر. بين سنتي 1970 و 1986، كان مقر الوزارة بسيدي امحمد بالجزائر العاصمة بشارع البشير بطار (المقر الذي تشغله حاليا دار الصحافة طاهر جاووت). وانتقلت الوزارة إلى شارع الإخوة عيسو إلى غاية منتصف التسعينيات (وتشغل المقر حاليا وزارة التكوين والتعليم المهني).
                </p>
              </MinistryCard>
            </div>
          </div>
        </section>
        
        {/* Statistics Section */}
        <section className="py-16 px-4 bg-white">
          <div className="container mx-auto max-w-5xl">
            <h2 className="text-3xl font-bold mb-12 text-center text-[#0F4C81]">إحصائيات القطاع</h2>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              <MinistryCard variant="outline" icon={<BookOpen className="h-6 w-6" />}>
                <div className="text-center">
                  <div className="text-4xl font-bold text-[#0F4C81] mb-2">106</div>
                  <div className="text-gray-600">مؤسسة جامعية</div>
                </div>
              </MinistryCard>
              <MinistryCard variant="outline" icon={<Users className="h-6 w-6" />}>
                <div className="text-center">
                  <div className="text-4xl font-bold text-[#0F4C81] mb-2">1.7M+</div>
                  <div className="text-gray-600">طالب</div>
                </div>
              </MinistryCard>
              <MinistryCard variant="outline" icon={<GraduationCap className="h-6 w-6" />}>
                <div className="text-center">
                  <div className="text-4xl font-bold text-[#0F4C81] mb-2">65K+</div>
                  <div className="text-gray-600">أستاذ باحث</div>
                </div>
              </MinistryCard>
              <MinistryCard variant="outline" icon={<Microscope className="h-6 w-6" />}>
                <div className="text-center">
                  <div className="text-4xl font-bold text-[#0F4C81] mb-2">36</div>
                  <div className="text-gray-600">مركز بحث علمي</div>
                </div>
              </MinistryCard>
            </div>
          </div>
        </section>

        {/* Enhanced Footer */}
        <footer className="bg-gradient-to-b from-[#0D3D66] to-[#092C4C] text-white pt-10 pb-4 mt-auto">
          {/* Top section with logo and info */}
          <div className="container mx-auto px-4">
            <div className="grid md:grid-cols-4 gap-8 mb-8">
              <div className="md:col-span-1 flex flex-col items-center md:items-start">
                <div className="bg-white p-2 rounded-lg shadow-lg mb-4">
                  <Image 
                    src="/وزارة_التعليم_العالي_والبحث_العلمي.svg.png" 
                    alt="شعار وزارة التعليم العالي والبحث العلمي" 
                    width={80} 
                    height={80}
                    className="h-16 w-auto"
                  />
                </div>
                <h3 className="text-lg font-bold mb-2">وزارة التعليم العالي والبحث العلمي</h3>
                <p className="text-sm text-gray-300 text-center md:text-right">الجمهورية الجزائرية الديمقراطية الشعبية</p>
              </div>
              
              <div className="md:col-span-1">
                <h3 className="text-lg font-semibold mb-4 flex items-center after:content-[''] after:h-[1px] after:flex-1 after:bg-yellow-400/30 after:mr-2">
                  <Calendar className="h-4 w-4 ml-2 text-yellow-300" />
                  روابط سريعة
                </h3>
                <ul className="space-y-2 text-sm">
                  <li>
                    <a href="https://www.mesrs.dz/" className="hover:text-yellow-300 transition-colors flex items-center" target="_blank" rel="noopener noreferrer">
                      <ArrowLeft className="h-3 w-3 ml-1" />
                      <span>الأخبار والإعلانات</span>
                    </a>
                  </li>
                  <li>
                    <a href="https://www.mesrs.dz/index.php/reseau-universitaire-ar/" className="hover:text-yellow-300 transition-colors flex items-center" target="_blank" rel="noopener noreferrer">
                      <ArrowLeft className="h-3 w-3 ml-1" />
                      <span>المؤسسات الجامعية</span>
                    </a>
                  </li>
                  <li>
                    <a href="https://www.mesrs.dz/index.php/fr/recherche-scientifique/" className="hover:text-yellow-300 transition-colors flex items-center" target="_blank" rel="noopener noreferrer">
                      <ArrowLeft className="h-3 w-3 ml-1" />
                      <span>مراكز البحث العلمي</span>
                    </a>
                  </li>
                </ul>
              </div>
              
              <div className="md:col-span-1">
                <h3 className="text-lg font-semibold mb-4 flex items-center after:content-[''] after:h-[1px] after:flex-1 after:bg-yellow-400/30 after:mr-2">
                  <BookOpenIcon className="h-4 w-4 ml-2 text-yellow-300" />
                  معلومات الاتصال
                </h3>
                <address className="not-italic text-sm space-y-2">
                  <p className="flex items-start">
                    <span className="inline-block w-5 ml-2 mt-1">🏢</span>
                    <span>11 شارع دودو مختار، بن عكنون، الجزائر</span>
                  </p>
                  <p className="flex items-start">
                    <span className="inline-block w-5 ml-2 mt-1">📧</span>
                    <span>contact@mesrs.dz</span>
                  </p>
                  <p className="flex items-start">
                    <span className="inline-block w-5 ml-2 mt-1">📞</span>
                    <span>+213 (0) 23 23 80 66</span>
                  </p>
                </address>
              </div>
              
              <div className="md:col-span-1">
                <h3 className="text-lg font-semibold mb-4 flex items-center after:content-[''] after:h-[1px] after:flex-1 after:bg-yellow-400/30 after:mr-2">
                  <GraduationCapIcon className="h-4 w-4 ml-2 text-yellow-300" />
                  تابعنا
                </h3>
                <div className="grid grid-cols-4 gap-2">
                  <a href="https://www.facebook.com/mesrs.dz" aria-label="Facebook" className="flex items-center justify-center h-10 w-10 rounded-full bg-white/10 hover:bg-white/20 transition-colors" target="_blank" rel="noopener noreferrer">
                    <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                      <path fillRule="evenodd" d="M22 12c0-5.523-4.477-10-10-10S2 6.477 2 12c0 4.991 3.657 9.128 8.438 9.878v-6.987h-2.54V12h2.54V9.797c0-2.506 1.492-3.89 3.777-3.89 1.094 0 2.238.195 2.238.195v2.46h-1.26c-1.243 0-1.63.771-1.63 1.562V12h2.773l-.443 2.89h-2.33v6.988C18.343 21.128 22 16.991 22 12z" clipRule="evenodd" />
                    </svg>
                  </a>
                  <a href="https://x.com/moheriom?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor" aria-label="Twitter" className="flex items-center justify-center h-10 w-10 rounded-full bg-white/10 hover:bg-white/20 transition-colors" target="_blank" rel="noopener noreferrer">
                    <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84" />
                    </svg>
                  </a>
                  <a href="https://www.instagram.com/accounts/login/?next=%2Fministry.official.page" aria-label="Instagram" className="flex items-center justify-center h-10 w-10 rounded-full bg-white/10 hover:bg-white/20 transition-colors" target="_blank" rel="noopener noreferrer">
                    <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                      <path fillRule="evenodd" d="M12.315 2c2.43 0 2.784.013 3.808.06 1.064.049 1.791.218 2.427.465a4.902 4.902 0 011.772 1.153 4.902 4.902 0 011.153 1.772c.247.636.416 1.363.465 2.427.048 1.067.06 1.407.06 4.123v.08c0 2.643-.012 2.987-.06 4.043-.049 1.064-.218 1.791-.465 2.427a4.902 4.902 0 01-1.153 1.772 4.902 4.902 0 01-1.772 1.153c-.636.247-1.363.416-2.427.465-1.067.048-1.407.06-4.123.06h-.08c-2.643 0-2.987-.012-4.043-.06-1.064-.049-1.791-.218-2.427-.465a4.902 4.902 0 01-1.772-1.153 4.902 4.902 0 01-1.153-1.772c-.247-.636-.416-1.363-.465-2.427-.047-1.024-.06-1.379-.06-3.808v-.63c0-2.43.013-2.784.06-3.808.049-1.064.218-1.791.465-2.427a4.902 4.902 0 011.153-1.772A4.902 4.902 0 015.45 2.525c.636-.247 1.363-.416 2.427-.465C8.901 2.013 9.256 2 11.685 2h.63zm-.081 1.802h-.468c-2.456 0-2.784.011-3.807.058-.975.045-1.504.207-1.857.344-.467.182-.8.398-1.15.748-.35.35-.566.683-.748 1.15-.137.353-.3.882-.344 1.857-.047 1.023-.058 1.351-.058 3.807v.468c0 2.456.011 2.784.058 3.807.045.975.207 1.504.344 1.857.182.466.399.8.748 1.15.35.35.683.566 1.15.748.353.137.882.3 1.857.344 1.054.048 1.37.058 4.041.058h.08c2.597 0 2.917-.01 3.96-.058.976-.045 1.505-.207 1.858-.344.466-.182.8-.398 1.15-.748.35-.35.566-.683.748-1.15.137-.353.3-.882.344-1.857.048-1.055.058-1.37.058-4.041v-.08c0-2.597-.01-2.917-.058-3.96-.045-.976-.207-1.505-.344-1.858a3.097 3.097 0 00-.748-1.15 3.098 3.098 0 00-1.15-.748c-.353-.137-.882-.3-1.857-.344-1.023-.047-1.351-.058-3.807-.058zM12 6.865a5.135 5.135 0 110 10.27 5.135 5.135 0 010-10.27zm0 1.802a3.333 3.333 0 100 6.666 3.333 3.333 0 000-6.666zm5.338-3.205a1.2 1.2 0 110 2.4 1.2 1.2 0 010-2.4z" clipRule="evenodd" />
                    </svg>
                  </a>
                  <a href="https://www.youtube.com/channel/UCLTBvSZAwawsSKqKHHSsVpw" aria-label="YouTube" className="flex items-center justify-center h-10 w-10 rounded-full bg-white/10 hover:bg-white/20 transition-colors" target="_blank" rel="noopener noreferrer">
                    <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                      <path fillRule="evenodd" d="M19.812 5.418c.861.23 1.538.907 1.768 1.768C21.998 8.746 22 12 22 12s0 3.255-.418 4.814a2.504 2.504 0 0 1-1.768 1.768c-1.56.419-7.814.419-7.814.419s-6.255 0-7.814-.419a2.505 2.505 0 0 1-1.768-1.768C2 15.255 2 12 2 12s0-3.255.417-4.814a2.507 2.507 0 0 1 1.768-1.768C5.744 5 11.998 5 11.998 5s6.255 0 7.814.418ZM15.194 12 10 15V9l5.194 3Z" clipRule="evenodd" />
                    </svg>
                  </a>
                </div>
              </div>
            </div>
            
            {/* Bottom copyright section */}
            <div className="border-t border-white/10 pt-4">
              <div className="flex flex-col md:flex-row justify-between items-center text-sm text-gray-300">
                <p>&copy; {new Date().getFullYear()} وزارة التعليم العالي والبحث العلمي. جميع الحقوق محفوظة.</p>
                <div className="flex mt-3 md:mt-0 space-x-4 space-x-reverse">
                  <Link href="/privacy-policy" className="hover:text-white transition-colors">سياسة الخصوصية</Link>
                  <span className="text-gray-500">|</span>
                  <Link href="/terms-of-use" className="hover:text-white transition-colors">شروط الاستخدام</Link>
                  <span className="text-gray-500">|</span>
                  <Link href="/site-map" className="hover:text-white transition-colors">خريطة الموقع</Link>
                </div>
              </div>
            </div>
          </div>
          
          {/* Footer decorative bar */}
          <div className="h-1 w-full bg-gradient-to-r from-transparent via-yellow-400 to-transparent mt-4 opacity-30"></div>
        </footer>
      </div>
    </React.Suspense>
  );
}
