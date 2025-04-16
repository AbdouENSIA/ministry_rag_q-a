import type { Metadata } from "next";
import "./globals.css";
import { Cairo } from "next/font/google";
import React from "react";
import { NuqsAdapter } from "nuqs/adapters/next/app";

const cairo = Cairo({
  subsets: ["arabic", "latin"],
  preload: true,
  display: "swap",
  variable: "--font-cairo",
});

export const metadata: Metadata = {
  title: "وزارة التعليم العالي والبحث العلمي - نظام استرجاع المعرفة",
  description: "واجهة التفاعل مع نظام استرجاع المعرفة المعزز بالذكاء الاصطناعي لوزارة التعليم العالي والبحث العلمي الجزائرية",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ar" dir="rtl" suppressHydrationWarning>
      <body className={`${cairo.className} ${cairo.variable} antialiased`}>
        <NuqsAdapter>{children}</NuqsAdapter>
      </body>
    </html>
  );
}
