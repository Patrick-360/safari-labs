import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Chord lab",
  description: "Live chord detection and full-track analysis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
