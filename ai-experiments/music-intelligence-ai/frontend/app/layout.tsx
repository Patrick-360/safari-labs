import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Live chord recognition",
  description: "Real-time chord and key detection from your microphone",
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
