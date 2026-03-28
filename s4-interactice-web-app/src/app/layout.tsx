import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sidewalk Analysis Viewer",
  description: "Interactive map viewer for urban sidewalk accessibility analysis",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
