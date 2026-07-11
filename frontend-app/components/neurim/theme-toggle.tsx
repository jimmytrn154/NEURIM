"use client";

import { useTheme } from "next-themes";
import { Sun, Moon } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Reusable light/dark toggle. The icon is chosen purely by CSS (`dark:`
 * utilities), which respond to the `.dark` class next-themes writes on <html>
 * *before* hydration — so the correct glyph shows on first paint with no
 * hydration mismatch and no `mounted` state. The click reads the live DOM class
 * so it flips correctly even before React has resolved the theme.
 */
export function ThemeToggle({ className }: { className?: string }) {
  const { setTheme } = useTheme();

  return (
    <button
      type="button"
      aria-label="Toggle theme"
      onClick={() => {
        const isDark = document.documentElement.classList.contains("dark");
        setTheme(isDark ? "light" : "dark");
      }}
      className={cn(
        "inline-grid place-items-center text-muted-foreground transition-colors hover:text-foreground",
        className
      )}
    >
      <Sun size={16} className="hidden dark:block" />
      <Moon size={16} className="block dark:hidden" />
      <span className="sr-only">Toggle light and dark theme</span>
    </button>
  );
}
