"use client";

import { useState } from "react";
import { ArrowUp } from "lucide-react";
import { cn } from "@/lib/utils";

export function SteerInput({
  onSubmit,
  disabled = false,
}: {
  onSubmit: (prompt: string) => void;
  disabled?: boolean;
}) {
  const [value, setValue] = useState("");

  const submit = () => {
    const trimmed = value.trim();
    if (trimmed && !disabled) {
      onSubmit(trimmed);
      setValue("");
    }
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className={cn(
        "w-full flex items-center gap-3 bg-card border border-border rounded-[32px] pl-5 pr-2 h-14",
        "shadow-[0_6px_20px_rgba(16,24,40,.05)] dark:shadow-[0_20px_60px_rgba(0,0,0,.45)]"
      )}
    >
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        disabled={disabled}
        placeholder="Steer the search, or lock it in…"
        className="flex-1 bg-transparent outline-none text-[15px] text-foreground placeholder:text-[#98a2b3] dark:placeholder:text-[#6f7178]"
      />

      <button
        type="submit"
        disabled={disabled || !value.trim()}
        className={cn(
          "h-9 w-9 rounded-full bg-primary text-white inline-flex items-center justify-center shrink-0",
          "shadow-[0_4px_16px_rgba(63,98,246,.35)]",
          "disabled:opacity-50 disabled:cursor-not-allowed"
        )}
      >
        <ArrowUp className="h-4 w-4" />
      </button>
    </form>
  );
}
