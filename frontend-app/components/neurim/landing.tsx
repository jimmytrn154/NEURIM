"use client";

import { useState } from "react";
import { Brain, Plus, ChevronDown, Mic, Loader2 } from "lucide-react";
import { examplePrompts } from "@/lib/mock-frame";
import { ThemeToggle } from "@/components/neurim/theme-toggle";

const USER_NAME = "Lucas";

export function Landing({
  onSubmit,
  isSubmitting,
}: {
  onSubmit: (prompt: string) => void;
  isSubmitting: boolean;
}) {
  const [value, setValue] = useState("");

  const submit = () => {
    const t = value.trim();
    if (t && !isSubmitting) onSubmit(t);
  };

  return (
    <div className="min-h-screen relative bg-landing-bg overflow-hidden">
      {/* Top bar */}
      <div className="absolute top-0 left-0 right-0 px-6 py-[18px] flex items-center justify-between z-20">
        <div className="flex items-center gap-2">
          <Brain size={20} className="text-approach" />
          <span className="font-sans font-semibold text-[16px] tracking-[-0.01em] text-foreground">
            NEURIM
          </span>
        </div>

        <div className="h-8 w-8 rounded-full bg-primary grid place-items-center text-white font-sans font-semibold text-[13px]">
          {USER_NAME[0]}
        </div>
      </div>

      {/* Centered hero */}
      <div className="absolute inset-0 flex flex-col items-center justify-center px-[60px] text-center">
        <div
          className="pointer-events-none absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[760px] h-[300px]"
          style={{
            background:
              "radial-gradient(ellipse 55% 55% at 50% 50%, rgba(63,98,246,.42), rgba(63,98,246,.16) 45%, transparent 72%)",
            filter: "blur(26px)",
          }}
        />

        <h1 className="relative font-serif font-normal text-[46px] leading-[1.1] tracking-[-0.01em] mb-9 text-[#1a2233] dark:text-[#eceef2]">
          What&apos;s on your mind?
        </h1>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
          className="relative z-10 w-full max-w-[720px] flex items-center gap-[14px] rounded-[32px] px-[22px] py-4 bg-card border border-transparent shadow-[0_10px_40px_rgba(16,24,40,.14),0_2px_6px_rgba(16,24,40,.06)] dark:border-white/10 dark:shadow-[0_30px_90px_rgba(0,0,0,.5),0_0_60px_rgba(63,98,246,.08)]"
        >
          <Plus size={24} className="shrink-0 text-[#5c6270] dark:text-muted-foreground" />

          <input
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Describe a visual to build with your mind…"
            className="flex-1 bg-transparent outline-none text-[17px] text-foreground placeholder:text-[#9aa0ac] dark:placeholder:text-[#6f7178]"
          />

          <div className="flex shrink-0 items-center gap-1 text-[15px] text-[#475467] dark:text-muted-foreground">
            <span>SD-Turbo</span>
            <ChevronDown size={18} />
          </div>

          {isSubmitting ? (
            <Loader2 size={22} className="shrink-0 ml-1.5 animate-spin text-[#101828] dark:text-foreground" />
          ) : (
            <Mic size={22} className="shrink-0 ml-1.5 text-[#101828] dark:text-foreground" />
          )}
        </form>

        <div className="relative mt-[26px] flex flex-wrap justify-center gap-2.5">
          {examplePrompts.map((prompt) => (
            <button
              key={prompt}
              type="button"
              onClick={() => setValue(prompt)}
              className="rounded-[20px] border border-[rgba(16,24,40,.1)] bg-white/70 px-[15px] py-2 font-sans text-[13px] font-medium text-[#475467] backdrop-blur transition hover:text-foreground dark:border-white/8 dark:bg-[#22242b] dark:text-[#9a9ba0]"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      {/* Footer hint */}
      <div className="absolute bottom-[18px] left-0 right-0 flex items-center justify-center gap-2 font-sans text-[12px] font-medium text-[#98a2b3] dark:text-muted-foreground">
        <span className="h-[7px] w-[7px] rounded-full bg-primary animate-blink" />
        EPOC X connected · baseline set · ready to search
      </div>

      {/* Floating light/dark toggle (bottom-right) */}
      <ThemeToggle className="fixed bottom-5 right-5 z-30 h-10 w-10 rounded-full border border-border bg-card/80 backdrop-blur shadow-sm" />
    </div>
  );
}
