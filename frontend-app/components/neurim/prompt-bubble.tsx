export function PromptBubble({ prompt }: { prompt: string }) {
  return (
    <div className="flex justify-end">
      <div
        className="max-w-[420px] border border-border bg-bubble px-4 py-[11px] text-[14.5px] leading-relaxed text-[#344054] shadow-sm dark:text-[#dfe1e6] dark:shadow-lg"
        style={{ borderRadius: "16px 16px 4px 16px" }}
      >
        {prompt}
      </div>
    </div>
  );
}
