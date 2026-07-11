export function ProcessingState({ statusText }: { statusText?: string }) {
  const headline = statusText ?? "Reading your signal — rendering the first frame";

  return (
    <div className="flex flex-col gap-8 py-16">
      <div className="flex flex-col items-center gap-4 text-center">
        <h2 className="bg-[image:linear-gradient(90deg,#98a2b3,#3f62f6,#98a2b3)] bg-[length:200%_auto] bg-clip-text font-serif text-[19px] font-medium text-transparent animate-txtshimmer dark:bg-[image:linear-gradient(90deg,#6f7178,#eceef2,#6f8bff,#6f7178)]">
          {headline}
        </h2>

        <div className="flex items-center gap-2">
          <span className="h-1 w-1 rounded-full bg-primary animate-dot" style={{ animationDelay: "0ms" }} />
          <span className="h-1 w-1 rounded-full bg-primary animate-dot" style={{ animationDelay: "160ms" }} />
          <span className="h-1 w-1 rounded-full bg-primary animate-dot" style={{ animationDelay: "320ms" }} />
        </div>
      </div>

      <div className="flex flex-col items-stretch gap-6 md:flex-row">
        <div
          className="relative aspect-square flex-[1.35] overflow-hidden rounded-[18px] border border-border bg-[#eef1f6] animate-breathe dark:bg-card"
          style={{ animationDelay: "0ms" }}
        >
          <div className="pointer-events-none absolute inset-0 overflow-hidden">
            <div className="absolute inset-y-0 left-0 h-full w-1/3 -skew-x-12 bg-[rgba(63,98,246,.12)] blur-xl animate-sweep" />
          </div>
        </div>

        <div className="flex flex-1 flex-col gap-4">
          <div
            className="h-6 w-2/3 rounded-md border border-border bg-[#e8ecf3] animate-breathe dark:bg-card"
            style={{ animationDelay: "120ms" }}
          />
          <div
            className="h-24 rounded-xl border border-border bg-[#eef1f6] animate-breathe dark:bg-card"
            style={{ animationDelay: "240ms" }}
          />
          <div className="flex gap-2">
            <div
              className="h-6 w-16 rounded-full border border-border bg-[#e8ecf3] animate-breathe dark:bg-card"
              style={{ animationDelay: "360ms" }}
            />
            <div
              className="h-6 w-16 rounded-full border border-border bg-[#e8ecf3] animate-breathe dark:bg-card"
              style={{ animationDelay: "420ms" }}
            />
            <div
              className="h-6 w-16 rounded-full border border-border bg-[#e8ecf3] animate-breathe dark:bg-card"
              style={{ animationDelay: "480ms" }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
