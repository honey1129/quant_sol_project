export const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

export const compactFormatter = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 2,
});

export function formatCurrency(value: number): string {
  return currencyFormatter.format(value);
}

export function formatOptionalCurrency(value: number | null | undefined): string {
  return value === null || value === undefined ? "--" : formatCurrency(value);
}

export function formatCompact(value: number): string {
  return compactFormatter.format(value);
}

export function formatPercent(value: number, digits = 2, withSign = true): string {
  const prefix = withSign && value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(digits)}%`;
}

export function formatNumber(value: number, digits = 2): string {
  return value.toFixed(digits);
}

export function formatOptionalNumber(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined ? "--" : formatNumber(value, digits);
}

export function formatDateTime(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export function formatClock(value: Date): string {
  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(value);
}

export function formatAxisTime(value: string, compact = false): string {
  return new Intl.DateTimeFormat("en-US", compact
    ? { month: "short", day: "2-digit" }
    : { hour: "2-digit", minute: "2-digit" }
  ).format(new Date(value));
}

export function formatCountdown(target: string, now: Date): string {
  const diff = Math.max(0, new Date(target).getTime() - now.getTime());
  const totalSeconds = Math.floor(diff / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}
