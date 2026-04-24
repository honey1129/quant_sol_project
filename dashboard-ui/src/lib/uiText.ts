import type {
  DataSource,
  LogLevel,
  RiskLevel,
  RiskSnapshot,
  SignalDirection,
  ThemeMode,
  TimeRange,
  TradeRow,
  StrategyStatus,
} from "../types";

export function getStrategyStatusLabel(status: StrategyStatus): string {
  if (status === "Running") {
    return "运行中";
  }
  if (status === "Paused") {
    return "已暂停";
  }
  return "异常";
}

export function getSignalDirectionLabel(direction: SignalDirection | TradeRow["side"]): string {
  if (direction === "Long") {
    return "做多";
  }
  if (direction === "Short") {
    return "做空";
  }
  return "观望";
}

export function getTradeStatusLabel(status: TradeRow["status"]): string {
  if (status === "Take Profit") {
    return "已止盈";
  }
  if (status === "Stopped") {
    return "已止损";
  }
  if (status === "Canceled") {
    return "已取消";
  }
  if (status === "Filled") {
    return "已成交";
  }
  return status;
}

export function getRiskLevelLabel(level: RiskLevel): string {
  if (level === "High") {
    return "高";
  }
  if (level === "Medium") {
    return "中";
  }
  return "低";
}

export function getConnectionStatusLabel(status: RiskSnapshot["apiStatus"] | RiskSnapshot["wsStatus"]): string {
  if (status === "Connected") {
    return "已连接";
  }
  if (status === "Degraded") {
    return "降级";
  }
  if (status === "Lagging") {
    return "延迟";
  }
  return "已断开";
}

export function getLogLevelLabel(level: LogLevel): string {
  if (level === "SUCCESS") {
    return "成功";
  }
  if (level === "WARN") {
    return "警告";
  }
  if (level === "ERROR") {
    return "错误";
  }
  return "信息";
}

export function getDataSourceLabel(dataSource: DataSource): string {
  if (dataSource === "live") {
    return "实时 API";
  }
  if (dataSource === "hybrid") {
    return "混合数据";
  }
  return "模拟回退";
}

export function getThemeLabel(theme: ThemeMode): string {
  return theme === "dark" ? "深色" : "浅色";
}

export function getTimeRangeLabel(range: TimeRange): string {
  return range === "All" ? "全部" : range;
}

export function getSignalSourceLabel(source: string): string {
  if (source === "ML Model") {
    return "ML 模型";
  }
  return source;
}
