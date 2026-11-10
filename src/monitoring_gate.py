"""显式门槛：默认安全准备模式，不自动抓取、调用 AI 或发送通知。"""
import os


MONITORING_ENABLED_ENV = "MONITORING_ENABLED"
_ENABLED_VALUES = {"1", "true", "yes", "on"}


def monitoring_enabled() -> bool:
    """未设置或非真值时保持准备模式。只读进程环境，不改任务或登录态数据。"""
    return os.getenv(MONITORING_ENABLED_ENV, "").strip().lower() in _ENABLED_VALUES


def monitoring_block_reason() -> str:
    return (
        "监控处于安全准备模式，未启动抓取、AI 分析或通知。"
        "确认任务后设置 MONITORING_ENABLED=true 并重启服务才会执行。"
        "任务、登录态文件和历史结果会保留。"
    )
