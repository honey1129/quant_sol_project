import time
import traceback
import logging

def safe_run(target_function, max_retry=5, retry_delay=60):
    """
    自动重试封装器：
    - 遇到API、网络等偶发错误自动重试；
    - 提高无人值守云端运行稳定性。
    """
    for attempt in range(max_retry):
        try:
            target_function()
            return  # 正常结束
        except Exception as e:
            logging.error(f"[异常] 第{attempt+1}次重试: {str(e)}")
            logging.error(traceback.format_exc())
            if attempt < max_retry - 1:
                time.sleep(retry_delay)
    logging.critical("⚠️ 多次重试失败，已放弃本轮任务！")
