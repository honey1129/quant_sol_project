#!/bin/bash
# 每小时运行胜率监控脚本

cd /root/quant_sol_project
source .venv/bin/activate
python -m monitoring.hourly_performance_report
