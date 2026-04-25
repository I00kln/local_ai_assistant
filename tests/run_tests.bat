@echo off
REM run_tests.bat
REM Windows 测试运行脚本

echo ========================================
echo 高风险场景测试
echo ========================================

cd /d %~dp0

if not exist "..\venv\Scripts\activate.bat" (
    echo [警告] 未找到虚拟环境，使用系统 Python
) else (
    call ..\venv\Scripts\activate.bat
)

echo.
echo [1/3] 检查测试依赖...
pip show pytest >nul 2>&1
if errorlevel 1 (
    echo 安装测试依赖...
    pip install -r requirements-test.txt
)

echo.
echo [2/3] 运行测试...
pytest test_high_risk_scenarios.py -v --tb=short

echo.
echo [3/3] 测试完成
pause
