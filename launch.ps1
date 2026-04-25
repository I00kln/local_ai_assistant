# ============================================================
# llama.cpp 极简启动脚本 (支持 CLI 和 Server 模式)
# ============================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "llama.cpp Launcher"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ===== 选择启动模式 =====
Write-Host "`n请选择启动模式:" -ForegroundColor Cyan
Write-Host "  1. 命令行交互模式 (llama-cli.exe)"
Write-Host "  2. API 服务器模式 (llama-server.exe)"
$ModeChoice = Read-Host "`n输入选择 (1 或 2)"

if ($ModeChoice -eq "2") {
    $ExeName = "llama-server.exe"
    $IsServer = $true
} else {
    $ExeName = if (Test-Path (Join-Path $ScriptDir "llama-cli.exe")) { "llama-cli.exe" } else { "main.exe" }
    $IsServer = $false
}

$ExePath = Join-Path $ScriptDir $ExeName
if (-not (Test-Path $ExePath)) {
    # 尝试在 build 目录查找
    $BuildPath = Join-Path $ScriptDir "build\bin\Release\$ExeName"
    if (Test-Path $BuildPath) {
        $ExePath = $BuildPath
    } else {
        Write-Host "错误: 未找到 $ExeName" -ForegroundColor Red
        Read-Host "按回车退出"
        exit
    }
}

Write-Host "使用: $ExePath" -ForegroundColor Green

# ===== 扫描模型 =====
$ModelDir = if (Test-Path (Join-Path $ScriptDir "Model")) { Join-Path $ScriptDir "Model" } 
            elseif (Test-Path (Join-Path $ScriptDir "models")) { Join-Path $ScriptDir "models" }
            else { $ScriptDir }

$Models = Get-ChildItem -Path $ModelDir -Filter "*.gguf" -File

if ($Models.Count -eq 0) {
    Write-Host "错误: 未找到 .gguf 模型" -ForegroundColor Red
    Read-Host "按回车退出"
    exit
}

# ===== 选择模型 =====
Write-Host "`n可用模型:" -ForegroundColor Cyan
for ($i = 0; $i -lt $Models.Count; $i++) {
    Write-Host "  $($i+1). $($Models[$i].Name)"
}

$Choice = Read-Host "`n选择模型 (1-$($Models.Count))"
$ModelPath = $Models[[int]$Choice - 1].FullName

# ============================================================
# 参数配置 (根据模式不同)
# ============================================================

if ($IsServer) {
    # ===== 服务器模式参数 =====
    $Args = @(
        "-m", "`"$ModelPath`"",                # 模型路径
        
        # ----- 服务器配置 -----
        "--host", "127.0.0.1",                  # 监听地址 (0.0.0.0 允许外部访问)
        "--port", "8080",                       # 监听端口
        # "--path", "/v1",                      # API 路径前缀 (默认 /)
        
        # ----- 上下文与性能 -----
        "-c", "8192",                           # 上下文长度
        "-b", "2048",                           # 批处理大小
        "-t", "4",                              # CPU 线程数
        "-np", "1",                             # 并行处理请求数 (默认 1)
        
        # ----- 采样参数 (服务器端默认值) -----
        "--temp", "0.3",                        # 温度
        "--top-p", "0.95",                      # Top-P
        "--top-k", "40",                        # Top-K
        "--min-p", "0.05",                      # Min-P
        "--repeat-penalty", "1.1",              # 重复惩罚
        
        # ----- 关闭 think 标签 (DeepSeek/Qwen 等) -----
        "--reasoning", "off",                   # 关闭思考输出
        
        # ----- 其他 -----
        # "--mlock",                            # 锁定内存
        # "--verbose",                          # 详细日志
        "--log-disable",                         # 禁用日志刷屏 (服务器模式推荐)
        
        # ===== 服务器模式扩展参数 =====

        # ----- API 配置 -----
        #"--api-key", "123",            # API 密钥验证
        #"--ssl-key", "key.pem",                    # SSL 私钥 (启用 HTTPS)
        #"--ssl-cert", "cert.pem",                  # SSL 证书

        # ----- 并发与队列 -----
        "--ubatch-size", "512",                    # 微批大小

        # ----- 上下文管理 -----
        "--cache-ram", "8192",                     # 缓存大小 (MiB)
        "--defrag-thold", "0.1"                   # KV 缓存碎片整理阈值

        # ----- 多模型 (热切换) -----
        #"--alias", "model1",                       # 模型别名 (可通过 /model1 访问)

        # ----- 日志与监控 -----
        #"--metrics"                               # 启用 Prometheus 指标 (端口 8081)
        #"--no-slots",                              # 禁用 slot 端点
        #"--no-webui"                               # 禁用内置 Web UI
    )
    
    Write-Host "`n服务器将启动在: http://127.0.0.1:8080" -ForegroundColor Green
    Write-Host "API 端点: http://127.0.0.1:8080/v1/chat/completions" -ForegroundColor Yellow
    Write-Host "按 Ctrl+C 停止服务器`n" -ForegroundColor Gray
    
} else {
    # ===== 命令行交互模式参数 =====
    $Args = @(
    # ===== 必选参数 =====
    "-m", "`"$ModelPath`"",                    # 模型路径
    
    # ===== 上下文与生成 =====
    "-c", "8192",                               # 上下文长度 (2048/4096/8192/32768)
    "-n", "4096",                                # 最大生成 token 数 (-1=无限)
    "-b", "2048",                               # 批处理大小 (影响 prompt 处理速度)
    
    # ===== CPU 线程 =====
    "-t", "2",                                  # CPU 线程数 (根据你的 CPU 核心数调整)
    # "-tb", "8",                               # 批处理线程数 (默认等于 -t)
    
    # ===== 采样参数 =====
    "--temp", "0.7",                            # 温度 (0=确定, 1=平衡, >1=创造性)
    "--top-p", "0.95",                          # 核采样 (1.0=禁用)
    "--top-k", "40",                            # Top-K 采样 (0=禁用)
    "--min-p", "0.05",                          # 最小概率过滤
    "--repeat-penalty", "1.1",                  # 重复惩罚 (>1 减少重复)
    "--repeat-last-n", "64",                    # 重复检测范围

#```powershell
    # ===== 关闭思考/推理 (针对 DeepSeek/Qwen 等有 think 标签的模型) =====
    "--reasoning", "off",                       # 关闭推理/思考输出 (去掉 <think> 内容)
    # "--reasoning-format", "none",             # 或使用此参数，none/deepseek/deepseek-legacy
    
    # ===== 交互与对话 =====
    "-cnv",                                     # 对话模式 (保留历史)
    # "-no-cnv",                                # 取消注释则关闭对话模式
    "--color", "auto",                                      # 彩色输出
    # "-mli",                                   # 多行输入模式 (取消注释启用)
    
    # ===== 系统提示词 =====
    #"-sys", "`"你是一个有帮助的助手`"",          # 系统提示词 (修改引号内内容)
    # "-sysf", "`"system.txt`"",                # 或从文件读取系统提示词
    
    # ===== 提示词 (如果指定则直接生成后退出) =====
    # "-p", "`"你好，请介绍一下自己`"",          # 单次问答 (取消注释则非交互)
    # "-f", "`"prompt.txt`"",                   # 或从文件读取提示词
    
    # ===== 调试与日志 =====
    "--show-timings"                           # 显示生成耗时
    # "--verbose-prompt",                       # 打印详细提示词 (调试用)
    # "-lv", "3",                               # 日志级别 (0-4, 调试时设为 3 或 4)
    
    # ===== 内存优化 =====
    # "--mlock",                                # 锁定内存防止交换 (需管理员权限)
    # "--no-mmap"                               # 禁用内存映射 (更慢但兼容性好)
    )
}

# ============================================================
# 启动
# ============================================================
$CmdLine = "$ExePath $($Args -join ' ')"
Write-Host "执行: $CmdLine`n" -ForegroundColor DarkGray

Set-Location $ScriptDir
cmd /c $CmdLine

Write-Host "`n程序已退出" -ForegroundColor Green
Read-Host "按回车关闭"
