#!/bin/bash

# Test all DS1000 dataset splitter runners
# 测试所有 DS1000 数据集分割器运行器

set -euo pipefail

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { printf "%b[INFO]%b %s\n" "${BLUE}" "${NC}" "$*"; }
success() { printf "%b[SUCCESS]%b %s\n" "${GREEN}" "${NC}" "$*"; }
warn()    { printf "%b[WARNING]%b %s\n" "${YELLOW}" "${NC}" "$*"; }
error()   { printf "%b[ERROR]%b %s\n" "${RED}" "${NC}" "$*"; }

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 测试计数
TESTS_PASSED=0
TESTS_FAILED=0

# 测试函数
test_runner() {
  local name="$1"
  local cmd="$2"
  
  info ""
  info "=========================================="
  info "测试: $name"
  info "=========================================="
  info "命令: $cmd"
  info ""
  
  if eval "$cmd" >/dev/null 2>&1; then
    success "$name 测试通过"
    ((TESTS_PASSED++))
  else
    error "$name 测试失败"
    ((TESTS_FAILED++))
  fi
}

# 检查文件是否存在
check_file() {
  local file="$1"
  if [ -f "$file" ]; then
    success "文件存在: $file"
    return 0
  else
    error "文件不存在: $file"
    return 1
  fi
}

# 主函数
main() {
  info "开始测试所有 DS1000 数据集分割器运行器..."
  info ""
  
  # 检查必要文件
  info "检查必要文件..."
  check_file "$PROJECT_ROOT/split_ds1000_by_domain.py" || return 1
  check_file "$PROJECT_ROOT/dataset/ds1000.jsonl" || return 1
  check_file "$SCRIPT_DIR/split_ds1000.sh" || return 1
  info ""
  
  # 测试 1: Python 脚本
  info "测试 Python 脚本..."
  if command -v python3 >/dev/null 2>&1; then
    test_runner "Python 脚本 (python3)" "python3 $PROJECT_ROOT/split_ds1000_by_domain.py --sample-size 10"
  elif command -v python >/dev/null 2>&1; then
    test_runner "Python 脚本 (python)" "python $PROJECT_ROOT/split_ds1000_by_domain.py --sample-size 10"
  else
    error "未找到 Python 解释器"
    ((TESTS_FAILED++))
  fi
  
  # 测试 2: Bash 脚本
  info ""
  info "测试 Bash 脚本..."
  test_runner "Bash 脚本" "bash $SCRIPT_DIR/split_ds1000.sh -s 10"
  
  # 测试 3: PowerShell 脚本 (仅在 Windows 上)
  if command -v powershell >/dev/null 2>&1; then
    info ""
    info "测试 PowerShell 脚本..."
    test_runner "PowerShell 脚本" "powershell -NoProfile -ExecutionPolicy Bypass -File $SCRIPT_DIR/split_ds1000.ps1 -SampleSize 10"
  else
    warn "PowerShell 不可用，跳过 PowerShell 脚本测试"
  fi
  
  # 测试 4: 批处理脚本 (仅在 Windows 上)
  if command -v cmd >/dev/null 2>&1; then
    info ""
    info "测试批处理脚本..."
    test_runner "批处理脚本" "cmd /c $SCRIPT_DIR/split_ds1000.bat -s 10"
  else
    warn "CMD 不可用，跳过批处理脚本测试"
  fi
  
  # 总结
  info ""
  info "=========================================="
  info "测试总结"
  info "=========================================="
  info "通过: $TESTS_PASSED"
  info "失败: $TESTS_FAILED"
  info ""
  
  if [ $TESTS_FAILED -eq 0 ]; then
    success "所有测试通过！"
    return 0
  else
    error "部分测试失败！"
    return 1
  fi
}

main "$@"
exit $?

