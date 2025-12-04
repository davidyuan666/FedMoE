#!/bin/bash

# Ensure running under bash even if invoked via sh
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# DS1000 Dataset Splitter Runner
# 作用：调用 split_ds1000_by_domain.py，将 DS1000 数据集按域拆分
# 兼容：Linux, macOS, Git Bash (Windows), WSL

set -euo pipefail

# 颜色（支持禁用）
if [ -t 1 ] && [ "${NO_COLOR:-}" != "1" ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'
  NC='\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BLUE=''
  NC=''
fi

# 默认参数（相对路径按项目根目录解析）
USE_WIN_PATHS=false
INPUT_FILE="dataset/ds1000.jsonl"
OUTPUT_DIR="dataset/domains"
DOMAINS="pandas,numpy,sklearn,sql,py_core"
CREATE_MISC=false
DEFAULT_DOMAIN="py_core"
VERBOSE=false
SAMPLE_SIZE=""

# 解析脚本与项目根路径（兼容 Windows 路径）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Python 解释器（兼容 Windows 上的 py -3）
PYEXE=""
PYARGS=()
PYBIN_DISPLAY=""

info()    { printf "%b[INFO]%b %s\n" "${BLUE}" "${NC}" "$*"; }
success() { printf "%b[SUCCESS]%b %s\n" "${GREEN}" "${NC}" "$*"; }
warn()    { printf "%b[WARNING]%b %s\n" "${YELLOW}" "${NC}" "$*"; }
error()   { printf "%b[ERROR]%b %s\n" "${RED}" "${NC}" "$*"; }

print_help() {
  cat <<EOF
${BLUE}DS1000 Dataset Splitter Runner${NC}

用法: $0 [选项]

选项:
  -i, --input FILE              输入 JSONL 文件 (默认: dataset/ds1000.jsonl)
  -o, --outdir DIR              输出目录 (默认: dataset/domains)
  -d, --domains LIST            逗号分隔的域列表 (默认: pandas,numpy,sklearn,sql,py_core)
  -m, --create-misc             额外生成 misc.jsonl
  --default-domain DOMAIN       默认回退域 (默认: py_core)
  -v, --verbose                 详细日志
  -s, --sample-size N           仅处理前 N 条
  -h, --help                    显示帮助

说明:
  - 所有相对路径统一相对“项目根目录”：$PROJECT_ROOT
  - 不会执行 cd，直接以绝对路径调用 Python 脚本
EOF
}

is_abs_path() {
  # POSIX 绝对路径或 Windows 盘符路径
  case "$1" in
    /*) return 0 ;;
    [A-Za-z]:/*) return 0 ;;
    [A-Za-z]:\\*) return 0 ;;
    *) return 1 ;;
  esac
}

abs_path_from_project() {
  local p="$1"
  if is_abs_path "$p"; then
    printf '%s' "$p"
  else
    printf '%s/%s' "$PROJECT_ROOT" "$p"
  fi
}

# 将 /d/xxx 形式的 MSYS 路径转换为 Windows 绝对路径 d:/xxx（保留正斜杠）
msys_to_win_path() {
  local p="$1"
  if [[ "$p" =~ ^/[A-Za-z]/ ]]; then
    local drive
    drive=$(printf '%s' "$p" | cut -c2)
    printf '%s' "${drive}:$(printf '%s' "$p" | cut -c3-)"
  else
    printf '%s' "$p"
  fi
}

# 将路径中的正斜杠替换为反斜杠（用于某些 Windows Python 环境）
to_backslash() {
  local p="$1"
  # 先标准化成 d:/xxx 再替换为 d:\xxx
  p="$(msys_to_win_path "$p")"
  printf '%s' "${p//\//\\}"
}

# 让 Python 检测给定路径是否存在（避免解释器路径风格不兼容）
python_path_exists() {
  local path="$1"
  "$PYEXE" ${PYARGS:+$PYARGS} -c "import os,sys; sys.exit(0 if os.path.exists(r'${path}') else 1)" >/dev/null 2>&1
}

# 判断给定 python 解释器是否为 Windows 平台
python_is_windows() {
  local plat
  plat=$($PYEXE ${PYARGS:-} - <<'PY'
import sys
print(sys.platform)
PY
)
  case "$plat" in
    win32|cygwin|msys|mingw*) return 0;;
    *) return 1;;
  esac
}

probe_python() {
  local exe="$1"
  local args="$2"
  "$exe" ${args:+$args} - <<'PY' >/dev/null 2>&1 || return 1
import sys
print(sys.version)
PY
  return 0
}

resolve_python() {
  # 候选解释器，按优先级尝试，并实际运行一次以验证可用
  local candidates=(
    "py|-3"
    "python3|"
    "python|"
  )

  PYEXE=""; PYARGS=""
  for c in "${candidates[@]}"; do
    local exe="${c%%|*}";
    local args="${c#*|}";
    if command -v "$exe" >/dev/null 2>&1; then
      if probe_python "$exe" "$args"; then
        PYEXE="$exe"; PYARGS="$args"; break
      fi
    fi
  done

  if [ -n "$PYEXE" ]; then
    if [ -n "$PYARGS" ]; then
      PYBIN_DISPLAY="$PYEXE $PYARGS"
    else
      PYBIN_DISPLAY="$PYEXE"
    fi
  else
    PYBIN_DISPLAY=""
  fi
}

print_python_info() {
  "$PYEXE" ${PYARGS:+$PYARGS} -c "import sys; print('executable=', sys.executable); print('version   =', sys.version.split()[0]); print('platform  =', sys.platform)" || true
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      -i|--input)        INPUT_FILE="$2"; shift 2 ;;
      -o|--outdir)       OUTPUT_DIR="$2"; shift 2 ;;
      -d|--domains)      DOMAINS="$2"; shift 2 ;;
      -m|--create-misc)  CREATE_MISC=true; shift ;;
      --default-domain)  DEFAULT_DOMAIN="$2"; shift 2 ;;
      -v|--verbose)      VERBOSE=true; shift ;;
      -s|--sample-size)  SAMPLE_SIZE="$2"; shift 2 ;;
      -h|--help)         print_help; exit 0 ;;
      *) error "未知选项: $1"; print_help; exit 1 ;;
    esac
  done
}

verify_environment() {
  info "验证环境..."
  resolve_python
  if [ -z "$PYBIN_DISPLAY" ]; then
    error "未找到可用的 Python 解释器 (python3/python/py)"
    return 1
  fi
  success "Python 可用: $PYBIN_DISPLAY"

  if [ "$VERBOSE" = "true" ]; then
    info "Python 环境信息:"
    print_python_info | sed 's/^/  /'
  fi

  local main_py="$PROJECT_ROOT/split_ds1000_by_domain.py"
  if [ ! -f "$main_py" ]; then
    error "主脚本不存在: $main_py"
    return 1
  fi
  success "主脚本已找到"

  # 解析路径（统一相对项目根目录）
  INPUT_FILE="$(abs_path_from_project "$INPUT_FILE")"
  OUTPUT_DIR="$(abs_path_from_project "$OUTPUT_DIR")"

  if [ ! -f "$INPUT_FILE" ]; then
    warn "输入文件不存在: $INPUT_FILE"
    warn "脚本将尝试处理，但可能会失败"
    warn "请确保输入文件存在或使用 -i/--input 指定正确的路径"
  else
    success "输入文件: $INPUT_FILE"
  fi
  success "输出目录: $OUTPUT_DIR"
}

run() {
  info "开始分割数据集..."
  info ""
  info "参数:"
  info "  输入文件: $INPUT_FILE"
  info "  输出目录: $OUTPUT_DIR"
  info "  域列表: $DOMAINS"
  info "  默认域: $DEFAULT_DOMAIN"
  info "  创建misc: $CREATE_MISC"
  if [ -n "$SAMPLE_SIZE" ]; then
    info "  样本大小: $SAMPLE_SIZE"
  fi
  info ""

  local posix_main_py="$PROJECT_ROOT/split_ds1000_by_domain.py"
  local posix_input="$INPUT_FILE"
  local posix_out="$OUTPUT_DIR"

  # 先构造 Windows 风格路径（d:/...），必要时回退到 POSIX 风格（/d/...）
  local win_main_py="$(msys_to_win_path "$posix_main_py")"
  local win_input="$(msys_to_win_path "$posix_input")"
  local win_out="$(msys_to_win_path "$posix_out")"

  # 组装命令（Windows 路径版本）
  local cmd=("$PYEXE")
  if [ -n "${PYARGS:-}" ]; then
    cmd+=("$PYARGS")
  fi
  cmd+=("$win_main_py" "--input" "$win_input" "--outdir" "$win_out" "--domains" "$DOMAINS" "--default-domain" "$DEFAULT_DOMAIN")
  if [ "$CREATE_MISC" = "true" ]; then
    cmd+=("--create-misc")
  fi
  if [ "$VERBOSE" = "true" ]; then
    cmd+=("--verbose")
  fi
  if [ -n "$SAMPLE_SIZE" ]; then
    cmd+=("--sample-size" "$SAMPLE_SIZE")
  fi

  if [ "$VERBOSE" = "true" ]; then
    info "执行命令: ${cmd[*]}"
    info ""
  fi

  # 执行 Python 脚本（捕获并回显日志）
  local exit_code=0
  local LOG_FILE
  if command -v mktemp >/dev/null 2>&1; then
    LOG_FILE="$(mktemp -t ds1000_split.XXXXXX.log)"
  else
    LOG_FILE="$OUTPUT_DIR/split_ds1000.log"
  fi

  # 使用进程替换保留 Python 的退出码，避免 PIPESTATUS 在 -u 下未绑定
  set +e
  "${cmd[@]}" > >(tee "$LOG_FILE") 2>&1
  exit_code=$?
  set -e

  # 如果失败且日志为空，尝试 POSIX 路径重试
  if [ $exit_code -ne 0 ] && [ ! -s "$LOG_FILE" ]; then
    warn "首次执行失败且无输出，尝试使用 POSIX 路径重试..."
    local cmd_posix=("$PYEXE")
    if [ -n "${PYARGS:-}" ]; then
      cmd_posix+=("$PYARGS")
    fi
    cmd_posix+=("$posix_main_py" "--input" "$posix_input" "--outdir" "$posix_out" "--domains" "$DOMAINS" "--default-domain" "$DEFAULT_DOMAIN")
    if [ "$CREATE_MISC" = "true" ]; then
      cmd_posix+=("--create-misc")
    fi
    if [ "$VERBOSE" = "true" ]; then
      cmd_posix+=("--verbose")
    fi
    if [ -n "$SAMPLE_SIZE" ]; then
      cmd_posix+=("--sample-size" "$SAMPLE_SIZE")
    fi

    if [ "$VERBOSE" = "true" ]; then
      info "重试命令: ${cmd_posix[*]}"
      info ""
    fi

    set +e
    "${cmd_posix[@]}" > >(tee -a "$LOG_FILE") 2>&1
    exit_code=$?
    set -e
  fi
  
  if [ $exit_code -eq 0 ]; then
    success "分割完成！"
    info ""
    info "输出文件位置: $OUTPUT_DIR"
    info "可以使用以下命令查看结果:"
    info "  ls -lh $OUTPUT_DIR/"
    info "Python 运行日志: $LOG_FILE"
    return 0
  else
    error "分割失败！(退出码: $exit_code)"
    error ""
    if [ -f "$LOG_FILE" ]; then
      error "Python 输出(最后50行):"
      tail -n 50 "$LOG_FILE" | sed 's/^/[PY] /'
    fi
    error ""
    error "故障排查建议:"
    error "  1. 使用 -v/--verbose 标志获取详细日志"
    error "  2. 检查输入文件格式是否为 JSONL 或 JSON 数组"
    error "  3. 检查输出目录是否有写入权限"
    error "  4. 运行: ${cmd[*]} --help 查看所有选项"
    error "  5. 查看完整日志: $LOG_FILE"
    return 1
  fi
}

main() {
  parse_args "$@"
  verify_environment || return 1
  # 提高日志实时性
  export PYTHONUNBUFFERED=1
  run
}

main "$@"
exit $?
