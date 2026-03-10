"""
run.py — 원클릭 서버 배포 스크립트
════════════════════════════════════
로컬에서 이 파일만 실행하면:
  1. 서버 SSH 접속
  2. 필요 패키지 자동 설치
  3. 프로젝트 파일 업로드 (scp)
  4. pip 라이브러리 설치
  5. tmux 세션에서 main_scheduler.py 자동 실행

사용법:
  python run.py              (서버 IP를 입력 프롬프트로 받음)
  python run.py --stop       (실행 중인 봇 중지)
  python run.py --status     (봇 상태 확인)
  python run.py --logs       (최근 로그 확인)
"""

import os
import sys
import subprocess
import getpass
import argparse
from pathlib import Path

# ── settings.py에서 서버 설정 로드 ──────────────────
try:
    import settings as cfg
    SERVER_CFG = cfg.SERVER
except ImportError:
    SERVER_CFG = {
        "host": "",
        "user": "root",
        "remote_dir": "/root/quant_trading_system",
        "tmux_session": "trade_bot",
    }

# ── 프로젝트 루트 ─────────────────────────────────
PROJECT_DIR = Path(__file__).parent

# 업로드할 파일/폴더 목록
UPLOAD_ITEMS = [
    "main_scheduler.py",
    "settings.py",
    "requirements.txt",
    ".env",
    "utils/",
    "scrapers/",
    "strategies/",
]


def get_server_info() -> tuple[str, str, str]:
    """서버 접속 정보를 입력받음."""
    host = SERVER_CFG.get("host", "")
    user = SERVER_CFG.get("user", "root")

    if not host:
        host = input("서버 IP 주소: ").strip()
    if not host:
        print("IP 주소가 필요합니다.")
        sys.exit(1)

    password = getpass.getpass(f"{user}@{host} 비밀번호: ")
    return host, user, password


def run_ssh(host: str, user: str, password: str, command: str, silent: bool = False) -> int:
    """sshpass를 사용하여 SSH 명령 실행."""
    ssh_cmd = [
        "sshpass", "-p", password,
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"{user}@{host}",
        command,
    ]
    if silent:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    else:
        result = subprocess.run(ssh_cmd)
    return result.returncode


def run_scp(host: str, user: str, password: str, local_path: str, remote_path: str) -> int:
    """sshpass를 사용하여 SCP 파일 전송."""
    is_dir = os.path.isdir(local_path)
    scp_cmd = [
        "sshpass", "-p", password,
        "scp",
        "-o", "StrictHostKeyChecking=no",
    ]
    if is_dir:
        scp_cmd.append("-r")
    scp_cmd += [local_path, f"{user}@{host}:{remote_path}"]
    result = subprocess.run(scp_cmd)
    return result.returncode


def check_sshpass():
    """sshpass가 설치되어 있는지 확인. 없으면 대체 방법 안내."""
    try:
        subprocess.run(["sshpass", "-V"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def deploy_with_sshpass(host: str, user: str, password: str):
    """sshpass 기반 배포."""
    remote_dir = SERVER_CFG["remote_dir"]
    tmux_session = SERVER_CFG["tmux_session"]

    # 1. 서버 패키지 설치
    print("\n[1/5] 서버 패키지 설치 중...")
    run_ssh(host, user, password,
            "dnf install -y python3 python3-pip tmux 2>/dev/null || "
            "apt-get update && apt-get install -y python3 python3-pip tmux 2>/dev/null || "
            "yum install -y python3 python3-pip tmux 2>/dev/null")

    # 2. 원격 디렉토리 생성
    print("\n[2/5] 원격 디렉토리 생성...")
    run_ssh(host, user, password,
            f"mkdir -p {remote_dir}/utils {remote_dir}/scrapers {remote_dir}/strategies")

    # 3. 파일 업로드
    print("\n[3/5] 프로젝트 파일 업로드 중...")
    for item in UPLOAD_ITEMS:
        local_path = str(PROJECT_DIR / item)
        if not os.path.exists(local_path):
            print(f"  ⚠ 건너뜀 (파일 없음): {item}")
            continue

        if os.path.isdir(local_path):
            remote_path = f"{remote_dir}/"
        else:
            remote_path = f"{remote_dir}/{item}"

        print(f"  📤 {item}")
        run_scp(host, user, password, local_path, remote_path)

    # 4. pip 라이브러리 설치
    print("\n[4/5] Python 라이브러리 설치 중...")
    run_ssh(host, user, password,
            f"cd {remote_dir} && pip3 install -r requirements.txt")

    # 5. tmux에서 실행
    print("\n[5/5] tmux 세션에서 봇 실행 중...")
    # 기존 세션이 있으면 종료
    run_ssh(host, user, password,
            f"tmux kill-session -t {tmux_session} 2>/dev/null; true")
    # 새 세션에서 실행
    run_ssh(host, user, password,
            f"tmux new-session -d -s {tmux_session} 'cd {remote_dir} && python3 main_scheduler.py'")

    print(f"""
╔══════════════════════════════════════════════════╗
║            🚀  배포 완료!                        ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  봇이 tmux 세션 '{tmux_session}'에서 실행 중      ║
║                                                  ║
║  상태 확인:  python run.py --status              ║
║  로그 확인:  python run.py --logs                ║
║  봇 중지:    python run.py --stop                ║
║                                                  ║
║  수동 접속:                                      ║
║    ssh {user}@{host}                       ║
║    tmux attach -t {tmux_session}                 ║
║                                                  ║
╚══════════════════════════════════════════════════╝
""")


def deploy_manual_instructions(host: str, user: str):
    """sshpass가 없을 때 수동 배포 스크립트 생성."""
    remote_dir = SERVER_CFG["remote_dir"]
    tmux_session = SERVER_CFG["tmux_session"]

    script_path = PROJECT_DIR / "deploy.sh"
    scp_lines = []
    for item in UPLOAD_ITEMS:
        local_path = str(PROJECT_DIR / item)
        if not os.path.exists(local_path):
            continue
        flag = "-r " if os.path.isdir(local_path) else ""
        if os.path.isdir(local_path):
            scp_lines.append(f"scp {flag}{item} {user}@{host}:{remote_dir}/")
        else:
            scp_lines.append(f"scp {flag}{item} {user}@{host}:{remote_dir}/{item}")

    script = f"""#!/bin/bash
# 자동 생성된 배포 스크립트 — run.py에 의해 생성됨
# 사용법: bash deploy.sh

echo "=== 1. 서버 접속 및 패키지 설치 ==="
ssh {user}@{host} "dnf install -y python3 python3-pip tmux 2>/dev/null || apt-get update && apt-get install -y python3 python3-pip tmux 2>/dev/null"

echo "=== 2. 원격 디렉토리 생성 ==="
ssh {user}@{host} "mkdir -p {remote_dir}/utils {remote_dir}/scrapers {remote_dir}/strategies"

echo "=== 3. 파일 업로드 ==="
{chr(10).join(scp_lines)}

echo "=== 4. 라이브러리 설치 ==="
ssh {user}@{host} "cd {remote_dir} && pip3 install -r requirements.txt"

echo "=== 5. 봇 실행 ==="
ssh {user}@{host} "tmux kill-session -t {tmux_session} 2>/dev/null; tmux new-session -d -s {tmux_session} 'cd {remote_dir} && python3 main_scheduler.py'"

echo ""
echo "배포 완료! tmux attach -t {tmux_session} 로 접속하세요."
"""
    with open(script_path, "w", newline="\n") as f:
        f.write(script)

    print(f"""
sshpass가 설치되어 있지 않습니다.

두 가지 방법 중 선택하세요:

[방법 1] sshpass 설치 후 다시 실행
  - Windows: choco install sshpass  또는  scoop install sshpass
  - Mac:     brew install hudochenkov/sshpass/sshpass
  - Linux:   sudo apt install sshpass

[방법 2] 생성된 배포 스크립트 사용
  배포 스크립트가 생성되었습니다: {script_path}
  Git Bash 또는 WSL에서 실행:
    cd {PROJECT_DIR}
    bash deploy.sh
  (비밀번호를 직접 입력해야 합니다)
""")


def stop_bot(host: str, user: str, password: str):
    """서버에서 실행 중인 봇 중지."""
    tmux_session = SERVER_CFG["tmux_session"]
    print(f"봇 중지 중 (세션: {tmux_session})...")
    run_ssh(host, user, password, f"tmux kill-session -t {tmux_session} 2>/dev/null")
    print("봇이 중지되었습니다.")


def check_status(host: str, user: str, password: str):
    """서버에서 봇 상태 확인."""
    tmux_session = SERVER_CFG["tmux_session"]
    print(f"봇 상태 확인 중...")
    returncode = run_ssh(host, user, password,
                         f"tmux has-session -t {tmux_session} 2>/dev/null && echo '✅ 봇 실행 중' || echo '❌ 봇 중지됨'")


def show_logs(host: str, user: str, password: str):
    """서버에서 최근 로그 출력."""
    remote_dir = SERVER_CFG["remote_dir"]
    print("최근 로그 (마지막 50줄):\n")
    run_ssh(host, user, password, f"tail -50 {remote_dir}/scheduler.log 2>/dev/null || echo '로그 파일 없음'")


def main():
    parser = argparse.ArgumentParser(description="퀀트 트레이딩 서버 배포 도구")
    parser.add_argument("--stop", action="store_true", help="실행 중인 봇 중지")
    parser.add_argument("--status", action="store_true", help="봇 상태 확인")
    parser.add_argument("--logs", action="store_true", help="최근 로그 확인")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║     QUANT TRADING SYSTEM — DEPLOY TOOL          ║")
    print("╚══════════════════════════════════════════════════╝")

    host, user, password = get_server_info()

    if args.stop:
        if check_sshpass():
            stop_bot(host, user, password)
        else:
            print("sshpass가 필요합니다. 위 안내를 참고하세요.")
        return

    if args.status:
        if check_sshpass():
            check_status(host, user, password)
        else:
            print("sshpass가 필요합니다.")
        return

    if args.logs:
        if check_sshpass():
            show_logs(host, user, password)
        else:
            print("sshpass가 필요합니다.")
        return

    # 기본: 배포
    if check_sshpass():
        deploy_with_sshpass(host, user, password)
    else:
        deploy_manual_instructions(host, user)


if __name__ == "__main__":
    main()
