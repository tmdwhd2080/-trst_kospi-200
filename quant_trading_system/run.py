"""
run.py 
-> python run.py 만 실행하면 로컬에서 봇 시작
-> python run.py --deploy 로 서버 배포 (SSH 키 인증 사용, sshpass 불필요)

사전 설정:
  1. ssh-keygen 으로 SSH 키 생성 (이미 있으면 건너뛰)
  2. ssh-copy-id root@<서버IP> 로 공개키 등록
     (Windows는: type $env:USERPROFILE\.ssh\id_rsa.pub | ssh root@<서버IP> "cat >> ~/.ssh/authorized_keys")
  3. settings.py SERVER["ssh_key"] 에 키 경로 설정 (선택, 기본 키 사용 시 비워둘 것)

사용법:
  python quant_trading_system/run.py              (로컬에서 봇 실행)
  python quant_trading_system/run.py --deploy     (서버에서 돌릴려면 얘 써야함)
  python quant_trading_system/run.py --stop       (실행 중인 봇 중지)
  python quant_trading_system/run.py --status     (봇 상태 확인)
  python quant_trading_system/run.py --logs       (최근 로그 확인)


!!!필수 -> 이거 해줘야 함
# 1. 키 생성 (이미 있으면 건너뛰기)
ssh-keygen -t rsa -b 4096

# 2. 서버에 공개키 등록 (Windows PowerShell)
type $env:USERPROFILE\.ssh\id_rsa.pub | ssh root@115.68.232.108 "mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys"
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).parent))
    import settings as cfg
    SERVER_CFG = cfg.SERVER
except ImportError:
    SERVER_CFG = {
        "host": "",
        "user": "root",
        "remote_dir": "/root/quant_trading_system",
        "tmux_session": "trade_bot",
    }

# 프로젝트 루트 
PROJECT_DIR = Path(__file__).parent

# 업로드할 파일 목록
UPLOAD_ITEMS = [
    "main_scheduler.py",
    "settings.py",
    "requirements.txt",
    ".env",
    "utils/",
    "scrapers/",
    "strategies/",
]


def get_server_info() -> tuple[str, str]:
    """서버 접속 정보를 반환. SSH 키 인증 사용 (비밀번호 불필요)."""
    host = SERVER_CFG.get("host", "")
    user = SERVER_CFG.get("user", "root")

    if not host:
        host = input("서버 IP 주소: ").strip()
    if not host:
        print("IP 주소가 필요합니다.")
        sys.exit(1)

    return host, user


def _ssh_key_args() -> list[str]:
    """SSH 키 파일 옵션을 반환. settings에 ssh_key가 설정되어 있으면 사용."""
    key_path = SERVER_CFG.get("ssh_key", "")
    if key_path and os.path.exists(key_path):
        return ["-i", key_path]
    return []


def run_ssh(host: str, user: str, command: str, silent: bool = False) -> int:
    """SSH 명령 실행 (키 인증)."""
    ssh_cmd = [
        "ssh",
        *_ssh_key_args(),
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


def run_scp(host: str, user: str, local_path: str, remote_path: str) -> int:
    """SCP 파일 전송 (키 인증)."""
    is_dir = os.path.isdir(local_path)
    scp_cmd = [
        "scp",
        *_ssh_key_args(),
        "-o", "StrictHostKeyChecking=no",
    ]
    if is_dir:
        scp_cmd.append("-r")
    scp_cmd += [local_path, f"{user}@{host}:{remote_path}"]
    result = subprocess.run(scp_cmd)
    return result.returncode


def deploy(host: str, user: str):
    """SSH 키 인증 기반 배포."""
    remote_dir = SERVER_CFG["remote_dir"]
    tmux_session = SERVER_CFG["tmux_session"]

    # 1. 서버 패키지 설치
    print("\n[1/5] 서버 패키지 설치 중...")
    run_ssh(host, user,
            "dnf install -y python3 python3-pip tmux 2>/dev/null || "
            "apt-get update && apt-get install -y python3 python3-pip tmux 2>/dev/null || "
            "yum install -y python3 python3-pip tmux 2>/dev/null")

    # 2. 원격 디렉토리 생성
    print("\n[2/5] 원격 디렉토리 생성...")
    run_ssh(host, user,
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
        run_scp(host, user, local_path, remote_path)

    # 4. pip 라이브러리 설치
    print("\n[4/5] Python 라이브러리 설치 중...")
    run_ssh(host, user,
            f"cd {remote_dir} && pip3 install -r requirements.txt")

    # 5. tmux에서 실행
    print("\n[5/5] tmux 세션에서 봇 실행 중...")
    # 기존 세션이 있으면 종료
    run_ssh(host, user,
            f"tmux kill-session -t {tmux_session} 2>/dev/null; true")
    # 새 세션에서 실행
    run_ssh(host, user,
            f"tmux new-session -d -s {tmux_session} 'cd {remote_dir} && python3 main_scheduler.py'")

    print(f"""
╔══════════════════════════════════════════════════╗
║              배포 완료                        ║
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


def stop_bot(host: str, user: str):
    """서버에서 실행 중인 봇 중지."""
    tmux_session = SERVER_CFG["tmux_session"]
    print(f"봇 중지 중 (세션: {tmux_session})...")
    run_ssh(host, user, f"tmux kill-session -t {tmux_session} 2>/dev/null")
    print("봇이 중지되었습니다.")


def check_status(host: str, user: str):
    """서버에서 봇 상태 확인."""
    tmux_session = SERVER_CFG["tmux_session"]
    print(f"봇 상태 확인 중...")
    returncode = run_ssh(host, user,
                         f"tmux has-session -t {tmux_session} 2>/dev/null && echo '✅ 봇 실행 중' || echo '❌ 봇 중지됨'")


def show_logs(host: str, user: str):
    """서버에서 최근 로그 출력."""
    remote_dir = SERVER_CFG["remote_dir"]
    print("최근 로그 (마지막 50줄):\n")
    run_ssh(host, user, f"tail -50 {remote_dir}/scheduler.log 2>/dev/null || echo '로그 파일 없음'")


def run_local():
    """로컬에서 메인 스케줄러 실행."""
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║     QUANT TRADING SYSTEM — LOCAL START           ║")
    print("╚══════════════════════════════════════════════════╝")
    os.chdir(PROJECT_DIR)
    from main_scheduler import main as scheduler_main
    scheduler_main()


def main():
    parser = argparse.ArgumentParser(description="퀀트 트레이딩 시스템 실행/배포 도구")
    parser.add_argument("--deploy", action="store_true", help="서버에 배포")
    parser.add_argument("--stop", action="store_true", help="서버 봇 중지")
    parser.add_argument("--status", action="store_true", help="서버 봇 상태 확인")
    parser.add_argument("--logs", action="store_true", help="서버 최근 로그 확인")
    args = parser.parse_args()

    # 서버 관련 명령
    if args.deploy or args.stop or args.status or args.logs:
        print()
        print("╔══════════════════════════════════════════════════╗")
        print("║     QUANT TRADING SYSTEM — DEPLOY TOOL          ║")
        print("╚══════════════════════════════════════════════════╝")

        host, user = get_server_info()

        if args.stop:
            stop_bot(host, user)
            return

        if args.status:
            check_status(host, user)
            return

        if args.logs:
            show_logs(host, user)
            return

        # --deploy
        deploy(host, user)
        return

    # 기본: 로컬 실행
    run_local()


if __name__ == "__main__":
    main()
# python quant_trading_system/run.py