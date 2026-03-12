"""
새 스크래퍼 자동 통합 스크립트
────────────────────────────────
사용법:
  python scrapers/integrate_scraper.py <새_스크래퍼_파일_경로>

예시:
  python scrapers/integrate_scraper.py scrapers/naver_news_scraper.py

이 스크립트는 새로 작성한 스크래퍼 파일의 TASK_META를 읽어
아래 파일들을 자동으로 수정합니다:
  1. main_scheduler.py → TASK_REGISTRY에 작업 등록
  2. settings.py → ENABLE_TASKS에 활성화 항목 추가
  3. settings.py → DEFAULT_SCHEDULE에 스케줄 추가

전제 조건:
  - 새 스크래퍼 파일에 TASK_META 딕셔너리가 정의되어 있어야 함
  - TASK_META 양식은 TEMPLATE_NEW_SCRAPER.py 참고
  - main_scheduler.py, settings.py에 통합 마커 주석이 존재해야 함
"""

import sys
import os
import re
import ast

# 프로젝트 루트 경로 (이 파일의 부모의 부모)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCHEDULER_PATH = os.path.join(PROJECT_ROOT, "main_scheduler.py")
SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.py")

MARKER_TASK_REGISTRY = "# __INTEGRATE_MARKER_TASK_REGISTRY__"
MARKER_ENABLE_TASKS = "# __INTEGRATE_MARKER_ENABLE_TASKS__"
MARKER_DEFAULT_SCHEDULE = "# __INTEGRATE_MARKER_DEFAULT_SCHEDULE__"


def extract_task_meta(filepath: str) -> dict:
    """파일에서 TASK_META 딕셔너리를 AST로 안전하게 추출."""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TASK_META":
                    return ast.literal_eval(node.value)
    raise ValueError(f"TASK_META를 찾을 수 없습니다: {filepath}")


def compute_module_path(filepath: str) -> str:
    """파일 경로에서 Python 모듈 경로를 계산 (프로젝트 루트 기준)."""
    rel = os.path.relpath(filepath, PROJECT_ROOT)
    module = rel.replace(os.sep, ".").replace("/", ".")
    if module.endswith(".py"):
        module = module[:-3]
    return module


def validate_meta(meta: dict):
    """TASK_META 유효성 검사."""
    required = ["task_id", "description", "type", "schedule"]
    for key in required:
        if key not in meta:
            raise ValueError(f"TASK_META에 필수 필드 '{key}'가 없습니다")

    if meta["type"] != "scraper":
        raise ValueError(
            f"TASK_META의 type이 'scraper'이어야 합니다 (현재: {meta['type']}). "
            f"전략이라면 strategies/integrate_strategy.py를 사용하세요."
        )

    if not re.match(r'^[a-z][a-z0-9_]*$', meta["task_id"]):
        raise ValueError(f"task_id는 영문 소문자와 언더스코어만 허용: {meta['task_id']}")

    for i, sched in enumerate(meta["schedule"]):
        if "time" not in sched:
            raise ValueError(f"schedule[{i}]에 'time' 필드가 필요합니다")
        if "force_kill_time" not in sched:
            raise ValueError(f"schedule[{i}]에 'force_kill_time' 필드가 필요합니다")
        for key in ["time", "force_kill_time"]:
            if not re.match(r'^\d{2}:\d{2}$', sched[key]):
                raise ValueError(f"schedule[{i}]['{key}'] 형식이 HH:MM이어야 합니다: {sched[key]}")


def insert_before_marker(content: str, marker: str, new_text: str) -> str:
    """마커 주석 바로 앞에 새 텍스트를 삽입 (마커의 들여쓰기 보존)."""
    # 마커가 파일 내에서 들여쓰기와 함께 있을 수 있으므로 패턴으로 찾기
    import re as _re
    pattern = _re.compile(r'^([ \t]*)' + _re.escape(marker), _re.MULTILINE)
    match = pattern.search(content)
    if not match:
        raise ValueError(
            f"마커를 찾을 수 없습니다: {marker}\n"
            f"main_scheduler.py 또는 settings.py에 해당 마커 주석이 있는지 확인하세요."
        )
    indent = match.group(1)
    full_marker = indent + marker
    return content.replace(full_marker, new_text + "\n" + full_marker, 1)


def check_duplicate(content: str, task_id: str, context: str):
    """이미 등록된 task_id인지 확인."""
    if f'"{task_id}"' in content:
        raise ValueError(f"'{task_id}'가 이미 {context}에 등록되어 있습니다")


def integrate(filepath: str):
    """새 스크래퍼를 전체 시스템에 통합."""
    filepath = os.path.abspath(filepath)

    if not os.path.exists(filepath):
        print(f"❌ 파일을 찾을 수 없습니다: {filepath}")
        return False

    # 0. 필수 파일 존재 확인
    for path, name in [(SCHEDULER_PATH, "main_scheduler.py"), (SETTINGS_PATH, "settings.py")]:
        if not os.path.exists(path):
            print(f"❌ {name}를 찾을 수 없습니다: {path}")
            return False

    # 1. TASK_META 추출 및 검증
    print(f"\n📂 파일 읽기: {filepath}")
    try:
        meta = extract_task_meta(filepath)
        validate_meta(meta)
    except (ValueError, SyntaxError) as e:
        print(f"❌ TASK_META 검증 실패: {e}")
        return False

    task_id = meta["task_id"]
    description = meta["description"]
    module_path = compute_module_path(filepath)
    enabled = meta.get("enabled", True)
    schedules = meta["schedule"]

    print(f"  task_id:     {task_id}")
    print(f"  module:      {module_path}")
    print(f"  description: {description}")
    print(f"  schedules:   {len(schedules)}개")
    print(f"  enabled:     {enabled}")

    # 2. main_scheduler.py 수정 — TASK_REGISTRY에 추가
    print(f"\n📝 main_scheduler.py 수정 중...")
    with open(SCHEDULER_PATH, "r", encoding="utf-8") as f:
        scheduler_content = f.read()

    try:
        check_duplicate(scheduler_content, task_id, "TASK_REGISTRY")
    except ValueError as e:
        print(f"❌ {e}")
        return False

    registry_entry = (
        f'    "{task_id}": {{\n'
        f'        "module": "{module_path}",\n'
        f'        "description": "{description}",\n'
        f'        "type": "scraper",\n'
        f'    }},'
    )
    try:
        scheduler_content = insert_before_marker(
            scheduler_content, MARKER_TASK_REGISTRY, registry_entry
        )
    except ValueError as e:
        print(f"❌ {e}")
        return False

    with open(SCHEDULER_PATH, "w", encoding="utf-8") as f:
        f.write(scheduler_content)
    print(f"  ✔ TASK_REGISTRY에 '{task_id}' 추가 완료")

    # 3. settings.py 수정 — ENABLE_TASKS + DEFAULT_SCHEDULE
    print(f"\n📝 settings.py 수정 중...")
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        settings_content = f.read()

    try:
        check_duplicate(settings_content, task_id, "settings.py")
    except ValueError as e:
        print(f"❌ {e}")
        return False

    # ENABLE_TASKS 추가
    enabled_str = "True" if enabled else "False"
    padding = max(1, 18 - len(task_id))
    enable_entry = f'    "{task_id}":{" " * padding}{enabled_str},   # {description}'
    try:
        settings_content = insert_before_marker(
            settings_content, MARKER_ENABLE_TASKS, enable_entry
        )
    except ValueError as e:
        print(f"❌ {e}")
        return False

    # DEFAULT_SCHEDULE 추가
    for sched in schedules:
        sched_time = sched["time"]
        sched_desc = sched.get("description", description)
        force_kill = sched.get("force_kill_time", "")

        if force_kill:
            schedule_entry = (
                f'    {{"time": "{sched_time}", "task": "{task_id}", '
                f'"description": "{sched_desc}", "force_kill_time": "{force_kill}"}},'
            )
        else:
            schedule_entry = (
                f'    {{"time": "{sched_time}", "task": "{task_id}", '
                f'"description": "{sched_desc}"}},'
            )
        try:
            settings_content = insert_before_marker(
                settings_content, MARKER_DEFAULT_SCHEDULE, schedule_entry
            )
        except ValueError as e:
            print(f"❌ {e}")
            return False

    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        f.write(settings_content)
    print(f"  ✔ ENABLE_TASKS에 '{task_id}' 추가 완료")
    print(f"  ✔ DEFAULT_SCHEDULE에 {len(schedules)}개 스케줄 추가 완료")

    # 완료 요약
    print(f"\n{'=' * 50}")
    print(f"✅ 통합 완료! '{task_id}' ({description})")
    print(f"  → main_scheduler.py  TASK_REGISTRY 등록")
    print(f"  → settings.py        ENABLE_TASKS 등록")
    print(f"  → settings.py        DEFAULT_SCHEDULE 등록")
    print(f"\n💡 비활성화: settings.py에서 ENABLE_TASKS[\"{task_id}\"] = False")
    print(f"💡 되돌리기: 위 3개 파일에서 '{task_id}' 관련 줄을 수동 삭제")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python scrapers/integrate_scraper.py <새_스크래퍼_파일_경로>")
        print("예시:   python scrapers/integrate_scraper.py scrapers/naver_news_scraper.py")
        sys.exit(1)

    success = integrate(sys.argv[1])
    sys.exit(0 if success else 1)
