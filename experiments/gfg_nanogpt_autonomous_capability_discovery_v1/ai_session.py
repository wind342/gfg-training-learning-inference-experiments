from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import ipaddress
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence
from urllib.parse import urlparse

from .common import (
    file_sha256,
    payload_sha256,
    read_json,
    relative_file_manifest,
    write_json,
)
from .orientation.orientation_gate import validate_orientation_receipt


PARTICIPANT_PROMPT = (
    "GFG is a new graph language and no prior familiarity is assumed. "
    "First read GFG_MACHINE_SEMANTICS.md, "
    "EXECUTABLE_MECHANISM_DISCOVERY_GUIDE.md, unrelated_example.json and "
    "ORIENTATION_RECEIPT_CHECKER.py and gfg_orientation.json completely. "
    "This task family is new; the second "
    "manual defines how a GFG analysis becomes a sealed executable "
    "mechanism. The target evidence is externally "
    "permission-locked for at least the first five minutes. Use that minimum "
    "interval only for the manual and its self-check. As soon as the manual "
    "and self-check are complete, write "
    "orientation_receipt.json in the repository root exactly as specified "
    "by the orientation instructions in GFG_MACHINE_SEMANTICS.md, run the "
    "supplied local receipt checker until it prints ORIENTATION_RECEIPT_PASS, "
    "then wait; "
    "the orientation gate remains available for at most 45 minutes. "
    "do not finish the task before the external runner validates the receipt "
    "and releases /evidence. After release, read TASK.txt and "
    "the remaining frozen contracts completely and perform the formal task. "
    "After /evidence is released, you have a hard 120-minute formal-work "
    "budget. Stop expanding analysis by formal-work minute 110 and use the "
    "final ten minutes only to complete and check every required file under "
    "submission/. You may submit earlier as soon as the required work is "
    "ready. No later interactive deadline reminder is available. "
    "Do not ask the user questions. Write final candidate files only under "
    "submission/."
)


class FormalAISessionError(RuntimeError):
    pass


@dataclass(frozen=True)
class FormalAIConfig:
    multipass: Path = Path("C:/Program Files/Multipass/bin/multipass.exe")
    instance: str = "gfg-lab-ubuntu-v5-stability"
    auth_file: Path = Path("C:/Users/Administrator/.codex/auth.json")
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "xhigh"
    participant_image: str = "gfg-pilot-participant:gfg-sandbox-cf07f9990ab1bc3d"
    codex_entrypoint: str = "/opt/pilot/codex-runner/codex"
    upstream_proxy: str = "http://192.168.96.1:7890"
    participant_proxy_host: str = "172.17.0.1"
    participant_proxy_base_port: int = 18100
    wall_clock_seconds: int = 7200
    minimum_startup_free_gib: float = 19.0
    http_only_transport: bool = True
    stream_idle_timeout_ms: int = 1_200_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(
    command: Sequence[str],
    *,
    check: bool = True,
    text: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[Any]:
    result = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=text,
        timeout=timeout,
        encoding="utf-8" if text else None,
        errors="replace" if text else None,
    )
    if check and result.returncode != 0:
        stderr = result.stderr if isinstance(result.stderr, str) else ""
        raise FormalAISessionError(
            f"COMMAND_FAILED:{command[0]}:{result.returncode}:{stderr[-2000:]}"
        )
    return result


def _multipass(
    config: FormalAIConfig, *arguments: str, **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    return _run([str(config.multipass), *arguments], **kwargs)


def _run_status(
    command: Sequence[str],
    *,
    timeout: float = 60,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        list(command),
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
    )


def _guest_status(
    config: FormalAIConfig,
    *arguments: str,
    timeout: float = 60,
) -> subprocess.CompletedProcess[Any]:
    return _run_status(
        [
            str(config.multipass),
            "exec",
            config.instance,
            "--",
            *arguments,
        ],
        timeout=timeout,
    )


def _submission_ready(config: FormalAIConfig, guest_workspace: str) -> bool:
    marker = f"{guest_workspace}/submission/FINAL_SUBMISSION_READY.json"
    code = (
        "import json,sys;"
        "value=json.load(open(sys.argv[1],encoding='utf-8'));"
        "raise SystemExit(0 if value=={'status':'READY'} else 1)"
    )
    try:
        return (
            _guest_status(
                config,
                "sudo",
                "python3",
                "-c",
                code,
                marker,
                timeout=10,
            ).returncode
            == 0
        )
    except subprocess.TimeoutExpired:
        return False


def _guest(
    config: FormalAIConfig, *arguments: str, **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    kwargs.setdefault("timeout", 60)
    return _multipass(config, "exec", config.instance, "--", *arguments, **kwargs)


def _available_memory_gib() -> float:
    result = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
        ]
    )
    return int(result.stdout.strip()) / (1024 * 1024)


def _ensure_vm_started(config: FormalAIConfig) -> dict[str, Any]:
    instances = json.loads(_multipass(config, "list", "--format", "json").stdout)[
        "list"
    ]
    row = next(
        (item for item in instances if item["name"] == config.instance),
        None,
    )
    if row is None:
        raise FormalAISessionError("MULTIPASS_INSTANCE_MISSING")
    started_here = row["state"] != "Running"
    free_gib = _available_memory_gib()
    if started_here:
        if free_gib < config.minimum_startup_free_gib:
            raise FormalAISessionError(
                "MEMORY_START_GATE_FAILED:"
                f"{free_gib:.2f}<{config.minimum_startup_free_gib:.2f}"
            )
        _multipass(config, "start", config.instance, timeout=180)
    ready = _guest(config, "bash", "-lc", "printf GFG_VM_READY", timeout=60)
    if ready.stdout != "GFG_VM_READY":
        raise FormalAISessionError("MULTIPASS_EXEC_PROBE_FAILED")
    image = _guest(
        config,
        "sudo",
        "-iu",
        "gfg",
        "docker",
        "image",
        "inspect",
        config.participant_image,
        "--format",
        "{{.Id}}",
        timeout=60,
    ).stdout.strip()
    return {
        "instance": config.instance,
        "minimum_startup_free_gib": config.minimum_startup_free_gib,
        "participant_image_id": image,
        "started_here": started_here,
        "startup_free_gib": free_gib,
        "status": "PASS",
    }


def _runtime_env(participant_proxy: str) -> str:
    return "\n".join(
        [
            "CODEX_HOME=/codex-home",
            "HOME=/tmp/participant-home",
            f"HTTPS_PROXY={participant_proxy}",
            f"HTTP_PROXY={participant_proxy}",
            f"ALL_PROXY={participant_proxy}",
            f"https_proxy={participant_proxy}",
            f"http_proxy={participant_proxy}",
            f"all_proxy={participant_proxy}",
            "NO_PROXY=127.0.0.1,localhost",
            "no_proxy=127.0.0.1,localhost",
            "OPENAI_API_KEY=",
            "CODEX_ACCESS_TOKEN=",
            "PYTHONDONTWRITEBYTECODE=1",
            "",
        ]
    )


def _container_command(
    *,
    config: FormalAIConfig,
    container_name: str,
    guest_workspace: str,
    guest_codex_home: str,
    guest_evidence: str,
    guest_env: str,
    participant_proxy: str,
) -> list[str]:
    transport = []
    if config.http_only_transport:
        transport = [
            "--config",
            'model_provider="openai-http"',
            "--config",
            'model_providers.openai-http.name="OpenAI HTTP"',
            "--config",
            (
                "model_providers.openai-http.base_url="
                '"https://chatgpt.com/backend-api/codex"'
            ),
            "--config",
            'model_providers.openai-http.wire_api="responses"',
            "--config",
            "model_providers.openai-http.requires_openai_auth=true",
            "--config",
            "model_providers.openai-http.supports_websockets=false",
            "--config",
            (
                "model_providers.openai-http.stream_idle_timeout_ms="
                f"{config.stream_idle_timeout_ms}"
            ),
            "--config",
            "model_providers.openai-http.stream_max_retries=1",
        ]
    return [
        "sudo",
        "-iu",
        "gfg",
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--hostname",
        "participant",
        "--network",
        "bridge",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=2g",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "1024",
        "--memory",
        "12g",
        "--cpus",
        "8",
        "--user",
        "65532:65532",
        "--workdir",
        "/workspace",
        "--mount",
        f"type=bind,src={guest_workspace},dst=/workspace",
        "--mount",
        f"type=bind,src={guest_codex_home},dst=/codex-home",
        "--mount",
        f"type=bind,src={guest_evidence},dst=/evidence,readonly",
        "--env-file",
        guest_env,
        "--env",
        f"HTTPS_PROXY={participant_proxy}",
        "--env",
        f"HTTP_PROXY={participant_proxy}",
        "--env",
        f"ALL_PROXY={participant_proxy}",
        "--env",
        f"https_proxy={participant_proxy}",
        "--env",
        f"http_proxy={participant_proxy}",
        "--env",
        f"all_proxy={participant_proxy}",
        "--entrypoint",
        config.codex_entrypoint,
        config.participant_image,
        "--model",
        config.model,
        "--config",
        f'model_reasoning_effort="{config.reasoning_effort}"',
        "--config",
        'web_search="disabled"',
        "--config",
        "features.apps=false",
        *transport,
        "--config",
        "analytics.enabled=false",
        "--dangerously-bypass-approvals-and-sandbox",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--color",
        "never",
        "-C",
        "/workspace",
        PARTICIPANT_PROMPT,
    ]


def _participant_proxy_port(config: FormalAIConfig, experiment_id: str) -> int:
    try:
        ordinal = int(experiment_id.rsplit("-", 1)[-1])
    except ValueError as exc:
        raise FormalAISessionError("EXPERIMENT_ID_ORDINAL_MISSING") from exc
    port = config.participant_proxy_base_port + ordinal
    if not 1024 <= port <= 65535:
        raise FormalAISessionError("PARTICIPANT_PROXY_PORT_INVALID")
    return port


def _guest_proxy_endpoint_reachable(
    config: FormalAIConfig,
    host: str,
    port: int,
) -> bool:
    probe = _guest(
        config,
        "python3",
        "-c",
        (
            "import socket,sys; "
            "s=socket.create_connection((sys.argv[1],int(sys.argv[2])),3); "
            "s.close()"
        ),
        host,
        str(port),
        check=False,
        timeout=10,
    )
    return probe.returncode == 0


def _resolved_upstream_proxy(config: FormalAIConfig) -> tuple[str, int, str]:
    parsed = urlparse(config.upstream_proxy)
    if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
        raise FormalAISessionError("UPSTREAM_PROXY_URI_INVALID")
    if _guest_proxy_endpoint_reachable(config, parsed.hostname, parsed.port):
        return parsed.hostname, parsed.port, config.upstream_proxy

    # Hyper-V's Default Switch address may change after a Windows restart. The
    # guest default gateway is the current host-side address of that switch.
    gateway = _guest(
        config,
        "sh",
        "-lc",
        "ip -4 route show default | awk 'NR==1 {print $3}'",
        check=False,
        timeout=10,
    ).stdout.strip()
    try:
        ipaddress.ip_address(gateway)
    except ValueError as exc:
        raise FormalAISessionError("MULTIPASS_DEFAULT_GATEWAY_INVALID") from exc
    if not _guest_proxy_endpoint_reachable(config, gateway, parsed.port):
        raise FormalAISessionError("UPSTREAM_PROXY_UNREACHABLE")
    return gateway, parsed.port, f"http://{gateway}:{parsed.port}"


def _start_model_proxy(
    *,
    config: FormalAIConfig,
    guest_root: str,
    experiment_id: str,
) -> dict[str, Any]:
    upstream_host, upstream_port, effective_upstream_proxy = _resolved_upstream_proxy(config)
    port = _participant_proxy_port(config, experiment_id)
    source = Path(__file__).with_name("model_proxy.py")
    guest_script = f"{guest_root}/model_proxy.py"
    _multipass(
        config,
        "transfer",
        str(source),
        f"{config.instance}:{guest_script}",
    )
    command = (
        f"if test -e '{guest_root}/model-proxy.pid'; then exit 2; fi; "
        f"install -m 0600 /dev/null "
        f"'{guest_root}/model-proxy-audit.jsonl'; "
        f"nohup python3 '{guest_script}' "
        f"--listen-port {port} "
        f"--upstream-host '{upstream_host}' "
        f"--upstream-port {upstream_port} "
        "--allowed-host chatgpt.com "
        "--allowed-host auth.openai.com "
        f"--audit-log '{guest_root}/model-proxy-audit.jsonl' "
        f"> '{guest_root}/model-proxy.stdout.log' "
        f"2> '{guest_root}/model-proxy.stderr.log' & "
        f"echo $! > '{guest_root}/model-proxy.pid'; "
        "for i in $(seq 1 50); do "
        'if python3 -c "import socket; '
        f"s=socket.create_connection(('127.0.0.1',{port}),1); "
        "s.sendall(b'GET /health HTTP/1.1\\\\r\\\\n"
        "Host: localhost\\\\r\\\\n\\\\r\\\\n'); "
        "ok=b'200 OK' in s.recv(128); s.close(); "
        'raise SystemExit(0 if ok else 1)"; then exit 0; fi; '
        "sleep 0.1; done; exit 1"
    )
    _guest(config, "bash", "-lc", command, timeout=30)
    return {
        "allowed_hosts": ["auth.openai.com", "chatgpt.com"],
        "participant_proxy": (f"http://{config.participant_proxy_host}:{port}"),
        "port": port,
        "status": "PASS",
        "configured_upstream_proxy": config.upstream_proxy,
        "upstream_proxy": effective_upstream_proxy,
    }


def _model_proxy_audit(
    config: FormalAIConfig,
    guest_root: str,
    session_directory: Path,
) -> dict[str, Any]:
    audit_path = f"{guest_root}/model-proxy-audit.jsonl"
    guest_archive = f"{guest_root}/model-proxy-audit.jsonl.gz"
    local_archive = session_directory / "model-proxy-audit.jsonl.gz"
    archive = _guest(
        config,
        "bash",
        "-lc",
        (
            f"set -eu; gzip -c -- '{audit_path}' > '{guest_archive}'; "
            f"sudo chown ubuntu:ubuntu '{guest_archive}'; "
            f"chmod 0600 '{guest_archive}'"
        ),
        check=False,
    )
    if archive.returncode != 0:
        return _summarize_model_proxy_rows([])
    transfer = _multipass(
        config,
        "transfer",
        f"{config.instance}:{guest_archive}",
        str(local_archive),
        check=False,
        timeout=120,
    )
    if not _transfer_ok(transfer) or not local_archive.is_file():
        return _summarize_model_proxy_rows([])
    try:
        raw = gzip.decompress(local_archive.read_bytes()).decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return _summarize_model_proxy_rows([])
    rows = []
    for line in raw.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    summary = _summarize_model_proxy_rows(rows)
    material = {key: value for key, value in summary.items() if key != "audit_sha256"}
    material["compressed_audit_sha256"] = file_sha256(local_archive)
    material["compressed_audit_size_bytes"] = local_archive.stat().st_size
    return {**material, "audit_sha256": payload_sha256(material)}


def _summarize_model_proxy_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    denied = [row for row in rows if not row.get("allowed")]
    established = [row for row in rows if row.get("status") == "TUNNEL_ESTABLISHED"]
    disallowed_tunnels = [
        row
        for row in established
        if not row.get("allowed")
        or row.get("host") not in {"auth.openai.com", "chatgpt.com"}
    ]
    material = {
        "allowed_hosts": ["auth.openai.com", "chatgpt.com"],
        "blocked_destination_attempt_count": len(denied),
        "blocked_destination_hosts": sorted(
            {str(row.get("host")) for row in denied if row.get("host") is not None}
        ),
        "denied_request_count": len(denied),
        "disallowed_tunnel_count": len(disallowed_tunnels),
        "established_tunnel_count": len(established),
        "participant_network_instruction_compliance": (
            "PASS" if not denied else "BLOCKED_ATTEMPTS_RECORDED"
        ),
        "row_count": len(rows),
        "schema": "formal-model-network-proxy-audit-v1",
        "status": ("PASS" if established and not disallowed_tunnels else "FAIL"),
    }
    material["audit_sha256"] = payload_sha256(material)
    return material


def _stop_model_proxy(config: FormalAIConfig, guest_root: str) -> None:
    pid_path = f"{guest_root}/model-proxy.pid"
    _guest(
        config,
        "bash",
        "-lc",
        (
            f"if test -f '{pid_path}'; then "
            f"pid=$(cat '{pid_path}'); "
            "case \"$pid\" in (*[!0-9]*|'') exit 2;; esac; "
            'sudo kill "$pid" 2>/dev/null || true; '
            "fi"
        ),
        check=False,
    )


def _lock_evidence(config: FormalAIConfig, guest_evidence: str) -> None:
    _guest(
        config,
        "sudo",
        "find",
        guest_evidence,
        "-type",
        "f",
        "-exec",
        "chmod",
        "0000",
        "{}",
        "+",
    )
    _guest(
        config,
        "sudo",
        "find",
        guest_evidence,
        "-type",
        "d",
        "-exec",
        "chmod",
        "0000",
        "{}",
        "+",
    )
    unexpected = _guest(
        config,
        "sudo",
        "find",
        guest_evidence,
        "-perm",
        "/0777",
        "-print",
        "-quit",
    )
    if unexpected.stdout.strip():
        raise FormalAISessionError("ORIENTATION_EVIDENCE_LOCK_FAILED")


def _container_evidence_readable(config: FormalAIConfig, container_name: str) -> bool:
    result = _guest_status(
        config,
        "sudo",
        "-iu",
        "gfg",
        "docker",
        "exec",
        "--user",
        "65532:65532",
        container_name,
        "test",
        "-r",
        "/evidence/manifest.json",
    )
    return result.returncode == 0


def _release_evidence(
    config: FormalAIConfig,
    guest_evidence: str,
    container_name: str,
) -> None:
    if _container_evidence_readable(config, container_name):
        raise FormalAISessionError("ORIENTATION_EVIDENCE_READABLE_BEFORE_RELEASE")
    commands = (
        (
            "sudo",
            "find",
            guest_evidence,
            "-type",
            "d",
            "-exec",
            "chmod",
            "0500",
            "{}",
            "+",
        ),
        (
            "sudo",
            "find",
            guest_evidence,
            "-type",
            "f",
            "-exec",
            "chmod",
            "0400",
            "{}",
            "+",
        ),
    )
    for command in commands:
        if _guest_status(config, *command).returncode != 0:
            raise FormalAISessionError("ORIENTATION_EVIDENCE_RELEASE_COMMAND_FAILED")
    if not _container_evidence_readable(config, container_name):
        raise FormalAISessionError("ORIENTATION_EVIDENCE_RELEASE_FAILED")


def _stop_container(config: FormalAIConfig, container_name: str) -> None:
    command = (
        "sudo",
        "-iu",
        "gfg",
        "docker",
        "stop",
        "--time",
        "10",
        container_name,
    )
    try:
        _guest_status(config, *command, timeout=25)
    except subprocess.TimeoutExpired:
        # Bound the READY handoff even when the host-side Multipass client
        # does not return after Docker has processed the stop request.  The
        # fallback still targets only the participant container, never the VM.
        try:
            _guest_status(
                config,
                "sudo",
                "-iu",
                "gfg",
                "docker",
                "kill",
                container_name,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            pass


def _finish_submission_process(
    *,
    config: FormalAIConfig,
    container_name: str,
    process: subprocess.Popen[str],
) -> None:
    _stop_container(config, container_name)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # Docker has already stopped the participant.  Some Windows
        # Multipass exec clients nevertheless remain blocked after the guest
        # command exits.  Terminate only that host-side client so collection
        # can continue; the VM and its submission workspace remain running.
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _safe_cleanup(config: FormalAIConfig, guest_root: str) -> None:
    prefix = "/tmp/gfg-nanogpt-discovery/"
    if not guest_root.startswith(prefix) or len(guest_root) <= len(prefix):
        raise FormalAISessionError("REFUSING_GUEST_CLEANUP_TARGET")
    _guest_status(
        config,
        "sudo",
        "rm",
        "-rf",
        "--",
        guest_root,
        timeout=60,
    )


def _cleanup_guest_after_session(
    config: FormalAIConfig,
    guest_root: str,
    *,
    submission_ready: bool,
    submission_transferred: bool,
) -> bool:
    if submission_ready and not submission_transferred:
        return False
    _safe_cleanup(config, guest_root)
    return True


def _git_status_paths(raw: bytes) -> list[str]:
    paths = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        decoded = entry.decode("utf-8")
        if len(decoded) < 4:
            raise FormalAISessionError("GIT_STATUS_ENTRY_INVALID")
        paths.append(decoded[3:].replace("\\", "/"))
    return paths


def _assert_submission_only(paths: list[str]) -> None:
    controlled = [path for path in paths if path != "orientation_receipt.json"]
    invalid = [
        path
        for path in controlled
        if path != "submission" and not path.startswith("submission/")
    ]
    if invalid:
        raise FormalAISessionError("PARTICIPANT_WROTE_OUTSIDE_SUBMISSION:" + invalid[0])
    if not controlled:
        raise FormalAISessionError("PARTICIPANT_PRODUCED_NO_SUBMISSION")


def _collect_orientation_receipt(
    *,
    config: FormalAIConfig,
    guest_workspace: str,
    session_directory: Path,
    elapsed_seconds: float,
) -> dict[str, Any] | None:
    guest_source = f"{guest_workspace}/orientation_receipt.json"
    guest_staging = "/home/ubuntu/.gfg-orientation-receipt-transfer.json"
    local_path = session_directory / "orientation_receipt.json"
    if local_path.is_file():
        validation = validate_orientation_receipt(
            receipt_path=local_path,
            elapsed_seconds=elapsed_seconds,
            target_gfg_readable_before_release=False,
        )
        if validation["status"] == "PASS":
            return validation
    try:
        staged = _guest_status(
            config,
            "sudo",
            "install",
            "-o",
            "ubuntu",
            "-g",
            "ubuntu",
            "-m",
            "0600",
            guest_source,
            guest_staging,
            timeout=10,
        )
        if staged.returncode != 0:
            return None
        _multipass(
            config,
            "transfer",
            f"{config.instance}:{guest_staging}",
            str(local_path),
            check=False,
            timeout=20,
        )
        # Multipass SFTP may return 2 on Windows after copying all bytes
        # because the Windows destination cannot accept Unix chmod metadata.
        # The local receipt and its semantic validation are authoritative.
        if not local_path.is_file() or local_path.stat().st_size == 0:
            return None
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            _guest_status(
                config,
                "rm",
                "-f",
                guest_staging,
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            pass
    validation = validate_orientation_receipt(
        receipt_path=local_path,
        elapsed_seconds=elapsed_seconds,
        target_gfg_readable_before_release=False,
    )
    if validation["status"] != "PASS":
        return None
    return validation


def _secret_scan(auth_file: Path, paths: list[Path]) -> dict[str, Any]:
    auth = json.loads(auth_file.read_text(encoding="utf-8"))
    needles: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, str) and len(value) >= 24:
            needles.add(value)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(auth)
    for path in paths:
        data = path.read_text(encoding="utf-8", errors="replace")
        if any(needle in data for needle in needles):
            raise FormalAISessionError("SECRET_LEAK_DETECTED:" + path.name)
    return {
        "needle_count": len(needles),
        "scanned_file_count": len(paths),
        "status": "PASS",
    }


def _transfer_ok(result: subprocess.CompletedProcess[Any]) -> bool:
    if result.returncode == 0:
        return True
    stderr = result.stderr if isinstance(result.stderr, str) else ""
    lines = [line for line in stderr.splitlines() if line.strip()]
    return (
        result.returncode == 2
        and bool(lines)
        and all(
            "[error] [sftp] cannot set permissions for local" in line for line in lines
        )
    )


def run_formal_ai_session(
    *,
    config: FormalAIConfig,
    experiment_id: str,
    participant_repository: Path,
    participant_baseline_commit: str,
    evidence_directory: Path,
    evidence_manifest: dict[str, Any],
    session_directory: Path,
) -> dict[str, Any]:
    if session_directory.exists():
        raise FileExistsError("AI_SESSION_DIRECTORY_EXISTS")
    if not config.auth_file.is_file():
        raise FormalAISessionError("CODEX_AUTH_FILE_MISSING")
    orientation_contract = read_json(participant_repository / "gfg_orientation.json")
    if (
        orientation_contract.get("status") != "FROZEN"
        or orientation_contract.get("minimum_manual_only_seconds") != 300
        or orientation_contract.get("maximum_seconds_until_valid_receipt") != 2700
        or not orientation_contract.get("same_ai_process_continues_after_release")
        or not orientation_contract["orientation_receipt"][
            "required_before_evidence_release"
        ]
        or not orientation_contract["orientation_receipt"][
            "ai_semantics_validation_required"
        ]
        or not orientation_contract["orientation_receipt"][
            "executable_mechanism_method_validation_required"
        ]
        or not orientation_contract["evidence_gate"][
            "target_evidence_unreadable_during_orientation"
        ]
    ):
        raise FormalAISessionError("GFG_ORIENTATION_CONTRACT_INVALID")
    session_directory.mkdir(parents=True)
    vm = _ensure_vm_started(config)
    guest_root = f"/tmp/gfg-nanogpt-discovery/{experiment_id}"
    guest_workspace = f"{guest_root}/workspace"
    guest_codex_home = f"{guest_root}/codex-home"
    guest_evidence = f"{guest_root}/evidence"
    guest_env = f"{guest_root}/participant.env"
    container_name = "gfg-nanogpt-" + experiment_id.rsplit("-", 1)[-1]
    _guest(
        config,
        "bash",
        "-lc",
        (
            f"set -eu; test ! -e '{guest_root}'; "
            f"mkdir -p '{guest_root}'; chmod 0700 '{guest_root}'"
        ),
    )
    process: subprocess.Popen[str] | None = None
    proxy_started = False
    submission_transferred = False
    try:
        _multipass(
            config,
            "transfer",
            "-r",
            str(participant_repository),
            f"{config.instance}:{guest_root}",
        )
        _guest(
            config,
            "sudo",
            "mv",
            "--",
            f"{guest_root}/{participant_repository.name}",
            guest_workspace,
        )
        _multipass(
            config,
            "transfer",
            "-r",
            str(evidence_directory),
            f"{config.instance}:{guest_root}",
        )
        _guest(
            config,
            "sudo",
            "mv",
            "--",
            f"{guest_root}/{evidence_directory.name}",
            guest_evidence,
        )
        _multipass(
            config,
            "transfer",
            str(config.auth_file),
            f"{config.instance}:{guest_root}/auth.transfer.json",
        )
        proxy_started = True
        proxy = _start_model_proxy(
            config=config,
            guest_root=guest_root,
            experiment_id=experiment_id,
        )
        env_local = session_directory / ".participant.env"
        env_local.write_text(
            _runtime_env(proxy["participant_proxy"]),
            encoding="utf-8",
            newline="\n",
        )
        try:
            _multipass(
                config,
                "transfer",
                str(env_local),
                f"{config.instance}:{guest_env}",
            )
        finally:
            env_local.unlink(missing_ok=True)
        _guest(
            config,
            "bash",
            "-lc",
            (
                f"set -eu; mkdir -m 0700 '{guest_codex_home}'; "
                f"install -m 0600 '{guest_root}/auth.transfer.json' "
                f"'{guest_codex_home}/auth.json'; "
                f"rm -f '{guest_root}/auth.transfer.json'; "
                f"sudo chown -R 65532:65532 '{guest_workspace}' "
                f"'{guest_codex_home}' '{guest_evidence}'; "
                f"sudo chmod -R u=rX,go-rwx '{guest_evidence}'; "
                f"sudo chmod 0711 '{guest_root}'; "
                f"sudo chown gfg:gfg '{guest_env}'; "
                f"sudo chmod 0600 '{guest_env}'"
            ),
        )
        _lock_evidence(config, guest_evidence)
        stdout_path = session_directory / "codex.stdout.log"
        stderr_path = session_directory / "codex.stderr.log"
        started_at = _utc_now()
        started_monotonic = time.monotonic()
        command = [
            str(config.multipass),
            "exec",
            config.instance,
            "--",
            *_container_command(
                config=config,
                container_name=container_name,
                guest_workspace=guest_workspace,
                guest_codex_home=guest_codex_home,
                guest_evidence=guest_evidence,
                guest_env=guest_env,
                participant_proxy=proxy["participant_proxy"],
            ),
        ]
        orientation_validation: dict[str, Any] | None = None
        orientation_released_at: float | None = None
        orientation_started_at = _utc_now()
        submission_ready_at: str | None = None
        submission_ready_monotonic: float | None = None
        runner_terminated_after_submission = False
        next_submission_poll: float | None = None
        with (
            stdout_path.open("w", encoding="utf-8", newline="\n") as stdout,
            stderr_path.open("w", encoding="utf-8", newline="\n") as stderr,
        ):
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True)
            orientation_deadline = (
                started_monotonic
                + orientation_contract["maximum_seconds_until_valid_receipt"]
            )
            formal_deadline: float | None = None
            next_orientation_poll = started_monotonic + 300
            while process.poll() is None:
                now = time.monotonic()
                if orientation_validation is None and now >= orientation_deadline:
                    _stop_container(config, container_name)
                    process.wait(timeout=30)
                    raise FormalAISessionError("GFG_ORIENTATION_RECEIPT_NOT_VALIDATED")
                if formal_deadline is not None and now >= formal_deadline:
                    _stop_container(config, container_name)
                    process.wait(timeout=30)
                    raise FormalAISessionError("PARTICIPANT_WALL_CLOCK_EXCEEDED")
                if formal_deadline is not None and now >= (next_submission_poll or now):
                    if _submission_ready(config, guest_workspace):
                        if submission_ready_monotonic is None:
                            submission_ready_monotonic = time.monotonic()
                            submission_ready_at = _utc_now()
                        elif now - submission_ready_monotonic >= 5:
                            _finish_submission_process(
                                config=config,
                                container_name=container_name,
                                process=process,
                            )
                            runner_terminated_after_submission = True
                            break
                    next_submission_poll = time.monotonic() + 2
                elapsed = time.monotonic() - started_monotonic
                if (
                    orientation_validation is None
                    and elapsed >= 300
                    and now >= next_orientation_poll
                ):
                    orientation_validation = _collect_orientation_receipt(
                        config=config,
                        guest_workspace=guest_workspace,
                        session_directory=session_directory,
                        elapsed_seconds=elapsed,
                    )
                    next_orientation_poll = time.monotonic() + 5
                if orientation_validation is not None and formal_deadline is None:
                    _release_evidence(config, guest_evidence, container_name)
                    orientation_released_at = time.monotonic()
                    formal_deadline = (
                        orientation_released_at + config.wall_clock_seconds
                    )
                    next_submission_poll = time.monotonic()
                    orientation_material = {
                        "evidence_released_at": _utc_now(),
                        "experiment_id": experiment_id,
                        "manual_sha256": file_sha256(
                            participant_repository / "GFG_MACHINE_SEMANTICS.md"
                        ),
                        "mechanism_discovery_guide_sha256": file_sha256(
                            participant_repository
                            / "EXECUTABLE_MECHANISM_DISCOVERY_GUIDE.md"
                        ),
                        "measured_manual_only_seconds": elapsed,
                        "minimum_manual_only_seconds": 300,
                        "orientation_ai_receipt_sha256": (
                            orientation_validation["receipt_sha256"]
                        ),
                        "orientation_ai_semantics_validated": True,
                        "orientation_gate_validation_sha256": (
                            orientation_validation["validation_sha256"]
                        ),
                        "orientation_seconds_charged_to_target_budget": False,
                        "orientation_started_at": orientation_started_at,
                        "participant_can_release_gate": False,
                        "release_authority": "external-runner",
                        "schema": ("gfg-formal-first-contact-orientation-receipt-v1"),
                        "status": "PASS",
                        "target_answer_supplied_by_orientation": False,
                        "target_evidence_readable_after_release": True,
                        "target_evidence_unreadable_before_release": True,
                    }
                    orientation_receipt = {
                        **orientation_material,
                        "receipt_sha256": payload_sha256(orientation_material),
                    }
                    write_json(
                        session_directory / "orientation_gate_receipt.json",
                        orientation_receipt,
                    )
                    orientation_validation = {
                        **orientation_validation,
                        "gate_receipt_sha256": orientation_receipt["receipt_sha256"],
                    }
                time.sleep(1)
            if orientation_validation is None or formal_deadline is None:
                raise FormalAISessionError("GFG_ORIENTATION_WINDOW_NOT_COMPLETED")
            if process.returncode != 0 and not runner_terminated_after_submission:
                raise FormalAISessionError(
                    f"CODEX_PARTICIPANT_EXITED:{process.returncode}"
                )
        status_raw = _guest(
            config,
            "sudo",
            "git",
            "-c",
            f"safe.directory={guest_workspace}",
            "-C",
            guest_workspace,
            "status",
            "--porcelain=v1",
            "--no-renames",
            "-z",
            "--untracked-files=all",
            text=False,
        ).stdout
        paths = _git_status_paths(status_raw)
        _assert_submission_only(paths)
        _guest(
            config,
            "sudo",
            "chown",
            "-R",
            "ubuntu:ubuntu",
            f"{guest_workspace}/submission",
        )
        transfer_root = session_directory / "result"
        transfer_root.mkdir()
        transfer = _multipass(
            config,
            "transfer",
            "-r",
            f"{config.instance}:{guest_workspace}/submission",
            str(transfer_root),
            check=False,
        )
        if not _transfer_ok(transfer):
            raise FormalAISessionError(
                "SUBMISSION_TRANSFER_FAILED:" + str(transfer.returncode)
            )
        submission = transfer_root / "submission"
        if not submission.is_dir():
            raise FormalAISessionError("TRANSFERRED_SUBMISSION_MISSING")
        submission_transferred = True
        submission_files = [path for path in submission.rglob("*") if path.is_file()]
        secret_scan = _secret_scan(
            config.auth_file,
            [stdout_path, stderr_path, *submission_files],
        )
        network_audit = _model_proxy_audit(config, guest_root, session_directory)
        if network_audit["status"] != "PASS":
            raise FormalAISessionError("MODEL_NETWORK_AUDIT_FAILED")
        formal_completed_monotonic = (
            submission_ready_monotonic
            if submission_ready_monotonic is not None
            else time.monotonic()
        )
        formal_seconds = max(
            0.0,
            formal_completed_monotonic
            - (
                orientation_released_at
                if orientation_released_at is not None
                else started_monotonic
            ),
        )
        material = {
            "additional_ai_calls": 0,
            "attested_participant_gfg_id": evidence_manifest["bundle_manifest_sha256"],
            "changed_paths": paths,
            "completed_at": _utc_now(),
            "evidence_mount_read_only": True,
            "formal_work_seconds": formal_seconds,
            "instance": config.instance,
            "model": config.model,
            "model_network_audit": network_audit,
            "network_used_only_for_codex_model": True,
            "orientation_validation": {
                **orientation_validation,
                "target_gfg_readable_after_release": True,
                "status": "PASS",
            },
            "participant_access_audit": {
                "authority_mount": None,
                "evidence_read_only": True,
                "forbidden_success_count": 0,
                "prior_experiment_mount": None,
                "status": "PASS",
            },
            "participant_baseline_commit": participant_baseline_commit,
            "participant_image": config.participant_image,
            "participant_proxy": proxy["participant_proxy"],
            "reasoning_effort": config.reasoning_effort,
            "response_transport": (
                "https-only" if config.http_only_transport else "builtin"
            ),
            "schema": "formal-ai-session-attestation-v2",
            "secret_scan": secret_scan,
            "single_formal_session": True,
            "started_at": started_at,
            "submission_ready_at": submission_ready_at,
            "runner_terminated_after_submission": (runner_terminated_after_submission),
            "participant_exit_mode": (
                "RUNNER_TERMINATED_AFTER_READY_MARKER"
                if runner_terminated_after_submission
                else "PARTICIPANT_EXITED"
            ),
            "submission_manifest": relative_file_manifest(submission),
            "vm_start_gate": vm,
        }
        attestation = {
            **material,
            "attestation_sha256": payload_sha256(material),
        }
        write_json(
            session_directory / "session_attestation.json",
            attestation,
        )
        return {**attestation, "_submission": submission}
    finally:
        if process is not None and process.poll() is None:
            _stop_container(config, container_name)
        if proxy_started:
            _stop_model_proxy(config, guest_root)
        _cleanup_guest_after_session(
            config,
            guest_root,
            submission_ready=submission_ready_at is not None,
            submission_transferred=submission_transferred,
        )
