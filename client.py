import json
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

import requests

waiting_for_answer = threading.Event()
question_ready = threading.Event()
pending_question: dict[str, Any] | None = None
_recv_buf = bytearray()


class TimeoutExpired(Exception):
    pass


def run_with_timeout(seconds: float, func, *args, **kwargs):
    def _handler(signum, frame):
        raise TimeoutExpired()

    old = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, max(0.0, float(seconds)))
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def encode_message(message: dict[str, Any]) -> bytes:
    return json.dumps(message).encode(encoding="utf-8")


def decode_message(data: bytes) -> dict[str, Any]:
    return json.loads(data.decode(encoding="utf-8"))


def send_message(connection: socket.socket, data: dict[str, Any]):
    connection.sendall(encode_message(data) + b"\n")


def receive_message(connection: socket.socket) -> dict[str, Any] | None:
    global _recv_buf
    while True:
        nl = _recv_buf.find(b"\n")
        if nl != -1:
            line = bytes(_recv_buf[:nl])
            _recv_buf = _recv_buf[nl + 1 :]
            try:
                return decode_message(line)
            except json.JSONDecodeError:
                continue

        chunk = connection.recv(4096)
        if not chunk:
            return None
        _recv_buf.extend(chunk)


def connect(host: str, port: int, username: str) -> socket.socket:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
    except OSError:
        print("Connection failed")
        sys.exit(0)
    send_message(sock, {"message_type": "HI", "username": username})
    return sock


def disconnect(connection: socket.socket | None):
    if not connection:
        return
    try:
        send_message(connection, {"message_type": "BYE"})
    finally:
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        connection.close()


def answer_question(
    question: str,
    short_question: str,
    client_mode: Literal["you", "auto", "ai"],
    time_limit: float,
    ollama_cfg: dict[str, Any] | None = None,
) -> str:
    if client_mode == "you":
        return input().strip()

    if client_mode == "auto":
        if "Mathematics" in question:
            total = 0
            num = 0
            op = 1
            for ch in short_question:
                if ch.isdigit():
                    num = num * 10 + (ord(ch) - 48)
                elif ch in "+-":
                    total += op * num
                    num = 0
                    op = 1 if ch == "+" else -1
                elif ch.isspace():
                    continue
            total += op * num
            return str(total)

        if "Roman Numerals" in question:
            table = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
            total = 0
            idx = 0
            while idx < len(short_question):
                value = table[short_question[idx]]
                if idx + 1 < len(short_question) and table[short_question[idx + 1]] > value:
                    total += table[short_question[idx + 1]] - value
                    idx += 2
                else:
                    total += value
                    idx += 1
            return str(total)

        if "Usable IP Addresses of a Subnet" in question:
            _, prefix = short_question.split("/")
            bits = 32 - int(prefix)
            return "0" if bits <= 1 else str((1 << bits) - 2)

        if "Network and Broadcast Address of a Subnet" in question:
            ip, prefix = short_question.split("/")
            prefix = int(prefix)
            octets = [int(part) for part in ip.split(".")]
            ip_int = (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]
            mask = (0xFFFFFFFF << (32 - prefix)) & 0xFFFFFFFF
            net = ip_int & mask
            broadcast = net | (~mask & 0xFFFFFFFF)
            network = f"{(net >> 24) & 255}.{(net >> 16) & 255}.{(net >> 8) & 255}.{net & 255}"
            bcast = f"{(broadcast >> 24) & 255}.{(broadcast >> 16) & 255}.{(broadcast >> 8) & 255}.{broadcast & 255}"
            return f"{network} and {bcast}"

        return ""

    if client_mode == "ai":
        try:
            return answer_question_ollama(
                question=question,
                ollama_host=ollama_cfg["ollama_host"],
                ollama_port=ollama_cfg["ollama_port"],
                ollama_model=ollama_cfg["ollama_model"],
                time_limit=time_limit,
            ).strip()
        except requests.Timeout:
            return "timeout"
        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError):
            return ""

    return ""


def get_client_config_path(argv: list[str]) -> Path:
    if (len(argv) < 2) or (argv[0] != "--config"):
        print("client.py: Configuration not provided", file=sys.stderr)
        sys.exit(1)
    path = Path(argv[1])
    if not path.exists():
        print(f"client.py: File {path} does not exist", file=sys.stderr)
        sys.exit(1)
    return path


def load_client_config(argv: list[str]) -> dict[str, Any]:
    path = get_client_config_path(argv)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("client_mode") == "ai":
        ollama_cfg = config.get("ollama_config")
        if not ollama_cfg or any(
            key not in ollama_cfg for key in ("ollama_host", "ollama_port", "ollama_model")
        ):
            print("client.py: Missing values for Ollama configuration", file=sys.stderr)
            sys.exit(1)
    return config


def answer_question_ollama(
    question: str,
    ollama_host: str,
    ollama_port: int,
    ollama_model: str,
    time_limit: float | int,
) -> str:
    url = f"http://{ollama_host}:{ollama_port}/api/chat"
    system_prompt = (
        "You are a trivia solver. Respond with ONLY the final answer, on a single line, "
        "with no commentary, no units, and no punctuation beyond what the answer needs.\n"
        "- If mathematics, output the integer result.\n"
        "- If a Roman numeral, output its decimal value as an integer.\n"
        "- If usable IPv4 addresses, output the integer count.\n"
        "- If network and broadcast addresses, output exactly: 'X.X.X.X and Y.Y.Y.Y'."
    )

    body = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "stream": False,
    }

    response = requests.post(url, json=body, timeout=(2, max(1, int(time_limit))))
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def handle_command(
    command: str,
    connection: socket.socket | None,
    username: str,
) -> tuple[socket.socket | None, bool]:
    parts = command.strip().split()
    cmd = parts[0].upper()

    if cmd == "DISCONNECT":
        if connection is not None:
            disconnect(connection)
        return None, False

    if cmd == "EXIT":
        if connection is not None:
            disconnect(connection)
        return None, True

    return connection, False


def handle_received_message(message: dict[str, Any]):
    if message.get("message_type") == "READY":
        print(message.get("info"))
        return

    if message.get("message_type") == "QUESTION":
        print(message.get("trivia_question"))
        return

    if message.get("message_type") == "RESULT":
        print(message.get("feedback"))
        return

    if message.get("message_type") == "LEADERBOARD":
        print(message.get("state"))
        return

    if message.get("message_type") == "FINISHED":
        print(message.get("final_standings"))
        return


def receiver_loop(connection, mode, cfg):
    def arm_timeout(seconds: float):
        waiting_for_answer.set()

        def _close():
            time.sleep(max(0.0, float(seconds)))
            waiting_for_answer.clear()

        threading.Thread(target=_close, daemon=True).start()

    while True:
        try:
            message = receive_message(connection)
            if not message:
                break
            handle_received_message(message)
            if message.get("message_type") == "QUESTION":
                tl = float(message.get("time_limit"))
                arm_timeout(tl)
                global pending_question
                pending_question = message
                question_ready.set()

        except Exception:
            waiting_for_answer.clear()
            continue


def main():
    cfg = load_client_config(sys.argv[1:])
    username = cfg.get("username")
    mode = cfg.get("client_mode")
    sock: socket.socket | None = None

    def read_line_with_small_timeout() -> str | None:
        try:
            return run_with_timeout(0.1, sys.stdin.readline)
        except TimeoutExpired:
            return None

    try:
        while True:
            if question_ready.is_set() and mode in ("auto", "ai") and sock is not None:
                q = pending_question
                question_ready.clear()
                try:
                    ans = run_with_timeout(
                        float(q.get("time_limit")),
                        answer_question,
                        q.get("trivia_question"),
                        q.get("short_question"),
                        mode,
                        float(q.get("time_limit")),
                        cfg.get("ollama_config") if mode == "ai" else None,
                    )
                    if waiting_for_answer.is_set():
                        try:
                            send_message(sock, {"message_type": "ANSWER", "answer": ans})
                        except OSError:
                            pass
                except TimeoutExpired:
                    pass
                finally:
                    waiting_for_answer.clear()

            line = read_line_with_small_timeout()
            if line is None:
                continue

            if not line:
                break

            text = line.strip()

            if text.upper() in ("DISCONNECT", "EXIT"):
                new_conn, should_exit = handle_command(text, sock, username)
                sock = new_conn
                waiting_for_answer.clear()
                if should_exit:
                    break
                continue

            if mode == "you" and waiting_for_answer.is_set():
                try:
                    send_message(sock, {"message_type": "ANSWER", "answer": text})
                except OSError:
                    pass
                finally:
                    waiting_for_answer.clear()
                continue

            parts = text.split()
            if parts and parts[0].upper() == "CONNECT":
                host, port_str = parts[1].split(":", 1)
                port = int(port_str)
                sock = connect(host, port, username)
                threading.Thread(target=receiver_loop, args=(sock, mode, cfg), daemon=True).start()
                continue

            new_conn, should_exit = handle_command(text, sock, username)
            sock = new_conn
            if should_exit:
                break

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if sock is not None:
            disconnect(sock)


if __name__ == "__main__":
    main()
