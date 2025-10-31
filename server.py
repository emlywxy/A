import json
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from questions import (
    generate_mathematics_question,
    generate_network_broadcast_question,
    generate_roman_numerals_question,
    generate_usable_addresses_question,
)


def jsend(conn: socket.socket, obj: Dict[str, Any]) -> None:
    conn.sendall(json.dumps(obj).encode("utf-8") + b"\n")


def jrecv(conn: socket.socket) -> Optional[Dict[str, Any]]:
    buf = bytearray()
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            return None
        nl = chunk.find(b"\n")
        if nl != -1:
            buf.extend(chunk[:nl])
            break
        buf.extend(chunk)
    return json.loads(buf.decode("utf-8"))


def require_config(argv: List[str]) -> Path:
    if len(argv) < 2 or argv[0] != "--config":
        print("server.py: Configuration not provided", file=sys.stderr)
        sys.exit(1)

    path = Path(argv[1])
    if not path.exists():
        print(f"server.py: File {path} does not exist", file=sys.stderr)
        sys.exit(1)
    return path


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def gen_short(qtype: str) -> str:
    if qtype == "Mathematics":
        return generate_mathematics_question()
    if qtype == "Roman Numerals":
        return generate_roman_numerals_question()
    if qtype == "Usable IP Addresses of a Subnet":
        return generate_usable_addresses_question()
    if qtype == "Network and Broadcast Address of a Subnet":
        return generate_network_broadcast_question()
    raise ValueError(qtype)


def usable_ipv4_host_count(prefix: int) -> str:
    host_bits = 32 - prefix
    if host_bits <= 0:
        # Spec treats /32 as having one usable address.
        return "1"
    if host_bits == 1:
        # Spec treats /31 as having two usable addresses.
        return "2"
    return str((1 << host_bits) - 2)


def compute_answer(qtype: str, short_q: str) -> str:
    if qtype == "Mathematics":
        total, num, op = 0, 0, 1
        for ch in short_q:
            if ch.isdigit():
                num = num * 10 + (ord(ch) - 48)
            elif ch in "+-":
                total += op * num
                num = 0
                op = 1 if ch == "+" else -1
        total += op * num
        return str(total)

    if qtype == "Roman Numerals":
        vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        s = short_q.strip().upper()
        i, tot = 0, 0
        while i < len(s):
            v = vals[s[i]]
            if i + 1 < len(s) and vals[s[i + 1]] > v:
                tot += vals[s[i + 1]] - v
                i += 2
            else:
                tot += v
                i += 1
        return str(tot)

    if qtype == "Usable IP Addresses of a Subnet":
        _, prefix = short_q.split("/")
        return usable_ipv4_host_count(int(prefix))

    if qtype == "Network and Broadcast Address of a Subnet":
        ip, prefix = short_q.split("/")
        a, b, c, d = map(int, ip.split("."))
        ipi = (a << 24) | (b << 16) | (c << 8) | d
        prefix_int = int(prefix)
        mask = (0xFFFFFFFF << (32 - prefix_int)) & 0xFFFFFFFF
        net = ipi & mask
        bcast = net | (~mask & 0xFFFFFFFF)

        def dec(x: int) -> str:
            return f"{(x >> 24) & 255}.{(x >> 16) & 255}.{(x >> 8) & 255}.{x & 255}"

        return f"{dec(net)} and {dec(bcast)}"

    raise ValueError(qtype)


def bind_or_die(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.listen()
        return sock
    except OSError:
        print(f"server.py: Binding to port {port} was unsuccessful", file=sys.stderr)
        sys.exit(1)


@dataclass
class Player:
    conn: socket.socket
    addr: tuple[str, int]
    username: Optional[str] = None
    points: int = 0
    connected: bool = True
    answered_this_round: bool = False


class TriviaServer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.sock = bind_or_die(int(cfg["port"]))
        self.players: List[Player] = []
        self.players_lock = threading.Lock()
        self.round_open = threading.Event()
        self.actual_correct_answer: Optional[str] = None
        self.correct_feedback_template: str = ""
        self.incorrect_feedback_template: str = ""
        self.game_started = False

    def run(self) -> None:
        while True:
            conn, addr = self.sock.accept()
            player = Player(conn, addr)
            with self.players_lock:
                self.players.append(player)
            threading.Thread(target=self.client_loop, args=(player,), daemon=True).start()

    def disconnect(self, player: Player) -> None:
        with self.players_lock:
            if not player.connected:
                return
            player.connected = False
        try:
            player.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        finally:
            try:
                player.conn.close()
            except OSError:
                pass

    def disconnect_all(self) -> None:
        with self.players_lock:
            players = [p for p in self.players if p.connected]
        for player in players:
            self.disconnect(player)

    def client_loop(self, player: Player) -> None:
        try:
            msg = jrecv(player.conn)
            if not msg or msg.get("message_type") != "HI":
                self.disconnect(player)
                return
            player.username = str(msg.get("username"))

            self.maybe_start_game()

            while True:
                msg = jrecv(player.conn)
                if msg is None:
                    self.disconnect(player)
                    return
                mtype = msg.get("message_type")
                if mtype == "BYE":
                    self.disconnect(player)
                    return
                if mtype == "ANSWER":
                    ans = str(msg.get("answer"))
                    self.handle_answer(player, ans)
        except Exception:
            self.disconnect(player)

    def maybe_start_game(self) -> None:
        with self.players_lock:
            if self.game_started:
                return
            joined = [p for p in self.players if p.connected]
            need = int(self.cfg["players"])
            if len(joined) < need:
                return
            self.game_started = True
            ready_targets = list(joined)

        info = self.cfg["ready_info"].format(**self.cfg)
        for player in ready_targets:
            jsend(player.conn, {"message_type": "READY", "info": info})

        threading.Thread(target=self.run_game, daemon=True).start()

    def run_game(self) -> None:
        time.sleep(float(self.cfg["question_interval_seconds"]))

        qword = self.cfg["question_word"]
        qtypes = list(self.cfg["question_types"])
        qfmt = self.cfg["question_formats"]
        qsecs = float(self.cfg["question_seconds"])

        for idx, qtype in enumerate(qtypes, start=1):
            short_q = gen_short(qtype)
            visible_q = qfmt[qtype].format(short_q)
            trivia_q = f"{qword} {idx} ({qtype}):\n{visible_q}"

            self.actual_correct_answer = compute_answer(qtype, short_q)
            self.correct_feedback_template = self.cfg["correct_answer"]
            self.incorrect_feedback_template = self.cfg["incorrect_answer"]

            with self.players_lock:
                for player in self.players:
                    player.answered_this_round = False

            with self.players_lock:
                targets = [player for player in self.players if player.connected]
            for player in targets:
                jsend(
                    player.conn,
                    {
                        "message_type": "QUESTION",
                        "question_type": qtype,
                        "trivia_question": trivia_q,
                        "short_question": short_q,
                        "time_limit": qsecs,
                    },
                )

            self.round_open.set()
            deadline = time.monotonic() + qsecs
            while self.round_open.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.round_open.clear()

            self.broadcast_leaderboard()

            last_question = idx == len(qtypes)
            if last_question:
                self.broadcast_finished()
                with self.players_lock:
                    self.game_started = False
                break
            time.sleep(float(self.cfg["question_interval_seconds"]))

    def all_connected_answered(self) -> bool:
        with self.players_lock:
            active = [p for p in self.players if p.connected]
            return len(active) > 0 and all(p.answered_this_round for p in active)

    def handle_answer(self, player: Player, ans: str) -> None:
        if not self.round_open.is_set():
            return

        with self.players_lock:
            if (not player.connected) or player.answered_this_round:
                return
            player.answered_this_round = True

        provided = ans.strip()
        correct_answer = self.actual_correct_answer
        if correct_answer is None:
            return
        correct = provided == correct_answer
        feedback_template = self.correct_feedback_template if correct else self.incorrect_feedback_template
        feedback = feedback_template.format(answer=provided, correct_answer=correct_answer)

        if correct:
            with self.players_lock:
                player.points += 1

        jsend(
            player.conn,
            {
                "message_type": "RESULT",
                "correct": bool(correct),
                "feedback": feedback,
            },
        )

        if self.all_connected_answered():
            self.round_open.clear()

    def sorted_all(self) -> List[Player]:
        with self.players_lock:
            players = list(self.players)
        return sorted(players, key=lambda p: (-p.points, p.username or ""))

    def fmt_place_lines(self, players: List[Player]) -> List[str]:
        lines: List[str] = []
        place = 0
        last_points: Optional[int] = None
        for index, player in enumerate(players):
            if last_points is None or player.points != last_points:
                place = index + 1
                last_points = player.points
            noun = (
                self.cfg["points_noun_singular"]
                if player.points == 1
                else self.cfg["points_noun_plural"]
            )
            lines.append(f"{place}. {player.username}: {player.points} {noun}")
        return lines

    def broadcast_leaderboard(self) -> None:
        players = self.sorted_all()
        state = "\n".join(self.fmt_place_lines(players))
        with self.players_lock:
            targets = [player for player in self.players if player.connected]
        for player in targets:
            jsend(player.conn, {"message_type": "LEADERBOARD", "state": state})

    def broadcast_finished(self) -> None:
        players = self.sorted_all()
        lines = [self.cfg["final_standings_heading"]]
        lines += self.fmt_place_lines(players)

        if players:
            top = players[0].points
            winners = sorted([player.username for player in players if player.points == top])
        else:
            winners = []

        if len(winners) == 1:
            tail = self.cfg["one_winner"].format(winners[0])
        else:
            tail = self.cfg["multiple_winners"].format(", ".join(winners))

        lines.append(tail)
        final = "\n".join(lines)

        with self.players_lock:
            targets = [player for player in self.players if player.connected]
        for player in targets:
            jsend(player.conn, {"message_type": "FINISHED", "final_standings": final})


def main() -> None:
    cfg = load_config(require_config(sys.argv[1:]))
    TriviaServer(cfg).run()


if __name__ == "__main__":
    main()
