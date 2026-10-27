"""
M.A.G.I.C. Core – User Interface
Lightweight Flask-based API + optional CLI fallback.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import threading
import time

try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

from memory_module import MemoryModule


class UserInterface:
    """
    Provides task input, progress monitoring, and feedback endpoints.
    When Flask is unavailable, a simple interactive CLI is used instead.
    """

    def __init__(self, task: Optional[str] = None, memory: Optional[MemoryModule] = None):
        self.task = task
        self.memory_module = memory or MemoryModule()
        self.progress: Dict[str, Any] = {
            "status": "idle",
            "generation": 0,
            "best_fitness": 0.0,
            "message": "System ready",
        }
        self._lock = threading.Lock()

        if HAS_FLASK:
            self.app = Flask(__name__)
            self._register_routes()
        else:
            self.app = None

    def _register_routes(self) -> None:
        @self.app.route("/task", methods=["POST"])
        def input_task():
            data = request.get_json(silent=True) or {}
            self.task = data.get("task", "")
            self.memory_module.store_information(
                {"event": "task_submitted", "task": self.task},
                task_signature=self.task or "general",
                tags=["task"],
            )
            with self._lock:
                self.progress["status"] = "task_received"
                self.progress["message"] = f"Task accepted: {self.task[:80]}"
            return jsonify({"message": "Task input successful", "task": self.task}), 200

        @self.app.route("/progress", methods=["GET"])
        def monitor_progress():
            with self._lock:
                return jsonify(self.progress), 200

        @self.app.route("/feedback", methods=["POST"])
        def provide_feedback():
            data = request.get_json(silent=True) or {}
            feedback = data.get("feedback", "")
            self.memory_module.store_information(
                {"event": "user_feedback", "feedback": feedback, "task": self.task},
                task_signature=self.task or "general",
                tags=["feedback"],
            )
            return jsonify({"message": "Feedback received"}), 200

        @self.app.route("/memory", methods=["GET"])
        def get_memory():
            limit = request.args.get("limit", 20, type=int)
            entries = self.memory_module.retrieve_information(limit=limit)
            return jsonify({"count": len(entries), "entries": entries}), 200

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok", "flask": True}), 200

    def update_progress(self, **kwargs) -> None:
        """Called by the orchestrator to push live telemetry."""
        with self._lock:
            self.progress.update(kwargs)
            self.progress["timestamp"] = time.time()

    def input_task(self, task: str) -> None:
        """Programmatic task submission (CLI / internal use)."""
        self.task = task
        self.memory_module.store_information(
            {"event": "task_submitted", "task": task},
            task_signature=task,
            tags=["task"],
        )
        self.update_progress(status="task_received", message=f"Task accepted: {task[:80]}")

    def monitor_progress(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.progress)

    def provide_feedback(self, feedback: str) -> None:
        self.memory_module.store_information(
            {"event": "user_feedback", "feedback": feedback, "task": self.task},
            task_signature=self.task or "general",
            tags=["feedback"],
        )

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
        if HAS_FLASK and self.app is not None:
            print(f"[M.A.G.I.C. UI] Flask server starting on http://{host}:{port}")
            self.app.run(host=host, port=port, debug=debug, use_reloader=False)
        else:
            self._run_cli()

    def _run_cli(self) -> None:
        """Simple interactive fallback when Flask is not installed."""
        print("\n=== M.A.G.I.C. Interactive CLI ===")
        print("Commands: task <text> | progress | feedback <text> | memory | quit")
        while True:
            try:
                cmd = input("magic> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not cmd:
                continue
            if cmd.lower() in ("quit", "exit", "q"):
                break
            if cmd.lower().startswith("task "):
                self.input_task(cmd[5:].strip())
                print("Task accepted.")
            elif cmd.lower() == "progress":
                print(self.monitor_progress())
            elif cmd.lower().startswith("feedback "):
                self.provide_feedback(cmd[9:].strip())
                print("Feedback stored.")
            elif cmd.lower() == "memory":
                entries = self.memory_module.retrieve_information(limit=5)
                for e in entries:
                    print(f"  [{e.get('task_signature')}] {e.get('content')}")
            else:
                print("Unknown command.")
        print("CLI session ended.")
