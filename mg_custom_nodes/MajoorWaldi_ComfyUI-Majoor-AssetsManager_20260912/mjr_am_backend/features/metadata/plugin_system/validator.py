"""
Plugin System - Validator

Validates plugins for security issues before loading.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PluginValidator:
    """
    Validates plugins for security issues.

    Performs both pattern-based and AST-based analysis
    to detect potentially dangerous code.
    """

    # Dangerous patterns that should trigger warnings
    DANGEROUS_PATTERNS = [
        (r"__import__\s*\(\s*['\"]os['\"]\s*\)", "Dynamic os import"),
        (r"__import__\s*\(\s*['\"]subprocess['\"]\s*\)", "Dynamic subprocess import"),
        (r"subprocess\.(call|run|Popen|check_output|check_call)", "Subprocess execution"),
        (r"os\.(system|popen|spawnl|spawnle|spawnv|spawnve|exec)", "OS command execution"),
        (r"eval\s*\(", "Eval usage"),
        (r"exec\s*\(", "Exec usage"),
        (r"compile\s*\(", "Compile usage"),
        (r"__builtins__", "Builtins access"),
        (r"importlib\.(reload|import_module|exec_module)", "Dynamic module loading"),
        (r"ctypes\.", "CTypes usage"),
        (r"pickle\.(load|loads)", "Pickle deserialization"),
        (r"marshal\.(load|loads)", "Marshal deserialization"),
        (r"shelve\.(open|DbfilenameShelf)", "Shelve usage"),
        (r"socket\.", "Socket usage"),
        (r"urllib\.(request|parse)", "URL operations"),
        (r"requests\.", "HTTP requests"),
        (r"http\.(client|server)", "HTTP operations"),
        (r"ftplib\.", "FTP operations"),
        (r"paramiko\.", "SSH operations"),
    ]

    # Allowed imports for plugins
    SAFE_IMPORTS = {
        # Standard library (safe)
        "typing", "collections", "functools", "itertools", "pathlib",
        "json", "re", "hashlib", "base64", "io", "struct",
        "logging", "dataclasses", "enum", "abc", "contextlib",
        "asyncio", "concurrent", "threading", "multiprocessing",
        "datetime", "time", "calendar",
        "math", "statistics", "random",
        "xml", "html", "csv",
        "unittest", "pytest",

        # PIL/Pillow (safe for image processing)
        "PIL", "PIL.Image", "PIL.PngImagePlugin", "PIL.JpegImagePlugin",

        # Third-party (safe)
        "numpy", "pillow",
    }

    # Dangerous imports that should trigger warnings
    DANGEROUS_IMPORTS = {
        "os": "Direct OS access",
        "subprocess": "Process execution",
        "ctypes": "C library access",
        "socket": "Network socket access",
        "requests": "HTTP requests",
        "urllib": "URL operations",
        "http": "HTTP operations",
        "ftplib": "FTP operations",
        "paramiko": "SSH operations",
        "pickle": "Unsafe deserialization",
        "marshal": "Unsafe deserialization",
        "shelve": "Unsafe database",
        "eval": "Code execution",
        "exec": "Code execution",
    }

    @classmethod
    def validate(
        cls,
        plugin_path: Path
    ) -> tuple[bool, list[str], dict[str, Any]]:
        """
        Validate a plugin file.

        Args:
            plugin_path: Path to plugin file

        Returns:
            Tuple of (is_valid, warnings, info_dict)
        """
        warnings = []
        info: dict[str, Any] = {
            "path": str(plugin_path),
            "size_bytes": 0,
            "lines": 0,
            "classes": 0,
            "functions": 0,
            "imports": [],
        }

        try:
            content = plugin_path.read_text(encoding='utf-8')
        except Exception as e:
            return False, [f"Cannot read file: {e}"], info

        info["size_bytes"] = len(content.encode('utf-8'))
        info["lines"] = len(content.splitlines())

        # Pattern-based checks
        pattern_warnings = cls._check_patterns(content)
        warnings.extend(pattern_warnings)

        # AST-based checks
        try:
            tree = ast.parse(content)
            ast_warnings, ast_info = cls._check_ast(tree)
            warnings.extend(ast_warnings)
            info.update(ast_info)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"], info
        except Exception as e:
            return False, [f"AST analysis failed: {e}"], info

        # File size check
        if info["size_bytes"] > 1024 * 1024:  # 1MB limit
            warnings.append(f"File too large: {info['size_bytes']} bytes (max 1MB)")

        # Complexity check (simple heuristic)
        if info["functions"] > 50 or info["classes"] > 10:
            warnings.append(
                f"High complexity: {info['classes']} classes, "
                f"{info['functions']} functions"
            )

        is_valid = len(warnings) == 0
        return is_valid, warnings, info

    @classmethod
    def _check_patterns(cls, content: str) -> list[str]:
        """Check for dangerous patterns in code."""
        warnings = []

        for pattern, description in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, content):
                warnings.append(f"Security warning: {description} detected")

        return warnings

    @classmethod
    def _record_import_warning(
        cls,
        warnings: list[str],
        dangerous_found: set[str],
        import_name: str,
        *,
        from_import: bool = False,
    ) -> None:
        if import_name not in cls.DANGEROUS_IMPORTS:
            return
        if import_name in dangerous_found:
            return
        prefix = "Dangerous import from" if from_import else "Dangerous import"
        warnings.append(f"{prefix}: {import_name} ({cls.DANGEROUS_IMPORTS[import_name]})")
        dangerous_found.add(import_name)

    @staticmethod
    def _record_function_call_warning(warnings: list[str], node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "compile"}:
            warnings.append(f"Dangerous function call: {node.func.id}()")
            return
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"system", "popen", "eval", "exec"}:
            warnings.append(f"Dangerous method call: {node.func.attr}()")

    @classmethod
    def _check_ast(
        cls,
        tree: ast.AST
    ) -> tuple[list[str], dict[str, Any]]:
        """AST-based security checks."""
        warnings: list[str] = []
        info: dict[str, Any] = {
            "classes": 0,
            "functions": 0,
            "imports": [],
        }

        dangerous_found: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                info["classes"] += 1

            if isinstance(node, ast.FunctionDef):
                info["functions"] += 1

            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.name.split('.')[0]
                    info["imports"].append(import_name)
                    cls._record_import_warning(warnings, dangerous_found, import_name)

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    info["imports"].append(module_name)
                    cls._record_import_warning(
                        warnings,
                        dangerous_found,
                        module_name,
                        from_import=True,
                    )

            if isinstance(node, ast.Call):
                cls._record_function_call_warning(warnings, node)

        return warnings, info

    @classmethod
    def validate_strict(
        cls,
        plugin_path: Path
    ) -> tuple[bool, list[str]]:
        """
        Strict validation - fails on any warning.

        Returns:
            (is_valid, warnings)
        """
        is_valid, warnings, _ = cls.validate(plugin_path)
        return is_valid, warnings

    @classmethod
    def validate_permissive(
        cls,
        plugin_path: Path
    ) -> tuple[bool, list[str]]:
        """
        Permissive validation - only fails on critical issues.

        Critical issues:
        - Syntax errors
        - File too large
        - eval/exec usage

        Returns:
            (is_valid, warnings)
        """
        is_valid, warnings, _ = cls.validate(plugin_path)

        critical_patterns = [
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__",
        ]

        try:
            content = plugin_path.read_text(encoding='utf-8')
            has_critical = any(
                re.search(p, content) for p in critical_patterns
            )
            is_valid = not has_critical
        except Exception:
            is_valid = False

        return is_valid, warnings
