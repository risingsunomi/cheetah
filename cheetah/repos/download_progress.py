from __future__ import annotations


class DownloadProgress:
    def __init__(self, label: str = "download") -> None:
        self.label = label
        self._frames = ("|", "/", "-", "\\")
        self._idx = 0

    def render(
        self,
        action: str,
        current: int,
        total: int,
        filename: str,
        *,
        downloaded_bytes: int | None = None,
        total_bytes: int | None = None,
    ) -> str:
        frame = self._frames[self._idx % len(self._frames)]
        self._idx += 1

        message = f"[{self.label} {frame}] {action} {current}/{total}: {filename}"
        if downloaded_bytes is None:
            return message

        if total_bytes is not None and total_bytes > 0:
            percent = min(100.0, (float(downloaded_bytes) / float(total_bytes)) * 100.0)
            return (
                f"{message} "
                f"({percent:5.1f}% {self._format_bytes(downloaded_bytes)}/{self._format_bytes(total_bytes)})"
            )

        return f"{message} ({self._format_bytes(downloaded_bytes)})"

    @staticmethod
    def _format_bytes(value: int) -> str:
        size = float(max(0, int(value)))
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PiB"
