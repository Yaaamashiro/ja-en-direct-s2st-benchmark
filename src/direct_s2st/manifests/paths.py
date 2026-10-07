"""Resolve corpus audio without basename searches or manifest-directory fallback."""
from pathlib import Path, PurePosixPath, PureWindowsPath


def resolve_audio_path(raw: str, corpus_root: Path) -> Path:
    from ..drive_staging import mapping, logical, local_path
    staged = mapping()
    root = Path(logical(corpus_root)) if staged else corpus_root.resolve()
    normalized = raw.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        raise ValueError(f"audio path traversal is forbidden: {raw}")
    absolute = PurePosixPath(normalized).is_absolute() or PureWindowsPath(raw).is_absolute()
    native = Path(raw)
    # Keep original logical paths in dataset identities, but validate only the
    # owned local copy. No remote stat is needed for every training example.
    candidates = [native] if absolute else [root / normalized]
    if absolute:
        for index in range(len(parts) - 3):
            if parts[index:index + 2] == ('audio', '16k') and parts[index + 2] in ('ja', 'en'):
                candidates += [root / Path(*parts[index:]), root / 'production' / Path(*parts[index:])]
    matches = {logical(p) for p in candidates if logical(p) in staged}
    if len(matches) > 1:
        raise ValueError(f'cannot uniquely rebase staged audio path {raw!r}')
    if matches:
        selected = Path(matches.pop())
        local_path(selected)
        return selected
    if absolute and native.is_absolute() and native.is_file():
        return native.resolve()
    if not absolute:
        if PureWindowsPath(raw).drive:
            raise ValueError(f"drive-relative audio path is forbidden: {raw}")
        candidate = (root / normalized).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"audio path escapes corpus_root: {raw}")
        if not candidate.is_file():
            raise FileNotFoundError(f"audio file does not exist: {candidate}")
        return candidate
    candidates = set()
    for index in range(len(parts) - 3):
        if parts[index:index + 2] != ("audio", "16k") or parts[index + 2] not in ("ja", "en"):
            continue
        suffix = Path(*parts[index:])
        for candidate in (root / suffix, root / "production" / suffix):
            resolved = candidate.resolve()
            if resolved.is_relative_to(root) and resolved.is_file():
                candidates.add(resolved)
    if len(candidates) != 1:
        raise ValueError(f"cannot uniquely rebase audio path {raw!r}: {len(candidates)} candidates under {root}")
    return candidates.pop()
