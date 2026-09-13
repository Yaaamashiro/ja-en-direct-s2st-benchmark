"""Implementation status is distinct from real-speech reproduction status."""


def require_implemented() -> None:
    from .model import Translatotron2  # Ensure the real neural dependency is available.
    assert Translatotron2 is not None


if __name__ == "__main__":
    require_implemented()
