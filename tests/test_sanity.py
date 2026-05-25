"""Smoke tests — verify the package's basic shape before any real tests exist.

These run early in Task 1.4 so `make check` (which calls pytest) does not exit 5
("no tests collected") on the empty scaffolding. The real test framework
(fixtures, parametrization, integration setup) lands in Task 1.5.
"""


def test_package_importable() -> None:
    """The sentry package must be importable — catches packaging-config breakage."""
    import sentry

    assert sentry.__name__ == "sentry"
