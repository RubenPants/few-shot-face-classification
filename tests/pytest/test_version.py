"""Test package version."""

from few_shot_face_classification import __version__


def test_version() -> None:
    """Test that the version string can be loaded."""
    assert isinstance(__version__, str)
