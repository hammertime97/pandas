"""Exception hierarchy for the clipper."""


class ClipperError(Exception):
    """Base class for every error raised by this package."""


class DependencyMissing(ClipperError):
    """A required external tool or optional Python package is not installed."""

    def __init__(self, what: str, how_to_install: str = "") -> None:
        message = f"{what} is required but was not found."
        if how_to_install:
            message += f" Install it with: {how_to_install}"
        super().__init__(message)
        self.what = what
        self.how_to_install = how_to_install


class IngestError(ClipperError):
    """The source video could not be resolved or downloaded."""


class TranscriptionError(ClipperError):
    """No usable transcript could be produced for the source."""


class RenderError(ClipperError):
    """ffmpeg failed while rendering a clip."""


class NoMomentsFound(ClipperError):
    """The source produced no candidate windows worth cutting."""
