"""Turn long viral videos into vertical short-form clips.

The package is deliberately layered so that each stage can be used on its
own:

``clipper.ingest``      resolve a URL or local path into a media file
``clipper.transcribe``  produce word level timestamps
``clipper.segment``     group words into sentences and candidate windows
``clipper.scoring``     rank candidate windows by predicted virality
``clipper.select``      pick a non-overlapping, diverse set of clips
``clipper.reframe``     plan a 9:16 crop path that follows the subject
``clipper.subtitles``   build word-highlight (karaoke) captions
``clipper.render``      burn everything into a platform ready MP4
``clipper.pipeline``    glue all of the above together
"""

from clipper.models import Clip, ScoreBreakdown, Sentence, Transcript, Word

__version__ = "0.1.0"

__all__ = ["Clip", "ScoreBreakdown", "Sentence", "Transcript", "Word", "__version__"]
