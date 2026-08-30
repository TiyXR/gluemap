from gluemap.estimators.track_inference import TrackInference as TrackInference

__all__ = ["TrackInference", "TrackSnapping"]


def __getattr__(name: str):
    if name == "TrackSnapping":
        from gluemap.estimators.track_snapping import TrackSnapping

        return TrackSnapping
    raise AttributeError(name)
