"""MAS dataset mapping for the fidgeting prototype.

Single source of truth for the label policy and the clip -> source-video
group mapping used by BOTH the landmark extractor and the trainer, so they
can never disagree.
"""
import json
import os

from . import paths


def load_annotations():
    """Return {clip_filename: [labels...]} from MAS annotations.json."""
    with open(paths.MAS_ANNOTATIONS, "r", encoding="utf-8") as f:
        return json.load(f)


def load_clip_to_source():
    """Return {clip_filename: source_video_name} built from jsons/videos.json.

    videos.json layout: {source_name: [url, [[start, dur, clipname], ...]]}
    """
    with open(paths.MAS_VIDEOS_JSON, "r", encoding="utf-8") as f:
        videos = json.load(f)
    clip2src = {}
    for source, (_url, segments) in videos.items():
        for seg in segments:
            clip2src[seg[2]] = source
    return clip2src


def list_local_clips():
    """Filenames of clips actually present on disk (never fabricate)."""
    return sorted(os.listdir(paths.MAS_CLIPS_DIR))


def usable_dataset():
    """Return (pos_clips, neg_clips, ignored_clips, clip_labels, clip2src).

    Label policy (see README):
      POSITIVE/FIDGET : annotation contains 'body stress'
      NEGATIVE        : annotation == ['no stress']
      IGNORED         : any local clip carrying stress labels but no body
                        stress (e.g. ['audio stress']) -- these are NOT
                        negatives. Clips absent from annotations or without a
                        source mapping are also ignored (not usable).
    """
    annotations = load_annotations()
    clip2src = load_clip_to_source()
    local = set(list_local_clips())

    pos, neg, ignored = [], [], []
    for clip in sorted(local):
        ann = annotations.get(clip)
        if ann is None or clip not in clip2src:
            ignored.append(clip)
            continue
        if "body stress" in ann:
            pos.append(clip)
        elif ann == ["no stress"]:
            neg.append(clip)
        else:
            ignored.append(clip)
    return pos, neg, ignored, annotations, clip2src


def source_group_stats(clips, clip2src):
    """Count distinct source videos among the given clips."""
    return len(set(clip2src[c] for c in clips))
