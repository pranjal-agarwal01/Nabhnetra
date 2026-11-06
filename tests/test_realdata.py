import numpy as np

from nabhnetra.realdata import (confusion_by_bin, fit_logistic_pd, iou_xywh, kfold_by_group, logmel,
                               match_frame, normalise_rows, parse_audio_name, parse_video_name,
                               split_by_group)


def test_parse_names():
    assert parse_video_name("IR_DRONE_001") == ("IR", "drone", 1)
    assert parse_video_name("V_AIRPLANE_059") == ("V", "aircraft", 59)
    assert parse_video_name("IR_DRONE_001_LABELS") is None
    assert parse_audio_name("BACKGROUND_030") == ("background", 30)


def test_split_keeps_whole_videos_and_every_class():
    stems = [f"IR_{c}_{i:03d}" for c in ("DRONE", "BIRD") for i in range(10)]
    labels = [s.split("_")[1] for s in stems]
    train, val = split_by_group(stems, labels, 0.2, seed=1)
    assert not set(train) & set(val)
    assert len(train) + len(val) == len(stems)
    assert {s.split("_")[1] for s in val} == {"DRONE", "BIRD"}
    folds = kfold_by_group(stems, labels, k=5)
    assert set(folds.values()) == set(range(5))


def test_matching_and_confusion():
    assert np.isclose(iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
    assert iou_xywh((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    gt = (100, 100, 10, 10)
    preds = [((101, 100, 10, 10), 1, 0.6), ((300, 300, 10, 10), 0, 0.9)]
    assert match_frame(gt, preds) == 1
    assert match_frame(gt, preds[1:]) is None
    cells = confusion_by_bin([(0, 0, 5), (0, 1, 5), (1, None, 30)], bins_px=(8, 20))
    assert cells["distant"]["counts"][0].tolist() == [1, 1, 0, 0]
    assert cells["close"]["total"][1] == 1 and cells["close"]["detected"][1] == 0
    m = normalise_rows(cells["distant"]["counts"])
    assert np.allclose(m.sum(axis=1), 1.0)
    prior = np.full((4, 4), 0.1) + np.eye(4) * 0.6
    one_sample = np.zeros((4, 4))
    one_sample[0, 3] = 1  # a single, possibly unlucky, sample
    shrunk = normalise_rows(one_sample, prior=prior, strength=20)
    assert np.allclose(shrunk.sum(axis=1), 1.0)
    assert shrunk[0, 0] > shrunk[0, 3]  # one sample does not overturn the prior
    many = np.zeros((4, 4))
    many[0] = [0, 0, 0, 1000]
    assert normalise_rows(many, prior=prior, strength=20)[0, 3] > 0.95  # lots of data wins


def test_logistic_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    px = rng.uniform(0, 20, 4000)
    hit = rng.random(4000) < 1 / (1 + np.exp(-(px - 6.0) / 1.5))
    fit = fit_logistic_pd(px, hit)
    assert abs(fit["px50"] - 6.0) < 0.6 and abs(fit["px_scale"] - 1.5) < 0.5


def test_shrink_frame_moves_box_with_target():
    import pytest

    pytest.importorskip("cv2")
    from nabhnetra.realdata import shrink_frame

    img = np.zeros((256, 320, 3), np.uint8)
    img[100:110, 200:220] = 255  # a 20 x 10 px bright target
    small, (x, y, w, h) = shrink_frame(img, (200, 100, 20, 10), 0.5)
    assert small.shape == img.shape
    assert np.isclose(w, 10) and np.isclose(h, 5)
    patch = small[int(y):int(y + h), int(x):int(x + w)]
    assert patch.mean() > 150  # the box still covers the (now smaller) target


def test_clip_score_counts_found_and_correct():
    from nabhnetra.detect import ClipScore

    s = ClipScore()
    truth = ((100, 100, 10, 10), "drone")
    s.add([((101, 100, 10, 10), "drone", 0.9)], truth)   # found, right class
    s.add([((101, 100, 10, 10), "bird", 0.9)], truth)    # found, wrong class
    s.add([], truth)                                      # missed
    s.add([((5, 5, 4, 4), "bird", 0.5)], None)            # unlabelled frame
    assert (s.labelled, s.found, s.correct) == (3, 2, 1)
    assert s.unlabelled_detections == 1


def test_decide_threshold_and_vote():
    from nabhnetra.realdata import decide

    # one recording: background, a lone confident drone blip, then three drone windows in a row
    probs = np.array([[0.1, 0.1, 0.8], [0.9, 0.05, 0.05], [0.1, 0.1, 0.8],
                      [0.7, 0.1, 0.2], [0.8, 0.1, 0.1], [0.9, 0.05, 0.05]])
    groups = ["a"] * 6
    assert decide(probs, groups).tolist() == [2, 0, 2, 0, 0, 0]
    assert decide(probs, groups, threshold=0.75).tolist() == [2, 0, 2, 2, 0, 0]
    voted = decide(probs, groups, k=2, n=3)
    assert voted.tolist() == [2, 2, 2, 0, 0, 0]  # the lone blip is suppressed
    # windows of another recording never vote for this one
    assert decide(probs[:2], ["a", "b"], k=1, n=3).tolist() == [2, 0]


def test_logmel_shape():
    sr = 22050
    t = np.arange(sr) / sr
    m = logmel(np.sin(2 * np.pi * 440 * t), sr, n_mels=64)
    assert m.shape[0] == 64 and m.shape[1] > 50
    assert np.isfinite(m).all()
