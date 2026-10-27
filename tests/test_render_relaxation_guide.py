"""Guard against vacuous or confounded solver-control figures."""
import numpy as np
import pytest


def test_solver_comparisons_isolate_controls_on_real_finite_geometry():
    from viz.render_relaxation_guide import collect_comparisons

    comparisons = collect_comparisons()
    assert {row["field"] for row in comparisons} == {
        "relax_iters", "relax_margin", "relax_smooth_passes",
    }
    for row in comparisons:
        samples = row["samples"]
        assert len(samples) >= 3
        assert len({sample["seed"] for sample in samples}) == 1
        assert all(sample["config"]["on_invalid"] == "keep" for sample in samples)
        first = samples[0]
        for sample in samples:
            changed = {key for key, value in sample["config"].items()
                       if value != first["config"][key]}
            assert changed <= {row["field"]}
            np.testing.assert_array_equal(sample["raw"], first["raw"])
            assert 3 <= sample["count"] <= sample["config"]["N_max"]
            assert sample["valid"] == (sample["fail_mask"] == 0)
            for field in ("center", "inner", "outer"):
                assert sample[field].shape == (sample["count"], 2)
                assert np.isfinite(sample[field]).all()
        # Repeated/default curves or a renderer that only relabels the same data
        # must fail. All rows share a fixed pre-solve count, so indices align.
        assert max(np.linalg.norm(sample["center"] - first["center"], axis=1).max()
                   for sample in samples[1:]) > 0.01
        if row["field"] == "relax_smooth_passes":
            assert all(sample["config"]["relax_smooth_spacing_iters"] == 0
                       for sample in samples)


def test_solver_control_manifest_records_measured_comparisons(tmp_path):
    import json
    import shutil
    import subprocess
    from viz import render_relaxation_guide as guide

    manifest = guide.render_guide(tmp_path)
    saved = json.loads((tmp_path / "relaxation-manifest.json").read_text())
    assert saved == manifest
    assert manifest["source"]["warp_version"]
    assert manifest["source"]["solver_sha256"]
    if (guide.ROOT / ".git").exists() and shutil.which("git"):
        commit = subprocess.run(["git", "rev-parse", "HEAD^{commit}"], cwd=guide.ROOT,
                                capture_output=True, text=True)
        if commit.returncode == 0:
            assert manifest["source"]["git_commit"] == commit.stdout.strip()
    assert manifest["selection"]["claims_yield"] is False
    for row in manifest["comparisons"]:
        assert (tmp_path / row["asset"]).is_file()
        assert row["max_center_displacement"] > 0.01
        assert all(sample["count"] > 3 for sample in row["samples"])
        assert all(sample["finite"] for sample in row["samples"])
        assert len({sample["seed"] for sample in row["samples"]}) == 1


@pytest.mark.parametrize("context", ("archive", "parent_repository", "git_unavailable"))
def test_renderer_retains_measured_provenance_without_a_source_repository(
        tmp_path, monkeypatch, context):
    import hashlib
    import json
    import shutil
    import subprocess
    from viz import render_relaxation_guide as guide

    if context == "parent_repository":
        git = shutil.which("git")
        if git is None:
            pytest.skip("the parent-repository counterexample requires Git")
        subprocess.run([git, "init", "--quiet", str(tmp_path)], check=True)
        (tmp_path / "unrelated.txt").write_text("unrelated parent repository\n")
        subprocess.run([git, "add", "unrelated.txt"], cwd=tmp_path, check=True)
        subprocess.run([git, "-c", "commit.gpgsign=false", "-c", "user.name=Antoine Richard", "-c",
                        "user.email=antoiner@nvidia.com", "commit", "--quiet", "-m",
                        "test fixture: unrelated parent"], cwd=tmp_path, check=True)

    archive = tmp_path / "source-archive"
    paths = ("track_gen/_src/warp_relax.py", "track_gen/seeds.py",
             "track_gen/ops/sampling.py", "track_gen/_src/warp_generate.py",
             "track_gen/_src/warp_pipeline.py")
    expected_hashes = {}
    for name in paths:
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(guide.ROOT / name, target)
        expected_hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    if context == "git_unavailable":
        # A metadata directory forces a Git probe, but no executable can be found.
        (archive / ".git").mkdir()
        missing_bin = tmp_path / "missing-bin"
        missing_bin.mkdir()
        monkeypatch.setenv("PATH", str(missing_bin))
    monkeypatch.setattr(guide, "ROOT", archive)

    output = tmp_path / "rendered"
    manifest = guide.render_guide(output)
    source = manifest["source"]
    assert source["git_commit"] is None
    assert source["runtime_reference"] is None
    assert source["runtime_worktree_changed"] is None
    assert source["solver_sha256"] == expected_hashes.pop("track_gen/_src/warp_relax.py")
    assert source["runtime_source_sha256"] == expected_hashes
    assert json.loads((output / "relaxation-manifest.json").read_text()) == manifest
    assert len(manifest["comparisons"]) == 3
    assert all((output / row["asset"]).is_file() for row in manifest["comparisons"])
