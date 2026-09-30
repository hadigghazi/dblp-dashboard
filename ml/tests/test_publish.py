"""
Publishing the dataset for download.

What is on the website is what people will cite and build on, so each way it could quietly be wrong
has a test: a checksum that does not match the bytes, a centrality table from a different graph, a
half-replaced folder, and anything from the models directory beyond the dataset becoming reachable.
"""
import hashlib
import json
import os
import stat

import pytest

from tests import test_pipeline as tp  # noqa: F401  (sets env, builds the serving db)
from ml import data  # noqa: E402
from ml.network import centrality as CE, export as EX, publish as PB  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    con, meta = data.connect()
    try:
        network = tmp_path_factory.mktemp("network-testfp0001")
        EX.export(con, meta, network)
    finally:
        con.close()
    centrality = tmp_path_factory.mktemp("centrality-testfp0001")
    CE.run(network, centrality, threads=1, seed=7, betweenness_samples=32, closeness_samples=4)
    return network, centrality


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_file_is_there_with_a_checksum_that_matches_its_bytes(built, tmp_path):
    network, centrality = built
    manifest = PB.publish(network, centrality, root=tmp_path)
    live = tmp_path / PB.SLUG
    assert {f["name"] for f in manifest["files"]} == {name for name, *_ in PB.files()}
    for f in manifest["files"]:
        on_disk = live / f["name"]
        assert on_disk.stat().st_size == f["bytes"], f["name"]
        assert sha(on_disk) == f["sha256"], f["name"]
    sums = dict(reversed(line.split("  ")) for line in (live / PB.CHECKSUMS).read_text().splitlines())
    assert sums == {f["name"]: f["sha256"] for f in manifest["files"]}
    assert json.loads((live / "manifest.json").read_text()) == manifest


def test_the_published_copy_is_the_exported_file(built, tmp_path):
    """The edge list on the website must be byte-for-byte the one the export wrote."""
    network, centrality = built
    PB.publish(network, centrality, root=tmp_path)
    name = f"{PB.config.NAME}.ungraph.txt.gz"
    assert sha(tmp_path / PB.SLUG / name) == sha(network / name)


def test_nothing_but_the_dataset_is_exposed(built, tmp_path):
    """The web container can read this folder. Anything else that landed in it would be public."""
    network, centrality = built
    (network / "stray-secret.txt").write_text("not for the website")
    try:
        PB.publish(network, centrality, root=tmp_path)
    finally:
        (network / "stray-secret.txt").unlink()
    published = {p.name for p in (tmp_path / PB.SLUG).iterdir()}
    assert published == {name for name, *_ in PB.files()} | {PB.CHECKSUMS, "manifest.json"}


def test_the_web_server_can_read_everything(built, tmp_path):
    """nginx runs as its own user, not as the one who published."""
    network, centrality = built
    PB.publish(network, centrality, root=tmp_path)
    live = (tmp_path / PB.SLUG).resolve()
    assert stat.S_IMODE(live.stat().st_mode) & 0o005 == 0o005, "others must be able to enter the folder"
    for f in live.iterdir():
        assert stat.S_IMODE(f.stat().st_mode) & 0o004, f"others must be able to read {f.name}"


def test_republishing_switches_versions_and_leaves_no_debris(built, tmp_path):
    network, centrality = built
    first = PB.publish(network, centrality, root=tmp_path)
    old_target = os.readlink(tmp_path / PB.SLUG)
    second = PB.publish(network, centrality, root=tmp_path)
    new_target = os.readlink(tmp_path / PB.SLUG)
    assert new_target != old_target
    assert not os.path.isabs(new_target), "relative, so it resolves inside the web container's mount"
    assert not (tmp_path / old_target).exists(), "the superseded version is removed"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == [], "no partial or temp debris"
    assert second["published_at"] >= first["published_at"]


def test_a_table_for_a_different_graph_is_refused_and_the_live_copy_is_untouched(built, tmp_path):
    network, centrality = built
    PB.publish(network, centrality, root=tmp_path)
    live_before = os.readlink(tmp_path / PB.SLUG)
    metrics_path = centrality / "metrics.json"
    original = metrics_path.read_text(encoding="utf-8")
    tampered = json.loads(original)
    tampered["edges"] = tampered["edges"] + 1
    metrics_path.write_text(json.dumps(tampered), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="different graph"):
            PB.publish(network, centrality, root=tmp_path)
    finally:
        metrics_path.write_text(original, encoding="utf-8")
    assert os.readlink(tmp_path / PB.SLUG) == live_before
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_a_missing_file_is_refused_before_anything_is_copied(built, tmp_path):
    network, centrality = built
    table = centrality / f"{PB.config.NAME}.centrality.tsv.gz"
    moved = table.with_suffix(".away")
    table.rename(moved)
    try:
        with pytest.raises(FileNotFoundError, match="centrality.tsv.gz"):
            PB.publish(network, centrality, root=tmp_path)
    finally:
        moved.rename(table)
    assert not (tmp_path / PB.SLUG).exists()


def test_the_manifest_says_what_the_page_needs(built, tmp_path):
    network, centrality = built
    manifest = PB.publish(network, centrality, root=tmp_path)
    assert manifest["scope"] == "all"
    assert manifest["summary"]["coauthorships"] > 0
    assert manifest["total_bytes"] == sum(f["bytes"] for f in manifest["files"] if f["kind"] == "data")
    assert all(f["description"] for f in manifest["files"])
    assert "CC0" in manifest["license"]


def test_the_datasheets_are_readable_pages_not_markdown_source(built, tmp_path):
    network, centrality = built
    PB.publish(network, centrality, root=tmp_path)
    page = (tmp_path / PB.SLUG / "datasheet-network.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert "<table>" in page and "<h2>" in page, "tables and headings are rendered"
    assert "\n## " not in page and "|---|" not in page, "no Markdown syntax left showing"
    assert 'href="/#network"' in page, "a way back to the site"


def test_the_exclusions_are_measured_not_described(built):
    network, _ = built
    stats = json.loads((network / "stats.json").read_text(encoding="utf-8"))
    b = stats["bins_effect"]
    assert b["author_slots"] > 0
    assert b["edges_with_bins"] >= stats["edges"], "keeping the bins can only add edges"
    text = (network / "README.md").read_text(encoding="utf-8")
    assert f"{b['author_slots']:,}" in text, "the datasheet states the measured size"
