from visualact_r.data.splits import assign_phash_clusters
from visualact_r.schemas import TaskRecord


def _record(image_id: str, phash: str) -> TaskRecord:
    return TaskRecord(
        image_id=image_id,
        image_path=f"/{image_id}",
        width=1,
        height=1,
        task_id="t",
        task_template_id="t",
        prompt="t",
        weather="clear",
        scene="test",
        timeofday="daytime",
        phash=phash,
    )


def test_complete_linkage_prevents_transitive_phash_chain_collapse():
    # A-B and B-C are within radius 1, but A-C are distance 2 and must not share a split cluster.
    records = [
        _record("a.png", "0000000000000000"),
        _record("b.png", "0000000000000001"),
        _record("c.png", "0000000000000003"),
    ]
    clustered = {
        record.image_id: record.phash_cluster_id for record in assign_phash_clusters(records, 1)
    }
    assert clustered["a.png"] == clustered["b.png"]
    assert clustered["c.png"] != clustered["a.png"]
