"""A query over several exemplars (e.g. frames of one track) sends every
exemplar's descriptors as one query to every open index."""
from viame.descriptors.query_service import QueryService


class FakeSession:
    def __init__(self, name, auto_results):
        self.index_dir = name
        self.auto_results = auto_results
        self.formulated = []
        self.queried = []

    def reset_query_state(self):
        pass

    def formulate(self, image_path, boxes):
        self.formulated.append((image_path, boxes))
        return {"results": list(self.auto_results), "descriptors": [f"d:{image_path}"]}

    def process_query(self, descriptors, threshold, model):
        self.queried.append(list(descriptors))
        return {"results": []}


def _service(tmp_path, auto_results):
    service = QueryService(None)
    primary = FakeSession("primary", auto_results)
    other = FakeSession("other", [])
    service._sessions = [primary, other]
    return service, primary, other


def _images(tmp_path, count):
    paths = []
    for i in range(count):
        path = tmp_path / f"f{i}.png"
        path.write_bytes(b"")
        paths.append(str(path))
    return paths


def test_exemplars_join_one_query(tmp_path):
    service, primary, other = _service(tmp_path, auto_results=["ignored"])
    paths = _images(tmp_path, 3)
    response = service.handle_request({
        "command": "formulate_query",
        "image_path": paths[0],
        "exemplars": [{"image_path": p, "boxes": [[0, 0, 5, 5]]} for p in paths],
    })
    assert [f[0] for f in primary.formulated] == paths
    expected = [f"d:{p}" for p in paths]
    assert primary.queried == [expected]
    assert other.queried == [expected]
    assert response["descriptor_count"] == 3


def test_single_image_without_auto_results_queries_every_index(tmp_path):
    service, primary, other = _service(tmp_path, auto_results=[])
    [path] = _images(tmp_path, 1)
    service.handle_request({
        "command": "formulate_query", "image_path": path, "boxes": [[0, 0, 5, 5]],
    })
    assert primary.formulated == [(path, [[0, 0, 5, 5]])]
    assert primary.queried == [[f"d:{path}"]]
    assert other.queried == [[f"d:{path}"]]
