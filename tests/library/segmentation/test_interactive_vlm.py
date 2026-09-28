"""VLM grounding output parsing and request handling, with Ollama stubbed out."""
import json
from unittest.mock import patch

import pytest

PIL = pytest.importorskip("PIL.Image")
from viame.segmentation import interactive_vlm as vlm  # noqa: E402


def test_qwen_fenced_list_is_rescaled_from_1000():
    text = '```json\n[{"bbox_2d": [100, 200, 500, 1000], "label": "fish"}]\n```'
    assert vlm.parse_grounding(text, 2000, 500) == [([200.0, 100.0, 1000.0, 500.0], "fish")]


def test_wrapped_object_thinking_and_bad_entries():
    text = ('<think>[not json]</think>{"objects": [{"bbox_2d": [0, 0, 10, 10]}, '
            '{"bbox_2d": [5, 5]}, "junk", {"box": [900, 900, 1200, 1100], "label": "crab"}]}')
    assert vlm.parse_grounding(text, 1000, 1000) == [
        ([0.0, 0.0, 10.0, 10.0], None),
        ([900.0, 900.0, 1000.0, 1000.0], "crab"),
    ]


def test_swapped_corners_and_degenerate_boxes():
    assert vlm.parse_grounding('[{"bbox_2d": [500, 500, 100, 100]}]', 1000, 1000) == [
        ([100.0, 100.0, 500.0, 500.0], None)]
    assert vlm.parse_grounding('[{"bbox_2d": [10, 10, 10, 400]}]', 1000, 1000) == []


def test_no_json_gives_no_boxes():
    assert vlm.parse_grounding("There are no fish here.", 100, 100) == []


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "frame.png"
    PIL.new("RGB", (400, 200)).save(path)
    return str(path)


def test_detect_labels_fall_back_to_query(image_path):
    service = vlm.VlmService()
    reply = '[{"bbox_2d": [0, 0, 500, 500]}, {"bbox_2d": [500, 500, 1000, 1000], "label": "eel"}]'
    with patch.object(vlm, "ollama_chat", return_value=reply) as chat:
        result = service.handle_request({
            "command": "vlm_detect", "model": "m", "image_path": image_path,
            "text": "fish", "max_detections": 1,
        })
    assert result["detections"] == [
        {"box": [0.0, 0.0, 200.0, 100.0], "bounds": [0.0, 0.0, 200.0, 100.0],
         "score": 1.0, "label": "fish"}]
    assert '"fish"' in chat.call_args[0][1][0]["content"]


def test_ask_sends_history_and_crops(image_path):
    service = vlm.VlmService()
    with patch.object(vlm, "ollama_chat", return_value="<think>hmm</think> A cod.") as chat:
        result = service.handle_request({
            "command": "vlm_ask", "model": "m", "question": "Species?",
            "images": [{"image_path": image_path, "box": [10, 10, 50, 50]},
                       {"image_path": image_path}],
            "history": [{"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello"}],
        })
    assert result == {"success": True, "answer": "A cod."}
    messages = chat.call_args[0][1]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert len(messages[-1]["images"]) == 2


def test_chat_posts_to_ollama_and_passes_think():
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"message": {"content": "ok"}}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        return Response()

    with patch.object(vlm.urllib.request, "urlopen", fake_urlopen), \
            patch.dict(vlm.os.environ, {"OLLAMA_HOST": "http://host:1/"}):
        answer = vlm.ollama_chat("qwen3-vl:8b", [{"role": "user", "content": "q"}], think=False)
    assert answer == "ok"
    assert captured["url"] == "http://host:1/api/chat"
    assert captured["body"]["think"] is False
    assert captured["body"]["stream"] is False


@pytest.mark.parametrize("host, expected", [
    ("", "http://localhost:11434"),
    ("0.0.0.0", "http://localhost:11434"),
    ("0.0.0.0:8080", "http://localhost:8080"),
    ("gpu-box", "http://gpu-box:11434"),
    ("https://ollama.example.com", "https://ollama.example.com"),
    ("http://gpu-box:9000/", "http://gpu-box:9000"),
])
def test_ollama_url_follows_ollama_host(host, expected):
    assert vlm.ollama_url(host) == expected
