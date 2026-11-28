import io
import json
import urllib.error

import pytest

from vektr_ops import runpod
from vektr_ops.runpod import RunpodError


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_send(answer=b"{}", error=None, seen=None):
    def send(request, timeout=None):
        if seen is not None:
            seen.append(request)
        if error is not None:
            raise urllib.error.HTTPError(request.full_url, error[0], "x", {}, io.BytesIO(error[1]))
        return Response(answer)

    return send


def test_create_posts_json_with_a_bearer_header():
    seen = []
    pod = runpod.create_pod("k123", {"name": "n"}, send=fake_send(b'{"id": "p1", "cost": 0.2}', seen=seen))
    request = seen[0]
    assert pod == {"id": "p1", "cost": 0.2}
    assert request.get_method() == "POST" and request.full_url == "https://api.runpod.io/v2/pods"
    assert request.get_header("Authorization") == "Bearer k123"
    assert json.loads(request.data) == {"name": "n"}


def test_create_error_carries_status_and_message():
    send = fake_send(error=(400, b'{"error": "no capacity"}'))
    with pytest.raises(RunpodError) as caught:
        runpod.create_pod("k", {}, send=send)
    assert caught.value.status == 400 and "no capacity" in caught.value.message


def test_get_of_a_deleted_pod_is_none():
    assert runpod.get_pod("k", "p1", send=fake_send(error=(404, b"{}"))) is None


def test_get_other_errors_raise():
    with pytest.raises(RunpodError) as caught:
        runpod.get_pod("k", "p1", send=fake_send(error=(403, b"{}")))
    assert caught.value.status == 403


def test_delete_treats_404_as_gone():
    seen = []
    runpod.delete_pod("k", "p1", send=fake_send(error=(404, b"{}"), seen=seen))
    runpod.delete_pod("k", "p1", send=fake_send(b"", seen=seen))
    assert [r.get_method() for r in seen] == ["DELETE", "DELETE"]


def test_list_returns_the_pods():
    assert runpod.list_pods("k", send=fake_send(b'[{"id": "a"}]')) == [{"id": "a"}]
