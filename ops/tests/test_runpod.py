import io
import json
import urllib.error
import urllib.parse

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


def test_list_reads_the_documented_object_shape():
    reply = b'{"pods": [{"id": "a"}], "pagination": {"nextCursor": null}}'
    assert runpod.list_pods("k", send=fake_send(reply)) == [{"id": "a"}]


def test_list_tolerates_a_bare_array():
    assert runpod.list_pods("k", send=fake_send(b'[{"id": "a"}]')) == [{"id": "a"}]


def test_list_follows_the_cursor_across_pages():
    pages = {
        None: b'{"pods": [{"id": "a"}], "pagination": {"nextCursor": "c1"}}',
        "c1": b'{"pods": [{"id": "b"}], "pagination": {"nextCursor": null}}',
    }
    seen = []

    def send(request, timeout=None):
        seen.append(request.full_url)
        cursor = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query).get("cursor", [None])[0]
        return Response(pages[cursor])

    assert runpod.list_pods("k", send=send) == [{"id": "a"}, {"id": "b"}]
    assert "cursor=c1" in seen[1] and "cursor" not in seen[0]


def test_list_stops_following_a_cursor_that_never_ends():
    reply = b'{"pods": [{"id": "a"}], "pagination": {"nextCursor": "again"}}'
    assert len(runpod.list_pods("k", send=fake_send(reply))) <= 50
