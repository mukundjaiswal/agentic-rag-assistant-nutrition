import pytest

from agentic_rag.ingestion.chunker import fixed_size_chunks


def test_short_text_yields_one_chunk():
    chunks = fixed_size_chunks("hello world", size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_long_text_is_split():
    chunks = fixed_size_chunks("a" * 250, size=100, overlap=0)
    assert len(chunks) == 3


def test_windows_overlap_so_a_boundary_does_not_orphan_a_sentence():
    text = "".join(str(i % 10) for i in range(300))
    chunks = fixed_size_chunks(text, size=100, overlap=20)
    assert chunks[0].text[-20:] == chunks[1].text[:20]


def test_offsets_are_recorded():
    chunks = fixed_size_chunks("x" * 300, size=100, overlap=0)
    assert [c.metadata["offset"] for c in chunks] == [0, 100, 200]


def test_empty_text_yields_nothing():
    assert fixed_size_chunks("   ", size=100, overlap=0) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (100, 100), (100, -1)])
def test_invalid_windows_are_rejected(size, overlap):
    with pytest.raises(ValueError, match=r"size|overlap"):
        fixed_size_chunks("text", size=size, overlap=overlap)
