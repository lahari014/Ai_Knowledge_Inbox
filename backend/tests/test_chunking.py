from app.services.chunking import Chunker


def test_blank_text_has_no_chunks() -> None:
    assert Chunker().split("  \n\n\t") == []


def test_short_text_stays_in_one_chunk() -> None:
    chunks = Chunker().split("Hello world.")

    assert [(chunk.index, chunk.text) for chunk in chunks] == [(0, "Hello world.")]


def test_chunks_stay_within_size_and_repeat_only_the_configured_tail() -> None:
    text = " ".join(f"s{index}." for index in range(100))
    chunker = Chunker(chunk_size=80, overlap=15)

    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert all(len(chunk.text) <= 80 for chunk in chunks)
    for left, right in zip(chunks, chunks[1:]):
        assert right.text.startswith(left.text[-15:])


def test_paragraph_order_is_preserved() -> None:
    chunks = Chunker().split("Alpha is first.\n\nBeta is second.")

    assert len(chunks) == 1
    assert chunks[0].text.index("Alpha") < chunks[0].text.index("Beta")


def test_oversized_text_is_split_on_character_boundaries() -> None:
    chunks = Chunker(chunk_size=10, overlap=0).split("abcdefghijklmnop")

    assert [chunk.text for chunk in chunks] == ["abcdefghij", "klmnop"]
    assert all(len(chunk.text) <= 10 for chunk in chunks)
