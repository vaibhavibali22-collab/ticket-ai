import random

import pytest

from nlp_engine import find_similar_tickets, ticket_text, tokenize
from seed_db import SEED_TICKETS


def _history():
    return [{"id": i + 1, "category": c, "priority": p, "title": t,
             "description": d, "resolution": r}
            for i, (c, p, t, d, r) in enumerate(SEED_TICKETS)]


def _ticket(i, title, desc, res="Restart the device and try again."):
    return {"id": i, "category": "Other", "priority": "Low", "title": title,
            "description": desc, "resolution": res}


def test_tokenize_stems_and_drops_stop_words():
    assert tokenize("The VPN keeps disconnecting") == ["vpn", "keep", "disconnect"]


def test_vpn_query_ranks_vpn_ticket_first():
    query = ticket_text("VPN keeps dropping",
                        "My VPN disconnects every few minutes when working from home")
    results = find_similar_tickets(query, _history())
    assert results[0]["title"] == "VPN disconnects frequently"
    assert results[0]["resolution"]  # historical resolution is returned


@pytest.mark.parametrize("title,desc,expected", [
    ("Printer offline", "The printer on floor 3 shows offline", "Printer shows offline"),
    ("Forgot password", "I forgot my login password", "Forgot password"),
    ("No storage", "Not enough disk space left on C drive", "Disk space full"),
    ("Mic problem", "Nobody can hear me on Teams calls, microphone issue",
     "Microphone not working in Teams"),
])
def test_common_issues_rank_expected_ticket_first(title, desc, expected):
    results = find_similar_tickets(ticket_text(title, desc), _history())
    assert results and results[0]["title"] == expected


def test_exact_text_scores_highest():
    hist = _history()
    target = hist[20]
    results = find_similar_tickets(ticket_text(target["title"], target["description"]), hist)
    assert results[0]["id"] == target["id"]
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-3)


def test_results_sorted_descending_and_limited():
    results = find_similar_tickets("printer printing offline paper jam", _history(),
                                   top_k=2, threshold=0.0)
    assert len(results) == 2
    scores = [r["similarity"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_returns_fewer_than_top_k_when_few_records():
    hist = [_ticket(1, "Printer jam", "Paper stuck in printer tray")]
    assert len(find_similar_tickets("printer paper stuck", hist, top_k=5)) == 1


def test_threshold_filters_irrelevant():
    for text in ["The coffee machine in the pantry is leaking water",
                 "Need a parking spot allocation for next month",
                 "My salary slip is wrong this month"]:
        assert find_similar_tickets(text, _history()) == []


def test_threshold_is_configurable():
    query = ticket_text("Laptop slow", "computer is extremely slow")
    low = find_similar_tickets(query, _history(), threshold=0.05)
    high = find_similar_tickets(query, _history(), threshold=0.9)
    assert len(low) > 0 and high == []
    assert all(r["similarity"] >= 0.05 for r in low)


def test_empty_query_raises():
    for bad in ["", "   ", None]:
        with pytest.raises(ValueError):
            find_similar_tickets(bad, _history())


def test_empty_history_returns_no_matches():
    assert find_similar_tickets("vpn not connecting", []) == []


def test_query_with_only_stop_words_returns_no_matches():
    assert find_similar_tickets("the and of it is", _history()) == []


def test_ties_broken_by_ticket_id():
    hist = [_ticket(7, "Scanner jam", "jam"), _ticket(3, "Printer jam", "jam"),
            _ticket(5, "Copier jam", "jam")]
    first = find_similar_tickets("jam", hist, threshold=0.0)
    assert len({r["similarity"] for r in first}) == 1  # genuinely tied
    assert [r["id"] for r in first] == [3, 5, 7]
    shuffled = hist[:]
    random.Random(1).shuffle(shuffled)
    assert [r["id"] for r in find_similar_tickets("jam", shuffled, threshold=0.0)] == [3, 5, 7]


def test_duplicates_collapsed():
    dup = _ticket(1, "VPN not connecting", "VPN client times out")
    hist = [dup, {**dup, "id": 2}, {**dup, "id": 3, "title": "  vpn NOT connecting "},
            _ticket(4, "VPN slow", "VPN client is slow")]
    results = find_similar_tickets("vpn client times out", hist, threshold=0.0)
    ids = [r["id"] for r in results]
    assert ids.count(1) == 1 and 2 not in ids and 3 not in ids
