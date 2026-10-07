"""The audit is the evidence for a signed declaration, so both its false
positives and its true positives are pinned down here."""

import pytest

from evcharge.validation.pii_audit import audit_column, classify_structure


@pytest.mark.parametrize(
    "value, expected",
    [
        ("5c36621bf9af8b4639a8e0b5", "mongo_object_id"),
        ("000000333", "zero_padded_numeric_id"),
        ("1-1-179-794", "dashed_numeric_id"),
        ("1_1_179_794_2018-09-05 11:08:08.945820", "acn_session_id"),
        ("Wed, 05 Sep 2018 11:08:09 GMT", "http_date"),
        ("America/Los_Angeles", "iana_timezone"),
        (7.114, "numeric"),
        (True, "boolean"),
        ("ada@example.com", None),
    ],
)
def test_structure_classification(value, expected):
    assert classify_structure(value) == expected


@pytest.mark.parametrize(
    "name, values",
    [
        ("_id", ["5c36621bf9af8b4639a8e0b5", "5b36621bf9af8b4639a8e0d2"]),
        ("kWhDelivered", [7.114, 9.283000000000001, 6.046]),
        ("stationID", ["1-1-179-794", "2-39-83-131"]),
        ("spaceID", ["AG-3F20", "CA-304"]),
        ("timezone", ["America/Los_Angeles"]),
    ],
)
def test_structured_identifiers_are_not_flagged_as_pii(name, values):
    """Mongo IDs, float reprs and dashed station IDs are not phone or card numbers."""
    result = audit_column(name, values)
    assert result["direct_pii_matches"] == {}
    assert result["classification"] == "NON_IDENTIFYING"


def test_real_pii_is_still_detected():
    result = audit_column("notes", ["contact ada@example.com", "call 415-555-0123"])
    assert "email" in result["direct_pii_matches"]
    assert "phone" in result["direct_pii_matches"]
    assert result["classification"] == "DIRECT_PII"


def test_luhn_filters_digit_runs_but_keeps_real_card_numbers():
    not_a_card = audit_column("blob", ["id 1234567890123456 end"])
    assert "credit_card_like" not in not_a_card["direct_pii_matches"]

    card = audit_column("blob", ["pan 4242424242424242 end"])
    assert "credit_card_like" in card["direct_pii_matches"]


def test_user_id_is_reported_as_pseudonymous_not_anonymous():
    result = audit_column("userID", ["000000333", "000000371"])
    assert result["classification"] == "PSEUDONYMOUS_IDENTIFIER"
