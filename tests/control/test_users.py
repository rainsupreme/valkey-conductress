"""Tests for the user/agent identity directory."""

import pytest

from conductress.control.users import UserDirectory


def _seed():
    return {
        "users": [
            {
                "login": "rain",
                "github": "rainsupreme",
                "kind": "human",
                "role": "owner",
                "quota_runner_minutes_per_day": 1440,
                "sources": ["valkey", "valkey-rainfall"],
            },
            {
                "login": "dante",
                "github": "xdk-amz",
                "kind": "human",
                "role": "collaborator",
                "quota_runner_minutes_per_day": 240,
            },
            # Agents name only a login and a sponsor; everything else is inherited.
            {"login": "rimuru", "kind": "agent", "sponsor": "rain"},
            {"login": "feraligatr", "kind": "agent", "sponsor": "dante"},
        ]
    }


def test_directory_parses_seed_and_roles():
    directory = UserDirectory.from_dict(_seed())
    assert directory.require("rain").is_owner is True
    assert directory.require("rain").is_approver is True  # owner is implicitly approver
    assert directory.require("dante").is_owner is False
    assert directory.require("dante").is_approver is False
    assert directory.require("rimuru").kind == "agent"
    assert directory.require("rimuru").sponsor == "rain"


def test_agent_inherits_everything_from_sponsor():
    directory = UserDirectory.from_dict(_seed())
    rain = directory.require("rain")
    rimuru = directory.require("rimuru")
    assert rimuru.github == rain.github == "rainsupreme"
    assert rimuru.role == rain.role == "owner"
    assert rimuru.is_owner is True and rimuru.is_approver is True
    assert rimuru.quota_runner_minutes_per_day == rain.quota_runner_minutes_per_day == 1440
    assert rimuru.sources == rain.sources == ("valkey", "valkey-rainfall")

    dante = directory.require("dante")
    feraligatr = directory.require("feraligatr")
    assert feraligatr.github == "xdk-amz"
    assert feraligatr.role == dante.role == "collaborator"
    assert feraligatr.is_owner is False
    assert feraligatr.quota_runner_minutes_per_day == 240
    assert feraligatr.sources == ()


def test_agent_quota_charges_sponsor_account():
    directory = UserDirectory.from_dict(_seed())
    assert directory.quota_account("rimuru") == "rainsupreme"
    assert directory.quota_minutes("rimuru") == 1440
    assert directory.quota_account("feraligatr") == "xdk-amz"
    assert directory.quota_minutes("feraligatr") == 240
    assert directory.quota_account("dante") == "xdk-amz"
    assert directory.quota_minutes("dante") == 240


@pytest.mark.parametrize(
    "field, value",
    [
        ("github", "someone-else"),
        ("role", "owner"),
        ("quota_runner_minutes_per_day", 99999),
        ("sources", ["valkey-rainfall"]),
    ],
)
def test_rejects_agent_that_sets_inherited_field(field, value):
    data = _seed()
    data["users"][3][field] = value  # feraligatr tries to widen its own grant
    with pytest.raises(ValueError, match=f"inherits .*{field}.* from its sponsor"):
        UserDirectory.from_dict(data)


def test_rejects_agent_without_sponsor():
    data = _seed()
    data["users"][2].pop("sponsor")
    with pytest.raises(ValueError, match="must name a sponsor"):
        UserDirectory.from_dict(data)


def test_rejects_human_with_sponsor():
    data = _seed()
    data["users"][1]["sponsor"] = "rain"
    with pytest.raises(ValueError, match="must not name a sponsor"):
        UserDirectory.from_dict(data)


def test_rejects_sponsor_that_is_not_human():
    data = _seed()
    # Point rimuru's sponsor at another agent.
    data["users"][2]["sponsor"] = "feraligatr"
    with pytest.raises(ValueError, match="must be a human"):
        UserDirectory.from_dict(data)


def test_rejects_unknown_sponsor():
    data = _seed()
    data["users"][2]["sponsor"] = "ghost"
    with pytest.raises(ValueError, match="not a known user"):
        UserDirectory.from_dict(data)


def test_agent_may_precede_sponsor_in_file():
    data = _seed()
    data["users"].reverse()  # agents now listed before the humans they act for
    directory = UserDirectory.from_dict(data)
    assert directory.require("rimuru").github == "rainsupreme"


def test_rejects_unknown_fields_and_bad_role():
    data = _seed()
    data["users"][0]["extra"] = "nope"
    with pytest.raises(ValueError, match="unknown user fields"):
        UserDirectory.from_dict(data)
    data = _seed()
    data["users"][2]["extra"] = "nope"
    with pytest.raises(ValueError, match="unknown user fields"):
        UserDirectory.from_dict(data)
    data = _seed()
    data["users"][0]["role"] = "wizard"
    with pytest.raises(ValueError, match="role must be"):
        UserDirectory.from_dict(data)


def test_rejects_duplicate_login():
    data = _seed()
    data["users"].append(dict(data["users"][0]))
    with pytest.raises(ValueError, match="duplicate user login"):
        UserDirectory.from_dict(data)
    data = _seed()
    data["users"].append({"login": "rain", "kind": "agent", "sponsor": "dante"})
    with pytest.raises(ValueError, match="duplicate user login"):
        UserDirectory.from_dict(data)


def test_from_file_reads_toml(tmp_path):
    path = tmp_path / "users.toml"
    path.write_text(
        "\n".join(
            [
                "[[users]]",
                'login = "rain"',
                'github = "rainsupreme"',
                'kind = "human"',
                'role = "owner"',
                "quota_runner_minutes_per_day = 60",
                'sources = ["valkey"]',
                "",
                "[[users]]",
                'login = "rimuru"',
                'kind = "agent"',
                'sponsor = "rain"',
            ]
        ),
        encoding="utf-8",
    )
    directory = UserDirectory.from_file(path)
    assert directory.require("rain").github == "rainsupreme"
    assert directory.require("rain").sources == ("valkey",)
    assert directory.require("rimuru").sources == ("valkey",)
    assert directory.require("rimuru").is_owner is True
