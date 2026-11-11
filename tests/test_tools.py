"""The three tools, called directly without the agent."""
import json

import pytest

from helpdesk import data, tools


# ---------------------------------------------------------------- find_instructions

@pytest.mark.parametrize("topic,guide", [("wifi", "GUIDE-WIFI"), ("Wi-Fi", "GUIDE-WIFI"),
                                         ("password", "GUIDE-PASSWORD")])
def test_find_instructions_returns_local_guide(topic, guide):
    # Catches: wrong topic mapping, or steps that no longer come from helpdesk.json.
    r = tools.find_instructions(topic)
    key = "wifi" if guide == "GUIDE-WIFI" else "password"
    assert r["instruction_id"] == guide
    assert r["steps"] == data.helpdesk()["instructions"][key]["steps"]


def test_find_instructions_unknown_topic_is_an_error_not_a_guess():
    assert "error" in tools.find_instructions("printer")


# ------------------------------------------------------------- check_service_status

@pytest.mark.parametrize("profile", ["normal", "wifi_outage", "recovery_outage", "status_unavailable"])
@pytest.mark.parametrize("service", ["wifi", "account_recovery"])
def test_status_comes_from_the_selected_snapshot(profile, service):
    # Catches: ignoring the selected profile (always 'normal'), or mixing up services.
    r = tools.check_service_status(service, status_profile=profile)
    assert r["status"] == data.service_status()["profiles"][profile][service]["status"]
    assert r["repair_time"] is None


def test_password_topic_uses_account_recovery():
    assert tools.check_service_status("password", status_profile="recovery_outage")["status"] == "outage"


def test_unavailable_status_is_unknown_not_operational():
    r = tools.check_service_status("wifi", status_profile="status_unavailable")
    assert r["status"] == "unknown"


def test_bad_profile_or_service_is_an_error():
    assert "error" in tools.check_service_status("wifi", status_profile="tuesday")
    assert "error" in tools.check_service_status("printer")


# ------------------------------------------------------------------- prepare_ticket

def _ticket(tmp_path, **over):
    args = dict(category="wifi", summary="Campus-Demo will not connect.",
                steps_tried=["Enabled Wi-Fi", "Forgot the saved network"], user_id="demo-user-a",
                thread_id="t1", status_profile="normal", service_status="operational",
                reference_ids=["GUIDE-WIFI"], ticket_dir=tmp_path)
    args.update(over)
    return tools.prepare_ticket(**args)


def test_ticket_is_a_local_draft_in_template_shape(tmp_path):
    # Catches: missing template fields, invented steps, or a ticket that claims to be sent.
    r = _ticket(tmp_path)
    saved = json.loads(open(r["file"], encoding="utf-8").read())["ticket"]
    assert set(saved) == set(data.ticket_template()["ticket"])
    assert saved["status"] == "draft" and r["sent"] is False
    assert saved["steps_tried"] == ["Enabled Wi-Fi", "Forgot the saved network"]
    assert saved["user_id"] == "demo-user-a" and saved["thread_id"] == "t1"
    assert saved["service_status"] == "operational" and saved["ticket_id"].startswith("TKT-DEMO-")


def test_ticket_ids_are_unique(tmp_path):
    assert _ticket(tmp_path)["ticket_id"] != _ticket(tmp_path)["ticket_id"]


@pytest.mark.parametrize("over,missing", [({"summary": ""}, "problem description"),
                                          ({"steps_tried": []}, "steps already tried")])
def test_missing_details_mean_no_file(tmp_path, over, missing):
    # Catches: a ticket filled with invented details instead of asking the user.
    r = _ticket(tmp_path, **over)
    assert r["error"] == "missing_information" and missing in r["missing"]
    assert not list(tmp_path.glob("*.json"))


def test_ticket_refuses_secrets(tmp_path):
    r = _ticket(tmp_path, summary="My password is hunter2 and it fails")
    assert r["error"] == "secret_information"
    assert not list(tmp_path.glob("*.json"))


def test_unknown_service_status_becomes_not_checked(tmp_path):
    r = _ticket(tmp_path, service_status="probably fine")
    assert r["service_status"] == "not_checked"
