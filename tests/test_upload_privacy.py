"""Unit tests for YouTube Shorts upload privacy status resolution and propagation.

Tests:
1. Default without env -> 'private' (sends privacyStatus: 'private')
2. YOUTUBE_PRIVACY_STATUS=public -> sends privacyStatus: 'public'
3. YOUTUBE_PRIVACY_STATUS=unlisted -> sends privacyStatus: 'unlisted'
4. Invalid value (e.g. YOUTUBE_PRIVACY_STATUS=invalid) -> safely fallbacks to 'private'
5. Parameter fallback when env is invalid or unset
6. Case and whitespace normalization
"""

import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from upload.uploader import (
    StrictUploadGate,
    UploadGateCheck,
    VALID_PRIVACY_STATUSES,
    YouTubeShortsUploader,
)


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    """Creates a dummy video file > 100KB for upload validation."""
    video = tmp_path / "test_privacy_short.mp4"
    video.write_bytes(b"\x00" * (128 * 1024))
    return video


@pytest.fixture
def valid_gate_check() -> UploadGateCheck:
    """Returns an approved UploadGateCheck with all 8 gates True."""
    return StrictUploadGate.evaluate(
        language_gate=True,
        semantic_clip_gate=True,
        visual_viability_gate=True,
        boundary_gate=True,
        render_success=True,
        technical_qc=True,
        visual_qc=True,
        perceptual_qc=True,
    )


# ==============================================================================
# 1. Pure Privacy Status Resolution Tests
# ==============================================================================

def test_resolve_privacy_default_without_env(monkeypatch):
    """Scenario 1 (pure): Without env var, defaults to 'private'."""
    monkeypatch.delenv("YOUTUBE_PRIVACY_STATUS", raising=False)
    assert YouTubeShortsUploader.resolve_privacy_status() == "private"
    assert YouTubeShortsUploader.resolve_privacy_status("private") == "private"


def test_resolve_privacy_public_via_env(monkeypatch):
    """Scenario 2 (pure): YOUTUBE_PRIVACY_STATUS=public resolves to 'public'."""
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "public")
    assert YouTubeShortsUploader.resolve_privacy_status() == "public"
    # Env takes precedence over parameter
    assert YouTubeShortsUploader.resolve_privacy_status("private") == "public"


def test_resolve_privacy_unlisted_via_env(monkeypatch):
    """Scenario 3 (pure): YOUTUBE_PRIVACY_STATUS=unlisted resolves to 'unlisted'."""
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "unlisted")
    assert YouTubeShortsUploader.resolve_privacy_status() == "unlisted"
    # Env takes precedence over parameter
    assert YouTubeShortsUploader.resolve_privacy_status("private") == "unlisted"


def test_resolve_privacy_invalid_env_fallback(monkeypatch):
    """Scenario 4 (pure): Invalid env var safely fallbacks to parameter or 'private'."""
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "invalid_status")
    # Default parameter is private
    assert YouTubeShortsUploader.resolve_privacy_status() == "private"
    # When parameter is valid, falls back to parameter
    assert YouTubeShortsUploader.resolve_privacy_status("unlisted") == "unlisted"
    # When parameter is also invalid, falls back to 'private'
    assert YouTubeShortsUploader.resolve_privacy_status("totally_broken") == "private"


def test_resolve_privacy_normalization(monkeypatch):
    """Ensures whitespace and casing are stripped and lowercased."""
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "  PUBLIC  ")
    assert YouTubeShortsUploader.resolve_privacy_status() == "public"

    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", " Unlisted\n")
    assert YouTubeShortsUploader.resolve_privacy_status() == "unlisted"

    monkeypatch.delenv("YOUTUBE_PRIVACY_STATUS", raising=False)
    assert YouTubeShortsUploader.resolve_privacy_status("  UNLISTED  ") == "unlisted"


# ==============================================================================
# 2. Integration with upload_short (Execution & Payload Verification)
# ==============================================================================

def test_upload_short_default_without_env_sends_private(
    dummy_video: Path, valid_gate_check: UploadGateCheck, monkeypatch
):
    """Scenario 1: Default without env -> sends privacyStatus: 'private'."""
    monkeypatch.delenv("YOUTUBE_PRIVACY_STATUS", raising=False)
    uploader = YouTubeShortsUploader()

    with patch.object(uploader, "_execute_oauth_upload") as mock_oauth:
        mock_oauth.return_value = {
            "status": "success",
            "video_id": "test_vid_01",
            "url": "https://youtube.com/shorts/test_vid_01",
            "privacy_status": "private",
        }

        res = uploader.upload_short(
            video_path=dummy_video,
            title="Video Default Privacy",
            description="Testing default privacy status",
            gate_check=valid_gate_check,
            dry_run=False,
        )

        assert res["status"] == "success"
        mock_oauth.assert_called_once()
        assert mock_oauth.call_args.kwargs["privacy_status"] == "private"


def test_upload_short_env_public_sends_public(
    dummy_video: Path, valid_gate_check: UploadGateCheck, monkeypatch
):
    """Scenario 2: YOUTUBE_PRIVACY_STATUS=public -> sends privacyStatus: 'public'."""
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "public")
    uploader = YouTubeShortsUploader()

    with patch.object(uploader, "_execute_oauth_upload") as mock_oauth:
        mock_oauth.return_value = {
            "status": "success",
            "video_id": "test_vid_02",
            "url": "https://youtube.com/shorts/test_vid_02",
            "privacy_status": "public",
        }

        res = uploader.upload_short(
            video_path=dummy_video,
            title="Video Public Privacy",
            description="Testing public privacy via env",
            gate_check=valid_gate_check,
            dry_run=False,
        )

        assert res["status"] == "success"
        mock_oauth.assert_called_once()
        assert mock_oauth.call_args.kwargs["privacy_status"] == "public"


def test_upload_short_env_unlisted_sends_unlisted(
    dummy_video: Path, valid_gate_check: UploadGateCheck, monkeypatch
):
    """Scenario 3: YOUTUBE_PRIVACY_STATUS=unlisted -> sends privacyStatus: 'unlisted'."""
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "unlisted")
    uploader = YouTubeShortsUploader()

    with patch.object(uploader, "_execute_oauth_upload") as mock_oauth:
        mock_oauth.return_value = {
            "status": "success",
            "video_id": "test_vid_03",
            "url": "https://youtube.com/shorts/test_vid_03",
            "privacy_status": "unlisted",
        }

        res = uploader.upload_short(
            video_path=dummy_video,
            title="Video Unlisted Privacy",
            description="Testing unlisted privacy via env",
            gate_check=valid_gate_check,
            dry_run=False,
        )

        assert res["status"] == "success"
        mock_oauth.assert_called_once()
        assert mock_oauth.call_args.kwargs["privacy_status"] == "unlisted"


def test_upload_short_invalid_env_fallbacks_to_private(
    dummy_video: Path, valid_gate_check: UploadGateCheck, monkeypatch
):
    """Scenario 4: YOUTUBE_PRIVACY_STATUS=invalid -> safely fallbacks to 'private'."""
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "invalid_status_xyz")
    uploader = YouTubeShortsUploader()

    with patch.object(uploader, "_execute_oauth_upload") as mock_oauth:
        mock_oauth.return_value = {
            "status": "success",
            "video_id": "test_vid_04",
            "url": "https://youtube.com/shorts/test_vid_04",
            "privacy_status": "private",
        }

        res = uploader.upload_short(
            video_path=dummy_video,
            title="Video Invalid Fallback Privacy",
            description="Testing invalid env fallback to default private",
            gate_check=valid_gate_check,
            dry_run=False,
        )

        assert res["status"] == "success"
        mock_oauth.assert_called_once()
        assert mock_oauth.call_args.kwargs["privacy_status"] == "private"


def test_dry_run_includes_effective_privacy(
    dummy_video: Path, valid_gate_check: UploadGateCheck, monkeypatch
):
    """Verifies dry-run returns effective privacy status reflecting env."""
    uploader = YouTubeShortsUploader()

    # Default
    monkeypatch.delenv("YOUTUBE_PRIVACY_STATUS", raising=False)
    res_def = uploader.upload_short(
        video_path=dummy_video,
        title="Dry Run Default",
        description="Dry run default",
        gate_check=valid_gate_check,
        dry_run=True,
    )
    assert res_def["status"] == "success"
    assert res_def["privacy_status"] == "private"

    # Public env
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "public")
    res_pub = uploader.upload_short(
        video_path=dummy_video,
        title="Dry Run Public",
        description="Dry run public",
        gate_check=valid_gate_check,
        dry_run=True,
    )
    assert res_pub["status"] == "success"
    assert res_pub["privacy_status"] == "public"


# ==============================================================================
# 3. Direct YouTube API v3 Body Verification (_execute_oauth_upload)
# ==============================================================================

@patch("googleapiclient.discovery.build")
@patch("google.oauth2.credentials.Credentials.from_authorized_user_file")
def test_oauth_upload_sends_exact_privacystatus_in_youtube_api_body(
    mock_creds_cls, mock_build, dummy_video: Path, tmp_path: Path
):
    """Verifies YouTube API videos().insert call receives exact privacyStatus in body."""
    # Mock credentials file
    token_file = tmp_path / "youtube_token.json"
    token_file.write_text('{"token": "fake_token"}')

    uploader = YouTubeShortsUploader(credentials_file=token_file)

    mock_creds = MagicMock()
    mock_creds.valid = True
    mock_creds_cls.return_value = mock_creds

    mock_youtube = MagicMock()
    mock_insert_req = MagicMock()
    mock_insert_req.next_chunk.return_value = (MagicMock(progress=lambda: 1.0), {"id": "yt_vid_999"})
    mock_youtube.videos().insert.return_value = mock_insert_req
    mock_build.return_value = mock_youtube

    # Test with privacy_status='public'
    res = uploader._execute_oauth_upload(
        video_file=dummy_video,
        title="Test Shorts Title #Shorts",
        description="Desc",
        tags=["shorts", "test"],
        privacy_status="public",
    )

    assert res["status"] == "success"
    assert res["video_id"] == "yt_vid_999"
    assert res["privacy_status"] == "public"

    # Inspect insert body
    insert_call = mock_youtube.videos().insert.call_args
    assert insert_call is not None
    body_passed = insert_call.kwargs.get("body") or insert_call[1].get("body")
    assert body_passed["status"]["privacyStatus"] == "public"


@patch("googleapiclient.discovery.build")
@patch("google.oauth2.credentials.Credentials.from_authorized_user_file")
def test_oauth_upload_env_overrides_param_in_youtube_api_body(
    mock_creds_cls, mock_build, dummy_video: Path, tmp_path: Path, monkeypatch
):
    """Verifies that YOUTUBE_PRIVACY_STATUS env overrides parameter in actual API body."""
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "unlisted")

    token_file = tmp_path / "youtube_token.json"
    token_file.write_text('{"token": "fake_token"}')

    uploader = YouTubeShortsUploader(credentials_file=token_file)

    mock_creds = MagicMock()
    mock_creds.valid = True
    mock_creds_cls.return_value = mock_creds

    mock_youtube = MagicMock()
    mock_insert_req = MagicMock()
    mock_insert_req.next_chunk.return_value = (MagicMock(progress=lambda: 1.0), {"id": "yt_vid_unlisted"})
    mock_youtube.videos().insert.return_value = mock_insert_req
    mock_build.return_value = mock_youtube

    res = uploader._execute_oauth_upload(
        video_file=dummy_video,
        title="Unlisted via Env Title #Shorts",
        description="Desc",
        tags=["shorts"],
        privacy_status="private",  # parameter is private, but env is unlisted
    )

    assert res["status"] == "success"
    assert res["privacy_status"] == "unlisted"

    insert_call = mock_youtube.videos().insert.call_args
    assert insert_call is not None
    body_passed = insert_call.kwargs.get("body") or insert_call[1].get("body")
    assert body_passed["status"]["privacyStatus"] == "unlisted"

