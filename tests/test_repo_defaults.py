"""Pin operational GitHub clients to the canonical repository."""

from scripts import (
    backfill_research_from_logs,
    cronjob_dispatch_setup,
    dispatch_watch,
    download_raw_research,
    download_research,
    download_run_logs,
    sync_all,
)
from scripts.research_sync import verify_completeness

CANONICAL_REPOSITORY = "No-Stream/nostreambot-metaculus-bot"


def test_operational_github_defaults_use_the_canonical_repository() -> None:
    assert dispatch_watch.REPO == CANONICAL_REPOSITORY
    assert sync_all.DEFAULT_REPO == CANONICAL_REPOSITORY
    assert download_run_logs.DEFAULT_REPO == CANONICAL_REPOSITORY
    assert download_raw_research.DEFAULT_REPO == CANONICAL_REPOSITORY
    assert download_research.DEFAULT_REPO == CANONICAL_REPOSITORY
    assert verify_completeness.DEFAULT_REPO == CANONICAL_REPOSITORY
    assert backfill_research_from_logs.DEFAULT_REPO == CANONICAL_REPOSITORY
    assert cronjob_dispatch_setup.GITHUB_DISPATCH_URL.startswith(
        f"https://api.github.com/repos/{CANONICAL_REPOSITORY}/"
    )
