"""Apply human verdicts from a review queue to fuselage.db.

Reads a verdicts.json (produced by editing verdicts.template.json from
replay_review.py), groups verdicts by (session_id, episode_id), resolves a
single final action per episode, and updates the canonical `episodes` table
that the build pipeline reads via `--from-fuselage`. Also appends an audit
row to `avea_episode_review_events` per applied verdict.

Episodes that don't exist in fuselage.db yet (e.g. freshly recorded sessions
that fuselage hasn't ingested) are reported as warnings and skipped — this
tool does not insert new episode rows.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import pathlib
import sqlite3
import sys
from collections import defaultdict


DEFAULT_FUSELAGE_DB = pathlib.Path("/home/stallion/fuselage.db")
VALID_ACTIONS = {"keep", "archive", "mark_bad", "edge_case", "skip", "TODO"}
ACTION_PRIORITY = {"archive": 3, "mark_bad": 2, "edge_case": 1, "keep": 0, "skip": -1, "TODO": -1}

logger = logging.getLogger(__name__)


def _resolve_episode_action(actions: list[str]) -> str:
    return max(actions, key=lambda a: ACTION_PRIORITY.get(a, -1))


def _episode_row(con: sqlite3.Connection, episode_pk: str) -> dict | None:
    row = con.execute(
        "SELECT id, session_id, quality, success, archived_at, notes FROM episodes WHERE id = ?",
        (episode_pk,),
    ).fetchone()
    if row is None:
        return None
    return {"id": row[0], "session_id": row[1], "quality": row[2], "success": row[3], "archived_at": row[4], "notes": row[5]}


def _apply_action(
    con: sqlite3.Connection,
    episode_pk: str,
    action: str,
    notes_to_append: list[str],
    *,
    reviewed_by: str,
    now_iso: str,
    dry_run: bool,
) -> dict:
    existing = _episode_row(con, episode_pk)
    if existing is None:
        return {"status": "missing", "episode_pk": episode_pk, "action": action}

    updates: dict[str, object] = {"reviewed_at": now_iso, "reviewed_by": reviewed_by}
    if action == "keep":
        updates["quality"] = "reviewed"
        updates["success"] = 1
    elif action == "archive":
        updates["archived_at"] = now_iso
        updates["archive_reason"] = "; ".join(notes_to_append) or "replay_review hotspot review"
    elif action == "mark_bad":
        updates["quality"] = "bad"
        updates["success"] = 0
        updates["failure_reason"] = "; ".join(notes_to_append) or "replay_review hotspot review"
    elif action == "edge_case":
        prior = existing.get("notes") or ""
        addendum = "[edge_case via replay_review] " + "; ".join(notes_to_append)
        new_notes = f"{prior}\n{addendum}".strip() if prior else addendum
        updates["notes"] = new_notes
        if existing.get("quality") in (None, "", "unreviewed"):
            updates["quality"] = "reviewed"
            updates["success"] = 1
    elif action in ("skip", "TODO"):
        return {"status": "skipped", "episode_pk": episode_pk, "action": action}
    else:
        return {"status": "unknown_action", "episode_pk": episode_pk, "action": action}

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    params = list(updates.values()) + [episode_pk]
    sql = f"UPDATE episodes SET {set_clause} WHERE id = ?"
    if not dry_run:
        con.execute(sql, params)
    return {
        "status": "applied" if not dry_run else "would_apply",
        "episode_pk": episode_pk,
        "action": action,
        "updates": updates,
    }


def _log_event(
    con: sqlite3.Connection,
    *,
    session_id: str,
    episode_id: str,
    action: str,
    payload: dict,
    now_iso: str,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    con.execute(
        "INSERT INTO avea_episode_review_events (session_id, episode_id, action, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, episode_id, f"replay_review:{action}", json.dumps(payload), now_iso),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verdicts_json", type=pathlib.Path)
    parser.add_argument("--fuselage-db", type=pathlib.Path, default=DEFAULT_FUSELAGE_DB)
    parser.add_argument("--reviewed-by", default="replay_review")
    parser.add_argument("--dry-run", action="store_true", help="show planned changes without writing")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname).1s] %(message)s")

    if not args.verdicts_json.exists():
        logger.error("verdicts file not found: %s", args.verdicts_json)
        return 2
    if not args.fuselage_db.exists():
        logger.error("fuselage.db not found: %s", args.fuselage_db)
        return 2

    data = json.loads(args.verdicts_json.read_text())
    verdicts = data.get("verdicts", [])
    logger.info("loaded %d verdicts from %s", len(verdicts), args.verdicts_json)

    by_episode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for v in verdicts:
        if v.get("action") not in VALID_ACTIONS:
            logger.warning("hotspot rank %s: invalid action %r — skipping", v.get("hotspot_rank"), v.get("action"))
            continue
        if v["action"] in ("skip", "TODO"):
            continue
        sess = v.get("session_id", "")
        ep = v.get("real_episode_id", "")
        if not sess or not ep:
            logger.warning("hotspot rank %s missing session_id/real_episode_id — skipping", v.get("hotspot_rank"))
            continue
        by_episode[(sess, ep)].append(v)

    if not by_episode:
        logger.info("nothing actionable in verdicts (all skip/TODO or invalid)")
        return 0

    now_iso = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    summary = {"applied": 0, "would_apply": 0, "skipped": 0, "missing": 0, "unknown_action": 0}
    missing_episodes: list[str] = []
    applied_episodes: list[dict] = []

    with sqlite3.connect(str(args.fuselage_db)) as con:
        for (sess, ep), hits in sorted(by_episode.items()):
            actions = [h["action"] for h in hits]
            final_action = _resolve_episode_action(actions)
            notes_to_append = [h["notes"] for h in hits if h.get("notes")]
            episode_pk = f"{sess}_{ep}"
            result = _apply_action(
                con, episode_pk, final_action, notes_to_append,
                reviewed_by=args.reviewed_by, now_iso=now_iso, dry_run=args.dry_run,
            )
            summary[result["status"]] = summary.get(result["status"], 0) + 1
            if result["status"] == "missing":
                missing_episodes.append(episode_pk)
                logger.warning(
                    "episode not in fuselage.db: %s — has fuselage ingested this session yet?",
                    episode_pk,
                )
            else:
                applied_episodes.append({
                    "episode_pk": episode_pk,
                    "final_action": final_action,
                    "hotspot_actions": actions,
                    "n_hotspots": len(hits),
                })
                logger.info(
                    "%s  %s  action=%s  notes=%d  frames=%s",
                    result["status"], episode_pk, final_action, len(notes_to_append),
                    ",".join(str(h["frame_index"]) for h in hits),
                )
                _log_event(
                    con, session_id=sess, episode_id=ep, action=final_action,
                    payload={
                        "frames": [h["frame_index"] for h in hits],
                        "notes": notes_to_append,
                        "ranks": [h["hotspot_rank"] for h in hits],
                    },
                    now_iso=now_iso, dry_run=args.dry_run,
                )
        if not args.dry_run:
            con.commit()

    print()
    print("=" * 60)
    print(f"summary: {summary}")
    if applied_episodes:
        print(f"\napplied to {len(applied_episodes)} episodes:")
        for e in applied_episodes:
            print(f"  {e['final_action']:10s}  {e['episode_pk']}  (from {e['n_hotspots']} hotspots)")
    if missing_episodes:
        print(f"\n{len(missing_episodes)} episodes NOT in fuselage.db (need ingestion first):")
        for pk in missing_episodes:
            print(f"  {pk}")
    if args.dry_run:
        print("\n[DRY RUN — no changes written]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
