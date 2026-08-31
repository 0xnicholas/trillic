"""Multi-turn dialogue scenario families: seeded synthesis of golden entries
(issue #5).

Multi-turn conversation history has no public dataset to copy from
(docs/data-strategy.md: 合成(种子场景族 × 变体)). This module is the
generator half of the pipeline, ported from the old repo's conversation
generator (0xnicholas/tokencamp compression/eval/conversation_gen.py — same
author) and extended with new families:

  1. eval/manifests/dialogue_families.toml — committed, human-curated
     registry: per family, DISJOINT eval/train integer seed ranges. It is
     the split authority (data-strategy hard constraint 4: the scenario
     templates and synthesis machinery below are shared with the phase-2
     training side, but seed pools never cross).
  2. FAMILY_TEMPLATES below — handwritten conversation templates: a
     multi-turn user/assistant HISTORY whose turns establish facts (names,
     dates, prior decisions), plus a FINAL user message that asks the
     assistant to recall those facts.
  3. `synthesize_entry` — deterministic slot synthesis for one
     (family, seed). Slots are picked from candidate pools with a
     sha256-derived RNG, so seeded values (names, ticket IDs, timestamps,
     amounts) vary per entry and are embedded in the key points.
  4. `build_dialogue_manifest` — records the seed allocation and, when a
     frozen golden file is given, cross-checks every entry against
     regeneration (content_sha1 drift check) and embeds the pilot record.

Serialization (inherited from the old corpus builder): one "role: content"
turn per line; turn content may span multiple lines (code blocks). The
prompt ends with the final user recall message.

key_points are HISTORY facts (名称/日期/先前决定 — never constraints, never
final-message-only statements): each point quotes the history verbatim, so
its survival is mechanically judgeable after compression.

Golden entries are drawn from EVAL seeds only; synthesizing or manifesting
a train seed raises.
"""

import hashlib
import json
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path

FACT_TYPES = ("name", "date", "decision")


class DialogueError(ValueError):
    """The registry, seed plan, or golden file violated the split discipline."""


@dataclass(frozen=True)
class DialogueTemplate:
    """One handwritten conversation family.

    history_turns: (role, content template) pairs — the conversation that
    establishes the facts. Roles alternate user/assistant.
    optional_round: extra turns spliced before the last two history turns
    when the "include_optional_round" slot draws True (length variation,
    mirroring the old generator's probabilistic deeper round).
    final_user_template: the last turn — a user message asking the
    assistant to recall history facts (考的是对话记忆压缩).
    """

    history_turns: tuple[tuple[str, str], ...]
    final_user_template: str
    key_point_templates: tuple[str, ...]
    slot_options: dict[str, tuple]
    optional_round: tuple[tuple[str, str], ...] = ()

    def render(self, slots: dict) -> tuple[str, list[str]]:
        turns = list(self.history_turns)
        if self.optional_round and slots.get("include_optional_round"):
            turns = turns[:-2] + list(self.optional_round) + turns[-2:]
        lines = [f"{role}: {tpl.format(**slots)}" for role, tpl in turns]
        lines.append(f"user: {self.final_user_template.format(**slots)}")
        prompt = "\n".join(lines)
        key_points = [kp.format(**slots).strip() for kp in self.key_point_templates]
        return prompt, key_points


FAMILY_TEMPLATES: dict[str, DialogueTemplate] = {
    # ── ported from 0xnicholas/tokencamp eval/conversation_gen.py ────────
    "support_ticket": DialogueTemplate(
        history_turns=(
            ("user",
             "Hi, this is {name} from {company}. We're seeing failed webhook "
             "deliveries from {product} since yesterday evening. Our endpoint "
             "hasn't changed in weeks. Ticket reference from your status page: "
             "{tid}. Can someone take a look?"),
            ("assistant",
             "Hi {name}, happy to look into this with you. Could you confirm "
             "your workspace ID and the approximate time (with timezone) of "
             "the most recent failed delivery?"),
            ("user",
             "Workspace is {ws}. Last failure I see is {failure_ts} UTC. The "
             "deliveries dashboard shows \"{err_code}\" on every attempt to "
             "our {endpoint} receiver."),
            ("assistant",
             "Thanks — that error code on all attempts narrows it down. First "
             "check: can you confirm your receiver allowlists our current "
             "egress ranges? The list is under Settings → Webhooks → "
             "Delivery IPs in your dashboard. What do you see there?"),
            ("user",
             "We allowlist by security group. Let me pull the current "
             "rules... The group {sg_id} allows 203.0.113.0/28 which I added "
             "about 8 months ago when we set this up. Is that still right?"),
            ("assistant",
             "That's the old range — we rotated egress IPs about six weeks "
             "ago and announced it in the changelog and by email to "
             "workspace admins. The current range is shown in the dashboard "
             "section I mentioned. Can you update the security group with "
             "the new range and tell me when it's applied?"),
            ("user",
             "Updated. I added 198.51.100.64/27 and removed the old range. "
             "Propagation is usually instant on our side. Do I need to "
             "redeliver the failed events manually?"),
            ("assistant",
             "No manual work needed. Failed deliveries retry automatically on "
             "an exponential backoff for 72 hours. Let's force one now to "
             "confirm the fix: in Deliveries, open the most recent failed "
             "event and click \"Redeliver\". What status do you get?"),
            ("user",
             "Redelivered at {redeliver_ts}. It shows 200 OK now, latency "
             "340ms. I see a handful of older ones automatically retrying "
             "and going green as well. Looks fixed."),
            ("assistant",
             "Confirmed on our side too — delivery success rate for your "
             "workspace is back to 100% in the last 10 minutes. Summary for "
             "the ticket: root cause was the egress IP rotation six weeks "
             "ago; your allowlist still referenced the retired range. Fix "
             "was updating the security group to 198.51.100.64/27. To avoid "
             "this in future, subscribe an ops distribution list to "
             "changelog emails (Settings → Notifications → Changelog) since "
             "admin-only emails can be missed. Anything else on this one?"),
            ("user",
             "That's all, thanks for the quick turnaround. Please note in the "
             "ticket that we were down for roughly 14 hours of event "
             "deliveries — our billing reconciliation job depends on "
             "{endpoint} events. Do we need to backfill anything?"),
            ("assistant",
             "Good question. All events from the failure window remain in the "
             "delivery log and will continue retrying until they succeed, so "
             "no data was lost and no backfill is required. If your "
             "reconciliation job has its own deduplication by event ID, the "
             "delayed deliveries will slot in normally. I've noted the "
             "14-hour impact window on {tid}. I'll mark this resolved — feel "
             "free to reopen if any delivery stays red."),
        ),
        optional_round=(
            ("user",
             "One follow-up: a couple of deliveries from two days ago show "
             "\"signature verification failed\" rather than the {err_code} "
             "we just fixed. Different problem?"),
            ("assistant",
             "Yes, that's separate. Signature failures mean the payload "
             "arrived but your receiver rejected the HMAC. That happens when "
             "the signing secret was rotated in {product} but not in your "
             "verifier. Current secret version for your workspace is "
             "{secret_version}, visible under Settings → Webhooks → "
             "Signing. Does your receiver pin a specific version?"),
            ("user",
             "Our verifier has the secret hardcoded from when we integrated, "
             "running version {sdk_version} of your SDK. Someone rotated it "
             "in the dashboard last quarter for a security drill and never "
             "told us. We'll update the env var and redeploy."),
            ("assistant",
             "That explains it. After you redeploy, redeliver one of the "
             "signature-failed events to confirm. Note the SDK accepts a "
             "list of secrets during rotation windows precisely to avoid "
             "this gap — worth using next time. I'll keep {tid} open until "
             "you confirm the redelivery is green."),
        ),
        final_user_template=(
            "Before we close {tid} completely: management will ask me what "
            "happened. Can you recap this case for the record — the "
            "timeline, the root cause you identified, the fix that was "
            "applied, and the impact window you noted on the ticket?"
        ),
        key_point_templates=(
            "reporter is {name} from {company} (ticket {tid})",
            "last failure I see is {failure_ts} UTC, {err_code} on every attempt to the {endpoint} receiver",
            "root cause: egress IPs were rotated about six weeks ago; the allowlist still referenced the retired range",
            "fix was updating the security group to 198.51.100.64/27",
            "failed deliveries retry automatically for 72 hours; no backfill is required",
            "impact window of roughly 14 hours of event deliveries was noted on the ticket",
        ),
        slot_options={
            "company": ("Northwind Analytics", "Bluepeak Systems", "Ferrostack"),
            "endpoint": ("/v1/events", "/v1/deploy-hooks", "/v1/billing-events"),
            "err_code": ("ERR_WEBHOOK_TIMEOUT", "ERR_SIGNATURE_MISMATCH", "ERR_DELIVERY_429"),
            "failure_ts": ("2026-07-18T21:47:12Z", "2026-07-19T08:23:51Z", "2026-07-21T17:05:38Z"),
            "include_optional_round": (True, True, False),
            "name": ("Maya", "Daniel", "Priya", "Aisha", "Kenji"),
            "product": ("CloudForge", "DataBridge", "Metricly"),
            "redeliver_ts": ("2026-07-22T09:14:03Z", "2026-07-22T10:41:26Z", "2026-07-23T08:02:47Z"),
            "sdk_version": ("2.14.3", "3.2.1", "2.7.0"),
            "secret_version": ("v3", "v4"),
            "sg_id": ("sg-72915", "sg-60438", "sg-28157"),
            "tid": ("SUP-48291", "INC-35017", "TKT-95726"),
            "ws": ("ws_583014", "ws_721946", "ws_330812"),
        },
    ),
    "pair_programming": DialogueTemplate(
        history_turns=(
            ("user",
             "{dev} and I have been chasing a flaky test in {svc} for two "
             "days. `test_concurrent_refund_idempotent` fails maybe 1 in 20 "
             "runs in CI, never locally. Here's the failure:\n\n"
             "```\n"
             "FAILED tests/test_refunds.py::test_concurrent_refund_idempotent\n"
             "AssertionError: expected 1 refund row, found 2\n"
             "  refund_id={refund_id}, amount=49.90, created twice within 3ms\n"
             "```\n\n"
             "Two refund rows for one request. Smells like a race, but the "
             "handler checks for an existing refund before inserting. Can "
             "you help us think through it?"),
            ("assistant",
             "Classic check-then-act race. The sequence `SELECT existing → "
             "if none → INSERT` is not atomic, so two concurrent requests "
             "both pass the SELECT and both insert. Locally you never get "
             "true concurrency; in CI the test harness fires the two "
             "requests in parallel tasks. Can you paste the handler's "
             "refund-creation code?"),
            ("user",
             "```python\n"
             "async def create_refund(db, order_id: int, amount: Decimal) -> Refund:\n"
             "    existing = await db.fetch_one(\n"
             "        \"SELECT * FROM refunds WHERE order_id = :oid AND status = 'pending'\",\n"
             "        {{\"oid\": order_id}},\n"
             "    )\n"
             "    if existing:\n"
             "        return Refund.from_row(existing)\n"
             "    row = await db.fetch_one(\n"
             "        \"INSERT INTO refunds (order_id, amount, status) VALUES (:oid, :amt, 'pending') RETURNING *\",\n"
             "        {{\"oid\": order_id, \"amt\": amount}},\n"
             "    )\n"
             "    return Refund.from_row(row)\n"
             "```\n\n"
             "We discussed wrapping it in {lock}, but that only serializes "
             "within one process — we run 4 workers in prod."),
            ("assistant",
             "Right — a process-local lock doesn't help across workers, and "
             "even across processes a lock just hides the bug behind "
             "latency. The correct fix is to let the database enforce "
             "uniqueness. Two options:\n\n"
             "1. Add a partial unique index: `CREATE UNIQUE INDEX "
             "refunds_one_pending ON refunds (order_id) WHERE status = "
             "'pending'` then catch the `UniqueViolationError` on insert and "
             "re-select the winner's row.\n"
             "2. Use `INSERT ... ON CONFLICT DO NOTHING RETURNING *` and "
             "fall back to a SELECT when RETURNING yields nothing.\n\n"
             "Option 2 is one round-trip and no exception handling. Want to "
             "draft that together?"),
            ("user",
             "```python\n"
             "async def create_refund(db, order_id: int, amount: Decimal) -> Refund:\n"
             "    row = await db.fetch_one(\n"
             "        \"INSERT INTO refunds (order_id, amount, status) VALUES (:oid, :amt, 'pending')\"\n"
             "        \" ON CONFLICT (order_id) WHERE status = 'pending' DO NOTHING RETURNING *\",\n"
             "        {{\"oid\": order_id, \"amt\": amount}},\n"
             "    )\n"
             "    if row is None:\n"
             "        row = await db.fetch_one(\n"
             "            \"SELECT * FROM refunds WHERE order_id = :oid AND status = 'pending'\",\n"
             "            {{\"oid\": order_id}},\n"
             "        )\n"
             "    return Refund.from_row(row)\n"
             "```\n\n"
             "Like this? The partial index needs to exist first, right?"),
            ("assistant",
             "Exactly — `ON CONFLICT (order_id) WHERE status = 'pending'` "
             "references the partial unique index, so the migration must add "
             "the index before this code ships. Two details: (1) the "
             "fallback SELECT can theoretically return None if the winner's "
             "refund was cancelled between statements — wrap the whole thing "
             "in a retry loop of 2 attempts or use REPEATABLE READ if you "
             "want it airtight; (2) check whether there are existing "
             "duplicate rows in prod, because the index creation will fail "
             "if there are. Run this first:\n\n"
             "```sql\n"
             "SELECT order_id, count(*) FROM refunds WHERE status = 'pending'\n"
             "GROUP BY order_id HAVING count(*) > 1;\n"
             "```"),
            ("user",
             "{dev} ran it on the prod read replica: 3 order_ids have "
             "duplicate pending refunds, worst one has 4 rows, all created "
             "during the incident window on {incident_date}. So the race "
             "already bit us in production, we just never noticed because "
             "the test only exists in the new suite. What's the cleanup "
             "plan?"),
            ("assistant",
             "Cleanup before the index, in one maintenance script:\n\n"
             "1. For each duplicated order_id, keep the earliest pending "
             "refund (that's what users saw first) and mark the rest "
             "`status = 'superseded'` — don't hard-delete, finance may need "
             "the trail.\n"
             "2. Cross-check each superseded row against the payment "
             "provider: if any actually captured money, those need manual "
             "refunds and a support ticket per customer, not a script.\n"
             "3. Then create the index, deploy the new handler, and re-run "
             "the flaky test 50 times in CI to confirm.\n\n"
             "Want me to draft the dedupe SQL? Also: how much money are we "
             "talking about across those dupes?"),
            ("user",
             "Total double-captured amount is {total_amount} across "
             "{dup_customers} customers. Small enough that support will just "
             "refund and apologize. Yes please, draft the dedupe SQL. And "
             "thanks — we were about to paper over this with {lock}, "
             "which would've broken the moment we scaled workers."),
            ("assistant",
             "```sql\n"
             "WITH ranked AS (\n"
             "  SELECT id, order_id,\n"
             "         ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY created_at ASC) AS rn\n"
             "  FROM refunds WHERE status = 'pending'\n"
             ")\n"
             "UPDATE refunds SET status = 'superseded', updated_at = now()\n"
             "WHERE id IN (SELECT id FROM ranked WHERE rn > 1);\n"
             "```\n\n"
             "Run it in a transaction with a `SELECT` of the affected rows "
             "first so you have the list for support. After the index is in, "
             "add a regression test that hammers the endpoint with 10 "
             "concurrent calls and asserts exactly one row — keep the "
             "existing 1-in-20 test too, it's cheap. Ping me when CI is "
             "green."),
        ),
        final_user_template=(
            "Great session. I'm writing the handover note for our team "
            "channel — can you restate what we found in production, the fix "
            "we decided on, the cleanup steps in order, and the money "
            "involved, so the note matches what we actually agreed?"
        ),
        key_point_templates=(
            "{dev} has been chasing a flaky test in {svc} that fails maybe 1 in 20 runs in CI (refund_id {refund_id})",
            "diagnosis: check-then-act race — two concurrent requests both pass the SELECT and both insert",
            "decision: fix via INSERT ... ON CONFLICT DO NOTHING with a partial unique index on (order_id) WHERE status = 'pending'",
            "3 order_ids have duplicate pending refunds, worst one has 4 rows, all created during the incident window on {incident_date}",
            "cleanup decision: keep the earliest pending refund and mark the rest superseded — don't hard-delete",
            "total double-captured amount is {total_amount} across {dup_customers} customers",
        ),
        slot_options={
            "dev": ("Daniel", "Kenji", "Felix", "Tom", "Leo"),
            "dup_customers": (5, 7, 9, 12),
            "incident_date": ("2026-07-12", "2026-07-15", "2026-07-18"),
            "lock": ("an asyncio.Lock", "a Redis lock", "an advisory lock"),
            "refund_id": (4821, 7163, 9547),
            "svc": ("orders-service", "ingest-worker", "notifications-api", "ledger-sync"),
            "total_amount": ("$612.47", "$483.20", "$874.95"),
        },
    ),
    "requirements_discussion": DialogueTemplate(
        history_turns=(
            ("user",
             "{eng}, kicking off requirements for {feature}. This is {pm}. "
             "Sales says three deals stalled last quarter partly because "
             "prospects couldn't self-serve this. I drafted the problem "
             "statement: mid-market admins need this capability without "
             "contacting support, currently ~40 tickets/month touch it. "
             "Before I write the PRD, let's align on scope and the nasty "
             "edge cases. What's your first reaction?"),
            ("assistant",
             "First reaction: agreed on the problem, but '{feature}' spans a "
             "wide scope range. Let's pin down the v1 boundary before "
             "acceptance criteria. Questions: (1) is this self-serve from "
             "the settings UI only, or does it also need API coverage? (2) "
             "who is allowed to use it — every member or admins only? (3) "
             "what's the failure mode the user sees when the underlying "
             "operation takes minutes rather than seconds?"),
            ("user",
             "Good pushes. (1) UI only for v1; API is a fast-follow if "
             "adoption justifies it — I'll put API in non-goals explicitly. "
             "(2) Admins only; auditors have been clear that broad access is "
             "a compliance problem. (3) That's exactly the edge case I "
             "wanted to discuss — walk me through what actually happens "
             "technically when it's slow."),
            ("assistant",
             "For the slow path: the request fans out over up to 90 days of "
             "records, worst case ~{max_rows} million rows for our largest "
             "tenant. Synchronous request will time out at the gateway's 30s "
             "limit for anything beyond a few hundred thousand rows. So v1 "
             "needs an async job: user requests, we enqueue, process in the "
             "background, and notify when the result is ready. That means we "
             "also need a 'jobs' status surface in the UI and a "
             "retention/expiry policy for the artifacts."),
            ("user",
             "Async it is. Retention: artifacts expire after 7 days, "
             "download links signed for 15 minutes. Now the permission model "
             "— admins only, but should there be an audit log entry each "
             "time someone triggers it? Compliance will ask."),
            ("assistant",
             "Yes — write an audit event with actor, timestamp, scope "
             "parameters, and row count, retention aligned with the existing "
             "audit log (400 days). Cheap to add since we reuse the existing "
             "audit pipeline. Two more edge cases to decide: (a) what "
             "happens on empty result sets — deliver an empty artifact or a "
             "'no data' notification? (b) do we rate-limit per workspace? A "
             "tenant could queue hundreds of jobs and starve the worker "
             "pool."),
            ("user",
             "(a) Deliver the empty artifact — 'no data' states that only "
             "exist in notifications confuse people; an empty file with "
             "correct headers is unambiguous. (b) Rate limit: max "
             "{queue_cap} queued jobs per workspace, excess rejected with a "
             "clear message and a Retry-After hint. Does that break any "
             "tenant you know of?"),
            ("assistant",
             "No tenant is near that today — top usage is 3 concurrent. One "
             "more: partial failures. If the job dies at 60%, do we deliver "
             "partial results flagged as partial, or fail the whole job? "
             "Partial data that's silently partial is the worst outcome "
             "here."),
            ("user",
             "Fail the whole job, retry once automatically, then surface a "
             "clear failure with a support link. No partial artifacts, ever "
             "— agreed on the silent-partial risk. I'll write user stories "
             "for: happy path, slow path via async job, empty result, "
             "rate-limit rejection, job failure + retry, audit entry, "
             "artifact expiry. Acceptance criteria in Given/When/Then for "
             "each. Effort guess?"),
            ("assistant",
             "Rough t-shirt: async job infrastructure exists from the "
             "reports work, so the new surface is the fan-out query (M), the "
             "status UI (S, reuse the jobs panel), audit + rate limiting "
             "(S), expiry via the existing lifecycle worker (XS). Call it "
             "{effort_weeks} weeks for one engineer with review. Main risk "
             "is the query planner on the 90-day fan-out for the top tenant "
             "— I'll spike that this week before you finalize the PRD."),
            ("user",
             "Perfect. Spike first, PRD after — I'll note the dependency. "
             "Sending you the draft stories by Thursday; flag anything where "
             "the acceptance criteria don't match what we just agreed. "
             "Summarizing decisions: admins only, UI only (API non-goal), "
             "async jobs with 7-day artifact expiry and 15-minute signed "
             "links, audit events on every trigger, empty artifacts "
             "delivered, per-workspace queue cap with clear rejection, no "
             "partial results, automatic single retry. Anything I missed?"),
            ("assistant",
             "One addition: put 'no partial artifacts, ever' into the PRD as "
             "an explicit invariant, not just a story — it's the kind of "
             "thing that gets negotiated away under deadline pressure and it "
             "protects the trust property that makes this feature worth "
             "building. Otherwise the list matches my notes."),
        ),
        final_user_template=(
            "That's a solid summary on your side. Before I circulate the "
            "decisions to the stakeholders: can you list them back as the "
            "final agreed scope and parameters — audience, boundaries, job "
            "behavior, retention, and the cap we settled on?"
        ),
        key_point_templates=(
            "v1 is admins only, UI only; API coverage is an explicit non-goal",
            "async jobs: artifacts expire after 7 days, download links signed for 15 minutes",
            "rate limit: max {queue_cap} queued jobs per workspace, excess rejected with a Retry-After hint",
            "decision: fail the whole job, retry once automatically — no partial artifacts, ever",
            "audit event with actor, timestamp, scope parameters, and row count; retention aligned with the existing audit log (400 days)",
            "effort estimate: {effort_weeks} weeks for one engineer with review",
        ),
        slot_options={
            "effort_weeks": ("2–3", "3–4"),
            "eng": ("Priya", "Kenji", "Sara"),
            "feature": ("CSV export of the usage dashboard", "SSO login via SAML", "per-project budget alerts"),
            "max_rows": (2, 5, 8),
            "pm": ("Nora", "Maya", "Aisha"),
            "queue_cap": (5, 10),
        },
    ),
    "data_pipeline": DialogueTemplate(
        history_turns=(
            ("user",
             "{eng}, the exec dashboard's {metric} metric dropped {pct}% "
             "overnight and leadership is asking. Nothing in the product "
             "changed. I suspect the pipeline, not reality. {table} was "
             "rebuilt yesterday in the dbt run. Where do we start?"),
            ("assistant",
             "Start by separating 'data missing' from 'data wrong'. Two "
             "quick checks: (1) row counts on {table} for the last 14 days — "
             "is yesterday's partition unusually small? (2) run the metric "
             "query directly on the warehouse, bypassing the dashboard "
             "cache. Can you run both and paste results?"),
            ("user",
             "```\n"
             "-- row counts, {table}\n"
             "2026-07-18   2,190,441\n"
             "2026-07-19   2,204,876\n"
             "yesterday      {low_count:,}   <-- way down\n"
             "today (partial)  214,908\n"
             "```\n\n"
             "Direct warehouse query agrees with the dashboard, so it's not "
             "a cache issue. Yesterday's partition really is small."),
            ("assistant",
             "Row count down by a similar proportion to the metric means "
             "we're losing events upstream, not mis-aggregating. Next "
             "suspect: the rebuild. Did yesterday's dbt run change any join "
             "or filter in the model? A one-to-many join exploding then "
             "getting deduped, or a new WHERE clause, are the usual "
             "culprits. What does `git log -p` on the model show?"),
            ("user",
             "There IS a change. Someone added an inner join to a new "
             "`dim_consent` table to filter out users who revoked analytics "
             "consent:\n\n"
             "```sql\n"
             "JOIN dim_consent c ON c.user_id = e.user_id AND c.analytics_allowed\n"
             "```\n\n"
             "Intent was privacy compliance. But `dim_consent` only has "
             "rows for users who touched the consent screen since the "
             "feature shipped last week — older users have no row at all, "
             "so the inner join silently drops them. That's our missing "
             "events."),
            ("assistant",
             "That's it. Inner join + incomplete dimension = silent row "
             "loss, and the metric drops without any error anywhere. The fix "
             "is a LEFT JOIN with a NULL-tolerant predicate, treating 'no "
             "consent row' as the legacy default — but the legacy default is "
             "a legal question, not an engineering one: does absence of a "
             "row mean consent (pre-feature users grandfathered) or no "
             "consent? Check with the privacy owner before I tell you which "
             "predicate to write."),
            ("user",
             "Privacy owner says: pre-feature users default to allowed (they "
             "accepted the old ToS), only explicit revocations block. So the "
             "predicate is: keep when no row, or row says allowed; drop only "
             "explicit revocations."),
            ("assistant",
             "```sql\n"
             "LEFT JOIN dim_consent c ON c.user_id = e.user_id\n"
             "WHERE (c.user_id IS NULL OR c.analytics_allowed)\n"
             "```\n\n"
             "Careful with the WHERE on a LEFT JOIN — putting "
             "`c.analytics_allowed` alone in WHERE would silently turn it "
             "back into an inner join. The `c.user_id IS NULL OR` is the "
             "whole point. After the fix: full-refresh the affected "
             "partitions, not just yesterday — every partition since the "
             "consent join shipped is undercounted. Then re-run the metric "
             "for the full window and hand finance the restated numbers."),
            ("user",
             "Full refresh running for the last 12 days. Estimated 40 "
             "minutes. While it runs: how do we stop this class of bug? A "
             "join that silently drops rows produced no failing test, no "
             "alert, nothing."),
            ("assistant",
             "Three layers, cheapest first:\n\n"
             "1. dbt test on the model: row-count-vs-source ratio within a "
             "band (e.g. 0.9–1.1 day over day). Would have caught this the "
             "morning after.\n"
             "2. A not-null/accepted-values audit on join keys that are "
             "supposed to be complete: if `dim_consent` is expected to "
             "cover all active users eventually, test coverage explicitly.\n"
             "3. Code-review rule: any new JOIN in a fct model must state "
             "in the PR description its intended row-count effect, with "
             "before/after counts from a dev run. Silent row loss is almost "
             "always a join; make joiners say the number out loud."),
            ("user",
             "Refresh done. Metric for yesterday is back within 2% of trend "
             "— remaining delta is genuinely revoked consent, which is "
             "correct behavior now. I've posted the restated numbers and the "
             "root cause in the incident channel. Adding all three "
             "safeguards to the backlog; the ratio test goes in this week. "
             "Thanks {eng}, this was a nasty one."),
            ("assistant",
             "Nice teamwork. One last thing: put the restated-window caveat "
             "in the dashboard as an annotation on those 12 days, or someone "
             "will 'discover' the dip again in six months and we'll "
             "re-investigate from scratch."),
        ),
        final_user_template=(
            "Agreed, I'll add the annotation. One thing before I write the "
            "incident recap: can you restate the root cause, the privacy "
            "ruling we got, and the fix exactly as we settled them, plus "
            "how many days of data we are restating?"
        ),
        key_point_templates=(
            "{metric} metric dropped {pct}% overnight; {table} was rebuilt yesterday in the dbt run",
            "root cause: an inner join to a new dim_consent table silently drops users without a consent row",
            "privacy ruling: pre-feature users default to allowed; only explicit revocations block",
            "fix: LEFT JOIN dim_consent with WHERE c.user_id IS NULL OR c.analytics_allowed",
            "full refresh covers the last 12 days, not just yesterday's partition",
        ),
        slot_options={
            "eng": ("Kenji", "Felix", "Leo"),
            "low_count": (612440, 733210, 848065),
            "metric": ("daily_active_workspaces", "net_revenue_retention", "weekly_report_runs"),
            "pct": (18, 26, 33),
            "table": ("fct_usage_daily", "fct_events_enriched", "fct_billing_daily"),
        },
    ),
    "devops_incident": DialogueTemplate(
        history_turns=(
            ("user",
             "PAGED: {svc} 5xx rate jumped to {err_pct}%. I'm {oncall}, "
             "on-call this week; {mate} is joining as second responder. Error "
             "rate still climbing. What's our first move?"),
            ("assistant",
             "Blast radius first: is it all routes on {svc} or "
             "concentrated? And what deployed in the last 2 hours? Check "
             "`kubectl rollout history deploy/{svc} -n prod` and the deploy "
             "channel. While that runs, pull the top error from the logs: "
             "`kubectl logs -n prod deploy/{svc} --tail=200 | grep -i error "
             "| sort | uniq -c | sort -rn | head`."),
            ("user",
             "Deploy at {deploy_ts} took {svc} to revision {deploy_rev}, 40 "
             "minutes before the alarm. Errors are concentrated on /v2/search "
             "— p99 latency there went from 300ms to timing out, and the 5xx "
             "are all 502/504 from the upstream. Top log line:\n\n"
             "```\n"
             "ERROR upstream connect error or disconnect/reset before headers. "
             "reset reason: connection failure\n"
             "  cluster=search-backend, addr={backend_ip}:9200\n"
             "```"),
            ("assistant",
             "Timeline fits: deploy → 40 min → 5xx on one route, upstream "
             "connection failures. Before root-causing the code diff, "
             "mitigate: roll back. `kubectl rollout undo deploy/{svc} -n "
             "prod`. The old revision was stable for days, so this restores "
             "service while we read the diff at leisure. Confirm the "
             "rollback and watch the error rate for 5 minutes."),
            ("user",
             "Rollback complete, pods cycling. {mate} — 5xx already "
             "dropping, at 2% and falling. Customers should be OK now. So "
             "what did {deploy_rev} change? Diffing {deploy_rev} against "
             "previous... The PR touched the search client: switched the "
             "connection pool from keep-alive persistent connections to "
             "per-request connections, and lowered the connect timeout from "
             "5s to 500ms. Intent was faster failure on dead nodes."),
            ("assistant",
             "That diff explains it without any code bug: per-request "
             "connections plus a 500ms connect timeout against a backend "
             "doing TLS handshake per connection. Under load, handshake + "
             "connect routinely exceeds 500ms, connections fail, retries "
             "amplify load, and the backend's listen queue saturates — a "
             "classic retry-storm feedback loop. It took 40 minutes because "
             "traffic had to ramp into the evening peak. Verify: check the "
             "backend's connection metrics during the window — you should "
             "see a cliff of connection churn starting at the deploy."),
            ("user",
             "Confirmed in Grafana: established connections went from a "
             "steady ~400 to oscillating between 50 and 4000 right after "
             "the deploy. Exactly the churn pattern. The 5xx are at 0.2% "
             "now, normal. Do we revert the PR or fix forward?"),
            ("assistant",
             "Fix forward, but not tonight. The PR's intent (bounded failure "
             "time) is reasonable; the implementation conflated 'connect "
             "timeout' with 'request timeout' and dropped connection reuse. "
             "The proper change: restore keep-alive pooling, keep an "
             "aggressive REQUEST timeout (not connect), and add a circuit "
             "breaker so one slow backend doesn't take the route down. "
             "That's a reviewed PR tomorrow, behind the load-test suite, not "
             "a midnight patch. Timeline so far — paged 40 minutes after the "
             "deploy at {deploy_ts}, rollback mitigated, root cause identified "
             "as connection-pool regression in {deploy_rev}. Next: anything "
             "still red on the board?"),
            ("user",
             "Board is green except a stale alert from the indexer that "
             "auto-resolved. I'll write the status-page update: 'Elevated "
             "errors on search between the deploy and rollback; fully "
             "recovered; no data loss.' Draft the customer-facing wording "
             "for me?"),
            ("assistant",
             "\"Between approximately [start] and [rollback] UTC, search "
             "requests experienced elevated error rates and timeouts for a "
             "subset of customers. The cause was a configuration regression "
             "in a routine deployment, which was rolled back. Service is "
             "fully restored; no data was lost. We're adding load-test "
             "coverage for connection-pool behavior to prevent recurrence.\" "
             "— keep it that plain; no percentages, no blame. Postmortem "
             "tomorrow 10:00? I'll bring the timeline."),
        ),
        final_user_template=(
            "Postmortem tomorrow at 10:00 works. I'm drafting the timeline "
            "doc now — walk me through the full sequence one more time: when "
            "we were paged, which deploy caused it, what the rollback "
            "decision was, and what the fix-forward plan is?"
        ),
        key_point_templates=(
            "{oncall} is on-call this week; {mate} is joining as second responder",
            "deploy at {deploy_ts} took {svc} to revision {deploy_rev}, 40 minutes before the alarm; 5xx rate jumped to {err_pct}%",
            "mitigation: before root-causing the code diff, roll back",
            "root cause: connection-pool regression — per-request connections plus a 500ms connect timeout caused a retry storm",
            "fix-forward plan: restore keep-alive pooling, aggressive request timeout, add a circuit breaker — a reviewed PR tomorrow, not a midnight patch",
        ),
        slot_options={
            "backend_ip": ("10.4.122.31", "10.12.8.207", "10.19.64.5"),
            "deploy_rev": ("a1b2c3d", "e5f6a7b", "9c8d7e6"),
            "deploy_ts": ("2026-07-22T11:20:00Z", "2026-07-22T14:05:00Z", "2026-07-23T09:45:00Z"),
            "err_pct": (8, 17, 26),
            "mate": ("Priya", "Sara", "Nora"),
            "oncall": ("Tom", "Kenji", "Felix"),
            "svc": ("api-gateway", "checkout-api", "search-indexer"),
        },
    ),
    # ── new families authored for this repo (issue #5) ───────────────────
    "vendor_procurement": DialogueTemplate(
        history_turns=(
            ("user",
             "I'm {lead} from procurement. Our {vendor} contract renews on "
             "{renewal_date} and their quote landed yesterday with a 9% "
             "uplift on last year. Leadership wants a counter-position "
             "settled this week. Where do we anchor?"),
            ("assistant",
             "Before anchoring, three inputs decide leverage: current annual "
             "spend and its trend, whether a credible alternative exists, "
             "and what switching would actually cost you in migration and "
             "risk. What do those look like for {vendor}?"),
            ("user",
             "Spend is $84k/year, growing about 15% as teams expand. We ran "
             "a bake-off last year with two alternatives; {vendor} won "
             "clearly on {advantage}. Switching is possible but would cost "
             "us a quarter of engineering effort."),
            ("assistant",
             "So: real but expensive alternative — that's negotiating power "
             "you can reference without bluffing. My recommendation: open at "
             "flat renewal (0% uplift), and concede toward a commitment "
             "discount — target {discount} off the quoted price in exchange "
             "for {term}. Your growth makes a term commitment credible."),
            ("user",
             "Workable. Legal separately flagged the old MSA: it allows "
             "termination on 30 days' notice — fine for us — but the "
             "auto-renewal clause renews for a full year unless we cancel "
             "{renewal_notice} days before {renewal_date}. We want that "
             "window widened."),
            ("assistant",
             "Reasonable ask and usually uncontested. Counter-proposal for "
             "the paper: termination for convenience on {notice_days} days' "
             "notice for both sides, auto-renewal notice window widened to "
             "{renewal_notice} days, and renewal capped at CPI + 2% so next "
             "year's quote can't repeat this. Keep them as a package — the "
             "capped uplift is worth more than the notice days."),
            ("user",
             "Security also has a condition: the SOC 2 Type II report on "
             "file is from {report_year} and stale. Signature should be "
             "conditional on receiving the current report."),
            ("assistant",
             "Agreed — make signature conditional on a current SOC 2 Type "
             "II report covering the last 12 months, delivered before "
             "execution. Standard ask; vendors expect it. One more decision: "
             "the data processing agreement. Their template assigns "
             "liability poorly for our data flows."),
            ("user",
             "On the DPA: we route it through {dpa_route} rather than "
             "signing their template as-is. Risk team already agreed. So "
             "the full position: flat-or-discounted renewal, term for "
             "discount trade, notice and cap changes, conditional SOC 2, "
             "DPA through our paper."),
            ("assistant",
             "Recorded position: renew with {vendor} for {term} at {discount} "
             "off the quoted price; termination for convenience on "
             "{notice_days} days; auto-renewal notice widened to "
             "{renewal_notice} days with uplift capped at CPI + 2%; signature "
             "conditional on a current SOC 2 Type II report; DPA routed "
             "through {dpa_route}. Deadline to close: {renewal_date}. "
             "Anything non-standard they'll push back on?"),
            ("user",
             "The cap, probably — everything else is market. If they hold, "
             "the fallback is CPI + 3% for the first renewal year only. "
             "That's the whole position; write it into the negotiation memo "
             "so the team speaks from one page."),
            ("assistant",
             "Memo skeleton ready: position, rationale per clause, fallback "
             "on the cap (CPI + 3%, first renewal year only), and the "
             "signature conditions. One flag: get the SOC 2 request in "
             "writing this week — report delivery is the long pole if their "
             "audit is late in the cycle."),
        ),
        final_user_template=(
            "Good catch on the SOC 2 timing. Before I send the memo: restate "
            "the agreed position with every number and date in it — the "
            "discount, the term, both notice windows, the cap, the fallback, "
            "and the signature conditions — so nothing drifts between this "
            "chat and the email?"
        ),
        key_point_templates=(
            "negotiation owner: {lead} from procurement; renewal deadline {renewal_date}",
            "target terms: renew for {term} at {discount} off the quoted price",
            "termination for convenience on {notice_days} days; auto-renewal notice widened to {renewal_notice} days",
            "renewal uplift capped at CPI + 2%; fallback is CPI + 3% for the first renewal year only",
            "signature conditional on a current SOC 2 Type II report (on-file report from {report_year} is stale)",
            "DPA routed through {dpa_route} rather than their template as-is",
        ),
        slot_options={
            "advantage": ("latency and the migration toolkit", "regional coverage", "the compliance certifications"),
            "discount": ("15%", "18%", "22%"),
            "dpa_route": ("our standard customer paper", "outside counsel's rider"),
            "lead": ("Iris Novak", "Tomas Weber", "Lena Fischer"),
            "notice_days": (60, 90),
            "renewal_date": ("2026-09-30", "2026-10-15", "2026-11-01"),
            "renewal_notice": (90, 120),
            "report_year": (2023, 2024),
            "term": ("a two-year commit", "a three-year commit"),
            "vendor": ("DatumGrid", "CirrusOps", "Vantage Labs"),
        },
    ),
    "onboarding_rollout": DialogueTemplate(
        history_turns=(
            ("user",
             "I'm {admin}, workspace admin for {company}. We're finally "
             "migrating off {old_system} onto your platform — {user_count} "
             "users, roughly {data_volume} TB of documents, plus the "
             "approval workflows. First question: when can we realistically "
             "go live?"),
            ("assistant",
             "Welcome aboard — I'm {csm}, your onboarding lead. For that "
             "size, a realistic timeline is five to seven weeks including a "
             "pilot wave. Before I propose dates: what hard constraints do "
             "you have? Freeze windows, audits, anything immovable?"),
            ("user",
             "One big one: fiscal-year close freezes ALL system changes from "
             "{freeze_start} to 2026-09-18. Nothing may change in that "
             "window, not even a migrated workflow. So go-live must be fully "
             "done before {freeze_start} or fully after 2026-09-18."),
            ("assistant",
             "Then the choice is: big-bang cutover before the freeze, or a "
             "phased rollout in three waves that completes after it. For "
             "{user_count} users with approval workflows, I recommend "
             "phased: pilot wave of ~50 users, two follow-on waves, full "
             "cutover after 2026-09-18. Big-bang before {freeze_start} "
             "leaves zero room for a rollback inside the freeze."),
            ("user",
             "Phased makes sense; a botched cutover during close would be a "
             "career event. So: pilot wave first — who's in it and when?"),
            ("assistant",
             "Pilot proposal: {pilot_group} — about 50 people, mixed "
             "departments, starts 2026-09-29. They run on the platform "
             "while {old_system} stays live as the system of record. Waves "
             "2 and 3 follow at two-week intervals; final cutover — "
             "{old_system} goes read-only — on {cutover_date}, safely after "
             "the freeze."),
            ("user",
             "Dates work. During the pilot, documents sync one-way? My "
             "worry is people editing in both systems and diverging."),
            ("assistant",
             "During pilot and waves: one-way sync from {old_system} nightly "
             "at 02:00, edits happen in the new platform only for pilot "
             "scope. At cutover we do a final delta migration on "
             "{cutover_date} starting 06:00, then {old_system} becomes "
             "read-only for {readonly_days} days as the rollback window — "
             "after that, writes disabled entirely."),
            ("user",
             "The {readonly_days}-day read-only rollback window — good. "
             "Training: my people are not early adopters, we need live "
             "sessions before each wave, not a video dump."),
            ("assistant",
             "Agreed: live 90-minute sessions, one per wave, run by our "
             "enablement team. First session 2026-09-22 for the pilot "
             "group; sessions for waves 2 and 3 scheduled as each wave "
             "confirms. All sessions recorded. I'll hold weekly slots "
             "starting 2026-09-22."),
            ("user",
             "Last thing — data residency. Our legal needs the data to stay "
             "in the EU region; the old contract with {old_system} promised "
             "Frankfurt."),
            ("assistant",
             "Noted and non-negotiable: your workspace is provisioned in the "
             "EU region (Frankfurt) from day one, including the nightly "
             "sync pipeline and backups. I'll put the residency clause in "
             "the onboarding spec so it's in writing. To recap the plan: "
             "phased rollout, pilot 2026-09-29, final cutover {cutover_date} "
             "after the {freeze_start}–2026-09-18 freeze, {old_system} "
             "read-only for {readonly_days} days, training starts "
             "2026-09-22, EU residency throughout."),
        ),
        final_user_template=(
            "That recap covers it. I need to brief our steering committee "
            "tomorrow — can you lay out the full plan once more with every "
            "date and the rollback arrangement exactly as agreed, so the "
            "deck matches this conversation?"
        ),
        key_point_templates=(
            "{admin}, workspace admin for {company}; onboarding lead {csm}",
            "hard constraint: system freeze from {freeze_start} to 2026-09-18; nothing may change in that window",
            "decision: phased rollout in three waves, not big-bang — final cutover after the freeze",
            "pilot wave of ~50 users starts 2026-09-29",
            "final cutover {cutover_date}: delta migration from 06:00, then {old_system} read-only for {readonly_days} days",
            "live 90-minute sessions, one per wave; first session 2026-09-22 for the pilot group",
            "workspace provisioned in the EU region (Frankfurt) from day one",
        ),
        slot_options={
            "admin": ("Marta Ilves", "Jonas Kreft", "Elif Aydın"),
            "company": ("Halden Logistics", "Brivant Insurance", "Sterner Manufacturing"),
            "csm": ("Ana Ruiz", "Devi Menon", "Karl Oja"),
            "cutover_date": ("2026-10-27", "2026-11-03"),
            "data_volume": (6, 12, 20),
            "freeze_start": ("2026-08-24", "2026-08-31"),
            "old_system": ("Docuflow", "Papertrail DMS", "Archivio"),
            "pilot_group": ("the finance department", "the legal and finance departments", "the operations team"),
            "readonly_days": (30, 45),
            "user_count": (420, 650, 1100),
        },
    ),
    "bug_triage": DialogueTemplate(
        history_turns=(
            ("user",
             "Weekly triage. First item: {bug_id}, filed by {reporter} on "
             "{report_date} — search results intermittently show stale data "
             "in {service}. Still reproducing this morning on {affected_version}. "
             "Who's seen it?"),
            ("assistant",
             "I have. Reproduces as a 30–90 second staleness window after a "
             "write, only under load. Before we argue severity: what's the "
             "customer-visible impact pattern — how often, and is there a "
             "workaround?"),
            ("user",
             "Support says about {affected_pct}% of search queries during "
             "peak windows show pre-update results. Workaround exists: a "
             "manual refresh pulls fresh data. No data loss, no wrong-row "
             "returns — just lag."),
            ("assistant",
             "Then classification is S2, not S1: significant degradation "
             "with a workaround and no correctness loss. S1 would require "
             "outage or wrong-data answers. I'd record it as: intermittent "
             "stale search results in {service}, degraded-not-down, "
             "workaround via manual refresh. Agreed?"),
            ("user",
             "Agreed, S2. Ownership: {assignee} knows the cache layer "
             "better than anyone — he's back from leave Monday. Assign to "
             "him."),
            ("assistant",
             "Assigned to {assignee}. Target milestone: the fix belongs in "
             "{milestone}, which gives him a full sprint. Next decision: "
             "backport. Most customers run {affected_version} in production, "
             "and it's affected. Do we backport the fix to the "
             "{affected_version} patch line once it lands?"),
            ("user",
             "Yes — backport to the {affected_version} patch line. Most "
             "customers sit on it; a fix that ships only in {milestone} "
             "leaves them exposed for months. Condition it on the same "
             "tests passing."),
            ("assistant",
             "Recorded: fix targets {milestone} with backport to the "
             "{affected_version} patch line gated on the same test suite. "
             "Root-cause hypothesis to confirm: the cache invalidation path "
             "drops invalidations under load (queue overflow, not timing), "
             "which matches the load-only reproduction."),
            ("user",
             "{reporter} should add a regression test for the stale-read "
             "path — she offered on the ticket this morning."),
            ("assistant",
             "Noted: {reporter} adds the stale-read regression test. Full "
             "record for the ticket then: {bug_id} classified S2 (stale "
             "search results in {service}, workaround manual refresh, "
             "affecting about {affected_pct}% of peak queries since "
             "{report_date}), owner {assignee}, fix in {milestone}, "
             "backport to the {affected_version} patch line, regression test "
             "by {reporter}."),
            ("user",
             "That's the complete picture. Post it to the ticket so the "
             "thread stops re-litigating severity — support kept asking why "
             "it isn't S1."),
            ("assistant",
             "Posted, with the S1/S2 rationale spelled out: no outage, no "
             "wrong data, workaround exists. I'll flag the ticket for "
             "re-triage if the impact share grows beyond {affected_pct}% "
             "or a correctness report comes in."),
        ),
        final_user_template=(
            "Perfect. One more pass before I close the triage meeting: "
            "restate this ticket's decisions in order — severity and why, "
            "owner, milestone, backport target, and the regression test "
            "commitment — so the minutes match the ticket?"
        ),
        key_point_templates=(
            "{bug_id}: stale search results in {service}, filed by {reporter} on {report_date}",
            "severity decision: S2, not S1 — degraded-not-down, workaround via manual refresh",
            "owner: {assignee}; fix targets {milestone}",
            "backport decision: backport the fix to the {affected_version} patch line",
            "impact: about {affected_pct}% of peak search queries show pre-update results",
            "{reporter} adds the stale-read regression test",
        ),
        slot_options={
            "affected_pct": (8, 12, 17),
            "affected_version": ("5.2.0", "5.3.0", "6.0.1"),
            "assignee": ("Ravi Chandran", "Marek Dvořák", "June Park"),
            "bug_id": ("SRCH-2148", "SRCH-2203", "SRCH-2317"),
            "milestone": ("v6.1.0", "v6.2.0", "v7.0.0"),
            "report_date": ("2026-08-11", "2026-08-13", "2026-08-18"),
            "reporter": ("Sofia Marchetti", "Ana Beltrán", "Ingrid Halvorsen"),
            "service": ("catalog-search", "orders-search", "docs-search"),
        },
    ),
    "access_review": DialogueTemplate(
        history_turns=(
            ("user",
             "Access review time. Item one: {requester} from the analytics "
             "team requested {scope} on {system}. Ticket {ticket}, manager "
             "pre-approval attached. Your read?"),
            ("assistant",
             "Three questions before any grant: purpose, duration, and least "
             "privilege. What's the stated purpose, does it need write or "
             "is read-only enough, and is a fixed expiry defined or is this "
             "indefinite?"),
            ("user",
             "Purpose: one-off migration analysis for the annual audit — "
             "tracing row lineage across the archive. Read-only genuinely "
             "suffices; she never needs write. Duration: two to four weeks, "
             "and the audit closes {audit_close}."),
            ("assistant",
             "Then my recommendation: approve time-boxed, read-only, with "
             "automatic expiry — grant on {grant_date}, expires "
             "{expiry_date} without manual tracking. Auto-expiry beats "
             "calendar reminders; orphaned read access to {system} is "
             "exactly what last year's audit finding was about."),
            ("user",
             "Agreed — approve until {expiry_date}, then auto-expire. If "
             "the audit slips she re-requests; that's cheap. One condition "
             "from security: access via the bastion only, no direct "
             "connection strings."),
            ("assistant",
             "Recorded: bastion-only access, no direct connections, MFA "
             "inherited from the bastion profile. Two more items. First, "
             "the analytics team's standing access was last reviewed "
             "{last_review_date} — overdue by our 180-day policy; the whole "
             "team goes into the next quarterly cycle. Second: should the "
             "expiry revoke silently or with a 48-hour warning email?"),
            ("user",
             "Warning email, obviously — silent revocation mid-audit helps "
             "nobody. And yes, put the whole team into the quarterly cycle, "
             "next review {next_review_date}. What does the ticket record "
             "look like now?"),
            ("assistant",
             "Decision record for {ticket}: {requester} granted read-only "
             "{scope} on {system}, active {grant_date} to {expiry_date}, "
             "bastion-only with MFA, auto-expiry with 48-hour warning, "
             "re-request allowed if the audit extends. Standing-access "
             "re-review for the analytics team scheduled {next_review_date}. "
             "Approver line still needs your name and title."),
            ("user",
             "Approving as {approver_title} — put '{approver_name}' on it. "
             "And keep the approval trail linked to the audit workspace, "
             "not just the ticket."),
            ("assistant",
             "Done — approval recorded under {approver_name}, "
             "{approver_title}, trail linked to the audit workspace with "
             "the ticket cross-referenced. Anything else on this item, or "
             "move to the service-account cleanups?"),
            ("user",
             "Service accounts next, but close this one first: post the "
             "final decision record to the ticket so {requester} sees the "
             "expiry date before she starts."),
            ("assistant",
             "Posted. {requester} will see: read-only, bastion-only, "
             "expires {expiry_date} with a warning before revocation. "
             "Moving to service accounts — first one up is the orphaned "
             "deploy key from the March migration."),
        ),
        final_user_template=(
            "Before we move on: for my compliance notes, restate this "
            "approval decision in full — who requested, exactly what was "
            "granted and where, the access method condition, the expiry, "
            "and the team re-review date — as it now stands on the ticket?"
        ),
        key_point_templates=(
            "requester: {requester} from the analytics team (ticket {ticket})",
            "grant: read-only, {scope} on {system}, active {grant_date} to {expiry_date}",
            "condition: bastion-only access, no direct connections, auto-expiry with 48-hour warning",
            "approved by {approver_name}, {approver_title}; trail linked to the audit workspace",
            "standing-access re-review for the analytics team scheduled {next_review_date}",
        ),
        slot_options={
            "approver_name": ("Marta Ilves", "Iris Novak", "Tomas Weber"),
            "approver_title": ("Director of Data Platform", "Head of Infrastructure", "VP Engineering"),
            "audit_close": ("2026-09-30", "2026-10-15"),
            "expiry_date": ("2026-09-11", "2026-09-18"),
            "grant_date": ("2026-08-21", "2026-08-28"),
            "last_review_date": ("2026-01-15", "2026-02-02"),
            "next_review_date": ("2026-10-01", "2026-10-15"),
            "requester": ("Sofia Marchetti", "Ana Beltrán", "Ingrid Halvorsen"),
            "scope": ("read access to the production archive", "read access to the billing replica"),
            "system": ("the events warehouse", "the reporting cluster"),
            "ticket": ("SEC-1042", "SEC-1067", "SEC-1093"),
        },
    ),
    "release_planning": DialogueTemplate(
        history_turns=(
            ("user",
             "Planning session for {version}. Ship date is locked: "
             "{ship_date}, because the {event} announcement happens the "
             "same week and marketing is committed. Walk the board with me "
             "— what's at risk?"),
            ("assistant",
             "Scope status: of nine planned items, seven are code-complete "
             "or on track. Two risks. The bigger one: {feature} sits at "
             "roughly 60% with its test suite flaky for two weeks. "
             "{owner}'s latest estimate is two more weeks of work — which "
             "lands exactly on the freeze, with zero slack."),
            ("user",
             "Two more weeks with no slack before a locked date is a cut "
             "conversation. What's the exposure if {feature} slips out of "
             "{version}?"),
            ("assistant",
             "Recommendation: cut {feature} from {version} and commit it to "
             "the next minor version instead. Exposure is limited — it's a "
             "power-user capability, marketing's materials don't feature "
             "it, and shipping it half-tested against a fixed date is how "
             "regressions escape. Everything else on the board fits the "
             "date."),
            ("user",
             "Cut it. I'll tell {owner} today, with the next-minor-version "
             "commitment in writing so it doesn't read as a demotion. "
             "What's the second risk?"),
            ("assistant",
             "The schema migration behind the new export service. It's a "
             "blocking step: it must run before canary traffic starts, and "
             "it takes the cluster partially read-only for up to 40 "
             "minutes. That can't happen during business hours."),
            ("user",
             "So the migration runs {migration_date} during the maintenance "
             "window, 02:00. {dba_owner} owns it end to end — she ran the "
             "rehearsal twice on staging last week."),
            ("assistant",
             "Recorded: migration owned by {dba_owner}, executes "
             "{migration_date} at 02:00 with the rehearsed rollback script "
             "on standby. Rollout plan then: canary at 5% of traffic for "
             "{canary_hours} hours starting {canary_date}, full rollout the "
             "next morning only if error budget and migration checks hold. "
             "Code freeze {freeze_date}; after that, fixes only, no new "
             "scope."),
            ("user",
             "Canary at 5% from {canary_date}, full rollout next morning if "
             "the numbers hold. Freeze {freeze_date} — put it in the "
             "release calendar so nobody 'discovers' it later. What's the "
             "communication plan for the cut?"),
            ("assistant",
             "Suggested comms: a short release-notes entry framing "
             "{feature} as 'coming in the next minor version', plus a "
             "heads-up to the customer advisory board since two design "
             "partners tested early builds. I'll draft both. Recap of "
             "decisions: {version} ships {ship_date}; {feature} cut to the "
             "next minor version; migration {migration_date} at 02:00 owned "
             "by {dba_owner}; canary 5% from {canary_date} for "
             "{canary_hours} hours; code freeze {freeze_date}."),
            ("user",
             "Recap matches my notes. One addition: the advisory board "
             "heads-up goes out this week, before they read the release "
             "notes — the design partners specifically asked not to be "
             "surprised."),
            ("assistant",
             "Added: advisory-board note drafts today, out this week, "
             "release-notes entry follows at freeze. The full decision "
             "record is on the release page for the ship-review meeting."),
        ),
        final_user_template=(
            "Good. Last thing for the ship-review doc: restate the complete "
            "release plan as decided — ship date, what was cut and where "
            "it lands, migration owner and time, canary plan, and the "
            "freeze date — exactly as we agreed today?"
        ),
        key_point_templates=(
            "ship date is locked: {ship_date}, because the {event} announcement happens the same week",
            "cut decision: {feature} moves out of {version}, committed to the next minor version",
            "schema migration runs {migration_date} at 02:00, owned by {dba_owner}",
            "rollout: canary at 5% starting {canary_date} for {canary_hours} hours, full rollout next morning if checks hold",
            "code freeze {freeze_date}: fixes only, no new scope",
        ),
        slot_options={
            "canary_date": ("2026-10-06", "2026-10-08"),
            "canary_hours": (12, 24),
            "dba_owner": ("Priya Raman", "Ingrid Halvorsen", "Marek Dvořák"),
            "event": ("KubeCon", "Industry Summit", "annual customer conference"),
            "feature": ("the advanced query builder", "the bulk import rework", "the custom dashboards feature"),
            "freeze_date": ("2026-09-29", "2026-10-01"),
            "migration_date": ("2026-09-30", "2026-10-05"),
            "owner": ("Ravi Chandran", "June Park", "Kenji Sato"),
            "ship_date": ("2026-10-14", "2026-10-20"),
            "version": ("v7.0", "v6.5", "v8.0-beta"),
        },
    ),
}


def load_families(path: Path) -> list[dict]:
    """Load and validate the scenario-family registry (split authority).

    Refuses: unknown families, bad fact types, and any overlap between a
    family's eval and train seed ranges (场景族/种子分流 is structural, not
    a reviewer's memory).
    """
    path = Path(path)
    try:
        registry = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise DialogueError(f"{path}: cannot read: {e}") from e

    families: list[dict] = []
    seen: set[str] = set()
    for entry in registry.get("family", []):
        name = entry.get("name", "")
        if name not in FAMILY_TEMPLATES:
            raise DialogueError(f"{path}: family {name!r} has no scenario template")
        if name in seen:
            raise DialogueError(f"{path}: duplicate family {name!r}")
        seen.add(name)

        types = entry.get("fact_types", [])
        if not isinstance(types, list) or len(types) < 2 or not set(types) <= set(
            FACT_TYPES
        ):
            raise DialogueError(
                f"{path}: family {name!r}: fact_types must list at least "
                f"two of {list(FACT_TYPES)}"
            )
        eval_range = _seed_range(path, name, entry, "eval_seed_range")
        train_range = _seed_range(path, name, entry, "train_seed_range")
        if set(range(*_inclusive(eval_range))) & set(range(*_inclusive(train_range))):
            raise DialogueError(
                f"{path}: family {name!r}: eval and train seed ranges overlap — "
                "seeds must never cross the split (data-strategy 硬约束 4)"
            )
        families.append(
            {
                "name": name,
                "description": entry.get("description", ""),
                "fact_types": list(types),
                "license": entry.get("license", ""),
                "train_use": bool(entry.get("train_use", False)),
                "eval_seed_range": eval_range,
                "train_seed_range": train_range,
                "note": entry.get("note", ""),
            }
        )
    if not families:
        raise DialogueError(f"{path}: no [[family]] entries found")
    return families


def synthesize_entry(family: dict, seed: int) -> dict:
    """Synthesize one golden entry for (family, seed) — eval seeds only."""
    name = family["name"]
    _check_seed_is_eval(family, seed)
    template = FAMILY_TEMPLATES[name]
    rng = _family_rng(name, seed)
    # sorted() keeps the RNG draw order stable regardless of dict order
    slots = {
        key: rng.choice(options)
        for key, options in sorted(template.slot_options.items())
    }
    prompt, key_points = template.render(slots)
    return {
        "id": f"dlg-{name}-s{seed}",
        "load_type": "dialogue",
        "prompt": prompt,
        "key_points": key_points,
        "source": {
            "dataset": "synthetic",
            "subset": name,
            "license": family["license"],
            "split": "eval",
            "family": name,
            "seed": seed,
            "content_sha1": hashlib.sha1(prompt.encode("utf-8")).hexdigest(),
        },
    }


def build_dialogue_entries(families: list[dict], seed_plan: dict[str, list[int]]) -> list[dict]:
    """Build golden entries from a {family: [seeds]} plan, registry order."""
    by_name = {f["name"]: f for f in families}
    missing = sorted(set(seed_plan) - set(by_name))
    if missing:
        raise DialogueError(f"seed plan references families not in the registry: {missing}")
    for name, seeds in seed_plan.items():
        if len(set(seeds)) != len(seeds):
            raise DialogueError(
                f"seed plan for {name!r} repeats a seed — each (family, seed) "
                "pair yields exactly one entry"
            )
    entries: list[dict] = []
    for family in families:
        for seed in sorted(seed_plan.get(family["name"], [])):
            entries.append(synthesize_entry(family, seed))
    return entries


def build_dialogue_manifest(
    families: list[dict],
    *,
    golden_path: Path | None = None,
    review: dict | None = None,
) -> dict:
    """Generate the dialogue manifest (allocation record; pilot block when a
    frozen golden file is provided).

    With golden_path, every entry is cross-checked against regeneration:
    registered family, eval-side seed, matching id, no duplicates, and
    matching content_sha1. This is the drift gate — the manifest never
    blesses a golden file that no longer matches the registry + generator.

    `review` overrides the pilot review block (the owner sign-off record);
    it defaults to pending. The review block is owner-owned metadata layered
    on top of the generator output, so regenerating never silently resets
    an approval.
    """
    manifest = {
        "schema_version": 1,
        "method": (
            "multi-turn conversation templates (src/trillic/dialogue.py — five "
            "families ported from 0xnicholas/tokencamp "
            "compression/eval/conversation_gen.py, same author, five new) + "
            "seeded slot synthesis; prompt = user/assistant history + final "
            "user recall message, serialized as 'role: content' lines; "
            "key_points are history facts (names/dates/prior decisions) that "
            "quote the history verbatim so survival is judgeable after "
            "compression; per-family DISJOINT eval/train integer seed ranges "
            "in dialogue_families.toml are the split authority — golden draws "
            "from eval seeds only; the phase-2 training side will reuse the "
            "same templates and machinery with train seeds (shared code, "
            "never shared seeds)"
        ),
        "entry_id_rule": "dlg-<family>-s<seed>",
        "rng": "random.Random(int.from_bytes(sha256('<family>:<seed>')[:8]))",
        "content_addressing": "sha1(prompt)",
        "families": [
            {
                "name": f["name"],
                "description": f["description"],
                "fact_types": f["fact_types"],
                "license": f["license"],
                "train_use": f["train_use"],
                "eval_seed_range": f["eval_seed_range"],
                "train_seed_range": f["train_seed_range"],
                "note": f["note"],
            }
            for f in families
        ],
    }
    if golden_path is not None:
        manifest["pilot"] = _pilot_record(families, Path(golden_path), review)
    return manifest


def _pilot_record(
    families: list[dict], golden_path: Path, review: dict | None
) -> dict:
    """Verify a frozen golden file against the registry and record it."""
    by_name = {f["name"]: f for f in families}
    rows: list[dict] = []
    for line_number, line in enumerate(
        golden_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise DialogueError(
                f"{golden_path}: line {line_number}: cannot parse JSON: {e}"
            ) from e

    seed_plan: dict[str, list[int]] = {}
    seen_ids: set[str] = set()
    for row in rows:
        entry_id = row.get("id", "?")
        source = row.get("source", {})
        name = source.get("family", "")
        if name not in by_name:
            raise DialogueError(
                f"{golden_path}: entry {entry_id!r}: family {name!r} is not in "
                "the registry"
            )
        seed = source.get("seed")
        _check_seed_is_eval(by_name[name], seed)
        regenerated = synthesize_entry(by_name[name], seed)
        if row.get("prompt") != regenerated["prompt"] or source.get(
            "content_sha1"
        ) != regenerated["source"]["content_sha1"]:
            raise DialogueError(
                f"{golden_path}: entry {entry_id!r}: content_sha1 mismatch — "
                "the golden file no longer matches regeneration from "
                "(registry, seed); re-freeze deliberately, never silently"
            )
        if entry_id != regenerated["id"]:
            raise DialogueError(
                f"{golden_path}: entry {entry_id!r}: id must be "
                f"{regenerated['id']!r}"
            )
        if entry_id in seen_ids:
            raise DialogueError(
                f"{golden_path}: entry {entry_id!r}: duplicate row — the pilot "
                "must list each (family, seed) exactly once"
            )
        seen_ids.add(entry_id)
        seed_plan.setdefault(name, []).append(seed)

    return {
        "file": Path(golden_path).name,  # stable across machines/checkout paths
        "entries": len(rows),
        "ids": [r["id"] for r in rows],
        "seed_plan": {name: sorted(seeds) for name, seeds in seed_plan.items()},
        "review": review if review is not None else _pending_review(),
    }


def _pending_review() -> dict:
    return make_review("pending")


def make_review(status: str, *, reviewer: str | None = None, notes: str = "") -> dict:
    """Build the pilot review (owner sign-off) block.

    The checklist is the durable record of what an approval vouches for; it
    lives here so the CLI flags and the default pending block can never
    drift apart.
    """
    return {
        "status": status,
        "reviewer": reviewer,
        "checked": [
            "each conversation reads as a realistic multi-turn exchange",
            "the final user message asks the assistant to recall history facts",
            "key_points are history facts (names/dates/prior decisions), not constraints or final-message-only claims",
            "seeded values embedded in key_points quote the history verbatim",
        ],
        "notes": notes,
    }


def _check_seed_is_eval(family: dict, seed) -> None:
    """Refuse train-side and out-of-pool seeds with a precise reason."""
    if not isinstance(seed, int):
        raise DialogueError(f"family {family['name']!r}: seed must be an int")
    name = family["name"]
    if seed in range(*_inclusive(family["train_seed_range"])):
        raise DialogueError(
            f"family {name!r}: seed {seed} is a TRAIN seed — golden sets draw "
            "from eval seeds only (data-strategy 硬约束 4: 种子分流)"
        )
    if seed not in range(*_inclusive(family["eval_seed_range"])):
        raise DialogueError(
            f"family {name!r}: seed {seed} is outside both seed pools "
            f"(eval {family['eval_seed_range']}, train {family['train_seed_range']})"
        )


def _family_rng(name: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{name}:{seed}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _seed_range(path: Path, name: str, entry: dict, field: str) -> list[int]:
    bounds = entry.get(field)
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or not all(isinstance(b, int) for b in bounds)
        or bounds[0] > bounds[1]
    ):
        raise DialogueError(
            f"{path}: family {name!r}: {field} must be [lo, hi] with lo <= hi"
        )
    return list(bounds)


def _inclusive(bounds: list[int]) -> tuple[int, int]:
    lo, hi = bounds
    return (lo, hi + 1)
