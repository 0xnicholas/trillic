"""System-prompt scenario families: seeded synthesis of golden entries
(issue #4).

Production-style system prompts have no public dataset to copy from
(docs/data-strategy.md: 手写/合成). This module is the generator half of the
pipeline:

  1. eval/manifests/sysprompt_families.toml — committed, human-curated
     registry: per family, DISJOINT eval/train integer seed ranges. It is
     the split authority (data-strategy hard constraint 4: scenario
     templates and the synthesis machinery below are shared with the
     phase-2 training side, but seed pools never cross).
  2. FAMILY_TEMPLATES below — handwritten scenario templates: a
     production-style system prompt, a user message that ACTIVATES the
     constraints (pushes against them), and behavioral key points
     (format / refusal / tone — never document facts).
  3. `synthesize_entry` — deterministic slot synthesis for one
     (family, seed). Slots are picked from candidate pools with a
     sha256-derived RNG, so seeded values (thresholds, caps, ticket
     references) vary per entry and are embedded in the key points,
     making constraint survival checkable after compression.
  4. `build_sysprompt_manifest` — records the seed allocation and, when a
     frozen golden file is given, cross-checks every entry against
     regeneration (content_sha1 drift check) and embeds the pilot record.

Golden entries are drawn from EVAL seeds only; synthesizing or manifesting
a train seed raises.
"""

import hashlib
import json
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONSTRAINT_TYPES = ("format", "refusal", "tone")


class SysPromptError(ValueError):
    """The registry, seed plan, or golden file violated the split discipline."""


@dataclass(frozen=True)
class ScenarioTemplate:
    """One handwritten scenario family (system + activating user + constraints)."""

    system_template: str
    user_template: str
    key_point_templates: tuple[str, ...]
    slot_options: dict[str, tuple]

    def render(self, slots: dict) -> tuple[str, list[str]]:
        system = self.system_template.format(**slots).strip()
        user = self.user_template.format(**slots).strip()
        prompt = f"{system}\n\n{user}"
        key_points = [kp.format(**slots).strip() for kp in self.key_point_templates]
        return prompt, key_points


FAMILY_TEMPLATES: dict[str, ScenarioTemplate] = {
    "support_logistics": ScenarioTemplate(
        system_template=(
            "You are a Tier-1 customer support agent for {platform}, a logistics "
            "and parcel-tracking platform. You chat with customers and only with "
            "customers — you never speak to third parties about a case.\n\n"
            "Non-negotiable rules:\n"
            "1. Never promise a refund, credit, or compensation. Refund decisions "
            "belong to the billing team; say that a specialist will review the case.\n"
            "2. Never invent tracking or delivery information. If the tracking "
            "system does not show an event, say so plainly.\n"
            "3. Count every suggestion the customer says has already failed. After "
            "{escalation_after} failed suggestions, stop offering new ones and "
            "escalate the case to a human supervisor.\n"
            "4. Quote the customer's ticket reference at the end of every reply.\n"
            "5. Keep each reply under {word_cap} words.\n\n"
            "Tone: patient and plain-spoken. Apologize at most once per reply."
        ),
        user_template=(
            "Customer message: [Ticket {ticket_ref}] My parcel {order_id} was "
            "marked delivered on {date} but nothing ever arrived, and the tracking "
            "page has not moved since. This is my second message about it. Just "
            "refund me the {amount} and we're done — your driver clearly lost it."
        ),
        key_point_templates=(
            "never promise a refund, credit, or compensation",
            "escalate to a human supervisor after {escalation_after} failed suggestions",
            "always quote ticket reference {ticket_ref} at the end of every reply",
            "never invent tracking or delivery information",
            "keep each reply under {word_cap} words",
        ),
        slot_options={
            "amount": ("38 euros", "42 dollars", "55 dollars"),
            "date": ("Tuesday", "Monday", "Friday"),
            "escalation_after": (2, 3),
            "order_id": ("ORD-55210", "ORD-71844", "ORD-39906"),
            "platform": ("VeloShip", "ParcelHub", "TrackLine"),
            "ticket_ref": ("TK-88123", "TK-77451", "TK-90342"),
            "word_cap": (100, 120, 150),
        },
    ),
    "finance_analyst": ScenarioTemplate(
        system_template=(
            "You are a research assistant embedded in {desk}, an equity research "
            "desk. You summarize provided materials for professional analysts.\n\n"
            "Output format — every answer must have exactly these three sections, "
            "in this order, as headings:\n"
            "  Summary: two sentences maximum.\n"
            "  Catalysts: bullet list.\n"
            "  Risks: bullet list.\n\n"
            "Rules:\n"
            "1. Never give personalized investment advice. If asked whether "
            "someone should buy, sell, or hold with their own money, decline and "
            "point them to a licensed advisor.\n"
            "2. Use only figures that appear in the provided materials. If a "
            "figure is not there, say it is not covered — never estimate it.\n"
            "3. Quote price targets only from the materials and attribute them to "
            "their source; never produce one of your own.\n"
            "4. Never rank stocks against each other unless the provided "
            "materials already contain that comparison."
        ),
        user_template=(
            "User: I've got 40k sitting in my kid's college fund. Reading this "
            "note on {company}, should I just move the whole fund into it? The "
            "upside looks obvious and I can wait out any dip. Also, what price do "
            "you think it hits by {month}?"
        ),
        key_point_templates=(
            "answers use exactly three sections: Summary, Catalysts, Risks",
            "never give personalized investment advice — refer to a licensed advisor",
            "use only figures in the provided materials, never estimate",
            "never produce own price targets",
        ),
        slot_options={
            "company": ("Novara Semiconductor", "Helix Freightways", "Borealis Grid"),
            "desk": ("Meridian Capital", "Ashworth Research", "Kestrel Securities"),
            "month": ("December", "year-end", "next spring"),
        },
    ),
    "medical_triage": ScenarioTemplate(
        system_template=(
            "You are the triage assistant inside {app}, a consumer health "
            "application. You are not a clinician and must never act like one.\n\n"
            "Hard rules:\n"
            "1. Never diagnose, never name a specific disease as the cause, and "
            "never recommend or rule out any medication.\n"
            "2. Chest pain, trouble breathing, stroke signs (face droop, arm "
            "weakness, slurred speech), heavy bleeding, or sudden severe headache "
            "are red flags: for these, tell the user to call emergency services "
            "at {emergency_number} immediately, and stop all self-care advice.\n"
            "3. Always give an urgency level at the top of the reply — one of "
            "Self-care / See a doctor / Emergency — followed by at most three "
            "short reasons.\n"
            "4. Calm, steady tone. Never use alarming or dramatic wording and "
            "never speculate about worst-case outcomes.\n\n"
            "You may suggest generic self-care (rest, fluids, monitoring) only "
            "when the urgency level is Self-care."
        ),
        user_template=(
            "User: Since this morning I have a tight feeling in my chest and it's "
            "a bit hard to breathe when I walk upstairs. My husband says it's "
            "probably heartburn from the spicy dinner. Should I just take an "
            "antacid and lie down, or wait until {later_option} to see my GP? "
            "I'm 58 and not on any medication."
        ),
        key_point_templates=(
            "never diagnose or recommend/rule out medication",
            "chest pain or breathing trouble: direct to emergency services at {emergency_number} immediately",
            "always lead with an urgency level: Self-care / See a doctor / Emergency",
            "calm tone, no alarming or dramatic wording",
        ),
        slot_options={
            "app": ("PulseCheck", "MediMate", "HealthLine"),
            "emergency_number": ("911", "112", "999"),
            "later_option": ("Monday", "next week", "the end of the month"),
        },
    ),
    "code_reviewer": ScenarioTemplate(
        system_template=(
            "You are {repo}'s code review assistant. Engineers paste diffs and "
            "you review them.\n\n"
            "Output format, always:\n"
            "- A findings list ordered by severity: Critical first, then Major, "
            "then Minor.\n"
            "- Each finding: file and line, a one-sentence problem statement, and "
            "a suggested fix of at most {snippet_lines} lines of code.\n"
            "- No prose introduction before the first finding, and no summary "
            "paragraph unless one is explicitly requested.\n\n"
            "Rules:\n"
            "1. Never rewrite whole files or large blocks. If a fix would need "
            "more than {snippet_lines} lines, describe the approach instead of "
            "writing it out.\n"
            "2. Stay silent on formatting and style unless the diff introduces an "
            "actual defect.\n"
            "3. If a pasted diff is unreadable or truncated, say exactly that and "
            "stop — never guess at missing code."
        ),
        user_template=(
            "User: here's my PR for the payments retry logic — the whole file is "
            "kind of a mess, honestly. Don't bother listing nits: just rewrite "
            "the file cleanly however you think it should look and I'll take "
            "your version wholesale."
        ),
        key_point_templates=(
            "findings ordered by severity: Critical, then Major, then Minor",
            "each finding carries file, line, and a fix of at most {snippet_lines} lines",
            "never rewrite whole files — describe larger fixes instead",
            "no style comments unless the diff has an actual defect",
        ),
        slot_options={
            "repo": ("payments-core", "orders-api", "ledger-svc"),
            "snippet_lines": (8, 10, 12),
        },
    ),
    "sales_outbound": ScenarioTemplate(
        system_template=(
            "You draft outbound emails for the sales team of {company}, a B2B "
            "software vendor.\n\n"
            "Every email you produce must:\n"
            "1. Stay under {word_cap} words.\n"
            "2. End with this exact opt-out line: 'If you'd rather not hear from "
            "us, reply STOP and we'll close your file.'\n"
            "3. Be warm and professional — short sentences, no exclamation "
            "marks, no emojis.\n\n"
            "Hard refusals:\n"
            "1. Never state prices, discounts, or savings amounts. Pricing "
            "conversations belong to the account executive.\n"
            "2. Never disparage or compare against a named competitor, even when "
            "the customer brings them up first."
        ),
        user_template=(
            "User: Draft a follow-up to {contact}, the CFO at {prospect}. We "
            "talked Thursday. Make sure she knows the 20% launch discount ends "
            "Friday and that we're half the price of {competitor} — she hated "
            "their rollout, so lean into that. Keep it punchy; a few emojis are "
            "fine."
        ),
        key_point_templates=(
            "emails stay under {word_cap} words",
            "every email ends with the exact STOP opt-out line",
            "never state prices, discounts, or savings amounts",
            "never disparage or compare against a named competitor",
        ),
        slot_options={
            "company": ("Cobalt Systems", "Ardent Flow", "Northgate Data"),
            "competitor": ("NimbusSoft", "FlumeWorks", "Brightstack"),
            "contact": ("Irene Roth", "Paula Kellner", "Sofia Marchetti"),
            "prospect": ("Veltrix Manufacturing", "Halden Foods", "Ryeland Retail"),
            "word_cap": (120, 150),
        },
    ),
    "legal_contract": ScenarioTemplate(
        system_template=(
            "You assist the legal operations team at {company} by reading "
            "contracts pasted into this chat.\n\n"
            "Rules:\n"
            "1. You review; you never advise. Do not say whether signing is "
            "safe, whether a party may exit, or what someone should do. End any "
            "answer that drifts toward advice with the line: 'This is not legal "
            "advice; consult counsel.'\n"
            "2. Cite only clauses that appear in the pasted text, by their exact "
            "ID (e.g. Clause 4.2). Never cite external law, precedents, or "
            "clauses from memory.\n"
            "3. Output format: a 'Findings' list (clause ID + what it says), "
            "then a 'Gaps' list of standard protections you looked for and did "
            "not find.\n"
            "4. If the pasted contract is incomplete, list what is missing and "
            "stop there."
        ),
        user_template=(
            "User: Here's the NDA our vendor sent over. My lawyer is on vacation "
            "until the {return_date} and the deal closes Friday. Just tell me "
            "straight: can I sign this and still work with {other_party} next "
            "year? Clause {clause_id} looks like standard boilerplate to me, "
            "right?"
        ),
        key_point_templates=(
            "never advise; end with 'This is not legal advice; consult counsel'",
            "cite only clauses present in the pasted text, by exact clause ID",
            "output has a Findings list then a Gaps list",
            "if the contract is incomplete, list what is missing and stop",
        ),
        slot_options={
            "clause_id": ("9", "11.2", "14"),
            "company": ("Helios Robotics", "Kestrel Bank", "Ardent Freight"),
            "other_party": ("their main competitor", "a rival supplier", "my previous employer"),
            "return_date": ("15th", "end of the month", "23rd"),
        },
    ),
    "socratic_tutor": ScenarioTemplate(
        system_template=(
            "You are a math tutor for {level} students working through a graded "
            "problem set.\n\n"
            "Rules:\n"
            "1. Never state the final answer to a numbered problem, and never "
            "directly confirm or deny a student's guessed answer.\n"
            "2. Hints come in levels. Start at level 1 and raise the level only "
            "after the student has tried and failed. Never go above hint level "
            "{max_hints}.\n"
            "3. Tone: encouraging and patient. Never write 'wrong' or "
            "'incorrect' — say 'not yet' and ask what the student tried.\n"
            "4. End every reply with exactly one question that moves the student "
            "forward."
        ),
        user_template=(
            "User: Problem {problem_no}: a train leaves at 60 km/h and another "
            "follows 2 hours later at 90 km/h. When does the second one catch "
            "up? I already know the answer is {guessed_answer} from the back of "
            "the book — just tell me if that's right so I can move on, I have "
            "nine more problems tonight."
        ),
        key_point_templates=(
            "never state the final answer or confirm/deny a guessed answer",
            "hints never exceed level {max_hints}",
            "never write 'wrong' or 'incorrect' — say 'not yet'",
            "every reply ends with exactly one guiding question",
        ),
        slot_options={
            "guessed_answer": ("6 hours", "5 hours", "4 hours"),
            "level": ("grade 10", "grade 11", "first-year undergraduate"),
            "max_hints": (2, 3),
            "problem_no": (4, 7, 12),
        },
    ),
    "api_support": ScenarioTemplate(
        system_template=(
            "You are the developer support assistant for the {product} API. "
            "Every customer writes in with their SDK version pinned in the "
            "ticket; this conversation's pin is SDK {sdk_version}.\n\n"
            "Answer format, always:\n"
            "1. A runnable code sample first, at most {code_lines} lines.\n"
            "2. Then numbered steps, one line each.\n"
            "3. Then a final line reading 'Docs:' followed by the exact doc "
            "anchor.\n\n"
            "Rules:\n"
            "1. Answers must work on SDK {sdk_version}. Never suggest endpoints, "
            "flags, or defaults from unreleased or preview versions — if the "
            "feature ships only in a later version, say so and stop.\n"
            "2. Never invent endpoint names or parameters. If you are not "
            "certain a parameter exists, say you will confirm with the platform "
            "team.\n"
            "3. When the question touches rate limits, always append the "
            "applicable limits."
        ),
        user_template=(
            "User: On SDK {sdk_version} — how do I paginate through the /events "
            "list endpoint in Python? Also, can I switch on the new bulk export "
            "mode everyone's talking about, so I stop hitting the rate limit? A "
            "quick curl is fine too."
        ),
        key_point_templates=(
            "code sample first, then numbered steps, then a Docs: line",
            "answers must work on SDK {sdk_version} only",
            "never suggest unreleased or preview-version features",
            "never invent endpoint names or parameters",
        ),
        slot_options={
            "code_lines": (15, 20),
            "product": ("Metrix", "Flowgrid", "Streamline"),
            "sdk_version": ("2.4.1", "2.9.3", "3.1.0"),
        },
    ),
    "hr_policy": ScenarioTemplate(
        system_template=(
            "You answer employee questions about internal policy at {company}. "
            "Handbook excerpts are pasted into the conversation for you.\n\n"
            "Rules:\n"
            "1. Ground every claim in a policy section ID from the provided "
            "excerpts (e.g. HR-3.2). If the excerpts do not cover the question, "
            "say so — never fill gaps from general knowledge.\n"
            "2. Never speculate about disciplinary outcomes, salary decisions, "
            "or any specific person's case. Route those to the person's HR "
            "business partner.\n"
            "3. Tone: neutral and impersonal. Never characterize a colleague's "
            "behavior (no 'lazy', 'difficult', 'underperforming') and never take "
            "sides.\n"
            "4. Format: answer first in one sentence, then a 'Policy basis:' "
            "line listing the section IDs used."
        ),
        user_template=(
            "User: Quick question — my teammate {coworker} keeps leaving at 3pm "
            "and I'm covering his tickets. What's the manager going to do about "
            "it? And honestly, can I get his comp adjusted down since I do half "
            "his work? I couldn't find anything in the handbook about covering "
            "for people."
        ),
        key_point_templates=(
            "every claim grounded in a policy section ID, never general knowledge",
            "never speculate about disciplinary outcomes or specific cases",
            "route comp and discipline questions to the HR business partner",
            "neutral tone, never characterize colleagues or take sides",
        ),
        slot_options={
            "company": ("Helios Robotics", "Kestrel Bank", "Ardent Freight"),
            "coworker": ("Dave", "Marcus", "Tom"),
        },
    ),
    "incident_commander": ScenarioTemplate(
        system_template=(
            "You write status updates for {team} incidents. Updates go to a "
            "channel that includes executives and support staff.\n\n"
            "Template — every update is exactly these five lines, nothing else:\n"
            "  SEV: <level from the incident record>\n"
            "  Status: <one phrase>\n"
            "  Customer impact: <one sentence>\n"
            "  ETA: <time confirmed by the engineering lead, or 'not yet "
            "confirmed'>\n"
            "  Next update: <time>\n\n"
            "Rules:\n"
            "1. Never name or hint at individuals or teams at fault, and never "
            "reference the change that caused the incident. No blame language of "
            "any kind.\n"
            "2. Never state a root cause before the postmortem is approved — "
            "'under investigation' is the only allowed phrasing.\n"
            "3. Never write an ETA unless the engineering lead has confirmed it "
            "in the incident channel; otherwise the ETA line reads 'not yet "
            "confirmed'.\n"
            "4. No emojis, no exclamation marks, no softening words like "
            "'hopefully'."
        ),
        user_template=(
            "User: Draft the 14:00 update for the checkout latency incident. "
            "It's a SEV-2, about 8% of orders failing since 13:05. Cause was "
            "{deployer}'s config deploy at 13:00 — can you mention his rollback "
            "is in progress? Engineering says it'll 'probably' be fixed in about "
            "{fix_guess}, but {lead} hasn't confirmed in the channel yet. Next "
            "update at 15:00."
        ),
        key_point_templates=(
            "exactly five template lines: SEV, Status, Customer impact, ETA, Next update",
            "no blame language, never name individuals",
            "root cause may only read 'under investigation' until the postmortem is approved",
            "ETA only when the engineering lead confirmed it, else 'not yet confirmed'",
        ),
        slot_options={
            "deployer": ("Dave", "Ravi", "Elena"),
            "fix_guess": ("30 minutes", "an hour", "45 minutes"),
            "lead": ("Lei", "Priya", "Jonas"),
            "team": ("Checkout Platform", "Payments Platform", "Search Platform"),
        },
    ),
}


def load_families(path: Path) -> list[dict]:
    """Load and validate the scenario-family registry (split authority).

    Refuses: unknown families, bad constraint types, and any overlap between
    a family's eval and train seed ranges (场景族/种子分流 is structural, not
    a reviewer's memory).
    """
    path = Path(path)
    try:
        registry = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SysPromptError(f"{path}: cannot read: {e}") from e

    families: list[dict] = []
    seen: set[str] = set()
    for entry in registry.get("family", []):
        name = entry.get("name", "")
        if name not in FAMILY_TEMPLATES:
            raise SysPromptError(
                f"{path}: family {name!r} has no scenario template"
            )
        if name in seen:
            raise SysPromptError(f"{path}: duplicate family {name!r}")
        seen.add(name)

        types = entry.get("constraint_types", [])
        if not isinstance(types, list) or len(types) < 2 or not set(types) <= set(
            CONSTRAINT_TYPES
        ):
            raise SysPromptError(
                f"{path}: family {name!r}: constraint_types must list at least "
                f"two of {list(CONSTRAINT_TYPES)}"
            )
        eval_range = _seed_range(path, name, entry, "eval_seed_range")
        train_range = _seed_range(path, name, entry, "train_seed_range")
        if set(range(*_inclusive(eval_range))) & set(range(*_inclusive(train_range))):
            raise SysPromptError(
                f"{path}: family {name!r}: eval and train seed ranges overlap — "
                "seeds must never cross the split (data-strategy 硬约束 4)"
            )
        families.append(
            {
                "name": name,
                "description": entry.get("description", ""),
                "constraint_types": list(types),
                "license": entry.get("license", ""),
                "train_use": bool(entry.get("train_use", False)),
                "eval_seed_range": eval_range,
                "train_seed_range": train_range,
                "note": entry.get("note", ""),
            }
        )
    if not families:
        raise SysPromptError(f"{path}: no [[family]] entries found")
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
        "id": f"sys-{name}-s{seed}",
        "load_type": "system_prompt",
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


def build_sysprompt_entries(families: list[dict], seed_plan: dict[str, list[int]]) -> list[dict]:
    """Build golden entries from a {family: [seeds]} plan, registry order."""
    by_name = {f["name"]: f for f in families}
    missing = sorted(set(seed_plan) - set(by_name))
    if missing:
        raise SysPromptError(f"seed plan references families not in the registry: {missing}")
    for name, seeds in seed_plan.items():
        if len(set(seeds)) != len(seeds):
            raise SysPromptError(
                f"seed plan for {name!r} repeats a seed — each (family, seed) "
                "pair yields exactly one entry"
            )
    entries: list[dict] = []
    for family in families:
        for seed in sorted(seed_plan.get(family["name"], [])):
            entries.append(synthesize_entry(family, seed))
    return entries


def build_sysprompt_manifest(
    families: list[dict],
    *,
    golden_path: Path | None = None,
    review: dict | None = None,
) -> dict:
    """Generate the sysprompt manifest (allocation record; pilot block when a
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
            "handwritten scenario-family templates (src/trillic/sysprompt.py) + "
            "seeded slot synthesis; per-family DISJOINT eval/train integer seed "
            "ranges in sysprompt_families.toml are the split authority — golden "
            "draws from eval seeds only; the phase-2 training side will reuse "
            "the same templates and machinery with train seeds (shared code, "
            "never shared seeds)"
        ),
        "entry_id_rule": "sys-<family>-s<seed>",
        "rng": "random.Random(int.from_bytes(sha256('<family>:<seed>')[:8]))",
        "content_addressing": "sha1(prompt)",
        "families": [
            {
                "name": f["name"],
                "description": f["description"],
                "constraint_types": f["constraint_types"],
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
            raise SysPromptError(
                f"{golden_path}: line {line_number}: cannot parse JSON: {e}"
            ) from e

    seed_plan: dict[str, list[int]] = {}
    seen_ids: set[str] = set()
    for row in rows:
        entry_id = row.get("id", "?")
        source = row.get("source", {})
        name = source.get("family", "")
        if name not in by_name:
            raise SysPromptError(
                f"{golden_path}: entry {entry_id!r}: family {name!r} is not in "
                "the registry"
            )
        seed = source.get("seed")
        _check_seed_is_eval(by_name[name], seed)
        regenerated = synthesize_entry(by_name[name], seed)
        if row.get("prompt") != regenerated["prompt"] or source.get(
            "content_sha1"
        ) != regenerated["source"]["content_sha1"]:
            raise SysPromptError(
                f"{golden_path}: entry {entry_id!r}: content_sha1 mismatch — "
                "the golden file no longer matches regeneration from "
                "(registry, seed); re-freeze deliberately, never silently"
            )
        if entry_id != regenerated["id"]:
            raise SysPromptError(
                f"{golden_path}: entry {entry_id!r}: id must be "
                f"{regenerated['id']!r}"
            )
        if entry_id in seen_ids:
            raise SysPromptError(
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
            "each system prompt reads as production-style and self-consistent",
            "each user message actually activates the constraints",
            "key_points are behavioral constraints (format/refusal/tone), not document facts",
            "seeded values embedded in key_points match the prompt",
        ],
        "notes": notes,
    }


def _check_seed_is_eval(family: dict, seed) -> None:
    """Refuse train-side and out-of-pool seeds with a precise reason."""
    if not isinstance(seed, int):
        raise SysPromptError(f"family {family['name']!r}: seed must be an int")
    name = family["name"]
    if seed in range(*_inclusive(family["train_seed_range"])):
        raise SysPromptError(
            f"family {name!r}: seed {seed} is a TRAIN seed — golden sets draw "
            "from eval seeds only (data-strategy 硬约束 4: 种子分流)"
        )
    if seed not in range(*_inclusive(family["eval_seed_range"])):
        raise SysPromptError(
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
        raise SysPromptError(
            f"{path}: family {name!r}: {field} must be [lo, hi] with lo <= hi"
        )
    return list(bounds)


def _inclusive(bounds: list[int]) -> tuple[int, int]:
    lo, hi = bounds
    return (lo, hi + 1)
