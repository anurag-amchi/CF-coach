"""The 6 database tables, written as SQLAlchemy 2 models.

One class = one table. One attribute = one column.
  Mapped[int]         -> NOT NULL integer column
  Mapped[int | None]  -> nullable integer column
  mapped_column(...)  -> extra details: primary key, foreign key, exact type, default

See the SQL these turn into:  uv run python -m cf_coach.db.models
Design reasons: docs/decisions.md #14.
"""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Parent class of every model. Base.metadata collects all tables; Alembic reads it."""


class User(Base):
    """One row per Codeforces handle we have synced."""
    __tablename__ = "users"

    handle: Mapped[str] = mapped_column(String(64), primary_key=True)
    rating: Mapped[int | None]                     # None for unrated users
    last_synced_sub_id: Mapped[int | None] = mapped_column(BigInteger)  # incremental sync stops here
    synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Problem(Base):
    """Every problem in the CF problemset. Shared by all users, refreshed daily."""
    __tablename__ = "problems"

    # A problem is identified by (contest_id, idx), e.g. (2000, "B") -> problem 2000B.
    contest_id: Mapped[int] = mapped_column(primary_key=True)
    idx: Mapped[str] = mapped_column(String(10), primary_key=True)  # CF calls it "index": "A", "C1"
    name: Mapped[str] = mapped_column(Text)
    rating: Mapped[int | None]                     # about 2% of problems have no rating
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)


class Submission(Base):
    """Every submission of every synced user."""
    __tablename__ = "submissions"

    # CF's own submission id. Using it as the primary key makes re-syncs idempotent:
    # inserting the same submission twice hits the key instead of creating a duplicate.
    # autoincrement=False: CF gives us the id; Postgres must not generate one.
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    handle: Mapped[str] = mapped_column(ForeignKey("users.handle", ondelete="CASCADE"))
    # No foreign key to problems: some submitted problems (gym, some contests) are not
    # in the problemset, and the insert would fail.
    contest_id: Mapped[int]
    idx: Mapped[str] = mapped_column(String(10))
    verdict: Mapped[str | None] = mapped_column(String(32))   # None while still being judged
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    participant_type: Mapped[str] = mapped_column(String(32))  # CONTESTANT, PRACTICE, VIRTUAL...
    relative_time_s: Mapped[int | None]            # seconds into the contest; None for practice

    __table_args__ = (
        # "This user's submissions, in time order": used by both sync and analytics.
        Index("ix_submissions_handle_created_at", "handle", "created_at"),
    )


class TagSkill(Base):
    """Output of the analytics engine: one row per (user, tag)."""
    __tablename__ = "tag_skills"

    handle: Mapped[str] = mapped_column(ForeignKey("users.handle", ondelete="CASCADE"), primary_key=True)
    tag: Mapped[str] = mapped_column(String(64), primary_key=True)
    skill: Mapped[float]                           # Elo-style estimate, e.g. 1450.0
    attempts: Mapped[int]
    abandon_rate: Mapped[float]                    # 0.0 to 1.0
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Plan(Base):
    """A weekly practice plan saved by the agent (only after validate_plan passes)."""
    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(primary_key=True)  # auto-increments
    handle: Mapped[str] = mapped_column(ForeignKey("users.handle", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    constraints: Mapped[dict] = mapped_column(JSONB)   # e.g. {"hours": 5, "avoid_tags": ["geometry"]}
    items: Mapped[list] = mapped_column(JSONB)         # e.g. [{"problem": "2000B", "reason": "..."}]
    status: Mapped[str] = mapped_column(String(16), default="active")  # active / done / replaced


class AgentRun(Base):
    """One row per agent run: the observability log for debugging, cost and evals."""
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    handle: Mapped[str | None] = mapped_column(ForeignKey("users.handle", ondelete="CASCADE"))
    # None when the run saved no plan (hit max_steps or failed). If the plan is
    # deleted, keep the run log and just clear the link.
    plan_id: Mapped[int | None] = mapped_column(ForeignKey("plans.id", ondelete="SET NULL"))
    model: Mapped[str] = mapped_column(String(64))
    steps: Mapped[list] = mapped_column(JSONB)         # every message and tool call in order
    input_tokens: Mapped[int]
    output_tokens: Mapped[int]
    latency_ms: Mapped[int]
    outcome: Mapped[str] = mapped_column(String(16))   # saved / max_steps / error
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ---------- Manual check: print the CREATE TABLE SQL for Postgres ----------

if __name__ == "__main__":
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateIndex, CreateTable

    dialect = postgresql.dialect()
    for table in Base.metadata.sorted_tables:      # sorted so referenced tables come first
        print(str(CreateTable(table).compile(dialect=dialect)).strip() + ";")
        for index in table.indexes:
            print(str(CreateIndex(index).compile(dialect=dialect)) + ";")
        print()
