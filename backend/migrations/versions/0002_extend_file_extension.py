"""Allow file extensions up to the filesystem filename limit."""

from alembic import op
import sqlalchemy as sa

revision = "0002_extend_file_extension"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "files",
        "extension",
        existing_type=sa.String(length=32),
        type_=sa.String(length=255),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "files",
        "extension",
        existing_type=sa.String(length=255),
        type_=sa.String(length=32),
        existing_nullable=True,
    )
