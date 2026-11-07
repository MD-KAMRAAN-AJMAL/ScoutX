"""Create the initial relational schema for file intelligence."""

from alembic import op

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("""\
    CREATE TABLE users (
    	id UUID NOT NULL, 
    	email VARCHAR(320), 
    	display_name VARCHAR(200), 
    	is_active BOOLEAN DEFAULT true NOT NULL, 
    	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	CONSTRAINT pk_users PRIMARY KEY (id), 
    	CONSTRAINT uq_users_email UNIQUE (email)
    )
    """)
    op.execute("""\
    CREATE TABLE agent_runs (
    	id UUID NOT NULL, 
    	user_id UUID NOT NULL, 
    	query TEXT NOT NULL, 
    	status VARCHAR(16) DEFAULT 'running' NOT NULL, 
    	plan JSONB DEFAULT '{}'::jsonb NOT NULL, 
    	outcome JSONB DEFAULT '{}'::jsonb NOT NULL, 
    	started_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	finished_at TIMESTAMP WITH TIME ZONE, 
    	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	CONSTRAINT pk_agent_runs PRIMARY KEY (id), 
    	CONSTRAINT ck_agent_runs_ck_agent_runs_status CHECK (status IN ('running','succeeded','failed','cancelled')), 
    	CONSTRAINT fk_agent_runs_user_id_users FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """)
    op.execute("""\
    CREATE INDEX ix_agent_runs_user_created ON agent_runs (user_id, created_at)
    """)
    op.execute("""\
    CREATE TABLE index_roots (
    	id UUID NOT NULL, 
    	user_id UUID NOT NULL, 
    	canonical_path TEXT NOT NULL, 
    	display_name VARCHAR(255), 
    	is_enabled BOOLEAN DEFAULT true NOT NULL, 
    	last_reconciled_at TIMESTAMP WITH TIME ZONE, 
    	settings JSONB DEFAULT '{}'::jsonb NOT NULL, 
    	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	CONSTRAINT pk_index_roots PRIMARY KEY (id), 
    	CONSTRAINT uq_index_roots_user_path UNIQUE (user_id, canonical_path), 
    	CONSTRAINT uq_index_roots_id_user UNIQUE (id, user_id), 
    	CONSTRAINT fk_index_roots_user_id_users FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """)
    op.execute("""\
    CREATE INDEX ix_index_roots_user_enabled ON index_roots (user_id, is_enabled)
    """)
    op.execute("""\
    CREATE TABLE projects (
    	id UUID NOT NULL, 
    	user_id UUID NOT NULL, 
    	name VARCHAR(255) NOT NULL, 
    	description TEXT, 
    	metadata_json JSONB DEFAULT '{}'::jsonb NOT NULL, 
    	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	CONSTRAINT pk_projects PRIMARY KEY (id), 
    	CONSTRAINT uq_projects_user_name UNIQUE (user_id, name), 
    	CONSTRAINT fk_projects_user_id_users FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """)
    op.execute("""\
    CREATE TABLE user_preferences (
    	user_id UUID NOT NULL, 
    	preferences JSONB DEFAULT '{}'::jsonb NOT NULL, 
    	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	CONSTRAINT pk_user_preferences PRIMARY KEY (user_id), 
    	CONSTRAINT fk_user_preferences_user_id_users FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """)
    op.execute("""\
    CREATE TABLE files (
    	id UUID NOT NULL, 
    	user_id UUID NOT NULL, 
    	root_id UUID NOT NULL, 
    	canonical_path TEXT NOT NULL, 
    	filename VARCHAR(1024) NOT NULL, 
    	extension VARCHAR(32), 
    	mime_type VARCHAR(255), 
    	size_bytes BIGINT NOT NULL, 
    	modified_at TIMESTAMP WITH TIME ZONE, 
    	device_id BIGINT, 
    	inode BIGINT, 
    	content_hash VARCHAR(128), 
    	version INTEGER DEFAULT '1' NOT NULL, 
    	indexing_status VARCHAR(24) DEFAULT 'pending' NOT NULL, 
    	last_indexed_at TIMESTAMP WITH TIME ZONE, 
    	extra_metadata JSONB DEFAULT '{}'::jsonb NOT NULL, 
    	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	CONSTRAINT pk_files PRIMARY KEY (id), 
    	CONSTRAINT uq_files_root_path UNIQUE (root_id, canonical_path), 
    	CONSTRAINT fk_files_root_user FOREIGN KEY(root_id, user_id) REFERENCES index_roots (id, user_id) ON DELETE CASCADE, 
    	CONSTRAINT ck_files_ck_files_size_nonnegative CHECK (size_bytes >= 0), 
    	CONSTRAINT ck_files_ck_files_version_positive CHECK (version >= 1), 
    	CONSTRAINT ck_files_ck_files_indexing_status CHECK (indexing_status IN ('pending','queued','processing','indexed','failed','excluded','deleted')), 
    	CONSTRAINT fk_files_user_id_users FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """)
    op.execute("""\
    CREATE INDEX ix_files_content_hash ON files (user_id, content_hash)
    """)
    op.execute("""\
    CREATE INDEX ix_files_filename_trgm ON files USING gin (filename gin_trgm_ops)
    """)
    op.execute("""\
    CREATE INDEX ix_files_root_status ON files (root_id, indexing_status)
    """)
    op.execute("""\
    CREATE INDEX ix_files_user_modified ON files (user_id, modified_at)
    """)
    op.execute("""\
    CREATE TABLE document_chunks (
    	id UUID NOT NULL, 
    	file_id UUID NOT NULL, 
    	chunk_index INTEGER NOT NULL, 
    	content TEXT NOT NULL, 
    	modality VARCHAR(32) DEFAULT 'text' NOT NULL, 
    	language VARCHAR(32), 
    	page_number INTEGER, 
    	line_start INTEGER, 
    	line_end INTEGER, 
    	start_time_ms BIGINT, 
    	end_time_ms BIGINT, 
    	section_title VARCHAR(512), 
    	content_version INTEGER DEFAULT '1' NOT NULL, 
    	qdrant_point_id UUID, 
    	chunk_metadata JSONB DEFAULT '{}'::jsonb NOT NULL, 
    	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	CONSTRAINT pk_document_chunks PRIMARY KEY (id), 
    	CONSTRAINT uq_chunks_file_version_index UNIQUE (file_id, content_version, chunk_index), 
    	CONSTRAINT ck_document_chunks_ck_chunks_index_nonnegative CHECK (chunk_index >= 0), 
    	CONSTRAINT ck_document_chunks_ck_chunks_version_positive CHECK (content_version >= 1), 
    	CONSTRAINT ck_document_chunks_ck_chunks_line_start_positive CHECK (line_start IS NULL OR line_start >= 1), 
    	CONSTRAINT ck_document_chunks_ck_chunks_line_end_positive CHECK (line_end IS NULL OR line_end >= 1), 
    	CONSTRAINT fk_document_chunks_file_id_files FOREIGN KEY(file_id) REFERENCES files (id) ON DELETE CASCADE, 
    	CONSTRAINT uq_document_chunks_qdrant_point_id UNIQUE (qdrant_point_id)
    )
    """)
    op.execute("""\
    CREATE INDEX ix_chunks_file_version ON document_chunks (file_id, content_version)
    """)
    op.execute("""\
    CREATE INDEX ix_chunks_modality_language ON document_chunks (modality, language)
    """)
    op.execute("""\
    CREATE TABLE file_activity (
    	id UUID NOT NULL, 
    	user_id UUID NOT NULL, 
    	file_id UUID, 
    	event_type VARCHAR(64) NOT NULL, 
    	details JSONB DEFAULT '{}'::jsonb NOT NULL, 
    	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	CONSTRAINT pk_file_activity PRIMARY KEY (id), 
    	CONSTRAINT fk_file_activity_user_id_users FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
    	CONSTRAINT fk_file_activity_file_id_files FOREIGN KEY(file_id) REFERENCES files (id) ON DELETE SET NULL
    )
    """)
    op.execute("""\
    CREATE INDEX ix_file_activity_user_time ON file_activity (user_id, created_at)
    """)
    op.execute("""\
    CREATE TABLE index_jobs (
    	id UUID NOT NULL, 
    	user_id UUID NOT NULL, 
    	root_id UUID, 
    	file_id UUID, 
    	job_type VARCHAR(32) NOT NULL, 
    	status VARCHAR(16) DEFAULT 'queued' NOT NULL, 
    	attempts INTEGER DEFAULT '0' NOT NULL, 
    	available_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	started_at TIMESTAMP WITH TIME ZONE, 
    	finished_at TIMESTAMP WITH TIME ZONE, 
    	error_message TEXT, 
    	payload JSONB DEFAULT '{}'::jsonb NOT NULL, 
    	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	CONSTRAINT pk_index_jobs PRIMARY KEY (id), 
    	CONSTRAINT ck_index_jobs_ck_index_jobs_status CHECK (status IN ('queued','running','succeeded','failed','cancelled')), 
    	CONSTRAINT ck_index_jobs_ck_index_jobs_attempts_nonnegative CHECK (attempts >= 0), 
    	CONSTRAINT fk_index_jobs_user_id_users FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
    	CONSTRAINT fk_index_jobs_root_id_index_roots FOREIGN KEY(root_id) REFERENCES index_roots (id) ON DELETE SET NULL, 
    	CONSTRAINT fk_index_jobs_file_id_files FOREIGN KEY(file_id) REFERENCES files (id) ON DELETE CASCADE
    )
    """)
    op.execute("""\
    CREATE INDEX ix_index_jobs_file_status ON index_jobs (file_id, status)
    """)
    op.execute("""\
    CREATE INDEX ix_index_jobs_status_created ON index_jobs (status, created_at)
    """)
    op.execute("""\
    CREATE TABLE project_files (
    	project_id UUID NOT NULL, 
    	file_id UUID NOT NULL, 
    	added_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	relationship_type VARCHAR(32) DEFAULT 'related' NOT NULL, 
    	CONSTRAINT pk_project_files PRIMARY KEY (project_id, file_id), 
    	CONSTRAINT fk_project_files_project_id_projects FOREIGN KEY(project_id) REFERENCES projects (id) ON DELETE CASCADE, 
    	CONSTRAINT fk_project_files_file_id_files FOREIGN KEY(file_id) REFERENCES files (id) ON DELETE CASCADE
    )
    """)
    op.execute("""\
    CREATE TABLE recommendation_feedback (
    	id UUID NOT NULL, 
    	user_id UUID NOT NULL, 
    	file_id UUID NOT NULL, 
    	agent_run_id UUID, 
    	rating INTEGER NOT NULL, 
    	feedback_text TEXT, 
    	action VARCHAR(32), 
    	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    	CONSTRAINT pk_recommendation_feedback PRIMARY KEY (id), 
    	CONSTRAINT ck_recommendation_feedback_ck_feedback_rating_range CHECK (rating >= 1 AND rating <= 5), 
    	CONSTRAINT fk_recommendation_feedback_user_id_users FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE, 
    	CONSTRAINT fk_recommendation_feedback_file_id_files FOREIGN KEY(file_id) REFERENCES files (id) ON DELETE CASCADE, 
    	CONSTRAINT fk_recommendation_feedback_agent_run_id_agent_runs FOREIGN KEY(agent_run_id) REFERENCES agent_runs (id) ON DELETE SET NULL
    )
    """)
    op.execute("""\
    CREATE INDEX ix_feedback_user_created ON recommendation_feedback (user_id, created_at)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS recommendation_feedback CASCADE")
    op.execute("DROP TABLE IF EXISTS project_files CASCADE")
    op.execute("DROP TABLE IF EXISTS index_jobs CASCADE")
    op.execute("DROP TABLE IF EXISTS file_activity CASCADE")
    op.execute("DROP TABLE IF EXISTS document_chunks CASCADE")
    op.execute("DROP TABLE IF EXISTS files CASCADE")
    op.execute("DROP TABLE IF EXISTS user_preferences CASCADE")
    op.execute("DROP TABLE IF EXISTS projects CASCADE")
    op.execute("DROP TABLE IF EXISTS index_roots CASCADE")
    op.execute("DROP TABLE IF EXISTS agent_runs CASCADE")
    op.execute("DROP TABLE IF EXISTS users CASCADE")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
