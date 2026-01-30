"""Create predictions table

Revision ID: 001
Revises: 
Create Date: 2026-01-30

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create predictions table"""
    op.create_table(
        'predictions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('input_text', sa.Text(), nullable=False),
        sa.Column('predicted_sentiment', sa.String(20), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('latency_ms', sa.Float(), nullable=False),
        sa.Column('model_version', sa.String(50), nullable=False, server_default='distilbert-v1'),
        sa.Column('cache_hit', sa.Boolean(), nullable=False, server_default='false'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('ix_predictions_id', 'predictions', ['id'])
    op.create_index('ix_predictions_timestamp', 'predictions', ['timestamp'])
    op.create_index('ix_predictions_predicted_sentiment', 'predictions', ['predicted_sentiment'])
    op.create_index('ix_predictions_timestamp_sentiment', 'predictions', ['timestamp', 'predicted_sentiment'])


def downgrade() -> None:
    """Drop predictions table"""
    op.drop_index('ix_predictions_timestamp_sentiment', table_name='predictions')
    op.drop_index('ix_predictions_predicted_sentiment', table_name='predictions')
    op.drop_index('ix_predictions_timestamp', table_name='predictions')
    op.drop_index('ix_predictions_id', table_name='predictions')
    op.drop_table('predictions')

