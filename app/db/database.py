"""Database setup with async SQLAlchemy."""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Create async engine
engine = create_async_engine(
    url=os.getenv("DATABASE_URL"),
    echo=os.getenv("DEBUG"),
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,      # recycle connections every 30 min
    pool_pre_ping=True,     # avoid stale connections
    future=True,
)

# Create async session factory
db_session = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for ORM models
Base = declarative_base()


async def get_db():
    """
    Provides a database session per request.
    Ensures proper cleanup.
    """
    async with db_session() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create all tables in the database (idempotent)."""
    from sqlalchemy.exc import IntegrityError, ProgrammingError
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except (IntegrityError, ProgrammingError):
        # Tables already exist - this is fine on redeployments
        pass
