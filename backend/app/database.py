# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker

# #DATABASE_URL = "postgresql://postgres:alex@localhost/career_guidance"
# DATABASE_URL = "postgresql+psycopg2://career_user:alex@localhost:5432/career_guidance_db"

# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 1. Get the URL from Render's environment variables
# 2. Fallback to your local address if you're developing at home
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://career_user:alex@localhost:5432/career_guidance_db")

# Fix for Render/Heroku: SQLAlchemy requires 'postgresql://', but some providers provide 'postgres://'
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
