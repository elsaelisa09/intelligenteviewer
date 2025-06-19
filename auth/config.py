# auth/config.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

import os

# Get the current directory (where config.py is)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to get to the project root directory
project_dir = os.path.dirname(current_dir)

# Create the database path
db_path = os.path.join(project_dir, 'app.db')

# Convert Windows path separators to forward slashes
db_uri = f'sqlite:///{db_path.replace(os.sep, "/")}'

AUTH_CONFIG = {
    'SECRET_KEY': 'admin123',  # Change this in production
    'SQLALCHEMY_DATABASE_URI': db_uri,
    'SESSION_TYPE': 'filesystem'
}

print(f"Database will be created at: {db_path}")
print(f"SQLAlchemy URI: {db_uri}")