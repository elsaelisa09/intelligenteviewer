# init_db.py

__author__ = "Ari S Negara"
__copyright__ = "Copyright (C) 2025 Ari S.Negara "
__version__ = "1.0"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from auth.models import Base, User
from auth.config import AUTH_CONFIG

def init_database():
    # Create database engine
    engine = create_engine(AUTH_CONFIG['SQLALCHEMY_DATABASE_URI'])
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check if admin user exists
        admin = session.query(User).filter_by(username='admin').first()
        
        if not admin:
            # Create admin user
            admin = User(
                username='admin',
                email='admin@example.com',
                role='admin',
                is_active=True
            )
            admin.set_password('admin123')  # Default password
            session.add(admin)
            
            # Create test user
            test_user = User(
                username='test',
                email='test@example.com',
                role='user',
                is_active=True
            )
            test_user.set_password('test123')  # Default password
            session.add(test_user)
            
            session.commit()
            print("Default users created successfully!")
            print("\nDefault login credentials:")
            print("Admin User:")
            print("  Username: admin")
            print("  Password: admin123")
            print("\nTest User:")
            print("  Username: test")
            print("  Password: test123")
        else:
            print("Default users already exist!")
            
    except Exception as e:
        print(f"Error creating default users: {str(e)}")
        session.rollback()
    finally:
        session.close()

if __name__ == '__main__':
    init_database()