MySQL Setup Guide for Travel Planner App
Step 1: Install MySQL Server
On Windows:
Download MySQL Community Server from MySQL Official Website
Run the installer and follow the setup wizard
Choose "Developer Default" setup type
Set a root password (remember this!)
Complete the installation
On macOS:
# Using Homebrew
brew install mysql

# Start MySQL service
brew services start mysql

# Secure installation
mysql_secure_installation

On Ubuntu/Debian:
# Update package index
sudo apt update

# Install MySQL Server
sudo apt install mysql-server

# Secure installation
sudo mysql_secure_installation

# Start MySQL service
sudo systemctl start mysql
sudo systemctl enable mysql

Step 2: Create Database and User
Connect to MySQL as root:
mysql -u root -p

Create the database:
CREATE DATABASE travel_app;

Create a dedicated user (recommended for security):
CREATE USER 'travel_user'@'localhost' IDENTIFIED BY 'your_secure_password';
GRANT ALL PRIVILEGES ON travel_app.* TO 'travel_user'@'localhost';
FLUSH PRIVILEGES;

Exit MySQL:
EXIT;

Step 3: Install Python Dependencies
Create a requirements.txt file:
Flask==2.3.3
Werkzeug==2.3.7
mysql-connector-python==8.1.0
google-generativeai==0.3.2
python-dotenv==1.0.0

Install dependencies:
pip install -r requirements.txt

Step 4: Update Configuration
Update the MySQL configuration in your Flask app:
# MySQL Configuration
MYSQL_CONFIG = {
    'host': 'localhost',
    'database': 'travel_app',
    'user': 'travel_user',  # Change to your username
    'password': 'your_secure_password'  # Change to your password
}

Step 5: Environment Variables (Optional but Recommended)
Create a .env file in your project root:
MYSQL_HOST=localhost
MYSQL_DATABASE=travel_app
MYSQL_USER=travel_user
MYSQL_PASSWORD=your_secure_password
SECRET_KEY=your-super-secret-key-change-this
GEMINI_API_KEY=your_gemini_api_key

Update your Flask app to use environment variables:
from dotenv import load_dotenv
load_dotenv()

MYSQL_CONFIG = {
    'host': os.environ.get('MYSQL_HOST', 'localhost'),
    'database': os.environ.get('MYSQL_DATABASE', 'travel_app'),
    'user': os.environ.get('MYSQL_USER', 'travel_user'),
    'password': os.environ.get('MYSQL_PASSWORD')
}

app.secret_key = os.environ.get('SECRET_KEY', 'fallback-secret-key')
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

Step 6: Test Database Connection
Create a simple test script test_db.py:
import mysql.connector
from mysql.connector import Error

MYSQL_CONFIG = {
    '


