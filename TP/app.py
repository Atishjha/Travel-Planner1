from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from contextlib import closing

from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import google.generativeai as genai
from utils.travel_functions import (
    get_destination_info,
    recommend_destinations_for_interests,
    calculate_budget_breakdown,
    generate_itinerary,
    common_interests
)
import os
import psycopg2
from psycopg2 import OperationalError, IntegrityError
from psycopg2.extras import DictCursor, RealDictCursor
from functools import wraps
import json

app = Flask(__name__)

# Configuration
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this!
genai.configure(api_key="AIzaSyDW4BCGnID5zsxrjDX1DNu23-Fn4tkH_Hw")

# Replace MySQL config with PostgreSQL
POSTGRES_CONFIG = {
    'host': 'dpg-d0sa4rili9vc73bhtfm0-a',
    'database': 'travel_planner_db_anjp',
    'user': 'travel_planner_db_anjp_user',
    'password': 'u86gSy2YHtUUqPjBNmz2cViZNcKQZ29x',
    'port': '5432'  # Default PostgreSQL port
}

# Database connection function
def get_db_connection():
    """Create and return a database connection"""
    try:
        connection = psycopg2.connect(**POSTGRES_CONFIG)
        return connection
    except OperationalError as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def init_database():
    """Create necessary tables if they don't exist"""
    connection = get_db_connection()
    if connection:
        cursor = connection.cursor()
        
        try:
            # Users table (PostgreSQL syntax)
            cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
            ''')
            
            # Travel history table (PostgreSQL syntax)
            cursor.execute('''
CREATE TABLE IF NOT EXISTS travel_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    destination VARCHAR(255) NOT NULL,
    country VARCHAR(100),
    start_date DATE,
    end_date DATE,
    budget DECIMAL(10, 2),
    num_people INTEGER,
    interests JSONB,
    itinerary JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)
            ''')
            
            connection.commit()
            print("Database initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing database: {e}")
            connection.rollback()
            return False
        finally:
            cursor.close()
            connection.close()
    return False

# Fixed database initialization check
def check_and_init_database():
    """Check if tables exist and initialize if needed"""
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'users'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                cursor.close()
                connection.close()
                return init_database()
            
            cursor.close()
            connection.close()
            return True
        except Exception as e:
            print(f"Error checking database: {e}")
            if connection:
                connection.close()
            return False
    return False

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# User management functions
def create_user(username, email, password):
    """Create a new user - FIXED VERSION"""
    connection = get_db_connection()
    if not connection:
        print("Failed to get database connection")
        return None
    
    try:
        cursor = connection.cursor()
        password_hash = generate_password_hash(password)
        
        # Use RETURNING clause to get the new user ID
        cursor.execute(
            'INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING id',
            (username, email, password_hash)
        )
        
        user_id = cursor.fetchone()[0]  # Get the returned ID
        connection.commit()
        
        print(f"Successfully created user with ID: {user_id}")
        return user_id
        
    except IntegrityError as e:
        print(f"Integrity error creating user: {e}")
        connection.rollback()
        return None
    except Exception as e:
        print(f"Error creating user: {e}")
        connection.rollback()
        return None
    finally:
        cursor.close()
        connection.close()

def get_user_by_username(username):
    """Get user by username"""
    connection = get_db_connection()
    if not connection:
        return None
        
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)  # Fixed cursor factory
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()
        return dict(user) if user else None  # Convert to regular dict
    except Exception as e:
        print(f"Error getting user by username: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

def get_user_by_id(user_id):
    """Get user by ID - FIXED VERSION"""
    connection = get_db_connection()
    if not connection:
        return None
        
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)  # Fixed: was using MySQL syntax
        cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
        user = cursor.fetchone()
        return dict(user) if user else None  # Convert to regular dict
    except Exception as e:
        print(f"Error getting user by ID: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

def save_travel_history(user_id, destination, country, start_date, end_date, budget, num_people, interests, itinerary):
    """Save travel itinerary to history"""
    connection = get_db_connection()
    if not connection:
        return False
        
    try:
        cursor = connection.cursor()
        
        # Convert itinerary to proper JSON format
        if isinstance(itinerary, str):
            try:
                # Try to parse if it's already JSON string
                json.loads(itinerary)
                itinerary_json = itinerary
            except json.JSONDecodeError:
                # If plain text, convert to JSON object
                itinerary_json = json.dumps({"content": itinerary})
        else:
            itinerary_json = json.dumps(itinerary)
        
        cursor.execute('''
            INSERT INTO travel_history 
            (user_id, destination, country, start_date, end_date, budget, num_people, interests, itinerary)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (user_id, destination, country, start_date, end_date, 
              budget, num_people, json.dumps(interests), itinerary_json))
        
        connection.commit()
        return True
    except Exception as e:
        print(f"Error saving travel history: {e}")
        connection.rollback()
        return False
    finally:
        cursor.close()
        connection.close()

def get_user_travel_history(user_id):
    """Get user's travel history - FIXED VERSION"""
    connection = get_db_connection()
    if not connection:
        return []
        
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)  # Fixed: was using MySQL syntax
        cursor.execute('''
            SELECT id, user_id, destination, country, 
                   to_char(start_date, 'YYYY-MM-DD') as start_date,
                   to_char(end_date, 'YYYY-MM-DD') as end_date,
                   budget, num_people, interests, itinerary, 
                   created_at
            FROM travel_history 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        ''', (user_id,))
        
        history = cursor.fetchall()
        history_list = [dict(row) for row in history]  # Convert to list of dicts
        
        # Parse JSON fields with error handling
        for item in history_list:
            try:
                if item['interests']:
                    item['interests'] = json.loads(item['interests'])
            except (json.JSONDecodeError, TypeError):
                item['interests'] = []
            
            try:
                if item['itinerary']:
                    item['itinerary'] = json.loads(item['itinerary'])
            except (json.JSONDecodeError, TypeError):
                # If not valid JSON, treat as plain text
                item['itinerary'] = {"content": item['itinerary']}
        
        return history_list
    except Exception as e:
        print(f"Error fetching travel history: {e}")
        return []
    finally:
        cursor.close()
        connection.close()

# Routes
@app.route('/')
def home():
    return render_template('index.html', common_interests=common_interests)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('signup.html')
        
        # Check if user already exists
        existing_user = get_user_by_username(username)
        if existing_user:
            flash('Username already exists.', 'error')
            return render_template('signup.html')
        
        # Check if email already exists
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                cursor.execute('SELECT id FROM users WHERE email = %s', (email,))
                if cursor.fetchone():
                    flash('Email already exists.', 'error')
                    return render_template('signup.html')
            except Exception as e:
                print(f"Error checking email: {e}")
                flash('Error checking email availability.', 'error')
                return render_template('signup.html')
            finally:
                cursor.close()
                connection.close()
        
        # Create user
        try:
            user_id = create_user(username, email, password)
            if user_id:
                session['user_id'] = user_id
                session['username'] = username
                flash('Account created successfully!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Error creating account. Please try again.', 'error')
        except Exception as e:
            print(f"Error during user creation: {e}")
            flash('Error creating account. Please try again.', 'error')
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('login.html')
        
        user = get_user_by_username(username)
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/healthcheck')
def healthcheck():
    """Endpoint to check basic app health"""
    try:
        # Test database connection
        conn = get_db_connection()
        if conn:
            conn.close()
            return jsonify({
                'status': 'healthy',
                'database': 'connected'
            }), 200
        return jsonify({
            'status': 'unhealthy',
            'database': 'disconnected'
        }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/test-signup', methods=['POST'])
def test_signup():
    """Test endpoint for signup functionality"""
    try:
        # Test user creation
        test_user = {
            'username': 'testuser_' + str(datetime.now().timestamp()),
            'email': f'test_{datetime.now().timestamp()}@example.com',
            'password': 'testpassword123'
        }
        
        user_id = create_user(test_user['username'], test_user['email'], test_user['password'])
        if user_id:
            return jsonify({
                'success': True,
                'user_id': user_id
            }), 200
        return jsonify({
            'success': False,
            'error': 'User creation failed'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/profile')
@login_required
def profile():
    user = get_user_by_id(session['user_id'])
    history = get_user_travel_history(session['user_id'])
    print("Retrieved history:", history)  # Debug output
    return render_template('profile.html', user=user, history=history)

@app.route('/destination', methods=['POST'])
def destination():
    destination = request.form.get('destination')
    country = request.form.get('country', None)
    
    if not destination:
        return redirect(url_for('home'))
    
    # Get destination info
    dest_info = get_destination_info(destination, country)
    
    return render_template('destination.html', 
                         destination=dest_info,
                         country=country)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    interests = request.form.getlist('interests')
    continent = request.form.get('continent', None)
    
    if not interests:
        flash('Please select at least one interest', 'error')
        return redirect(url_for('home'))
    
    try:
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Get recommendations using your function
        recs = recommend_destinations_for_interests(
            model=model,
            interests=interests,
            continent=continent
        )
        
        # Convert the dictionary format to a list for template compatibility
        recommendations_list = []
        for interest, destinations in recs.items():
            for dest in destinations:
                dest['interest'] = interest.replace('_', ' ').title()
                recommendations_list.append(dest)
        
        return render_template('recommendations.html',
                            recommendations=recommendations_list,
                            interests=interests,
                            continent=continent)
                            
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        flash('Error generating recommendations. Showing sample destinations.', 'error')
        # Fallback to basic recommendations
        basic_recs = [
            {
                'name': 'Paris', 
                'country': 'France', 
                'continent': 'Europe',
                'description': 'City of Lights with rich culture',
                'best_time': 'April to October',
                'cost': 'mid-range',
                'match_reason': 'Perfect for art, history, and romance',
                'interest': 'Culture'
            },
            # Add more fallback destinations as needed
        ]
        return render_template('recommendations.html',
                            recommendations=basic_recs,
                            interests=interests,
                            continent=continent)
        
 
@app.route('/itinerary', methods=['POST'])
@login_required
def itinerary():
    try:
        # Get form data
        destination = request.form.get('destination')
        country = request.form.get('country', None)
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')
        interests = request.form.getlist('interests')
        budget = float(request.form.get('budget', 0))
        num_people = int(request.form.get('num_people', 1))

        # Validate inputs first
        validate_trip_input(destination, start_date_str, end_date_str, budget, num_people)

        # Convert to datetime objects (keep original strings for saving)
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None
        
        # Calculate number of days
        if start_date and end_date:
            num_days = (end_date - start_date).days + 1
        else:
            num_days = 1  # default value if dates aren't provided (though validation should prevent this)
        # Validate inputs
        

        

        # Get destination info
        dest_info = get_destination_info(destination, country) or {}
        if not dest_info:
            flash('Could not retrieve destination information', 'error')
            return redirect(url_for('home'))

        # Calculate budget
        budget_info = calculate_budget_breakdown(
            destination_info=dest_info,
            total_budget=budget,
            num_days=num_days,
            num_people=num_people
        )

        # Generate itinerary - pass dates as strings to avoid the error
        itinerary = generate_itinerary(
            destination=destination,
            start_date=start_date_str,  # Pass as string instead of datetime
            end_date=end_date_str,      # Pass as string instead of datetime
            interests=interests,
            num_people=num_people,
            budget_info=budget_info,
            country=country
        )

        # Save to history - use the original string dates
        save_success = save_travel_history(
            user_id=session['user_id'],
            destination=destination,
            country=country,
            start_date=start_date_str,
            end_date=end_date_str,
            budget=budget,
            num_people=num_people,
            interests=interests,
            itinerary=itinerary
        )

        if not save_success:
            flash('Itinerary generated but failed to save history', 'warning')

        return render_template('itinerary.html',
                            destination=dest_info,
                            country=country,
                            budget_info=budget_info,
                            itinerary=itinerary,
                            num_days=num_days,
                            num_people=num_people,
                            remaining_abs=abs(budget_info.get('remaining', 0)))

    except ValueError as e:
        flash(str(e), "error")
        return redirect(url_for('home'))
    except Exception as e:
        app.logger.error(f"Error generating itinerary: {str(e)}", exc_info=True)
        flash("An unexpected error occurred while generating your itinerary", "error")
        return redirect(url_for('home'))
    


# API endpoints for AJAX calls
# API endpoints for AJAX calls (example)
@app.route('/api/destination_info', methods=['POST'])
def api_destination_info():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON payload'}), 400
            
        destination = data.get('destination')
        country = data.get('country', None)
        
        if not destination:
            return jsonify({'error': 'Destination is required'}), 400
        
        # model = genai.GenerativeModel('gemini-1.5-flash') # Initialize if get_destination_info needs it
        info = get_destination_info(destination, country) # Pass model if needed
        
        if "error" in info: # Assuming get_destination_info returns a dict with an 'error' key on failure
            return jsonify({'error': info['error']}), 500
        return jsonify(info)

    except Exception as e:
        print(f"API Error (destination_info): {e}")
        return jsonify({'error': 'Failed to get destination information due to an internal error.'}), 500


@app.route('/api/recommendations', methods=['POST'])
def api_recommendations():
    try:
        data = request.get_json()
        app.logger.info(f"Received data for /api/recommendations: {data}") 

        if not data:
            app.logger.error("Invalid JSON payload received for /api/recommendations.")
            return jsonify({'error': 'Invalid JSON payload'}), 400

        interests = data.get('interests', [])
        continent = data.get('continent', None)
        if continent == "": continent = None 
        max_results = data.get('max_results', 3)
        
        if not interests:
            app.logger.warning("No interests provided for /api/recommendations.")
            return jsonify({'error': 'At least one interest is required'}), 400
        
        app.logger.info(f"Calling recommend_destinations_for_interests with: interests={interests}, continent={continent}, max_results={max_results}")
        model = genai.GenerativeModel('gemini-1.5-flash') # Changed model
        
        recs = recommend_destinations_for_interests(
            model, 
            interests, 
            continent=continent, 
            max_results=max_results
        )
        app.logger.info(f"Response from recommend_destinations_for_interests: {recs}")

        if not recs: 
            app.logger.warning("recommend_destinations_for_interests returned no data (None or empty).")
            return jsonify({'recommendations': {}}) 
        if isinstance(recs, dict) and recs.get("error"):
            app.logger.error(f"Error from recommend_destinations_for_interests: {recs['error']}")
            return jsonify({'error': recs["error"]}), 500

        return jsonify({'recommendations': recs})

    except Exception as e:
        app.logger.error(f"Unhandled exception in /api/recommendations: {e}", exc_info=True) 
        return jsonify({'error': 'Failed to generate recommendations due to an internal server error.'}), 500


# Utility function for input validation (as provided by you, slightly modified)
def validate_trip_input(destination, start_date_str, end_date_str, budget, num_people):
    """Validate all trip input parameters"""
    if not destination or not destination.strip():
        raise ValueError("Destination is required.")
    if not start_date_str or not end_date_str:
        raise ValueError("Start and end dates are required.")
    
    # Budget and num_people are likely already validated as float/int before this call,
    # but good to have checks if called directly.
    if not isinstance(budget, (int, float)) or budget <= 0:
        raise ValueError("Budget must be a positive number.")
    if not isinstance(num_people, int) or num_people <= 0:
        raise ValueError("Number of travelers must be a positive integer.")
    
    try:
        start = datetime.strptime(start_date_str, '%Y-%m-%d')
        end = datetime.strptime(end_date_str, '%Y-%m-%d')
        if end < start: # Allow same day trips, so use < not <=
            raise ValueError("End date must be on or after start date.")
        if (end - start).days > 365: # Example limit
            raise ValueError("Trip duration cannot exceed 365 days.")
    except ValueError as e:
        # Re-raise with a more user-friendly message if it's a date parsing error
        if "time data" in str(e) or "format" in str(e):
            raise ValueError("Invalid date format. Please use YYYY-MM-DD.")
        raise e # Re-raise other ValueErrors (like end date before start)


if __name__ == '__main__':
    # Initialize database on startup
    check_and_init_database()
    
    # Set Flask environment and debug mode from environment variables
    flask_env = os.environ.get('FLASK_ENV', 'development')
    debug_mode = os.environ.get('FLASK_DEBUG', 'True' if flask_env == 'development' else 'False').lower() == 'true'

    if flask_env == 'production':
        app.config.update(
            TEMPLATES_AUTO_RELOAD=False,
            SESSION_COOKIE_SECURE=True,
            SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE='Lax',
        )
    
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
