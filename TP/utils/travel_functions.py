from google.oauth2 import service_account
from google.auth.transport.requests import Request
import google.generativeai as genai
import os
import requests
import numpy as np
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import google.auth
from typing import List, Dict, Any
import json
import time
from functools import lru_cache
# --- Authentication Setup ---
# Removed dotenv imports and load_dotenv() call.
# The API key is now hardcoded directly in this file.
# WARNING: Hardcoding API keys is NOT recommended for production environments
# due to security risks. For production, use environment variables
# or a secure secrets management system.

GEMINI_API_KEY = "AIzaSyDW4BCGnID5zsxrjDX1DNu23-Fn4tkH_Hw" # Your actual API key, now hardcoded.
                                                          # Ensure there are no trailing spaces.

# --- Attempt to configure genai using either ADC or the API Key ---
try:
    # Try to get ADC. This will not directly configure genai,
    # but it helps verify if ADC is set up and available.
    # The genai library itself will try to pick these up if available.
    credentials, project_id = google.auth.default(
        scopes=[
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/generative-language.generateContent',
            'https://www.googleapis.com/auth/generative-language' # A broader Generative Language scope
        ]
    )
    if credentials.valid:
        print("Successfully obtained Google Cloud Application Default Credentials.")
        if not credentials.token or credentials.expired:
            credentials.refresh(Request())
        # If ADC is valid, we'll rely on it. Configure genai without API key, letting it use ADC.
        genai.configure() 
        print("Gemini API configured using Application Default Credentials.")
    else:
        print("Google Cloud Application Default Credentials not valid. Falling back to API Key.")
        # If ADC is not valid, use the hardcoded API key.
        if not GEMINI_API_KEY:
            # This should ideally not happen if GEMINI_API_KEY is hardcoded,
            # but kept for robustness if it's accidentally empty.
            raise ValueError("GEMINI_API_KEY is empty and ADC not found/valid.")
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured using hardcoded GEMINI_API_KEY.")

except Exception as e:
    print(f"Error checking Google Cloud credentials: {e}")
    print("Will attempt to use hardcoded API Key for Gemini configuration as a last resort.")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured using hardcoded GEMINI_API_KEY.")
    else:
        # If ADC failed and no API key (even hardcoded), raise a critical error
        raise ValueError("Neither Application Default Credentials nor hardcoded GEMINI_API_KEY found for Gemini configuration.")

# It's good to test if a model can be initialized after configuration
try:
    _ = genai.GenerativeModel(model_name="gemini-1.5-flash")
    print("Gemini model 'gemini-1.5-flash' initialized successfully (connection test).")
except Exception as e:
    print(f"FAILED to initialize Gemini model: {e}")
    print("Please ensure your hardcoded API Key is correct and has access, or ADC is properly set up with sufficient IAM roles.")
    # Exit or handle this critical failure appropriately in a real app


# Sample global destinations - used as suggestions rather than limitations
sample_destinations = [
    {"name": "Bali", "country": "Indonesia", "budget": "medium", "interests": ["beach", "culture", "adventure"], "best_season": "dry(April-October)", 
     "flight_cost": 35000, "daily_hotel": 1500, "daily_food": 1000, "daily_activities": 1200, "visa_cost": 3000,
     "visa_type": "Visa on Arrival", "visa_process": "Available at major airports for 30 days stay", "visa_chance": "high"},
    {"name": "Tokyo", "country": "Japan", "budget": "high", "interests": ["culture", "food", "technology"], "best_season": "spring(March-May)",
     "flight_cost": 50000, "daily_hotel": 4000, "daily_food": 2000, "daily_activities": 2500, "visa_cost": 2500,
     "visa_type": "Tourist Visa", "visa_process": "Apply at embassy/consulate with documents 1-2 months in advance", "visa_chance": "medium"},
    {"name": "Lisbon", "country": "Portugal", "budget": "medium", "interests": ["history", "food", "beach"], "best_season": "summer(June-August)",
     "flight_cost": 45000, "daily_hotel": 2500, "daily_food": 1500, "daily_activities": 1500, "visa_cost": 0,
     "visa_type": "Schengen Visa", "visa_process": "Apply at VFS center with documents 3-4 weeks in advance", "visa_chance": "medium"},
    {"name": "Marrakech", "country": "Morocco", "budget": "low", "interests": ["culture", "history", "shopping"], "best_season": "spring(March-May)",
     "flight_cost": 40000, "daily_hotel": 1200, "daily_food": 800, "daily_activities": 1000, "visa_cost": 0,
     "visa_type": "Visa-Free", "visa_process": "No visa required for stays up to 90 days", "visa_chance": "high"},
    {"name": "Queenstown", "country": "New Zealand", "budget": "high", "interests": ["adventure", "nature", "hiking"], "best_season": "summer(December-February)",
     "flight_cost": 70000, "daily_hotel": 3500, "daily_food": 1800, "daily_activities": 3000, "visa_cost": 4500,
     "visa_type": "Tourist Visa", "visa_process": "Online application with documents 1-2 months in advance", "visa_chance": "medium"}
]

# Common interests for travel
common_interests = [
    "beach", "mountains", "culture", "history", "food", "adventure", "relaxation", 
    "wildlife", "photography", "shopping", "architecture", "art", "music", "festivals",
    "nightlife", "luxury", "budget", "family-friendly", "romantic", "solo travel", 
    "hiking", "water sports", "winter sports", "nature", "urban exploration", "technology"
]

DEFAULT_DESTINATION_INFO = {
    "name": "",
    "country": "Unknown",
    "budget": "medium",
    "interests": ["sightseeing", "culture", "food"],
    "best_season": "year-round",
    "flight_cost": 40000,
    "daily_hotel": 2500,
    "daily_food": 1500,
    "daily_activities": 1500,
    "visa_cost": 3000,
    "visa_type": "Tourist Visa",
    "visa_process": "Check embassy requirements",
    "visa_chance": "medium"
}
'''
def get_destination_info(destination: str, country: str = None) -> Dict[str, Any]:
    """
    Get detailed information about any global destination
    
    Args:
        destination: Name of the destination
        country: Optional country name for disambiguation
    
    Returns:
        Dictionary with destination details
    """
    try:
        # Construct prompt with country if provided
        location = f"{destination}, {country}" if country else destination
        
        # Use Gemini to get destination information
        prompt = f"""
        Provide detailed travel information for {location} in JSON format for Indian citizens.
        Include the following details:
        - Average flight cost from India in INR
        - Daily hotel cost (budget, mid-range, luxury) in INR
        - Daily food cost in INR
        - Daily activities cost in INR
        - Visa cost for Indian citizens in INR
        - Visa type (e.g., Visa on Arrival, eVisa, Visa-Free, etc.)
        - Visa process description (application method, required documents, processing time)
        - Visa approval chance (low/medium/high)
        - Best season to visit
        - Top interests/attractions
        - Budget category (low/medium/high)
        - Country where this destination is located
        
        Format your response as valid JSON only, no explanations.
        Example format:
        {{
          "country": "Italy",
          "flight_cost": 35000,
          "daily_hotel": {{
            "budget": 1000,
            "mid_range": 2500, 
            "luxury": 6000
          }},
          "daily_food": 1200,
          "daily_activities": 1500,
          "visa_cost": 3000,
          "visa_type": "Schengen Visa",
          "visa_process": "Apply at VFS center with documents including bank statements, itinerary, and hotel bookings 3-4 weeks in advance",
          "visa_chance": "medium",
          "best_season": "April-October",
          "interests": ["beach", "culture", "hiking", "food"],
          "budget_category": "medium"
        }}
        """
        
        # Use the newer recommended model
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        response = model.generate_content(prompt)
        
        # Process the response
        import json
        try:
            # Try to parse the JSON response
            dest_info = json.loads(response.text)
            
            # Handle the case where daily_hotel might be a dictionary or a single value
            if isinstance(dest_info.get("daily_hotel", {}), dict):
                daily_hotel = dest_info["daily_hotel"].get("mid_range", 2500)
            else:
                daily_hotel = dest_info.get("daily_hotel", 2500)
            
            # Get country information if available
            country_info = dest_info.get("country", country if country else "Unknown")
                
            # Create a standardized destination info dictionary
            return {
                "name": destination,
                "country": country_info,
                "budget": dest_info.get("budget_category", "medium"),
                "interests": dest_info.get("interests", ["sightseeing"]),
                "best_season": dest_info.get("best_season", "year-round"),
                "flight_cost": dest_info.get("flight_cost", 40000),
                "daily_hotel": daily_hotel,
                "daily_food": dest_info.get("daily_food", 1500),
                "daily_activities": dest_info.get("daily_activities", 1500),
                "visa_cost": dest_info.get("visa_cost", 0),
                "visa_type": dest_info.get("visa_type", "Tourist Visa"),
                "visa_process": dest_info.get("visa_process", "Check with embassy for latest requirements"),
                "visa_chance": dest_info.get("visa_chance", "medium")
            }
            
        except json.JSONDecodeError:
            # If JSON parsing fails, use fallback values
            print(f"Error parsing destination data for {location}. Using default values.")
            return {
                "name": destination,
                "country": country if country else "Unknown",
                "budget": "medium",
                "interests": ["sightseeing", "culture", "food"],
                "best_season": "year-round",
                "flight_cost": 40000,
                "daily_hotel": 2500,
                "daily_food": 1500,
                "daily_activities": 1500,
                "visa_cost": 3000,
                "visa_type": "Tourist Visa",
                "visa_process": "Check with embassy for latest requirements",
                "visa_chance": "medium"
            }
            
    except Exception as e:
        print(f"Error getting destination info: {str(e)}")
        # Return default values if an error occurs
        return {
            "name": destination,
            "country": country if country else "Unknown",
            "budget": "medium",
            "interests": ["sightseeing", "culture", "food"],
            "best_season": "year-round",
            "flight_cost": 40000,
            "daily_hotel": 2500,
            "daily_food": 1500,
            "daily_activities": 1500,
            "visa_cost": 3000,
            "visa_type": "Tourist Visa",
            "visa_process": "Check with embassy for latest requirements",
            "visa_chance": "medium"
        }
'''
# ==== New Media Functions ====
def get_destination_images_with_gemini(destination: str, country: str = None, num_images: int = 3) -> List[Dict[str, Any]]:
    """
    Get image descriptions for a destination using Gemini AI
    
    Args:
        destination: Name of the destination
        country: Optional country name for disambiguation
        num_images: Number of images to retrieve
        
    Returns:
        List of image information dictionaries
    """
    try:
        # Avoid redundancy for continents
        if destination.lower() in ["europe", "asia", "africa", "north america", "south america", 
                                  "australia", "antarctica", "oceania"]:
            location = destination
        else:
            # Construct location string with country if provided
            location = f"{destination}, {country}" if country else destination
        
        print(f"Fetching image data for location: {location}")
        
        # Create prompt for Gemini
        prompt = f"""
        Describe {num_images} stunning landmarks or scenic spots in {location} that travelers would want to photograph.
        
        For each location, provide:
        1. A descriptive name for the landmark/spot
        2. A brief description of what makes it visually impressive (max 100 characters)
        3. A suggestion for the best time of day to photograph it
        
        Format your response as valid JSON only, no explanations.
        Example format:
        [
          {{
            "name": "Example Landmark",
            "description": "A stunning vista point with panoramic city views",
            "best_time_to_photograph": "sunset"
          }},
          ...
        ]
        """
        
        # Use Gemini to generate image descriptions
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        # Process the response - add more debugging
        import json
        try:
            # Print the response to debug
            print(f"Raw response from Gemini: {response.text[:100]}...")  # First 100 chars
            
            # Try to parse the JSON response
            image_data = json.loads(response.text)
            
            # Transform the data into the expected format
            return [{
                "url": f"https://source.unsplash.com/featured/?{destination},{spot['name'].replace(' ', '%20')}",
                "description": f"{spot['name']}: {spot['description']}",
                "credit": f"Best photographed during {spot.get('best_time_to_photograph', 'daytime')}",
                "is_generated": True
            } for spot in image_data]
            
        except json.JSONDecodeError as json_err:
            # If JSON parsing fails, provide more detailed error info
            print(f"JSON parsing error for {location}: {str(json_err)}")
            print(f"Response content sample: {response.text[:150]}...")  # Show more of the response
            
            # Create a more informative fallback
            return [{
                "url": f"https://source.unsplash.com/featured/?{destination},{i}",
                "description": f"Beautiful view of {destination} - landmark {i}",
                "credit": "Generated description - API response could not be parsed",
                "is_generated": True
            } for i in range(num_images)]
            
    except Exception as e:
        print(f"Error getting destination images with Gemini: {str(e)}")
        # Return default image descriptions
        return [{
            "url": f"https://source.unsplash.com/featured/?{destination},{i}",
            "description": f"Scenic {destination} location - view {i}",
            "credit": "Generated description - API request failed",
            "is_generated": True
        } for i in range(num_images)]
def generate_traveler_insights(destination: str, country: str = None) -> str:
    """Generate traveler comments using Gemini"""
    try:
        prompt = f"""
        Generate 3-5 authentic traveler reviews for {destination}, {country if country else ''} 
        from Indian tourists perspective. Include:
        - Both positive and constructive feedback
        - Ratings (1-5 stars)
        - Month of visit
        - Brief experiences
        Format as bullet points with emojis.
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text
    except Exception as e:
        return f"\n🚩 Could not generate traveler insights: {str(e)}"

def generate_traveler_insights(destination: str, country: str = None) -> str:
    """Generate traveler comments using Gemini"""
    try:
        prompt = f"""
        Generate 3-5 authentic traveler reviews for {destination}, {country if country else ''} 
        from Indian tourists perspective. Include:
        - Both positive and constructive feedback
        - Ratings (1-5 stars)
        - Month of visit
        - Brief experiences
        Format as bullet points with emojis.
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating traveler insights: {str(e)}")
        return f"\n🚩 Could not generate traveler insights: {str(e)}"

FALLBACK_DATA_FOR_DESTINATIONS = {
    "country": "Unknown",
    "budget": "medium",
    "interests": ["travel", "sightseeing", "culture"],
    "best_season": "year-round",
    "flight_cost": 0,  # Default flight cost
    "daily_hotel": 3500,
    "daily_food": 2000,
    "daily_activities": 2000,
    "visa_cost": 5000,
    "visa_type": "Varies by country",
    "visa_process": "Check with specific country embassies",
    "visa_chance": "medium",
    "images": [],
    "traveler_insights": "No traveler insights available."
}

def get_continent_info(continent: str) -> dict:
    """
    Get predefined information for continental destinations.
    This function is assumed to be correct and include 'flight_cost'.
    """
    continent_data = {
        "europe": {
            "name": "Europe", "country": "Multiple Countries", "budget": "high",
            "interests": ["culture", "history", "architecture", "cuisine", "art"],
            "best_season": "April-October", "flight_cost": 50000, "daily_hotel": 4000,
            "daily_food": 2000, "daily_activities": 2500, "visa_cost": 6000,
            "visa_type": "Schengen Visa (for most countries)",
            "visa_process": "Apply at VFS center with documents including bank statements, itinerary, and hotel bookings 3-4 weeks in advance",
            "visa_chance": "medium"
        },
        "asia": {
            "name": "Asia", "country": "Multiple Countries", "budget": "medium",
            "interests": ["culture", "food", "temples", "beaches", "shopping"],
            "best_season": "November-February", "flight_cost": 30000, "daily_hotel": 2000,
            "daily_food": 1200, "daily_activities": 1500, "visa_cost": 3000,
            "visa_type": "Varies by country",
            "visa_process": "Varies by country, many offer visa on arrival or e-visa for Indian citizens",
            "visa_chance": "high"
        },
        "north america": {
            "name": "North America", "country": "Multiple Countries", "budget": "high",
            "interests": ["nature", "cities", "theme parks", "shopping", "national parks"],
            "best_season": "May-September", "flight_cost": 80000, "daily_hotel": 6000,
            "daily_food": 2500, "daily_activities": 3000, "visa_cost": 12000,
            "visa_type": "Tourist Visa (B1/B2 for USA, eTA for Canada)",
            "visa_process": "Complex process requiring documentation of ties to India, financial capability, and interview",
            "visa_chance": "medium"
        },
        "south america": {
            "name": "South America", "country": "Multiple Countries", "budget": "medium",
            "interests": ["nature", "culture", "adventure", "beaches", "hiking"],
            "best_season": "June-August", "flight_cost": 90000, "daily_hotel": 3000,
            "daily_food": 1800, "daily_activities": 2000, "visa_cost": 5000,
            "visa_type": "Varies by country",
            "visa_process": "Varies by country, some offer visa-free entry for Indian citizens",
            "visa_chance": "medium"
        },
        "africa": {
            "name": "Africa", "country": "Multiple Countries", "budget": "medium",
            "interests": ["safari", "wildlife", "culture", "adventure", "beaches"],
            "best_season": "June-October", "flight_cost": 60000, "daily_hotel": 3500,
            "daily_food": 1500, "daily_activities": 3000, "visa_cost": 4000,
            "visa_type": "Varies by country",
            "visa_process": "Varies by country, some offer e-visa",
            "visa_chance": "medium"
        },
        "australia": {
            "name": "Australia", "country": "Australia", "budget": "high",
            "interests": ["beaches", "wildlife", "cities", "outback", "marine life"],
            "best_season": "December-February", "flight_cost": 70000, "daily_hotel": 5000,
            "daily_food": 2500, "daily_activities": 3000, "visa_cost": 15000,
            "visa_type": "Tourist Visa (Subclass 600)",
            "visa_process": "Online application with financial documentation",
            "visa_chance": "high"
        },
        "oceania": {
            "name": "Oceania", "country": "Multiple Countries", "budget": "high",
            "interests": ["islands", "beaches", "culture", "water activities", "nature"],
            "best_season": "May-October", "flight_cost": 75000, "daily_hotel": 4500,
            "daily_food": 2300, "daily_activities": 2800, "visa_cost": 10000,
            "visa_type": "Varies by country",
            "visa_process": "Varies by country, some offer visa on arrival",
            "visa_chance": "medium"
        },
        "antarctica": {
            "name": "Antarctica", "country": "No country (governed by Antarctic Treaty)",
            "budget": "very high",
            "interests": ["wildlife", "landscapes", "expedition", "photography", "adventure"],
            "best_season": "November-March", "flight_cost": 200000, "daily_hotel": 25000,
            "daily_food": 5000, "daily_activities": 10000, "visa_cost": 0,
            "visa_type": "No visa required, but permits needed",
            "visa_process": "Tour operator handles permits, no direct visa needed",
            "visa_chance": "high"
        }
    }
    continent_key = continent.lower()
    if continent_key in continent_data:
        return continent_data[continent_key]
    else:
        print(f"Unknown continent: {continent}. Using default values for continent.")
        # Fallback for unknown continents, ensuring all keys including flight_cost are present
        return {
            "name": continent,
            "country": "Multiple Countries",
            "budget": FALLBACK_DATA_FOR_DESTINATIONS["budget"],
            "interests": list(FALLBACK_DATA_FOR_DESTINATIONS["interests"]),
            "best_season": FALLBACK_DATA_FOR_DESTINATIONS["best_season"],
            "flight_cost": FALLBACK_DATA_FOR_DESTINATIONS["flight_cost"],
            "daily_hotel": FALLBACK_DATA_FOR_DESTINATIONS["daily_hotel"],
            "daily_food": FALLBACK_DATA_FOR_DESTINATIONS["daily_food"],
            "daily_activities": FALLBACK_DATA_FOR_DESTINATIONS["daily_activities"],
            "visa_cost": FALLBACK_DATA_FOR_DESTINATIONS["visa_cost"],
            "visa_type": FALLBACK_DATA_FOR_DESTINATIONS["visa_type"],
            "visa_process": FALLBACK_DATA_FOR_DESTINATIONS["visa_process"],
            "visa_chance": FALLBACK_DATA_FOR_DESTINATIONS["visa_chance"]
        }

def get_destination_info(destination: str, country: str = None) -> dict:
    """
    Get detailed information about any global destination, including images and traveler insights.
    Ensures 'flight_cost' and other essential keys are always present in the returned dictionary.
    """
    continents = ["europe", "asia", "africa", "north america", "south america",
                  "australia", "antarctica", "oceania"]
    destination_lower = destination.lower()

    # Handle continents separately as they have predefined data
    if destination_lower in continents:
        continent_info = get_continent_info(destination)
        try:
            # For continents, country is not applicable for these generic image/insight functions
            # Ensure the external functions are called correctly based on their definitions
            continent_info["images"] = get_destination_images_with_gemini(destination, num_images=3)
        except Exception as img_e:
            print(f"Error adding images for continent {destination}: {img_e}")
            continent_info["images"] = list(FALLBACK_DATA_FOR_DESTINATIONS["images"]) # Use a copy
        try:
            continent_info["traveler_insights"] = generate_traveler_insights(destination)
        except Exception as insight_e:
            print(f"Error adding traveler insights for continent {destination}: {insight_e}")
            continent_info["traveler_insights"] = FALLBACK_DATA_FOR_DESTINATIONS["traveler_insights"]
        return continent_info

    # For non-continent destinations
    location_query = f"{destination}, {country}" if country else destination
    result = {}  # Initialize result

    try:
        prompt = f"""
        Provide detailed travel information for {location_query} in JSON format for Indian citizens.
        Include the following details:
        - country: Name of the country where the destination is located (string)
        - flight_cost: Average flight cost from India in INR (integer)
        - daily_hotel: {{ "budget": int, "mid_range": int, "luxury": int }} (object with integer values in INR)
        - daily_food: Average daily food cost in INR (integer)
        - daily_activities: Average daily activities cost in INR (integer)
        - visa_cost: Visa cost for Indian citizens in INR (integer)
        - visa_type: Visa type (e.g., "Visa on Arrival", "eVisa", "Visa-Free") (string)
        - visa_process: Visa process description (string)
        - visa_chance: Visa approval chance ("low", "medium", "high") (string)
        - best_season: Best season to visit (string)
        - interests: Top interests/attractions (list of strings)
        - budget_category: Overall budget category ("low", "medium", "high") (string)

        Format your response as valid JSON only, with no explanations or surrounding text.
        Example:
        {{
          "country": "France",
          "flight_cost": 45000,
          "daily_hotel": {{ "budget": 3000, "mid_range": 7000, "luxury": 15000 }},
          "daily_food": 2500,
          "daily_activities": 3000,
          "visa_cost": 7000,
          "visa_type": "Schengen Visa",
          "visa_process": "Apply via VFS with required documents like bank statements, itinerary, bookings. Processing time: 3-4 weeks.",
          "visa_chance": "medium",
          "best_season": "April to June, September to October",
          "interests": ["Eiffel Tower", "Louvre Museum", "cuisine", "fashion"],
          "budget_category": "high"
        }}
        """
        # Assuming genai is imported and configured (e.g., import google.generativeai as genai)
        # Use the model name from your original code or update if necessary
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash", # Or "gemini-2.0-flash" as per latest guidelines
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        response = model.generate_content(prompt)
        dest_info = json.loads(response.text)

        # Safely extract daily_hotel (mid-range preferred)
        daily_hotel_val = FALLBACK_DATA_FOR_DESTINATIONS["daily_hotel"]  # Default
        daily_hotel_obj = dest_info.get("daily_hotel")
        if isinstance(daily_hotel_obj, dict):
            if isinstance(daily_hotel_obj.get("mid_range"), (int, float)):
                daily_hotel_val = daily_hotel_obj["mid_range"]
            elif isinstance(daily_hotel_obj.get("budget"), (int, float)):
                daily_hotel_val = daily_hotel_obj["budget"]
            elif isinstance(daily_hotel_obj.get("luxury"), (int, float)):
                daily_hotel_val = daily_hotel_obj["luxury"]
        elif isinstance(daily_hotel_obj, (int, float)): # if API returns a single number for hotel
            daily_hotel_val = daily_hotel_obj

        result = {
            "name": destination,
            "country": dest_info.get("country", country if country else FALLBACK_DATA_FOR_DESTINATIONS["country"]),
            "budget": dest_info.get("budget_category", FALLBACK_DATA_FOR_DESTINATIONS["budget"]),
            "interests": dest_info.get("interests", list(FALLBACK_DATA_FOR_DESTINATIONS["interests"])),
            "best_season": dest_info.get("best_season", FALLBACK_DATA_FOR_DESTINATIONS["best_season"]),
            "flight_cost": dest_info.get("flight_cost", FALLBACK_DATA_FOR_DESTINATIONS["flight_cost"]),
            "daily_hotel": daily_hotel_val,
            "daily_food": dest_info.get("daily_food", FALLBACK_DATA_FOR_DESTINATIONS["daily_food"]),
            "daily_activities": dest_info.get("daily_activities", FALLBACK_DATA_FOR_DESTINATIONS["daily_activities"]),
            "visa_cost": dest_info.get("visa_cost", FALLBACK_DATA_FOR_DESTINATIONS["visa_cost"]),
            "visa_type": dest_info.get("visa_type", FALLBACK_DATA_FOR_DESTINATIONS["visa_type"]),
            "visa_process": dest_info.get("visa_process", FALLBACK_DATA_FOR_DESTINATIONS["visa_process"]),
            "visa_chance": dest_info.get("visa_chance", FALLBACK_DATA_FOR_DESTINATIONS["visa_chance"])
        }

    except (json.JSONDecodeError, AttributeError, Exception) as e:
        print(f"Error processing LLM data for {location_query} (Error: {type(e).__name__} - {str(e)}). Using fallback data.")
        result = {
            "name": destination,
            "country": country if country else FALLBACK_DATA_FOR_DESTINATIONS["country"],
            "budget": FALLBACK_DATA_FOR_DESTINATIONS["budget"],
            "interests": list(FALLBACK_DATA_FOR_DESTINATIONS["interests"]), # Use a copy
            "best_season": FALLBACK_DATA_FOR_DESTINATIONS["best_season"],
            "flight_cost": FALLBACK_DATA_FOR_DESTINATIONS["flight_cost"], # Ensure flight_cost is present
            "daily_hotel": FALLBACK_DATA_FOR_DESTINATIONS["daily_hotel"],
            "daily_food": FALLBACK_DATA_FOR_DESTINATIONS["daily_food"],
            "daily_activities": FALLBACK_DATA_FOR_DESTINATIONS["daily_activities"],
            "visa_cost": FALLBACK_DATA_FOR_DESTINATIONS["visa_cost"],
            "visa_type": FALLBACK_DATA_FOR_DESTINATIONS["visa_type"],
            "visa_process": FALLBACK_DATA_FOR_DESTINATIONS["visa_process"],
            "visa_chance": FALLBACK_DATA_FOR_DESTINATIONS["visa_chance"]
        }

    # Add images and insights, using data from `result`
    try:
        country_for_img_insight = result.get("country", FALLBACK_DATA_FOR_DESTINATIONS["country"])
        # Call image/insight functions appropriately based on whether country is specific
        if country_for_img_insight in ["Unknown", "Multiple Countries", FALLBACK_DATA_FOR_DESTINATIONS["country"]]:
            result["images"] = get_destination_images_with_gemini(destination, num_images=3)
            result["traveler_insights"] = generate_traveler_insights(destination)
        else:
            result["images"] = get_destination_images_with_gemini(destination, country_for_img_insight, num_images=3)
            result["traveler_insights"] = generate_traveler_insights(destination, country_for_img_insight)
    except Exception as img_insight_e:
        print(f"Error adding images or insights for {destination}: {img_insight_e}")
        result["images"] = result.get("images", list(FALLBACK_DATA_FOR_DESTINATIONS["images"]))
        result["traveler_insights"] = result.get("traveler_insights", FALLBACK_DATA_FOR_DESTINATIONS["traveler_insights"])

    # Final check to ensure all fallback keys are present and flight_cost is numeric
    for key, default_val in FALLBACK_DATA_FOR_DESTINATIONS.items():
        if key not in result:
            result[key] = list(default_val) if isinstance(default_val, list) else default_val
    
    if not isinstance(result.get("flight_cost"), (int, float)):
        print(f"Warning: flight_cost for {destination} was not numeric, defaulting.")
        result["flight_cost"] = int(FALLBACK_DATA_FOR_DESTINATIONS["flight_cost"])
    if not isinstance(result.get("interests"), list):
        result["interests"] = list(FALLBACK_DATA_FOR_DESTINATIONS["interests"])


    return result

import json
import re
from typing import List, Dict, Optional

def recommend_destinations_for_interests(
    model, 
    interests: List[str], 
    continent: Optional[str] = None,
    max_results: int = 3
) -> Dict[str, List[Dict]]:
    """
    Get travel recommendations from Gemini based on user interests, optionally filtered by continent.
    
    Args:
        model: Initialized Gemini model
        interests: List of user interests
        continent: Optional continent to filter by (e.g., "Europe", "Asia")
        max_results: Maximum number of recommendations per interest
        
    Returns:
        Dictionary mapping interests to recommended destinations
    """
    continent_filter = f" in {continent}" if continent else ""
    examples_per_interest = min(max_results, 5)  # Cap at 5 examples
    
    prompt = f"""
    Act as a world-class travel recommendation system. For each of these interests: {', '.join(interests)},
    provide {examples_per_interest} specific destination recommendations{continent_filter}.
    
    IMPORTANT: Return ONLY a valid JSON object with no additional text, explanations, or formatting.
    
    Format the response as a JSON object where:
    - Keys are the interest categories (lowercase, no spaces, use underscores)
    - Values are lists of objects with destination details
    
    Required fields for each destination:
    - name: destination name
    - country: country name
    - continent: continent name
    - description: brief description (1 sentence)
    - best_time: best time to visit
    - cost: one of "budget", "mid-range", or "luxury"
    - match_reason: why it matches the interest
    
    Example format:
    {{
        "beaches": [
            {{
                "name": "Maldives",
                "country": "Maldives", 
                "continent": "Asia",
                "description": "Pristine white sand beaches with overwater bungalows.",
                "best_time": "November to April",
                "cost": "luxury",
                "match_reason": "Some of the world's most beautiful coral beaches."
            }}
        ]
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            print("Error: Empty response from Gemini")
            return {}
        
        response_text = response.text.strip()
        
        # Remove any markdown formatting or extra text
        response_text = clean_response_text(response_text)
        
        # Parse JSON response
        try:
            recommendations = json.loads(response_text)
            
            if not isinstance(recommendations, dict):
                print("Error: Response is not a dictionary")
                return {}
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response text: {response_text[:500]}...")
            return {}
        
        # Validate and clean the recommendations
        cleaned_recommendations = validate_recommendations(recommendations, interests)
        
        # Apply continent filtering if specified
        if continent:
            cleaned_recommendations = filter_by_continent(cleaned_recommendations, continent)
        
        # Limit results per interest
        for interest in cleaned_recommendations:
            cleaned_recommendations[interest] = cleaned_recommendations[interest][:max_results]
        
        return cleaned_recommendations
        
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return {}


def clean_response_text(text: str) -> str:
    """Clean response text to extract JSON."""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    
    # Find JSON object boundaries
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx + 1]
    
    return text.strip()


def validate_recommendations(recommendations: Dict, interests: List[str]) -> Dict[str, List[Dict]]:
    """Validate and clean recommendation data."""
    required_fields = ['name', 'country', 'continent', 'description', 'best_time', 'cost', 'match_reason']
    valid_costs = ['budget', 'mid-range', 'luxury']
    
    cleaned = {}
    
    for interest, destinations in recommendations.items():
        if not isinstance(destinations, list):
            continue
            
        valid_destinations = []
        for dest in destinations:
            if not isinstance(dest, dict):
                continue
                
            # Check required fields
            if all(field in dest for field in required_fields):
                # Validate cost field
                if dest['cost'].lower() in valid_costs:
                    dest['cost'] = dest['cost'].lower()
                else:
                    dest['cost'] = 'mid-range'  # Default fallback
                
                # Clean string fields
                for field in ['name', 'country', 'continent', 'description', 'best_time', 'match_reason']:
                    if isinstance(dest[field], str):
                        dest[field] = dest[field].strip()
                
                valid_destinations.append(dest)
        
        if valid_destinations:
            cleaned[interest.lower().replace(' ', '_')] = valid_destinations
    
    return cleaned


def filter_by_continent(recommendations: Dict[str, List[Dict]], continent: str) -> Dict[str, List[Dict]]:
    """Filter recommendations by continent."""
    continent_lower = continent.lower()
    filtered = {}
    
    for interest, destinations in recommendations.items():
        filtered_destinations = [
            dest for dest in destinations 
            if dest.get('continent', '').lower() == continent_lower
        ]
        if filtered_destinations:
            filtered[interest] = filtered_destinations
    
    return filtered


# Example usage function for testing
def test_recommendations():
    """Test function - replace with your actual Gemini model initialization."""
    # Mock response for testing
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    
    
    interests = ["beaches", "mountains"]
    interests = ["beaches", "mountains"]
    
    result = recommend_destinations_for_interests(
        model, 
        interests, 
        continent="Europe", 
        max_results=2
    )
    
    if result:
        print(json.dumps(result, indent=2))
    
    





def calculate_budget_breakdown(destination_info: Dict[str, Any], total_budget: float, num_days: int, num_people: int) -> Dict[str, Any]:
    """Calculate budget breakdown with safe defaults."""
    # Default costs (in INR)
    DEFAULTS = {
        'flight_cost': 0,          # Free for local destinations
        'daily_hotel': 1500,       # ₹1500/night
        'daily_food': 500,         # ₹500/day
        'daily_activities': 800,   # ₹800/day
        'visa_cost': 0             # No visa needed
    }
    
    # Get costs with defaults
    costs = {key: destination_info.get(key, default) 
             for key, default in DEFAULTS.items()}
    
    # Calculate breakdown
    flight_cost = costs['flight_cost'] * num_people
    accommodation = costs['daily_hotel'] * num_days * num_people
    meals = costs['daily_food'] * num_days * num_people
    activities = costs['daily_activities'] * num_days * num_people
    visa = costs['visa_cost'] * num_people
    local_transport = 500 * num_days * num_people
    miscellaneous = 500 * num_days * num_people
    
    total_estimated = flight_cost + accommodation + meals + activities + visa + local_transport + miscellaneous
    
    return {
        "flight_cost": flight_cost,
        "accommodation": accommodation,
        "meals": meals,
        "activities": activities,
        "visa": visa,
        "local_transport": local_transport,
        "miscellaneous": miscellaneous,
        "total_estimated": total_estimated,
        "remaining": total_budget - total_estimated,
        "per_person": total_estimated / num_people,
        "per_person_per_day": total_estimated / (num_people * num_days)
    }

def generate_itinerary(destination: str, 
                      start_date: Union[str, datetime], 
                      end_date: Union[str, datetime],
                      interests: List[str], 
                      num_people: int, 
                      budget_info: Dict[str, Any], 
                      country: str = None) -> str:
    """Enhanced with visual elements and improved formatting"""
    try:
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        num_days = (end_date - start_date).days + 1
        flight_cost = budget_info.get('flight_cost', 0)
        # Get destination info again for visa details
        dest_info = get_destination_info(destination, country)
        start_date_str = start_date.strftime('%d %B %Y')
        end_date_str = end_date.strftime('%d %B %Y')
        # Construct location string with country if provided
        location = f"{destination}, {country}" if country else destination
        
        # Format budget status message with visual indicators
        if budget_info['remaining'] >= 0:
            budget_status = f"✅ Within budget (remaining: ₹{budget_info['remaining']:,.2f})"
            budget_emoji = "💰"
        else:
            budget_status = f"⚠️ Over budget by ₹{abs(budget_info['remaining']):,.2f}"
            budget_emoji = "💸"
        
        # Create a visually appealing header
        header = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           🌍 TRAVEL ITINERARY 2025 🌍                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  📍 Destination: {location:<30}                                      ║
║  📅 Duration: {start_date} to {end_date} ({num_days} days)                    ║
║  👥 Travelers: {num_people} people                                            ║
║  {budget_emoji} Budget: ₹{budget_info['total_estimated']:,.2f} - {budget_status}     ║
║  🎯 Interests: {', '.join(interests)}                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        
        prompt = f"""
        Create a beautifully formatted {num_days}-day travel itinerary for {location} for {num_people} people focusing on {', '.join(interests)}.
        Travel dates: {start_date} to {end_date}.
        Total budget: ₹{budget_info['total_estimated']:,.2f} ({budget_status}).
        
        {'' if budget_info['remaining'] >= 0 else 'Please suggest ways to reduce costs while maintaining a good experience.'}
        
        Format the itinerary with the following enhanced sections:
        
        {'' if budget_info['remaining'] >= 0 else '''
        ╭─────────────────────────────────────────────────────────────╮
        │                   💡 BUDGET OPTIMIZATION                    │
        ╰─────────────────────────────────────────────────────────────╯
        
        🎯 COST-CUTTING STRATEGIES:
        • Areas where costs can be reduced
        • 🏨 Alternative accommodations (hostels, guesthouses)
        • 🍽️ Budget dining options
        • 🎫 Free/low-cost activities
        • 🚌 Public transport vs private transport
        • 📱 Money-saving apps and deals
        
        '''}
        ╭─────────────────────────────────────────────────────────────╮
        │                     🛂 VISA REQUIREMENTS                    │
        ╰─────────────────────────────────────────────────────────────╯
        
        📋 VISA DETAILS (for Indian Citizens):
        ┌─────────────────────────────────────────────────────────────┐
        │ Visa Type      │ {dest_info['visa_type']}                    │
        │ Cost           │ ₹{dest_info['visa_cost']}                   │
        │ Process        │ {dest_info['visa_process']}                 │
        │ Success Rate   │ {dest_info['visa_chance']}                  │
        └─────────────────────────────────────────────────────────────┘
        
        ✅ REQUIRED DOCUMENTS:
        • Valid passport (6+ months validity)
        • Completed application form
        • Recent passport-size photographs
        • Flight itinerary
        • Hotel bookings
        • Bank statements (last 3 months)
        • Employment letter/NOC
        
        ⏰ PROCESSING TIME: [Specify timeline]
        💡 SUCCESS TIPS: [Application tips]
        
        ╭─────────────────────────────────────────────────────────────╮
        │                   📅 DAILY ITINERARY                       │
        ╰─────────────────────────────────────────────────────────────╯
        
        For each day, format as:
        
        ┌── DAY X │ [DATE] ─────────────────────────────────────────────┐
        │                                                             │
        │ 🌅 MORNING (9:00 AM - 12:00 PM)                           │
        │   • [Activity with location pin 📍]                        │
        │   • ⏱️ Duration: [Time needed]                             │
        │   • 💰 Cost: [Estimated cost]                              │
        │                                                             │
        │ ☀️ AFTERNOON (12:00 PM - 6:00 PM)                         │
        │   • [Activity with location pin 📍]                        │
        │   • 🍽️ Lunch recommendation                                │
        │   • ⏱️ Duration: [Time needed]                             │
        │                                                             │
        │ 🌆 EVENING (6:00 PM - 10:00 PM)                           │
        │   • [Activity with location pin 📍]                        │
        │   • 🍽️ Dinner recommendation                               │
        │   • 📸 Photo spot of the day                               │
        │                                                             │
        │ 🏨 ACCOMMODATION: [Hotel/Area name]                        │
        └─────────────────────────────────────────────────────────────┘
        
        ╭─────────────────────────────────────────────────────────────╮
        │                    ✈️ FLIGHT GUIDE                         │
        ╰─────────────────────────────────────────────────────────────╯
        
        🏆 RECOMMENDED AIRLINES:
        • [Airline 1] - [Brief description]
        • [Airline 2] - [Brief description]
        
        📈 BOOKING STRATEGY:
        • Best time to book: [Timeline]
        • Price tracking apps: [Apps]
        • Flexible date options
        
        ╭─────────────────────────────────────────────────────────────╮
        │                   🏨 ACCOMMODATION                          │
        ╰─────────────────────────────────────────────────────────────╯
        
        💎 LUXURY (₹5000+/night):
        • [Hotel name] - [Description]
        
        🏨 MID-RANGE (₹2000-5000/night):
        • [Hotel name] - [Description]
        
        🎒 BUDGET (₹500-2000/night):
        • [Hostel/Guesthouse name] - [Description]
        
        📍 BEST AREAS TO STAY:
        • [Area 1]: [Why it's good]
        • [Area 2]: [Why it's good]
        
        ╭─────────────────────────────────────────────────────────────╮
        │                    🍽️ CULINARY JOURNEY                     │
        ╰─────────────────────────────────────────────────────────────╯
        
        🌟 MUST-TRY DISHES:
        • [Dish 1] - [Description]
        • [Dish 2] - [Description]
        • [Dish 3] - [Description]
        
        🍴 RESTAURANT CATEGORIES:
        
        💰 BUDGET EATS (₹200-500):
        • [Restaurant] - [Specialty]
        
        🍽️ MID-RANGE (₹500-1500):
        • [Restaurant] - [Specialty]
        
        🌟 FINE DINING (₹1500+):
        • [Restaurant] - [Specialty]
        
        ╭─────────────────────────────────────────────────────────────╮
        │                   🚗 TRANSPORTATION                         │
        ╰─────────────────────────────────────────────────────────────╯
        
        🚌 PUBLIC TRANSPORT:
        • [Options and costs]
        
        🚗 PRIVATE TRANSPORT:
        • [Options and costs]
        
        📱 USEFUL APPS:
        • [App 1] - [Purpose]
        • [App 2] - [Purpose]
        
        ╭─────────────────────────────────────────────────────────────╮
        │                  🎫 ACTIVITY BOOKINGS                       │
        ╰─────────────────────────────────────────────────────────────╯
        
        🌐 BOOKING PLATFORMS:
        • [Platform 1] - [What to book]
        • [Platform 2] - [What to book]
        
        💡 MONEY-SAVING TIPS:
        • [Tip 1]
        • [Tip 2]
        
        ╭─────────────────────────────────────────────────────────────╮
        │                    ✨ TRAVEL WISDOM                        │
        ╰─────────────────────────────────────────────────────────────╯
        
        🤝 CULTURAL ETIQUETTE:
        • [Important customs]
        
        🏥 HEALTH & SAFETY:
        • [Important advice]
        
        🎒 PACKING ESSENTIALS:
        • [Climate-appropriate items]
        • [Special items for activities]
        
        🗣️ LANGUAGE TIPS:
        • Essential phrases
        • Translation apps
        
        Use beautiful box-drawing characters, emojis, and clear section dividers.
        Make each section visually distinct and easy to scan.
        """
        
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        itinerary = response.text
        
        # Add the header to the beginning
        final_itinerary = header + itinerary
        
        # Add enhanced image and traveler insights sections
        final_itinerary += """
        
╭─────────────────────────────────────────────────────────────╮
│                  📸 VISUAL DESTINATION GUIDE                │
╰─────────────────────────────────────────────────────────────╯

🖼️ INSTAGRAM-WORTHY SPOTS:
"""
        
        # Make sure the images key exists and has data
        if dest_info.get('images') and len(dest_info['images']) > 0:
            for i, img in enumerate(dest_info['images'][:3], 1):
                final_itinerary += f"""
┌─ PHOTO SPOT {i} ─────────────────────────────────────────────┐
│ 📷 {img.get('description', 'Scenic view'):<50} │
│ 📍 [Location details]                                       │
│ ⏰ Best time: [Golden hour/Blue hour]                       │
│ 💡 Photo tip: [Composition/lighting advice]                │"""
                if 'credit' in img:
                    final_itinerary += f"""
│ 📸 Credit: {img['credit']}                                  │"""
                final_itinerary += """
└─────────────────────────────────────────────────────────────┘
"""
        else:
            final_itinerary += """
┌─────────────────────────────────────────────────────────────┐
│ ⚠️  No image data available for this destination            │
│ 💡 Please check Instagram, Pinterest, or Google Images     │
│     for visual inspiration and photo opportunities          │
└─────────────────────────────────────────────────────────────┘
"""
        
        # Add enhanced traveler insights
        final_itinerary += """
╭─────────────────────────────────────────────────────────────╮
│                   🌟 TRAVELER INSIGHTS                      │
╰─────────────────────────────────────────────────────────────╯

💬 INSIDER TIPS FROM FELLOW TRAVELERS:
"""
        
        if dest_info.get('traveler_insights'):
            # Format the insights in a box
            insights_lines = dest_info['traveler_insights'].split('\n')
            for line in insights_lines:
                if line.strip():
                    final_itinerary += f"   • {line.strip()}\n"
        else:
            final_itinerary += """
┌─────────────────────────────────────────────────────────────┐
│ 📝 No specific traveler insights available                  │
│ 💡 Check TripAdvisor, Reddit, or travel blogs for         │
│    real experiences and tips from other travelers           │
└─────────────────────────────────────────────────────────────┘
"""
        
        # Add a beautiful footer
        final_itinerary += """

╔══════════════════════════════════════════════════════════════════════════════╗
║                         🎉 HAPPY TRAVELS! 🎉                                ║
║                                                                              ║
║  Remember: The best trips are made of unexpected moments and new friendships ║
║  📱 Save this itinerary offline • 🔄 Stay flexible • ✨ Enjoy every moment  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        return final_itinerary
        
    except Exception as e:
        error_message = f"""
╔══════════════════════════════════════════════════════════════╗
║                        ⚠️ ERROR OCCURRED ⚠️                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  We encountered an issue generating your itinerary:          ║
║  {str(e):<58} ║
║                                                              ║
║  💡 Please try:                                              ║
║  • Checking your destination spelling                        ║
║  • Using a different destination                             ║
║  • Contacting our support team                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        return error_message

def display_destination_profile(info: Dict[str, Any]):
    """Show destination info with images"""
    print(f"\n{'🌍 ' + info['name'].upper() + ' PROFILE ':-^50}")
    print(f"📍 {info['country']} | 💰 {info['budget'].capitalize()} Budget")
    print(f"🌤 Best Season: {info['best_season']}")
    print(f"🎯 Top Interests: {', '.join(info['interests'])}")
    
    if info.get('images'):
        print("\n📷 Recommended Photo Spots:")
        for img in info['images'][:3]:
            print(f"  - {img['description']}")

def get_user_input():
    """Get detailed user input for travel planning"""
    print("\n===== Global AI Travel Planner =====")
    genai.configure(api_key="AIzaSyDW4BCGnID5zsxrjDX1DNu23-Fn4tkH_Hw")
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Get basic destination information or use interests to find destinations
    print("\nYou can either:")
    print("1. Enter a specific destination")
    print("2. Get destination recommendations based on your interests")
    choice = input("\nEnter your choice (1 or 2): ")
    
    destination = ""
    country = None
    interests = []
    
    if choice == "1":
        # Get specific destination
        destination = input("\nEnter your destination (e.g., 'Paris', 'Kyoto', 'New York'): ")
        country_input = input("Enter the country (optional, helps with disambiguation): ")
        if country_input.strip():
            country = country_input
            
        # Get interests for specific destination too
        print("\nCommon travel interests:")
        for i, interest in enumerate(common_interests, 1):
            print(f"{i}. {interest.capitalize()}")
        
        while True:
            try:
                choices = input("\nEnter interest numbers (comma separated) or type custom interests separated by commas: ")
                
                if ',' in choices and all(c.strip().isdigit() for c in choices.split(',')):
                    selected = [int(c.strip()) for c in choices.split(",")]
                    if all(1 <= s <= len(common_interests) for s in selected):
                        interests = [common_interests[s-1] for s in selected]
                        break
                    print(f"Please enter numbers 1-{len(common_interests)}")
                else:
                    interests = [interest.strip().lower() for interest in choices.split(",")]
                    break
            except ValueError:
                print("Invalid input. Try again.")
                
    else:  # choice == "2"
        # Show common interests
        print("\nCommon travel interests:")
        for i, interest in enumerate(common_interests, 1):
            print(f"{i}. {interest.capitalize()}")
        
        # Get user interests
        while True:
            try:
                choices = input("\nEnter interest numbers (comma separated) or type custom interests separated by commas: ")
                
                if ',' in choices and all(c.strip().isdigit() for c in choices.split(',')):
                    selected = [int(c.strip()) for c in choices.split(",")]
                    if all(1 <= s <= len(common_interests) for s in selected):
                        interests = [common_interests[s-1] for s in selected]
                        break
                    print(f"Please enter numbers 1-{len(common_interests)}")
                else:
                    interests = [interest.strip().lower() for interest in choices.split(",")]
                    break
            except ValueError:
                print("Invalid input. Try again.")
        
        # Ask if user wants to filter by continent
        continent_filter = None
        continent_choice = input("\nWould you like to filter destinations by continent? (y/n): ").lower()
        if continent_choice.startswith('y'):
            continents = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
            for i, cont in enumerate(continents, 1):
                print(f"{i}. {cont}")
            while True:
                try:
                    cont_choice = int(input("\nEnter continent number: "))
                    if 1 <= cont_choice <= len(continents):
                        continent_filter = continents[cont_choice-1]
                        break
                    print(f"Please enter a number between 1 and {len(continents)}")
                except ValueError:
                    print("Please enter a valid number.")
                
        # Get recommendations
        print("\nGetting destination recommendations based on your interests...")
        recommendations = recommend_destinations_for_interests(model, interests, continent=continent_filter, max_results=5)
        
        # Debug output
        print(f"Debug - recommendations type: {type(recommendations)}")
        print(f"Debug - recommendations content: {recommendations}")
        
        # Check if we got recommendations
        if not recommendations:
            print("No recommendations found. Please try again.")
            return None

        # Display recommendations properly
        print("\nRecommended destinations based on your interests:")
        all_destinations = []
        
        for interest, destinations in recommendations.items():
            if not destinations:
                continue
                
            print(f"\n{interest.title()} destinations:")
            for rec in destinations:
                if isinstance(rec, dict):
                    all_destinations.append(rec)
                    idx = len(all_destinations)
                    name = rec.get('name', 'Unknown')
                    country_name = rec.get('country', 'Unknown')
                    continent = rec.get('continent', 'Unknown')
                    description = rec.get('description', 'No description')
                    
                    print(f"{idx}. {name}, {country_name} ({continent})")
                    print(f"   Description: {description}")
                    
                    # Handle additional fields that might not exist
                    if 'best_time' in rec:
                        print(f"   Best time to visit: {rec['best_time']}")
                    if 'cost' in rec:
                        print(f"   Budget category: {rec['cost']}")
                    if 'match_reason' in rec:
                        print(f"   Why it matches: {rec['match_reason']}")
                    print()
                    
        if not all_destinations:
            print("No valid destinations found. Please try again.")
            return None
            
        # Ask user to select one of the recommendations
        while True:
            try:
                rec_choice = int(input(f"Enter the number of your chosen destination (1-{len(all_destinations)}): "))
                if 1 <= rec_choice <= len(all_destinations):
                    selected_dest = all_destinations[rec_choice-1]
                    destination = selected_dest['name']
                    country = selected_dest['country']
                    print(f"\nSelected destination: {destination}, {country}")
                    break
                print(f"Please enter a number between 1 and {len(all_destinations)}")
            except ValueError:
                print("Please enter a valid number.")
    
    # Get budget
    while True:
        try:
            budget = float(input("\nEnter total budget (INR): "))
            if budget > 0:
                break
            print("Budget must be positive.")
        except ValueError:
            print("Invalid amount. Please enter a number.")

    # Get dates
    print("\nEnter travel dates (format: DD Month YYYY, e.g., 01 June 2025)")
    while True:
        try:
            start_date = input("Start date: ")
            end_date = input("End date: ")
            start = datetime.strptime(start_date, "%d %B %Y")
            end = datetime.strptime(end_date, "%d %B %Y")
            if end >= start:
                break
            print("End date must be after start date.")
        except ValueError:
            print("Invalid format. Use '01 June 2025' format.")

    # Get travelers
    while True:
        try:
            num_people = int(input("\nNumber of travelers: "))
            if num_people > 0:
                break
            print("Must have at least 1 traveler.")
        except ValueError:
            print("Please enter a number.")

    return {
        "destination": destination,
        "country": country,
        "budget": budget,
        "start_date": start_date,
        "end_date": end_date,
        "num_people": num_people,
        "interests": interests
    }
    
    
    
def main():
    """Main function to run the travel planner"""
    user_input = get_user_input()
    
    # Get destination information
    print(f"\nGetting information about {user_input['destination']}{', ' + user_input['country'] if user_input['country'] else ''}...")
    destination_info = get_destination_info(user_input["destination"], user_input["country"])
    
    # Calculate number of days
    num_days = (datetime.strptime(user_input["end_date"], "%d %B %Y") - 
                datetime.strptime(user_input["start_date"], "%d %B %Y")).days + 1
    
    # Calculate budget
    budget = calculate_budget_breakdown(
        destination_info,
        user_input["budget"],
        num_days,
        user_input["num_people"]
    )
    
    # Generate itinerary
    print("\nGenerating your personalized travel plan...")
    itinerary = generate_itinerary(
        user_input["destination"], 
        user_input["start_date"],
        user_input["end_date"],
        user_input["interests"],
        user_input["num_people"],
        budget,
        user_input["country"]
    )
    
    # Display results
    print("\n" + "=" * 50)
    print(f"⭐ TRAVEL PLAN FOR {user_input['destination'].upper()}{', ' + user_input['country'].upper() if user_input['country'] else ''} ⭐")
    print("=" * 50)
    
    print(f"\n📍 Destination: {user_input['destination']}{', ' + destination_info['country'] if destination_info['country'] != 'Unknown' else ''}")
    print(f"📅 Dates: {user_input['start_date']} to {user_input['end_date']} ({num_days} days)")
    print(f"👥 Travelers: {user_input['num_people']}")
    print(f"💰 Your Budget: ₹{user_input['budget']:,.2f}")
    
    print(f"\n💵 ESTIMATED COSTS:")
    print(f"✈️ Flights: ₹{budget['flight_cost']:,.2f}")
    print(f"🏨 Accommodation: ₹{budget['accommodation']:,.2f}")
    print(f"🍽️ Meals: ₹{budget['meals']:,.2f}")
    print(f"🎯 Activities: ₹{budget['activities']:,.2f}")
    print(f"🛂 Visa: ₹{budget['visa']:,.2f} ({destination_info['visa_type']}, Approval chance: {destination_info['visa_chance']})")
    print(f"🚕 Local Transport: ₹{budget['local_transport']:,.2f}")
    print(f"🛍️ Miscellaneous: ₹{budget['miscellaneous']:,.2f}")
    print(f"\n📊 Total Estimated: ₹{budget['total_estimated']:,.2f}")
    print(f"💸 Remaining: ₹{budget['remaining']:,.2f}")
    print(f"💰 Cost per person: ₹{budget['per_person']:,.2f}")
    print(f"📈 Cost per person per day: ₹{budget['per_person_per_day']:,.2f}")
    
    # Budget status message
    if budget['remaining'] >= 0:
        print(f"\n✅ Your budget is sufficient! You have ₹{budget['remaining']:,.2f} extra.")
    else:
        print(f"\n⚠️ You are over budget by ₹{abs(budget['remaining']):,.2f}. Check the itinerary for cost-saving tips.")
    
    # Save itinerary to file
    try:
        filename = f"{user_input['destination'].replace(' ', '_')}_itinerary.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"⭐ TRAVEL PLAN FOR {user_input['destination'].upper()}{', ' + user_input['country'].upper() if user_input['country'] else ''} ⭐\n\n")
            f.write(f"📍 Destination: {user_input['destination']}{', ' + destination_info['country'] if destination_info['country'] != 'Unknown' else ''}\n")
            f.write(f"📅 Dates: {user_input['start_date']} to {user_input['end_date']} ({num_days} days)\n")
            f.write(f"👥 Travelers: {user_input['num_people']}\n")
            f.write(f"💰 Your Budget: ₹{user_input['budget']:,.2f}\n\n")
            
            f.write(f"💵 ESTIMATED COSTS:\n")
            f.write(f"✈️ Flights: ₹{budget['flight_cost']:,.2f}\n")
            f.write(f"🏨 Accommodation: ₹{budget['accommodation']:,.2f}\n")
            f.write(f"🍽️ Meals: ₹{budget['meals']:,.2f}\n")
            f.write(f"🎯 Activities: ₹{budget['activities']:,.2f}\n")
            f.write(f"🛂 Visa: ₹{budget['visa']:,.2f} ({destination_info['visa_type']}, Approval chance: {destination_info['visa_chance']})\n")
            f.write(f"🚕 Local Transport: ₹{budget['local_transport']:,.2f}\n")
            f.write(f"🛍️ Miscellaneous: ₹{budget['miscellaneous']:,.2f}\n\n")
            f.write(f"📊 Total Estimated: ₹{budget['total_estimated']:,.2f}\n")
            f.write(f"💸 Remaining: ₹{budget['remaining']:,.2f}\n")
            f.write(f"💰 Cost per person: ₹{budget['per_person']:,.2f}\n")
            f.write(f"📈 Cost per person per day: ₹{budget['per_person_per_day']:,.2f}\n\n")
            
            # Budget status message
            if budget['remaining'] >= 0:
                f.write(f"✅ Your budget is sufficient! You have ₹{budget['remaining']:,.2f} extra.\n\n")
            else:
                f.write(f"⚠️ You are over budget by ₹{abs(budget['remaining']):,.2f}. Check the itinerary for cost-saving tips.\n\n")
            
            f.write("\n" + itinerary)
            
        print(f"\n💾 Itinerary saved to {filename}")
    except Exception as e:
        print(f"\nCouldn't save itinerary to file: {str(e)}")
    
    print("\n" + "=" * 50)
    print("\n" + itinerary)

if __name__ == "__main__":
    main()
