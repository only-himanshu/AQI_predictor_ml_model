"""
aqi_utils.py
============
Shared utilities for mapping a numeric AQI value to the official CPCB
National AQI category, a display color, and a health advisory message.

Reference: CPCB National Air Quality Index categories (2014 standard).
"""

AQI_CATEGORIES = [
    (0, 50, "Good", "#009865",
     "Minimal impact. Air quality is considered satisfactory for everyone."),
    (51, 100, "Satisfactory", "#a3c853",
     "Minor breathing discomfort to sensitive people (asthma, lung/heart disease)."),
    (101, 200, "Moderate", "#ffd400",
     "Breathing discomfort to people with lung, asthma, and heart disease, "
     "children, and older adults."),
    (201, 300, "Poor", "#ff7e27",
     "Breathing discomfort to most people on prolonged exposure. "
     "Sensitive groups should limit outdoor exertion."),
    (301, 400, "Very Poor", "#ff0000",
     "Respiratory illness on prolonged exposure. Avoid outdoor activity, "
     "especially for children, elderly, and those with respiratory/heart conditions."),
    (401, 10_000, "Severe", "#7e0023",
     "Affects healthy people and seriously impacts those with existing diseases. "
     "Avoid all outdoor physical activity; remain indoors with air purification if possible."),
]


def categorize_aqi(aqi: float):
    """Return (category, color_hex, health_advice) for a numeric AQI value."""
    aqi = max(0, aqi)
    for low, high, category, color, advice in AQI_CATEGORIES:
        if low <= aqi <= high:
            return category, color, advice
    return "Severe", "#7e0023", AQI_CATEGORIES[-1][4]
