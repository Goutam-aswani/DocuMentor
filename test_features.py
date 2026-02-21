#!/usr/bin/env python3
"""
Test script to verify model configurations and web search functionality
"""

from chatbot_service import MODELS, get_model_instance
from web_search_service import TavilySearchService
from config import settings


def test_models():
    """Test that all configured models can be instantiated"""
    print("Testing model configurations...")
    
    for model_name in MODELS.keys():
        try:
            model = get_model_instance(model_name)
            print(f"✅ {model_name}: OK")
        except Exception as e:
            print(f"❌ {model_name}: {str(e)}")
    
    print()


def test_web_search():
    """Test web search functionality"""
    print("Testing web search...")
    
    try:
        search_service = TavilySearchService()
        results = search_service.search("Python programming", max_results=3)
        
        if results:
            print(f"✅ Web search: Got {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['title'][:50]}...")
        else:
            print("❌ Web search: No results returned")
            
    except Exception as e:
        print(f"❌ Web search: {str(e)}")
    
    print()


def test_environment_variables():
    """Test that all required environment variables are set"""
    print("Testing environment variables...")
    
    required_vars = [
        ('GOOGLE_API_KEY', settings.google_api_key),
        ('GROK_API_KEY', settings.grok_api_key),
        ('OPENROUTER_API_KEY', settings.openrouter_api_key),
        ('TAVILY_API_KEY', settings.tavily_api_key),
    ]
    
    for var_name, var_value in required_vars:
        if var_value and len(var_value) > 10:
            print(f"✅ {var_name}: Set (length: {len(var_value)})")
        else:
            print(f"❌ {var_name}: Not set or too short")
    
    print()


if __name__ == "__main__":
    print("🔧 Chatbot Configuration Test\n")
    
    test_environment_variables()
    test_models()
    test_web_search()
    
    print("✨ Test completed!")
