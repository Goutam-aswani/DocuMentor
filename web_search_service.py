import requests
from typing import List, Dict
from config import settings


class TavilySearchService:
    def __init__(self):
        self.api_key = settings.tavily_api_key
        self.base_url = "https://api.tavily.com/search"
    
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search the web using Tavily API
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, content, and URL
        """
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "include_images": False,
                "include_raw_content": False,
                "max_results": max_results
            }
            
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the results
            results = []
            if "results" in data:
                for result in data["results"]:
                    results.append({
                        "title": result.get("title", ""),
                        "content": result.get("content", ""),
                        "url": result.get("url", ""),
                        "score": result.get("score", 0)
                    })
            
            return results
            
        except Exception as e:
            print(f"Error in web search: {e}")
            return []
    
    def format_search_results(self, results: List[Dict]) -> str:
        """
        Format search results for inclusion in the prompt
        """
        if not results:
            return "No web search results found."
        
        formatted = "**Web Search Results:**\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"**{i}. {result['title']}**\n"
            formatted += f"Source: {result['url']}\n"
            formatted += f"{result['content']}\n\n"
        
        return formatted
