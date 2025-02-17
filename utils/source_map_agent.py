from pprint import pprint
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import openai

# Define Pydantic models for structured output
class WebsiteSummary(BaseModel):
    title: str = Field(..., description="Title of the website")
    description: Optional[str] = Field(None, description="Brief description of the website")

class WebsiteObject(BaseModel):
    object_type: str = Field(..., description="Type of the object (e.g., table, image, text)")
    hierarchy_path: str = Field(..., description="XPath or CSS path in the website hierarchy")
    relations: List[str] = Field(..., description="Relations with other objects on the website")

class ProcessedData(BaseModel):
    summary: WebsiteSummary
    objects: List[WebsiteObject]

class AIDataAgent:
    """AI Agent for analyzing and organizing scraped website data."""

    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key

    def analyze_summary(self, soup: BeautifulSoup) -> WebsiteSummary:
        """Analyzes and extracts a brief description of the website."""
        title = soup.title.string if soup.title else "Unknown Title"
        description_tag = soup.find("meta", attrs={"name": "description"})
        description = description_tag["content"] if description_tag else None

        return WebsiteSummary(title=title, description=description)

    def analyze_objects(self, soup: BeautifulSoup) -> List[WebsiteObject]:
        """Analyzes and extracts objects of interest from the website."""
        objects = []

        # Example: Extracting tables
        tables = soup.find_all("table")
        for i, table in enumerate(tables):
            objects.append(
                WebsiteObject(
                    object_type="table",
                    hierarchy_path=f"table[{i + 1}]",
                    relations=[]  # Add relations based on further analysis
                )
            )

        # Example: You can add logic for other objects like images, forms, etc.

        return objects

    def analyze_relations(self, objects: List[WebsiteObject]) -> List[WebsiteObject]:
        """Analyzes relationships between objects and updates them."""
        for obj in objects:
            # Placeholder: Add logic to determine relations between objects
            obj.relations = ["Related to object X"]
        return objects

    def process(self, soup: BeautifulSoup) -> ProcessedData:
        """Processes the soup object to extract and organize website data."""
        summary = self.analyze_summary(soup)
        objects = self.analyze_objects(soup)
        objects_with_relations = self.analyze_relations(objects)

        return ProcessedData(summary=summary, objects=objects_with_relations)

# Example usage
if __name__ == "__main__":
    html_example = """<html><head><title>Sample Website</title><meta name='description' content='This website demonstrates analysis and cleaning of web data.'></head><body><table><tr><td>Sample Table</td></tr></table></body></html>"""
    soup = BeautifulSoup(html_example, "html.parser")

    agent = AIDataAgent(openai_api_key="your-openai-api-key")
    processed_data = agent.process(soup)
    pprint(processed_data.model_json_schema()['properties'])
