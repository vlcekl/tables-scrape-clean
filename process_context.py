from pydantic import BaseModel, Field

class ContextSummary(BaseModel):
    """Structured summary of provided text including text along with identiried time and location information"""
    summary: str = Field(description="Informative summary of a provided text. Focus on agronomic information, such as crop or yield.")
    products: list[str] = Field(default_factory=list, description="Identified agricultural products, e.g., crops or animal products, the text is about.")
    time: str = Field(description="Time point or time span in ISO format the summarized text refers to.")
    location: str = Field(description="Geographic location the summarized text refers to.")
