"""
Graph state definition for the LinkedIn Ads Sales Agent LangGraph workflow.
This module defines the state structure that flows through the agent graph.
"""

from typing import TypedDict, List, Optional, Dict, Any
from sales_agent.models import (
    CustomerProfile, 
    HistoricalPerformance, 
    CompetitorData, 
    MarketBenchmarks, 
    Recommendation,
    SalesPitch
)


class GraphState(TypedDict):
    """
    State object that flows through the LangGraph nodes for the LinkedIn Ads Sales Agent.
    
    The GraphState class defines the data structure that persists throughout the entire
    sales pitch generation workflow. It acts as a shared memory between different nodes
    in the LangGraph, allowing each node to read from and write to this state as the
    workflow progresses from data collection to final presentation generation.
    
    This state enables the agent to:
    - Maintain context across different analysis steps
    - Pass data between nodes (data collection → analysis → visualization → recommendations)
    - Track workflow progress and handle errors
    - Store intermediate results for use in subsequent steps
    """
    # Core customer information
    customer_profile: Optional[CustomerProfile]
    
    # Performance data for historical analysis
    historical_data: List[HistoricalPerformance]
    
    # Competitor intelligence data
    competitor_data: List[CompetitorData]
    
    # Industry benchmarks for comparison
    market_benchmarks: Optional[MarketBenchmarks]
    
    # Analysis results from data processing
    analysis_results: Dict[str, Any]
    
    # File paths to generated charts and visualizations
    visualizations: List[str]
    
    # AI-generated recommendations for the customer
    recommendations: List[Recommendation]
    
    # Path to the final PowerPoint presentation
    presentation_path: Optional[str]
    
    # Current step in the workflow for tracking progress
    current_step: str
    
    # Error handling
    error_message: Optional[str]
    
    # Raw input data from the user
    input_data: Dict[str, Any]