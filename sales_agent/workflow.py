"""
LangGraph workflow definition for the LinkedIn Ads Sales Agent.
This module defines the complete workflow graph that orchestrates the sales pitch
generation process from data collection through final presentation creation.
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import os

from sales_agent.graph_state import GraphState
from sales_agent.agents import (
    DataAnalysisAgent,
    VisualizationAgent, 
    RecommendationAgent,
    PresentationAgent
)
from sales_agent.data_generator import DataGenerator
from sales_agent.models import Industry


class SalesAgentWorkflow:
    """
    Main workflow class that orchestrates the LinkedIn Ads Sales Agent process.
    Uses LangGraph to create a stateful workflow that progresses through:
    1. Data collection and preparation
    2. Performance analysis 
    3. Visualization creation
    4. Recommendation generation
    5. Final presentation assembly
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the sales agent workflow with necessary components.
        
        Args:
            openai_api_key: OpenAI API key for LLM-powered agents
        """
        # Initialize language model
        self.llm = None
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            try:
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    max_tokens=2000
                )
            except Exception as e:
                print(f"Warning: Could not initialize LLM: {e}")
                self.llm = None
        
        # Initialize agents
        self.data_generator = DataGenerator()
        self.analysis_agent = DataAnalysisAgent(self.llm)
        self.visualization_agent = VisualizationAgent()
        self.recommendation_agent = RecommendationAgent(self.llm)
        self.presentation_agent = PresentationAgent()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow that defines the sales agent process.
        
        Returns:
            Configured StateGraph ready for execution
        """
        # Create the state graph
        workflow = StateGraph(GraphState)
        
        # Add nodes for each step in the process
        workflow.add_node("prepare_data", self._prepare_data_node)
        workflow.add_node("analyze_performance", self._analyze_performance_node)
        workflow.add_node("create_visualizations", self._create_visualizations_node)
        workflow.add_node("generate_recommendations", self._generate_recommendations_node)
        workflow.add_node("create_presentation", self._create_presentation_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define the workflow flow
        workflow.set_entry_point("prepare_data")
        
        # Normal flow
        workflow.add_edge("prepare_data", "analyze_performance")
        workflow.add_edge("analyze_performance", "create_visualizations")
        workflow.add_edge("create_visualizations", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "create_presentation")
        workflow.add_edge("create_presentation", END)
        
        # Error handling edges
        workflow.add_conditional_edges(
            "prepare_data",
            self._check_for_errors,
            {
                "continue": "analyze_performance",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "analyze_performance", 
            self._check_for_errors,
            {
                "continue": "create_visualizations",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "create_visualizations",
            self._check_for_errors, 
            {
                "continue": "generate_recommendations",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_recommendations",
            self._check_for_errors,
            {
                "continue": "create_presentation", 
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _prepare_data_node(self, state: GraphState) -> GraphState:
        """
        Prepare data for analysis by generating or loading customer profiles,
        historical performance, competitor data, and market benchmarks.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with prepared data
        """
        try:
            state["current_step"] = "preparing_data"
            
            # Check if customer profile is provided in input
            input_data = state.get("input_data", {})
            
            if "customer_profile" in input_data:
                # Use provided customer profile
                customer_profile = input_data["customer_profile"]
                state["customer_profile"] = customer_profile
            else:
                # Generate sample customer profile
                customer_profiles = self.data_generator.generate_customer_profiles(count=1)
                customer_profile = customer_profiles[0]
                state["customer_profile"] = customer_profile
            
            # Generate historical performance data
            customer_id = customer_profile.company_name.replace(" ", "_").lower()
            historical_data = self.data_generator.generate_historical_performance(
                customer_id=customer_id,
                periods=8
            )
            state["historical_data"] = historical_data
            
            # Generate competitor data for the industry
            competitor_data = self.data_generator.generate_competitor_data(
                industry=customer_profile.industry,
                count=5
            )
            state["competitor_data"] = competitor_data
            
            # Generate market benchmarks
            market_benchmarks = self.data_generator.generate_market_benchmarks(
                industry=customer_profile.industry
            )
            state["market_benchmarks"] = market_benchmarks
            
            # Initialize other state fields
            state["analysis_results"] = {}
            state["visualizations"] = []
            state["recommendations"] = []
            state["presentation_path"] = None
            state["error_message"] = None
            
            state["current_step"] = "data_prepared"
            
        except Exception as e:
            state["error_message"] = f"Data preparation failed: {str(e)}"
            state["current_step"] = "data_preparation_error"
        
        return state
    
    def _analyze_performance_node(self, state: GraphState) -> GraphState:
        """
        Analyze customer performance data using the DataAnalysisAgent.
        
        Args:
            state: Current graph state with prepared data
            
        Returns:
            Updated state with analysis results
        """
        return self.analysis_agent.analyze_performance(state)
    
    def _create_visualizations_node(self, state: GraphState) -> GraphState:
        """
        Create visualizations using the VisualizationAgent.
        
        Args:
            state: Current graph state with analysis results
            
        Returns:
            Updated state with visualization file paths
        """
        return self.visualization_agent.create_visualizations(state)
    
    def _generate_recommendations_node(self, state: GraphState) -> GraphState:
        """
        Generate recommendations using the RecommendationAgent.
        
        Args:
            state: Current graph state with analysis and visualizations
            
        Returns:
            Updated state with generated recommendations
        """
        return self.recommendation_agent.generate_recommendations(state)
    
    def _create_presentation_node(self, state: GraphState) -> GraphState:
        """
        Create final presentation using the PresentationAgent.
        
        Args:
            state: Current graph state with all generated content
            
        Returns:
            Updated state with presentation file path
        """
        return self.presentation_agent.create_presentation(state)
    
    def _handle_error_node(self, state: GraphState) -> GraphState:
        """
        Handle errors that occur during workflow execution.
        
        Args:
            state: Current graph state with error information
            
        Returns:
            Updated state with error handling results
        """
        error_message = state.get("error_message", "Unknown error occurred")
        print(f"Error in workflow: {error_message}")
        
        # Log error details
        state["current_step"] = "error_handled"
        
        # Could implement error recovery logic here
        # For now, just ensure the workflow can end gracefully
        
        return state
    
    def _check_for_errors(self, state: GraphState) -> str:
        """
        Check if the current state contains any errors.
        
        Args:
            state: Current graph state
            
        Returns:
            "error" if errors are present, "continue" otherwise
        """
        if state.get("error_message") or "error" in state.get("current_step", ""):
            return "error"
        return "continue"
    
    def run(self, input_data: Dict[str, Any] = None) -> GraphState:
        """
        Execute the complete sales agent workflow.
        
        Args:
            input_data: Optional input data including customer profile
            
        Returns:
            Final state with all results including presentation path
        """
        # Initialize state
        initial_state = GraphState(
            customer_profile=None,
            historical_data=[],
            competitor_data=[],
            market_benchmarks=None,
            analysis_results={},
            visualizations=[],
            recommendations=[],
            presentation_path=None,
            current_step="starting",
            error_message=None,
            input_data=input_data or {}
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return final_state
    
    def run_for_industry(self, industry: Industry, company_name: str = None) -> GraphState:
        """
        Run the workflow for a specific industry with optional company name.
        
        Args:
            industry: Target industry for the analysis
            company_name: Optional specific company name
            
        Returns:
            Final state with industry-specific results
        """
        # Generate a customer profile for the specified industry
        profiles = self.data_generator.generate_customer_profiles(count=5)
        industry_profiles = [p for p in profiles if p.industry == industry]
        
        if not industry_profiles:
            # Generate new profiles if none match the industry
            profiles = self.data_generator.generate_customer_profiles(count=10)
            industry_profiles = [p for p in profiles if p.industry == industry]
        
        selected_profile = industry_profiles[0] if industry_profiles else profiles[0]
        
        if company_name:
            selected_profile.company_name = company_name
        
        input_data = {
            "customer_profile": selected_profile
        }
        
        return self.run(input_data)
    
    def generate_sample_datasets(self) -> Dict[str, str]:
        """
        Generate and save sample datasets to CSV files for manual inspection.
        
        Returns:
            Dictionary with paths to generated CSV files
        """
        # Generate sample data
        customers = self.data_generator.generate_customer_profiles(count=20)
        
        # Generate historical data for a sample customer
        sample_customer = customers[0]
        historical_data = self.data_generator.generate_historical_performance(
            customer_id=sample_customer.company_name.replace(" ", "_").lower(),
            periods=8
        )
        
        # Generate competitor data for each industry
        all_competitors = []
        for industry in Industry:
            competitors = self.data_generator.generate_competitor_data(industry, count=3)
            all_competitors.extend(competitors)
        
        # Save to CSV files
        file_paths = {}
        
        try:
            file_paths["customers"] = self.data_generator.save_to_csv(
                "customers", customers, "sample_customers.csv"
            )
            file_paths["historical"] = self.data_generator.save_to_csv(
                "historical", historical_data, "sample_historical_performance.csv"
            )
            file_paths["competitors"] = self.data_generator.save_to_csv(
                "competitors", all_competitors, "sample_competitors.csv"
            )
        except Exception as e:
            print(f"Error saving CSV files: {e}")
        
        return file_paths