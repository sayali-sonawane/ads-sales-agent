"""
Main module for the LinkedIn Ads Sales Agent application.
This module provides the entry point for running the complete sales pitch generation workflow.
"""

import os
from dotenv import load_dotenv
from sales_agent.workflow import SalesAgentWorkflow
from sales_agent.models import Industry

# Load environment variables
load_dotenv()


def check_sample_datasets_exist() -> bool:
    """
    Check if sample datasets already exist in the data directory.
    
    Returns:
        True if datasets exist, False otherwise
    """
    required_files = [
        "data/sample_customers.csv",
        "data/sample_historical_performance.csv", 
        "data/sample_competitors.csv"
    ]
    
    return all(os.path.exists(file_path) for file_path in required_files)


def main():
    """
    Main entry point for the LinkedIn Ads Sales Agent application.
    Demonstrates the complete workflow from data generation to presentation creation.
    """
    print("🚀 LinkedIn Ads Sales Agent Starting...")
    print("=" * 60)
    
    try:
        # Initialize the workflow
        # Note: Set your OpenAI API key in .env file or environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            print("⚠️  Warning: No OpenAI API key found. Recommendations will use fallback logic.")
            print("   To use AI-powered recommendations, set OPENAI_API_KEY in your environment.")
            print()
        
        workflow = SalesAgentWorkflow(openai_api_key=openai_api_key)
        
        # Check if sample datasets exist, generate only if missing
        if not check_sample_datasets_exist():
            print("📊 Sample datasets not found. Generating new datasets...")
            dataset_paths = workflow.generate_sample_datasets()
            print("✅ Sample datasets created:")
            for data_type, path in dataset_paths.items():
                print(f"   - {data_type.title()}: {path}")
        else:
            print("📊 Using existing sample datasets from data/ directory")
        print()
        
        # Run a complete workflow example for Technology industry
        print("🔍 Running complete sales pitch generation for Technology industry...")
        print("   This will create analysis, visualizations, and a PowerPoint presentation.")
        print()
        
        # Execute the workflow
        final_state = workflow.run_for_industry(
            industry=Industry.TECHNOLOGY,
            company_name="TechFlow Innovations"
        )
        
        # Display results
        if final_state.get("error_message"):
            print(f"❌ Error occurred: {final_state['error_message']}")
        else:
            print("🎉 Sales pitch generation completed successfully!")
            print()
            
            # Show customer profile
            customer = final_state["customer_profile"]
            print(f"👤 Customer Profile:")
            print(f"   Company: {customer.company_name}")
            print(f"   Industry: {customer.industry.value}")
            print(f"   Size: {customer.company_size}")
            print(f"   Annual Revenue: ${customer.annual_revenue:,.0f}")
            print(f"   Current Ad Spend: ${customer.current_ad_spend:,.0f}")
            print()
            
            # Show key analysis results
            analysis = final_state["analysis_results"]
            print(f"📈 Key Performance Insights:")
            if "revenue_growth" in analysis:
                print(f"   Revenue Growth: {analysis['revenue_growth']:+.1f}%")
            if "roas_vs_benchmark" in analysis:
                print(f"   ROAS vs Benchmark: {analysis['roas_vs_benchmark']:+.1f}%")
            if "ctr_vs_benchmark" in analysis:
                print(f"   CTR vs Benchmark: {analysis['ctr_vs_benchmark']:+.1f}%")
            print()
            
            # Show generated visualizations
            visualizations = final_state["visualizations"]
            print(f"📊 Generated Visualizations:")
            for i, viz_path in enumerate(visualizations, 1):
                print(f"   {i}. {viz_path}")
            print()
            
            # Show recommendations
            recommendations = final_state["recommendations"]
            print(f"💡 Strategic Recommendations ({len(recommendations)} total):")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec.title}")
                print(f"      Priority: {rec.priority} | Est. ROI: +{rec.estimated_roi:.1f}%")
                print(f"      Timeline: {rec.implementation_timeline}")
                print()
            
            # Show final presentation
            presentation_path = final_state["presentation_path"]
            if presentation_path:
                print(f"📋 Final Presentation: {presentation_path}")
                print("   Open this PowerPoint file to view the complete sales pitch!")
            
        print()
        print("=" * 60)
        print("✨ LinkedIn Ads Sales Agent finished!")
        
    except Exception as e:
        print(f"❌ Application error: {str(e)}")
        print("Please check your setup and try again.")


def generate_datasets_only():
    """
    Utility function to generate and save sample datasets without running the full workflow.
    Useful for pre-generating data or refreshing datasets.
    """
    print("📊 Generating sample datasets...")
    
    try:
        # Import here to avoid loading the full workflow
        from sales_agent.data_generator import DataGenerator
        from sales_agent.models import Industry
        
        data_generator = DataGenerator()
        
        # Generate sample data
        customers = data_generator.generate_customer_profiles(count=20)
        
        # Generate historical data for a sample customer
        sample_customer = customers[0]
        historical_data = data_generator.generate_historical_performance(
            customer_id=sample_customer.company_name.replace(" ", "_").lower(),
            periods=8
        )
        
        # Generate competitor data for each industry
        all_competitors = []
        for industry in Industry:
            competitors = data_generator.generate_competitor_data(industry, count=3)
            all_competitors.extend(competitors)
        
        # Save to CSV files
        dataset_paths = {}
        dataset_paths["customers"] = data_generator.save_to_csv(
            "customers", customers, "sample_customers.csv"
        )
        dataset_paths["historical"] = data_generator.save_to_csv(
            "historical", historical_data, "sample_historical_performance.csv"
        )
        dataset_paths["competitors"] = data_generator.save_to_csv(
            "competitors", all_competitors, "sample_competitors.csv"
        )
        
        print("✅ Sample datasets generated successfully:")
        for data_type, path in dataset_paths.items():
            print(f"   - {data_type.title()}: {path}")
            
    except Exception as e:
        print(f"❌ Dataset generation error: {str(e)}")


def demo_different_industries():
    """
    Demonstrate the sales agent workflow across different industries.
    This function shows how the agent adapts its analysis and recommendations
    based on industry-specific characteristics.
    """
    print("🌍 Multi-Industry Demo Starting...")
    print("=" * 60)
    
    # Industries to demonstrate
    demo_industries = [
        Industry.TECHNOLOGY,
        Industry.HEALTHCARE, 
        Industry.FINANCE,
        Industry.RETAIL
    ]
    
    try:
        workflow = SalesAgentWorkflow()
        
        for industry in demo_industries:
            print(f"\n🔍 Analyzing {industry.value} industry...")
            
            # Run workflow for this industry
            final_state = workflow.run_for_industry(industry)
            
            if not final_state.get("error_message"):
                customer = final_state["customer_profile"]
                recommendations = final_state["recommendations"]
                
                print(f"   Company: {customer.company_name}")
                print(f"   Revenue: ${customer.annual_revenue:,.0f}")
                print(f"   Ad Spend: ${customer.current_ad_spend:,.0f}")
                print(f"   Top Recommendation: {recommendations[0].title if recommendations else 'N/A'}")
                
                presentation_path = final_state["presentation_path"]
                if presentation_path:
                    print(f"   Presentation: {presentation_path}")
            else:
                print(f"   ❌ Error: {final_state['error_message']}")
        
        print("\n" + "=" * 60)
        print("✨ Multi-Industry Demo completed!")
        
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")


if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--generate-data":
            generate_datasets_only()
        elif sys.argv[1] == "--demo":
            demo_different_industries()
        elif sys.argv[1] == "--help":
            print("LinkedIn Ads Sales Agent - Usage:")
            print("  python -m sales_agent.main                # Run main application")
            print("  python -m sales_agent.main --generate-data # Generate sample datasets only") 
            print("  python -m sales_agent.main --demo          # Run multi-industry demo")
            print("  python -m sales_agent.main --help          # Show this help message")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Run the main application
        main()
