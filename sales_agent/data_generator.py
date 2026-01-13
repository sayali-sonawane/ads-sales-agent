"""
Data generator for creating sample datasets for the LinkedIn Ads Sales Agent.
This module generates realistic customer profiles, historical performance data,
competitor information, and market benchmarks for demonstration purposes.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict
from sales_agent.models import (
    Industry, 
    CustomerProfile, 
    PerformanceMetrics, 
    HistoricalPerformance,
    CompetitorData,
    MarketBenchmarks
)


class DataGenerator:
    """
    Generates synthetic but realistic data for LinkedIn advertising scenarios.
    This includes customer profiles across various industries, their historical
    ad performance, competitor data, and industry benchmarks.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_customer_profiles(self, count: int = 50) -> List[CustomerProfile]:
        """
        Generate diverse customer profiles across different industries and company sizes.
        
        Args:
            count: Number of customer profiles to generate
            
        Returns:
            List of CustomerProfile objects with realistic business data
        """
        profiles = []
        company_sizes = ["Small", "Medium", "Large", "Enterprise"]
        
        # Sample company names by industry
        company_names = {
            Industry.TECHNOLOGY: ["TechFlow", "DataSync", "CloudEdge", "DevCore", "NextGen Solutions"],
            Industry.HEALTHCARE: ["MedCare Plus", "HealthTech", "WellnessPro", "CareConnect", "VitalSystems"],
            Industry.FINANCE: ["FinanceFirst", "CreditMax", "InvestPro", "BankTech", "MoneyWise"],
            Industry.RETAIL: ["ShopSmart", "RetailHub", "FashionForward", "MarketPlace", "StyleCo"],
            Industry.MANUFACTURING: ["IndustrialPro", "ManufactureTech", "ProductionLine", "FactoryFlow", "MakeCorp"],
            Industry.EDUCATION: ["EduTech", "LearnPlus", "SkillBuilder", "AcademyPro", "StudyHub"],
            Industry.REAL_ESTATE: ["PropertyPro", "RealtyTech", "HomeFinder", "EstateMax", "PropertyHub"],
            Industry.AUTOMOTIVE: ["AutoTech", "CarPro", "VehicleMax", "MotorCorp", "AutoSolutions"]
        }
        
        locations = ["New York, NY", "San Francisco, CA", "Chicago, IL", "Austin, TX", 
                    "Boston, MA", "Seattle, WA", "Los Angeles, CA", "Miami, FL", "Denver, CO"]
        
        for i in range(count):
            industry = random.choice(list(Industry))
            company_name = random.choice(company_names[industry]) + f" {i+1}"
            company_size = random.choice(company_sizes)
            
            # Generate realistic revenue based on company size
            revenue_ranges = {
                "Small": (500_000, 5_000_000),
                "Medium": (5_000_000, 50_000_000),
                "Large": (50_000_000, 500_000_000),
                "Enterprise": (500_000_000, 10_000_000_000)
            }
            
            annual_revenue = random.uniform(*revenue_ranges[company_size])
            
            # Generate ad spend as percentage of revenue (typically 2-10%)
            ad_spend_percentage = random.uniform(0.02, 0.10)
            current_ad_spend = annual_revenue * ad_spend_percentage
            
            # Industry-specific target audiences and goals
            target_audiences = {
                Industry.TECHNOLOGY: ["Software Developers", "IT Managers", "Tech Executives", "Startups"],
                Industry.HEALTHCARE: ["Healthcare Professionals", "Patients", "Medical Administrators", "Insurance Companies"],
                Industry.FINANCE: ["Investors", "Financial Advisors", "Business Owners", "Individual Consumers"],
                Industry.RETAIL: ["Consumers", "Fashion Enthusiasts", "Online Shoppers", "Local Customers"],
                Industry.MANUFACTURING: ["Procurement Managers", "Supply Chain Professionals", "Industrial Buyers"],
                Industry.EDUCATION: ["Students", "Parents", "Educators", "Corporate Training Managers"],
                Industry.REAL_ESTATE: ["Home Buyers", "Real Estate Investors", "Property Developers", "Renters"],
                Industry.AUTOMOTIVE: ["Car Buyers", "Fleet Managers", "Auto Enthusiasts", "Service Customers"]
            }
            
            goals = {
                Industry.TECHNOLOGY: ["Lead Generation", "Brand Awareness", "Product Adoption", "Talent Acquisition"],
                Industry.HEALTHCARE: ["Patient Acquisition", "Service Awareness", "Provider Network Growth"],
                Industry.FINANCE: ["Client Acquisition", "Investment Growth", "Risk Management", "Financial Education"],
                Industry.RETAIL: ["Sales Growth", "Customer Acquisition", "Brand Recognition", "Seasonal Campaigns"],
                Industry.MANUFACTURING: ["B2B Lead Generation", "Partnership Development", "Industry Leadership"],
                Industry.EDUCATION: ["Student Enrollment", "Course Promotion", "Corporate Training Sales"],
                Industry.REAL_ESTATE: ["Property Sales", "Client Acquisition", "Market Presence", "Listing Promotion"],
                Industry.AUTOMOTIVE: ["Vehicle Sales", "Service Bookings", "Brand Loyalty", "New Model Launches"]
            }
            
            profile = CustomerProfile(
                company_name=company_name,
                industry=industry,
                company_size=company_size,
                location=random.choice(locations),
                annual_revenue=annual_revenue,
                target_audience=random.choice(target_audiences[industry]),
                current_ad_spend=current_ad_spend,
                primary_goals=random.sample(goals[industry], random.randint(2, 3))
            )
            
            profiles.append(profile)
        
        return profiles
    
    def generate_historical_performance(self, customer_id: str, periods: int = 8) -> List[HistoricalPerformance]:
        """
        Generate historical performance data for a customer over multiple quarters.
        
        Args:
            customer_id: Identifier for the customer
            periods: Number of quarters of historical data to generate
            
        Returns:
            List of HistoricalPerformance objects showing trends over time
        """
        historical_data = []
        base_date = datetime.now() - timedelta(days=periods * 90)
        
        # Base performance metrics that will trend over time
        base_impressions = random.randint(50_000, 500_000)
        base_ctr = random.uniform(0.015, 0.035)  # 1.5% to 3.5% CTR
        base_conversion_rate = random.uniform(0.02, 0.08)  # 2% to 8% conversion rate
        base_cpc = random.uniform(2.0, 8.0)  # $2-8 CPC
        
        for i in range(periods):
            # Add some trend and seasonality
            trend_factor = 1 + (i * 0.05)  # 5% improvement per quarter
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * i / 4)  # Seasonal variation
            noise_factor = random.uniform(0.9, 1.1)  # Random variation
            
            total_factor = trend_factor * seasonal_factor * noise_factor
            
            # Calculate metrics with realistic relationships
            impressions = int(base_impressions * total_factor)
            ctr = max(0.01, min(0.05, base_ctr * random.uniform(0.8, 1.2)))
            clicks = int(impressions * ctr)
            
            conversion_rate = max(0.01, min(0.12, base_conversion_rate * random.uniform(0.7, 1.3)))
            conversions = int(clicks * conversion_rate)
            
            cpc = max(1.0, base_cpc * random.uniform(0.8, 1.2))
            ad_spend = clicks * cpc
            
            # Cost per conversion
            cpc_conversion = ad_spend / max(1, conversions)
            
            # Revenue and ROAS
            avg_order_value = random.uniform(50, 500)
            revenue_generated = conversions * avg_order_value
            roas = revenue_generated / max(1, ad_spend)
            
            # Additional metrics
            reach = int(impressions * random.uniform(0.7, 0.9))
            engagement_rate = random.uniform(0.02, 0.08)
            
            metrics = PerformanceMetrics(
                impressions=impressions,
                clicks=clicks,
                conversions=conversions,
                cost_per_click=cpc,
                cost_per_conversion=cpc_conversion,
                click_through_rate=ctr,
                conversion_rate=conversion_rate,
                return_on_ad_spend=roas,
                reach=reach,
                engagement_rate=engagement_rate
            )
            
            quarter = (base_date.month - 1) // 3 + 1
            period_name = f"Q{quarter} {base_date.year}"
            
            historical_performance = HistoricalPerformance(
                customer_id=customer_id,
                period=period_name,
                metrics=metrics,
                ad_spend=ad_spend,
                revenue_generated=revenue_generated
            )
            
            historical_data.append(historical_performance)
            base_date += timedelta(days=90)
        
        return historical_data
    
    def generate_competitor_data(self, industry: Industry, count: int = 5) -> List[CompetitorData]:
        """
        Generate competitor data for market analysis within a specific industry.
        
        Args:
            industry: The industry to generate competitor data for
            count: Number of competitors to generate
            
        Returns:
            List of CompetitorData objects with market intelligence
        """
        competitor_names = {
            Industry.TECHNOLOGY: ["TechGiant", "InnovaCorp", "DigitalPro", "CodeMasters", "TechVision"],
            Industry.HEALTHCARE: ["HealthLeader", "MedTech Solutions", "CarePlus", "WellnessCorp", "VitalCare"],
            Industry.FINANCE: ["FinanceMax", "MoneyPro", "InvestCorp", "BankingPlus", "CapitalFlow"],
            Industry.RETAIL: ["RetailKing", "ShopMax", "StyleLeader", "MarketPro", "FashionHub"],
            Industry.MANUFACTURING: ["IndustrialMax", "ProductionPro", "ManufactureLeader", "FactoryTech", "MakeMaster"],
            Industry.EDUCATION: ["EduLeader", "LearningPro", "SkillMax", "AcademyPlus", "StudyMaster"],
            Industry.REAL_ESTATE: ["PropertyMax", "RealtyLeader", "HomePro", "EstateKing", "PropertyPlus"],
            Industry.AUTOMOTIVE: ["AutoLeader", "CarMax", "VehiclePro", "MotorKing", "AutoPlus"]
        }
        
        platforms = ["LinkedIn", "Facebook", "Google Ads", "Twitter", "Instagram", "YouTube"]
        
        competitors = []
        for i in range(count):
            # Generate realistic market share (should sum to reasonable total)
            market_share = random.uniform(5, 25)  # 5% to 25% market share
            
            # Ad spend correlated with market share
            base_spend = random.uniform(100_000, 2_000_000)
            estimated_ad_spend = base_spend * (market_share / 15)  # Scale by market share
            
            # Engagement rate varies by industry and competitor strength
            base_engagement = {
                Industry.TECHNOLOGY: 0.035,
                Industry.HEALTHCARE: 0.025,
                Industry.FINANCE: 0.020,
                Industry.RETAIL: 0.045,
                Industry.MANUFACTURING: 0.015,
                Industry.EDUCATION: 0.040,
                Industry.REAL_ESTATE: 0.030,
                Industry.AUTOMOTIVE: 0.025
            }
            
            engagement_rate = base_engagement[industry] * random.uniform(0.7, 1.5)
            
            # Reach correlated with ad spend
            estimated_reach = int(estimated_ad_spend * random.uniform(10, 50))
            
            competitor = CompetitorData(
                competitor_name=competitor_names[industry][i],
                industry=industry,
                estimated_ad_spend=estimated_ad_spend,
                market_share=market_share,
                avg_engagement_rate=engagement_rate,
                estimated_reach=estimated_reach,
                primary_platforms=random.sample(platforms, random.randint(2, 4))
            )
            
            competitors.append(competitor)
        
        return competitors
    
    def generate_market_benchmarks(self, industry: Industry) -> MarketBenchmarks:
        """
        Generate industry-specific market benchmarks for comparison analysis.
        
        Args:
            industry: The industry to generate benchmarks for
            
        Returns:
            MarketBenchmarks object with industry averages
        """
        # Industry-specific benchmark ranges based on real LinkedIn advertising data
        benchmarks = {
            Industry.TECHNOLOGY: {
                "avg_ctr": random.uniform(0.025, 0.040),
                "avg_conversion_rate": random.uniform(0.035, 0.065),
                "avg_cost_per_click": random.uniform(4.0, 8.0),
                "avg_engagement_rate": random.uniform(0.030, 0.045),
                "avg_roas": random.uniform(3.5, 6.0)
            },
            Industry.HEALTHCARE: {
                "avg_ctr": random.uniform(0.020, 0.035),
                "avg_conversion_rate": random.uniform(0.025, 0.050),
                "avg_cost_per_click": random.uniform(3.5, 7.0),
                "avg_engagement_rate": random.uniform(0.020, 0.035),
                "avg_roas": random.uniform(4.0, 7.0)
            },
            Industry.FINANCE: {
                "avg_ctr": random.uniform(0.018, 0.030),
                "avg_conversion_rate": random.uniform(0.020, 0.045),
                "avg_cost_per_click": random.uniform(5.0, 12.0),
                "avg_engagement_rate": random.uniform(0.015, 0.030),
                "avg_roas": random.uniform(3.0, 5.5)
            },
            Industry.RETAIL: {
                "avg_ctr": random.uniform(0.030, 0.050),
                "avg_conversion_rate": random.uniform(0.040, 0.080),
                "avg_cost_per_click": random.uniform(2.5, 6.0),
                "avg_engagement_rate": random.uniform(0.035, 0.055),
                "avg_roas": random.uniform(4.5, 8.0)
            },
            Industry.MANUFACTURING: {
                "avg_ctr": random.uniform(0.015, 0.025),
                "avg_conversion_rate": random.uniform(0.015, 0.035),
                "avg_cost_per_click": random.uniform(6.0, 15.0),
                "avg_engagement_rate": random.uniform(0.010, 0.025),
                "avg_roas": random.uniform(2.5, 4.5)
            },
            Industry.EDUCATION: {
                "avg_ctr": random.uniform(0.025, 0.045),
                "avg_conversion_rate": random.uniform(0.030, 0.060),
                "avg_cost_per_click": random.uniform(3.0, 7.0),
                "avg_engagement_rate": random.uniform(0.030, 0.050),
                "avg_roas": random.uniform(3.5, 6.5)
            },
            Industry.REAL_ESTATE: {
                "avg_ctr": random.uniform(0.020, 0.035),
                "avg_conversion_rate": random.uniform(0.025, 0.055),
                "avg_cost_per_click": random.uniform(4.0, 9.0),
                "avg_engagement_rate": random.uniform(0.025, 0.040),
                "avg_roas": random.uniform(4.0, 7.5)
            },
            Industry.AUTOMOTIVE: {
                "avg_ctr": random.uniform(0.022, 0.038),
                "avg_conversion_rate": random.uniform(0.020, 0.045),
                "avg_cost_per_click": random.uniform(3.5, 8.0),
                "avg_engagement_rate": random.uniform(0.020, 0.035),
                "avg_roas": random.uniform(3.0, 5.5)
            }
        }
        
        industry_benchmarks = benchmarks[industry]
        
        return MarketBenchmarks(
            industry=industry,
            avg_ctr=industry_benchmarks["avg_ctr"],
            avg_conversion_rate=industry_benchmarks["avg_conversion_rate"],
            avg_cost_per_click=industry_benchmarks["avg_cost_per_click"],
            avg_engagement_rate=industry_benchmarks["avg_engagement_rate"],
            avg_roas=industry_benchmarks["avg_roas"]
        )
    
    def save_to_csv(self, data_type: str, data: List, filename: str) -> str:
        """
        Save generated data to CSV files for persistence and analysis.
        
        Args:
            data_type: Type of data being saved (for appropriate formatting)
            data: List of data objects to save
            filename: Name of the CSV file to create
            
        Returns:
            Path to the saved CSV file
        """
        # Convert data objects to dictionaries for pandas
        if data_type == "customers":
            df_data = []
            for customer in data:
                df_data.append({
                    "company_name": customer.company_name,
                    "industry": customer.industry.value,
                    "company_size": customer.company_size,
                    "location": customer.location,
                    "annual_revenue": customer.annual_revenue,
                    "target_audience": customer.target_audience,
                    "current_ad_spend": customer.current_ad_spend,
                    "primary_goals": ", ".join(customer.primary_goals)
                })
        elif data_type == "historical":
            df_data = []
            for hist in data:
                df_data.append({
                    "customer_id": hist.customer_id,
                    "period": hist.period,
                    "impressions": hist.metrics.impressions,
                    "clicks": hist.metrics.clicks,
                    "conversions": hist.metrics.conversions,
                    "cost_per_click": hist.metrics.cost_per_click,
                    "cost_per_conversion": hist.metrics.cost_per_conversion,
                    "click_through_rate": hist.metrics.click_through_rate,
                    "conversion_rate": hist.metrics.conversion_rate,
                    "return_on_ad_spend": hist.metrics.return_on_ad_spend,
                    "reach": hist.metrics.reach,
                    "engagement_rate": hist.metrics.engagement_rate,
                    "ad_spend": hist.ad_spend,
                    "revenue_generated": hist.revenue_generated
                })
        elif data_type == "competitors":
            df_data = []
            for comp in data:
                df_data.append({
                    "competitor_name": comp.competitor_name,
                    "industry": comp.industry.value,
                    "estimated_ad_spend": comp.estimated_ad_spend,
                    "market_share": comp.market_share,
                    "avg_engagement_rate": comp.avg_engagement_rate,
                    "estimated_reach": comp.estimated_reach,
                    "primary_platforms": ", ".join(comp.primary_platforms)
                })
        
        df = pd.DataFrame(df_data)
        file_path = f"data/{filename}"
        df.to_csv(file_path, index=False)
        
        return file_path