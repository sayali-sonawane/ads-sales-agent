from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class Industry(Enum):
    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare"
    FINANCE = "Finance"
    RETAIL = "Retail"
    MANUFACTURING = "Manufacturing"
    EDUCATION = "Education"
    REAL_ESTATE = "Real Estate"
    AUTOMOTIVE = "Automotive"


@dataclass
class CustomerProfile:
    company_name: str
    industry: Industry
    company_size: str  # "Small", "Medium", "Large", "Enterprise"
    location: str
    annual_revenue: float
    target_audience: str
    current_ad_spend: float
    primary_goals: List[str]
    
    
@dataclass
class PerformanceMetrics:
    impressions: int
    clicks: int
    conversions: int
    cost_per_click: float
    cost_per_conversion: float
    click_through_rate: float
    conversion_rate: float
    return_on_ad_spend: float
    reach: int
    engagement_rate: float


@dataclass
class HistoricalPerformance:
    customer_id: str
    period: str  # "Q1 2024", "Q2 2024", etc.
    metrics: PerformanceMetrics
    ad_spend: float
    revenue_generated: float


@dataclass
class CompetitorData:
    competitor_name: str
    industry: Industry
    estimated_ad_spend: float
    market_share: float
    avg_engagement_rate: float
    estimated_reach: int
    primary_platforms: List[str]


@dataclass
class MarketBenchmarks:
    industry: Industry
    avg_ctr: float
    avg_conversion_rate: float
    avg_cost_per_click: float
    avg_engagement_rate: float
    avg_roas: float


@dataclass
class Recommendation:
    title: str
    description: str
    expected_impact: str
    priority: str  # "High", "Medium", "Low"
    estimated_roi: float
    implementation_timeline: str


@dataclass
class SalesPitch:
    customer_profile: CustomerProfile
    historical_performance: List[HistoricalPerformance]
    competitor_analysis: List[CompetitorData]
    market_benchmarks: MarketBenchmarks
    recommendations: List[Recommendation]
    presentation_slides: List[str]  # File paths to generated slides