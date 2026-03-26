"""
Utility module for extracting and formatting Azure metrics data.
Loads metrics from JSON files and provides structured data for frontend consumption.
"""

import json
import os
from typing import Dict, Optional


class MetricsExtractor:
    """Extracts and formats Azure metrics and cost data."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize metrics extractor.
        
        Args:
            data_dir: Directory containing azure_metrics.json and azure_cost.json
        """
        self.data_dir = data_dir
        self.metrics_file = os.path.join(data_dir, "azure_metrics.json")
        self.cost_file = os.path.join(data_dir, "azure_cost.json")
    
    def load_metrics(self) -> Dict:
        """
        Load Azure metrics from JSON file.
        
        Returns:
            Dictionary containing Azure metrics or empty dict if file not found
        """
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading metrics file: {str(e)}")
        
        return {}
    
    def load_costs(self) -> Dict:
        """
        Load Azure cost data from JSON file.
        
        Returns:
            Dictionary containing Azure costs or empty dict if file not found
        """
        try:
            if os.path.exists(self.cost_file):
                with open(self.cost_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading cost file: {str(e)}")
        
        return {}
    
    def get_app_service_metrics(self) -> Dict:
        """Extract App Service metrics."""
        metrics = self.load_metrics()
        app_service = metrics.get("app_service", {})
        
        return {
            "service": "App Service",
            "cpu_utilization": app_service.get("cpu_utilization", "N/A"),
            "memory_utilization": app_service.get("memory_utilization", "N/A"),
            "instance_count": app_service.get("instance_count", 0),
            "autoscaling_enabled": app_service.get("autoscaling_enabled", False)
        }
    
    def get_sql_db_metrics(self) -> Dict:
        """Extract SQL Database metrics."""
        metrics = self.load_metrics()
        sql_db = metrics.get("sql_database", {})
        costs = self.load_costs()
        sql_cost = costs.get("sql_database", {})
        
        return {
            "service": "SQL Database",
            "utilization": sql_db.get("utilization", "N/A"),
            "dtus": sql_db.get("dtus", 0),
            "cost_per_month": sql_cost.get("cost_per_month", sql_db.get("cost_per_month", "N/A"))
        }
    
    def get_storage_metrics(self) -> Dict:
        """Extract Storage Account metrics."""
        metrics = self.load_metrics()
        storage = metrics.get("storage_account", {})
        costs = self.load_costs()
        storage_cost = costs.get("storage_account", {})
        
        total_gb = storage.get("total_storage_gb", 0)
        active_gb = storage.get("active_storage_gb", 0)
        utilization_percent = (active_gb / total_gb * 100) if total_gb > 0 else 0
        
        return {
            "service": "Storage Account",
            "total_storage_gb": total_gb,
            "active_storage_gb": active_gb,
            "utilization_percent": f"{utilization_percent:.2f}%",
            "unused_storage_gb": total_gb - active_gb,
            "cost_per_month": storage_cost.get("cost_per_month", storage.get("cost_per_month", "N/A"))
        }
    
    def get_total_monthly_cost(self) -> Dict:
        """Calculate total monthly cost across all services."""
        costs = self.load_costs()
        
        total = 0
        services_cost = {}
        
        # Extract costs from all services
        for service, data in costs.items():
            cost_str = data.get("cost_per_month", "$0")
            # Parse cost (remove $ and convert to float)
            try:
                cost_value = float(cost_str.replace("$", "").replace(",", ""))
                services_cost[service] = cost_value
                total += cost_value
            except (ValueError, AttributeError):
                services_cost[service] = 0
        
        return {
            "total_monthly_cost": f"${total:,.2f}",
            "total_monthly_cost_value": total,
            "services_breakdown": {
                "app_service": f"${services_cost.get('app_service', 0):,.2f}",
                "sql_database": f"${services_cost.get('sql_database', 0):,.2f}",
                "storage_account": f"${services_cost.get('storage_account', 0):,.2f}"
            }
        }
    
    def get_all_metrics(self) -> Dict:
        """
        Get all metrics combined into a single comprehensive object.
        
        Returns:
            Dictionary containing all Azure metrics and costs
        """
        return {
            "app_service": self.get_app_service_metrics(),
            "sql_database": self.get_sql_db_metrics(),
            "storage": self.get_storage_metrics(),
            "costs": self.get_total_monthly_cost(),
            "timestamp": self._get_timestamp()
        }
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


# Singleton instance
_metrics_extractor = None


def get_metrics_extractor(data_dir: str = "data") -> MetricsExtractor:
    """Get or create the metrics extractor instance."""
    global _metrics_extractor
    if _metrics_extractor is None:
        _metrics_extractor = MetricsExtractor(data_dir)
    return _metrics_extractor
