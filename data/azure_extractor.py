"""
Azure Extractor Module
Dynamically extracts Azure metrics and cost data from a specified resource group.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from azure.identity import ClientSecretCredential
from azure.mgmt.monitor import MonitorManagementClient
from azure.mgmt.costmanagement import CostManagementClient
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.web import WebSiteManagementClient
from azure.mgmt.sql import SqlManagementClient
from azure.mgmt.storage import StorageManagementClient


class AzureExtractor:
    """Extracts Azure metrics and cost data using Azure SDK."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        subscription_id: str,
        resource_group_name: str
    ):
        """
        Initialize Azure extractor with service principal credentials.
        
        Args:
            client_id: Azure AD application (client) ID
            client_secret: Azure AD client secret
            tenant_id: Azure AD tenant ID
            subscription_id: Azure subscription ID
            resource_group_name: Resource group to analyze
        """
        self.subscription_id = subscription_id
        self.resource_group_name = resource_group_name
        
        # Create credential
        self.credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        
        # Initialize clients
        self.monitor_client = MonitorManagementClient(self.credential, subscription_id)
        self.resource_client = ResourceManagementClient(self.credential, subscription_id)
        self.web_client = WebSiteManagementClient(self.credential, subscription_id)
        self.sql_client = SqlManagementClient(self.credential, subscription_id)
        self.storage_client = StorageManagementClient(self.credential, subscription_id)
        self.cost_client = CostManagementClient(self.credential)
    
    def _get_metric_value(
        self,
        resource_id: str,
        metric_name: str,
        aggregation: str = "Average",
        timespan: Optional[timedelta] = None
    ) -> Optional[float]:
        """
        Get a specific metric value for a resource.
        
        Args:
            resource_id: Full Azure resource ID
            metric_name: Name of the metric (e.g., "CpuPercentage")
            aggregation: Aggregation type (Average, Maximum, Minimum, Total)
            timespan: Time range (defaults to last 1 hour)
            
        Returns:
            Metric value or None if not available
        """
        try:
            if timespan is None:
                timespan = timedelta(hours=1)
            
            end_time = datetime.utcnow()
            start_time = end_time - timespan
            
            metrics_data = self.monitor_client.metrics.list(
                resource_id,
                timespan=f"{start_time.isoformat()}/{end_time.isoformat()}",
                metricnames=metric_name,
                aggregation=aggregation
            )
            
            for metric in metrics_data.value:
                if metric.timeseries and len(metric.timeseries) > 0:
                    timeseries = metric.timeseries[0]
                    if timeseries.data and len(timeseries.data) > 0:
                        # Get the most recent data point
                        data_point = timeseries.data[-1]
                        if aggregation.lower() == "average" and data_point.average is not None:
                            return data_point.average
                        elif aggregation.lower() == "maximum" and data_point.maximum is not None:
                            return data_point.maximum
                        elif aggregation.lower() == "total" and data_point.total is not None:
                            return data_point.total
            
            return None
        except Exception as e:
            print(f"⚠️ Error fetching metric {metric_name}: {str(e)}")
            return None
    
    def get_app_service_metrics(self) -> Dict:
        """
        Extract App Service metrics.
        
        Returns:
            Dictionary with CPU utilization, memory utilization, instance count, autoscaling status
        """
        try:
            # List all App Service plans in the resource group
            app_service_plans = list(self.web_client.app_service_plans.list_by_resource_group(
                self.resource_group_name
            ))
            
            if not app_service_plans:
                return {
                    "cpu_utilization": "N/A",
                    "memory_utilization": "N/A",
                    "instance_count": 0,
                    "autoscaling_enabled": False
                }
            
            # Use the first App Service Plan
            plan = app_service_plans[0]
            resource_id = plan.id
            
            # Get CPU percentage
            cpu_percent = self._get_metric_value(resource_id, "CpuPercentage")
            cpu_str = f"{cpu_percent:.1f}%" if cpu_percent is not None else "N/A"
            
            # Get Memory percentage
            memory_percent = self._get_metric_value(resource_id, "MemoryPercentage")
            memory_str = f"{memory_percent:.1f}%" if memory_percent is not None else "N/A"
            
            # Get instance count
            instance_count = plan.sku.capacity if plan.sku and plan.sku.capacity else 1
            
            # Check autoscaling (simplified - checking if plan supports scaling)
            autoscaling_enabled = plan.sku and plan.sku.tier in ["Standard", "Premium", "PremiumV2", "PremiumV3"]
            
            return {
                "cpu_utilization": cpu_str,
                "memory_utilization": memory_str,
                "instance_count": instance_count,
                "autoscaling_enabled": autoscaling_enabled
            }
        except Exception as e:
            print(f"⚠️ Error fetching App Service metrics: {str(e)}")
            return {
                "cpu_utilization": "N/A",
                "memory_utilization": "N/A",
                "instance_count": 0,
                "autoscaling_enabled": False
            }
    
    def get_sql_database_metrics(self) -> Dict:
        """
        Extract SQL Database metrics.
        
        Returns:
            Dictionary with DTU utilization and cost per month
        """
        try:
            # List all SQL servers in the resource group
            sql_servers = list(self.sql_client.servers.list_by_resource_group(
                self.resource_group_name
            ))
            
            if not sql_servers:
                return {
                    "utilization": "N/A",
                    "dtus": 0,
                    "cost_per_month": "$0"
                }
            
            # Use the first SQL server and its first database
            server = sql_servers[0]
            databases = list(self.sql_client.databases.list_by_server(
                self.resource_group_name,
                server.name
            ))
            
            # Filter out 'master' database
            databases = [db for db in databases if db.name.lower() != 'master']
            
            if not databases:
                return {
                    "utilization": "N/A",
                    "dtus": 0,
                    "cost_per_month": "$0"
                }
            
            database = databases[0]
            resource_id = database.id
            
            # Get CPU utilization (works for both DTU and vCore-based databases)
            cpu_percent = self._get_metric_value(resource_id, "cpu_percent")
            utilization_str = f"{cpu_percent:.1f}%" if cpu_percent is not None else "N/A"
            
            # Extract DTU count from SKU (simplified)
            dtus = 0
            if database.sku:
                # DTU count is typically in the capacity field for DTU-based SKUs
                dtus = database.sku.capacity if database.sku.capacity else 0
            
            return {
                "utilization": utilization_str,
                "dtus": dtus,
                "cost_per_month": "$0"  # Cost will be populated separately
            }
        except Exception as e:
            print(f"⚠️ Error fetching SQL Database metrics: {str(e)}")
            return {
                "utilization": "N/A",
                "dtus": 0,
                "cost_per_month": "$0"
            }
    
    def get_storage_account_metrics(self) -> Dict:
        """
        Extract Storage Account metrics.
        
        Returns:
            Dictionary with storage usage and cost per month
        """
        try:
            # List all storage accounts in the resource group
            storage_accounts = list(self.storage_client.storage_accounts.list_by_resource_group(
                self.resource_group_name
            ))
            
            if not storage_accounts:
                return {
                    "total_storage_gb": 0,
                    "active_storage_gb": 0,
                    "cost_per_month": "$0"
                }
            
            # Use the first storage account
            storage_account = storage_accounts[0]
            resource_id = storage_account.id
            
            # Get used capacity in bytes
            used_capacity_bytes = self._get_metric_value(
                resource_id,
                "UsedCapacity",
                aggregation="Average"
            )
            
            if used_capacity_bytes is not None:
                # Convert bytes to GB
                active_storage_gb = used_capacity_bytes / (1024 ** 3)
            else:
                active_storage_gb = 0
            
            # Total storage is typically much larger; use a reasonable estimate
            # In reality, Azure storage is virtually unlimited, but we'll use a conservative estimate
            total_storage_gb = max(active_storage_gb * 10, 500)  # At least 10x active or 500GB
            
            return {
                "total_storage_gb": int(total_storage_gb),
                "active_storage_gb": int(active_storage_gb),
                "cost_per_month": "$0"  # Cost will be populated separately
            }
        except Exception as e:
            print(f"⚠️ Error fetching Storage Account metrics: {str(e)}")
            return {
                "total_storage_gb": 0,
                "active_storage_gb": 0,
                "cost_per_month": "$0"
            }
    
    def get_resource_costs(self) -> Dict:
        """
        Extract cost data for resources in the resource group.
        
        Returns:
            Dictionary with cost per month for each service
        """
        try:
            # Define the scope for cost query
            scope = f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group_name}"
            
            # Define time period (last 30 days)
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            
            # Create query definition
            query_definition = {
                "type": "ActualCost",
                "timeframe": "Custom",
                "time_period": {
                    "from": start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "to": end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                },
                "dataset": {
                    "granularity": "None",
                    "aggregation": {
                        "totalCost": {
                            "name": "Cost",
                            "function": "Sum"
                        }
                    },
                    "grouping": [
                        {
                            "type": "Dimension",
                            "name": "ResourceType"
                        }
                    ]
                }
            }
            
            # Query costs
            result = self.cost_client.query.usage(scope, query_definition)
            
            # Parse results
            costs = {
                "app_service": {"cost_per_month": "$0"},
                "sql_database": {"cost_per_month": "$0"},
                "storage_account": {"cost_per_month": "$0"}
            }
            
            if result and result.rows:
                for row in result.rows:
                    if len(row) >= 2:
                        cost = row[0]  # Total cost
                        resource_type = row[1]  # Resource type
                        
                        # Map Azure resource types to our service names
                        if "microsoft.web/serverfarms" in resource_type.lower():
                            costs["app_service"]["cost_per_month"] = f"${cost:.2f}"
                        elif "microsoft.sql/servers/databases" in resource_type.lower():
                            costs["sql_database"]["cost_per_month"] = f"${cost:.2f}"
                        elif "microsoft.storage/storageaccounts" in resource_type.lower():
                            costs["storage_account"]["cost_per_month"] = f"${cost:.2f}"
            
            return costs
        except Exception as e:
            error_msg = str(e)
            # Check if it's a known unsupported subscription error
            if "offer" in error_msg.lower() and "not supported" in error_msg.lower():
                print(f"⚠️ Cost Management API not available for this subscription type. Costs will be displayed as $0.")
            else:
                print(f"⚠️ Error fetching resource costs: {error_msg}")
            return {
                "app_service": {"cost_per_month": "$0"},
                "sql_database": {"cost_per_month": "$0"},
                "storage_account": {"cost_per_month": "$0"}
            }
    
    def extract_all_metrics(self) -> Dict:
        """
        Extract all Azure metrics and costs.
        
        Returns:
            Dictionary with metrics data (compatible with existing azure_metrics.json format)
        """
        app_service_metrics = self.get_app_service_metrics()
        sql_db_metrics = self.get_sql_database_metrics()
        storage_metrics = self.get_storage_account_metrics()
        
        return {
            "app_service": app_service_metrics,
            "sql_database": sql_db_metrics,
            "storage_account": storage_metrics
        }
    
    def extract_all_costs(self) -> Dict:
        """
        Extract all Azure costs.
        
        Returns:
            Dictionary with cost data (compatible with existing azure_cost.json format)
        """
        return self.get_resource_costs()


def create_azure_extractor(
    client_id: str,
    client_secret: str,
    tenant_id: str,
    subscription_id: str,
    resource_group_name: str
) -> AzureExtractor:
    """
    Factory function to create an Azure extractor instance.
    
    Args:
        client_id: Azure AD application (client) ID
        client_secret: Azure AD client secret
        tenant_id: Azure AD tenant ID
        subscription_id: Azure subscription ID
        resource_group_name: Resource group to analyze
        
    Returns:
        AzureExtractor instance
    """
    return AzureExtractor(
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id,
        subscription_id=subscription_id,
        resource_group_name=resource_group_name
    )
