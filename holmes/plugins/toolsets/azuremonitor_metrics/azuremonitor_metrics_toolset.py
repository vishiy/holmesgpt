"""Azure Monitor Metrics toolset for HolmesGPT."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from pydantic import BaseModel, field_validator
from requests import RequestException

from holmes.core.tools import (
    CallablePrerequisite,
    StructuredToolResult,
    Tool,
    ToolParameter,
    ToolResultStatus,
    Toolset,
    ToolsetTag,
)
from holmes.plugins.toolsets.consts import STANDARD_END_DATETIME_TOOL_PARAM_DESCRIPTION
from holmes.plugins.toolsets.utils import (
    get_param_or_raise,
    process_timestamps_to_rfc3339,
    standard_start_datetime_tool_param_description,
)
try:
    from holmes.utils.cache import TTLCache
except ImportError:
    # Fallback for older versions or if cache is not available
    class TTLCache:
        def __init__(self, *args, **kwargs):
            pass

from .utils import (
    check_if_running_in_aks,
    extract_cluster_name_from_resource_id,
    get_aks_cluster_resource_id,
    get_azure_monitor_workspace_for_cluster,
    enhance_promql_with_cluster_filter,
)

DEFAULT_TIME_SPAN_SECONDS = 3600

class AzureMonitorMetricsConfig(BaseModel):
    """Configuration for Azure Monitor Metrics toolset."""
    azure_monitor_workspace_endpoint: Optional[str] = None
    cluster_name: Optional[str] = None
    cluster_resource_id: Optional[str] = None
    auto_detect_cluster: bool = True
    cache_duration_seconds: Optional[int] = 1800
    tool_calls_return_data: bool = True
    headers: Dict = {}

    @field_validator("azure_monitor_workspace_endpoint")
    def ensure_trailing_slash(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.endswith("/"):
            return v + "/"
        return v

class BaseAzureMonitorMetricsTool(Tool):
    """Base class for Azure Monitor Metrics tools."""
    toolset: "AzureMonitorMetricsToolset"

class CheckAKSClusterContext(BaseAzureMonitorMetricsTool):
    """Tool to check if running in AKS cluster context."""
    
    def __init__(self, toolset: "AzureMonitorMetricsToolset"):
        super().__init__(
            name="check_aks_cluster_context",
            description="Check if the current environment is running inside an AKS cluster",
            parameters={},
            toolset=toolset,
        )

    def _invoke(self, params: Any) -> StructuredToolResult:
        try:
            is_aks = check_if_running_in_aks()
            
            data = {
                "running_in_aks": is_aks,
                "message": "Running in AKS cluster" if is_aks else "Not running in AKS cluster",
            }
            
            return StructuredToolResult(
                status=ToolResultStatus.SUCCESS,
                data=data,
                params=params,
            )
            
        except Exception as e:
            return StructuredToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Failed to check AKS cluster context: {str(e)}",
                params=params,
            )

    def get_parameterized_one_liner(self, params) -> str:
        return "Check if running in AKS cluster"

class GetAKSClusterResourceID(BaseAzureMonitorMetricsTool):
    """Tool to get the Azure resource ID of the current AKS cluster."""
    
    def __init__(self, toolset: "AzureMonitorMetricsToolset"):
        super().__init__(
            name="get_aks_cluster_resource_id",
            description="Get the full Azure resource ID of the current AKS cluster",
            parameters={},
            toolset=toolset,
        )

    def _invoke(self, params: Any) -> StructuredToolResult:
        try:
            cluster_resource_id = get_aks_cluster_resource_id()
            
            if cluster_resource_id:
                cluster_name = extract_cluster_name_from_resource_id(cluster_resource_id)
                
                data = {
                    "cluster_resource_id": cluster_resource_id,
                    "cluster_name": cluster_name,
                    "message": f"Found AKS cluster: {cluster_name}",
                }
                
                return StructuredToolResult(
                    status=ToolResultStatus.SUCCESS,
                    data=data,
                    params=params,
                )
            else:
                return StructuredToolResult(
                    status=ToolResultStatus.ERROR,
                    error="Could not determine AKS cluster resource ID. Make sure you are running in an AKS cluster or have proper Azure credentials configured.",
                    params=params,
                )
                
        except Exception as e:
            return StructuredToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Failed to get AKS cluster resource ID: {str(e)}",
                params=params,
            )

    def get_parameterized_one_liner(self, params) -> str:
        return "Get AKS cluster Azure resource ID"

class CheckAzureMonitorPrometheusEnabled(BaseAzureMonitorMetricsTool):
    """Tool to check if Azure Monitor managed Prometheus is enabled for the AKS cluster."""
    
    def __init__(self, toolset: "AzureMonitorMetricsToolset"):
        super().__init__(
            name="check_azure_monitor_prometheus_enabled",
            description="Check if Azure Monitor managed Prometheus is enabled for the current AKS cluster",
            parameters={
                "cluster_resource_id": ToolParameter(
                    description="Azure resource ID of the AKS cluster (optional, will auto-detect if not provided)",
                    type="string",
                    required=False,
                ),
            },
            toolset=toolset,
        )

    def _invoke(self, params: Any) -> StructuredToolResult:
        try:
            cluster_resource_id = params.get("cluster_resource_id")
            
            # Auto-detect cluster resource ID if not provided
            if not cluster_resource_id:
                cluster_resource_id = get_aks_cluster_resource_id()
                
            if not cluster_resource_id:
                return StructuredToolResult(
                    status=ToolResultStatus.ERROR,
                    error="Could not determine AKS cluster resource ID. Please provide cluster_resource_id parameter or ensure you are running in an AKS cluster.",
                    params=params,
                )
            
            # Get Azure Monitor workspace details
            workspace_info = get_azure_monitor_workspace_for_cluster(cluster_resource_id)
            
            if workspace_info:
                cluster_name = extract_cluster_name_from_resource_id(cluster_resource_id)
                
                data = {
                    "azure_monitor_prometheus_enabled": True,
                    "cluster_resource_id": cluster_resource_id,
                    "cluster_name": cluster_name,
                    "prometheus_query_endpoint": workspace_info.get("prometheus_query_endpoint"),
                    "azure_monitor_workspace_resource_id": workspace_info.get("azure_monitor_workspace_resource_id"),
                    "location": workspace_info.get("location"),
                    "associated_grafanas": workspace_info.get("associated_grafanas", []),
                    "message": f"Azure Monitor managed Prometheus is enabled for cluster {cluster_name}",
                }
                
                # Update toolset configuration with discovered information
                if self.toolset.config:
                    self.toolset.config.azure_monitor_workspace_endpoint = workspace_info.get("prometheus_query_endpoint")
                    self.toolset.config.cluster_name = cluster_name
                    self.toolset.config.cluster_resource_id = cluster_resource_id
                
                return StructuredToolResult(
                    status=ToolResultStatus.SUCCESS,
                    data=data,
                    params=params,
                )
            else:
                cluster_name = extract_cluster_name_from_resource_id(cluster_resource_id)
                return StructuredToolResult(
                    status=ToolResultStatus.ERROR,
                    error=f"Azure Monitor managed Prometheus is not enabled for AKS cluster {cluster_name}. Please enable Azure Monitor managed Prometheus in the Azure portal.",
                    params=params,
                )
                
        except Exception as e:
            return StructuredToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Failed to check Azure Monitor Prometheus status: {str(e)}",
                params=params,
            )

    def get_parameterized_one_liner(self, params) -> str:
        cluster_id = params.get("cluster_resource_id", "auto-detect")
        return f"Check Azure Monitor Prometheus status for cluster: {cluster_id}"

class ExecuteAzureMonitorPrometheusQuery(BaseAzureMonitorMetricsTool):
    """Tool to execute instant PromQL queries against Azure Monitor workspace."""
    
    def __init__(self, toolset: "AzureMonitorMetricsToolset"):
        super().__init__(
            name="execute_azuremonitor_prometheus_query",
            description="Execute an instant PromQL query against Azure Monitor managed Prometheus workspace",
            parameters={
                "query": ToolParameter(
                    description="The PromQL query to execute",
                    type="string",
                    required=True,
                ),
                "description": ToolParameter(
                    description="Description of what the query is meant to find or analyze",
                    type="string",
                    required=True,
                ),
                "auto_cluster_filter": ToolParameter(
                    description="Automatically add cluster filtering to the query (default: true)",
                    type="boolean",
                    required=False,
                ),
            },
            toolset=toolset,
        )

    def _invoke(self, params: Any) -> StructuredToolResult:
        if not self.toolset.config or not self.toolset.config.azure_monitor_workspace_endpoint:
            return StructuredToolResult(
                status=ToolResultStatus.ERROR,
                error="Azure Monitor workspace is not configured. Run check_azure_monitor_prometheus_enabled first.",
                params=params,
            )
            
        try:
            query = params.get("query", "")
            description = params.get("description", "")
            auto_cluster_filter = params.get("auto_cluster_filter", True)
            
            # Enhance query with cluster filtering if enabled and cluster name is available
            if auto_cluster_filter and self.toolset.config.cluster_name:
                query = enhance_promql_with_cluster_filter(query, self.toolset.config.cluster_name)
            
            url = urljoin(self.toolset.config.azure_monitor_workspace_endpoint, "api/v1/query")
            
            payload = {"query": query}
            
            response = requests.post(
                url=url,
                headers=self.toolset.config.headers,
                data=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                error_message = None
                
                if status == "success":
                    result_data = data.get("data", {})
                    if not result_data.get("result"):
                        status = "no_data"
                        error_message = "The query returned no results. The metric may not exist or the cluster filter may be too restrictive."
                else:
                    error_message = data.get("error", "Unknown error from Prometheus endpoint")
                
                response_data = {
                    "status": status,
                    "error_message": error_message,
                    "tool_name": self.name,
                    "description": description,
                    "query": query,
                    "cluster_name": self.toolset.config.cluster_name,
                    "auto_cluster_filter_applied": auto_cluster_filter and bool(self.toolset.config.cluster_name),
                }
                
                if self.toolset.config.tool_calls_return_data:
                    response_data["data"] = data.get("data")
                
                result_status = ToolResultStatus.SUCCESS
                if status == "no_data":
                    result_status = ToolResultStatus.NO_DATA
                elif status != "success":
                    result_status = ToolResultStatus.ERROR
                
                return StructuredToolResult(
                    status=result_status,
                    data=json.dumps(response_data, indent=2),
                    params=params,
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return StructuredToolResult(
                    status=ToolResultStatus.ERROR,
                    error=f"Azure Monitor Prometheus query failed: {error_msg}",
                    params=params,
                )
                
        except RequestException as e:
            return StructuredToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Connection error to Azure Monitor workspace: {str(e)}",
                params=params,
            )
        except Exception as e:
            return StructuredToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Unexpected error executing query: {str(e)}",
                params=params,
            )

    def get_parameterized_one_liner(self, params) -> str:
        query = params.get("query", "")
        description = params.get("description", "")
        return f"Execute Azure Monitor Prometheus Query (instant): promql='{query}', description='{description}'"

class ExecuteAzureMonitorPrometheusRangeQuery(BaseAzureMonitorMetricsTool):
    """Tool to execute range PromQL queries against Azure Monitor workspace."""
    
    def __init__(self, toolset: "AzureMonitorMetricsToolset"):
        super().__init__(
            name="execute_azuremonitor_prometheus_range_query",
            description="Execute a PromQL range query against Azure Monitor managed Prometheus workspace",
            parameters={
                "query": ToolParameter(
                    description="The PromQL query to execute",
                    type="string",
                    required=True,
                ),
                "description": ToolParameter(
                    description="Description of what the query is meant to find or analyze",
                    type="string",
                    required=True,
                ),
                "start": ToolParameter(
                    description=standard_start_datetime_tool_param_description(DEFAULT_TIME_SPAN_SECONDS),
                    type="string",
                    required=False,
                ),
                "end": ToolParameter(
                    description=STANDARD_END_DATETIME_TOOL_PARAM_DESCRIPTION,
                    type="string",
                    required=False,
                ),
                "step": ToolParameter(
                    description="Query resolution step width in duration format or float number of seconds",
                    type="number",
                    required=True,
                ),
                "output_type": ToolParameter(
                    description="Specifies how to interpret the Prometheus result. Use 'Plain' for raw values, 'Bytes' to format byte values, 'Percentage' to scale 0–1 values into 0–100%, or 'CPUUsage' to convert values to cores (e.g., 500 becomes 500m, 2000 becomes 2).",
                    type="string",
                    required=True,
                ),
                "auto_cluster_filter": ToolParameter(
                    description="Automatically add cluster filtering to the query (default: true)",
                    type="boolean",
                    required=False,
                ),
            },
            toolset=toolset,
        )

    def _invoke(self, params: Any) -> StructuredToolResult:
        if not self.toolset.config or not self.toolset.config.azure_monitor_workspace_endpoint:
            return StructuredToolResult(
                status=ToolResultStatus.ERROR,
                error="Azure Monitor workspace is not configured. Run check_azure_monitor_prometheus_enabled first.",
                params=params,
            )
            
        try:
            query = get_param_or_raise(params, "query")
            description = params.get("description", "")
            auto_cluster_filter = params.get("auto_cluster_filter", True)
            
            # Enhance query with cluster filtering if enabled and cluster name is available
            if auto_cluster_filter and self.toolset.config.cluster_name:
                query = enhance_promql_with_cluster_filter(query, self.toolset.config.cluster_name)
            
            (start, end) = process_timestamps_to_rfc3339(
                start_timestamp=params.get("start"),
                end_timestamp=params.get("end"),
                default_time_span_seconds=DEFAULT_TIME_SPAN_SECONDS,
            )
            step = params.get("step", "")
            output_type = params.get("output_type", "Plain")
            
            url = urljoin(self.toolset.config.azure_monitor_workspace_endpoint, "api/v1/query_range")
            
            payload = {
                "query": query,
                "start": start,
                "end": end,
                "step": step,
            }
            
            response = requests.post(
                url=url,
                headers=self.toolset.config.headers,
                data=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                error_message = None
                
                if status == "success":
                    result_data = data.get("data", {})
                    if not result_data.get("result"):
                        status = "no_data"
                        error_message = "The query returned no results. The metric may not exist or the cluster filter may be too restrictive."
                else:
                    error_message = data.get("error", "Unknown error from Prometheus endpoint")
                
                response_data = {
                    "status": status,
                    "error_message": error_message,
                    "tool_name": self.name,
                    "description": description,
                    "query": query,
                    "start": start,
                    "end": end,
                    "step": step,
                    "output_type": output_type,
                    "cluster_name": self.toolset.config.cluster_name,
                    "auto_cluster_filter_applied": auto_cluster_filter and bool(self.toolset.config.cluster_name),
                }
                
                if self.toolset.config.tool_calls_return_data:
                    response_data["data"] = data.get("data")
                
                result_status = ToolResultStatus.SUCCESS
                if status == "no_data":
                    result_status = ToolResultStatus.NO_DATA
                elif status != "success":
                    result_status = ToolResultStatus.ERROR
                
                return StructuredToolResult(
                    status=result_status,
                    data=json.dumps(response_data, indent=2),
                    params=params,
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return StructuredToolResult(
                    status=ToolResultStatus.ERROR,
                    error=f"Azure Monitor Prometheus range query failed: {error_msg}",
                    params=params,
                )
                
        except RequestException as e:
            return StructuredToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Connection error to Azure Monitor workspace: {str(e)}",
                params=params,
            )
        except Exception as e:
            return StructuredToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Unexpected error executing range query: {str(e)}",
                params=params,
            )

    def get_parameterized_one_liner(self, params) -> str:
        query = params.get("query", "")
        start = params.get("start", "")
        end = params.get("end", "")
        step = params.get("step", "")
        description = params.get("description", "")
        return f"Execute Azure Monitor Prometheus Range Query: promql='{query}', start={start}, end={end}, step={step}, description='{description}'"

class AzureMonitorMetricsToolset(Toolset):
    """Azure Monitor Metrics toolset for querying Azure Monitor managed Prometheus metrics."""
    
    def __init__(self):
        super().__init__(
            name="azuremonitormetrics",
            description="Azure Monitor Metrics integration to query Azure Monitor managed Prometheus metrics for AKS cluster analysis and troubleshooting",
            docs_url="https://docs.robusta.dev/master/configuration/holmesgpt/toolsets/azuremonitor-metrics.html",
            icon_url="https://raw.githubusercontent.com/robusta-dev/holmesgpt/master/images/integration_logos/azure-managed-prometheus.png",
            prerequisites=[CallablePrerequisite(callable=self.prerequisites_callable)],
            tools=[
                CheckAKSClusterContext(toolset=self),
                GetAKSClusterResourceID(toolset=self),
                CheckAzureMonitorPrometheusEnabled(toolset=self),
                ExecuteAzureMonitorPrometheusQuery(toolset=self),
                ExecuteAzureMonitorPrometheusRangeQuery(toolset=self),
            ],
            tags=[
                ToolsetTag.CORE
            ],
            is_default=True,  # Enable by default like internet toolset
        )
        self._cache = None
        self._reload_llm_instructions()

    def _reload_llm_instructions(self):
        """Load LLM instructions from Jinja template."""
        try:
            template_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "azuremonitor_metrics_instructions.jinja2")
            )
            self._load_llm_instructions(jinja_template=f"file://{template_file_path}")
        except Exception as e:
            # Ignore any errors in loading instructions
            logging.debug(f"Failed to load LLM instructions: {e}")

    def prerequisites_callable(self, config: dict[str, Any]) -> Tuple[bool, str]:
        """Check prerequisites for the Azure Monitor Metrics toolset."""
        try:
            if not config:
                self.config = AzureMonitorMetricsConfig()
            else:
                self.config = AzureMonitorMetricsConfig(**config)
            
            return True, ""
            
        except Exception as e:
            logging.debug(f"Azure Monitor toolset config initialization failed: {str(e)}")
            self.config = AzureMonitorMetricsConfig()
            return True, ""

    def get_example_config(self) -> Dict[str, Any]:
        """Return example configuration for the toolset."""
        example_config = AzureMonitorMetricsConfig(
            azure_monitor_workspace_endpoint="https://your-workspace.prometheus.monitor.azure.com",
            cluster_name="your-aks-cluster-name",
            auto_detect_cluster=True,
            cache_duration_seconds=1800,
            tool_calls_return_data=True
        )
        return example_config.model_dump()
