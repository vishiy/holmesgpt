"""Utility functions for Azure Monitor Metrics toolset."""

import json
import logging
import re
from typing import Dict, Optional, Tuple

from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
import requests


def get_aks_cluster_resource_id() -> Optional[str]:
    """
    Get the Azure resource ID of the current AKS cluster.
    
    Returns:
        str: The full Azure resource ID of the AKS cluster if found, None otherwise
    """
    try:
        # Try to get cluster info from Azure Instance Metadata Service
        metadata_url = "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
        headers = {"Metadata": "true"}
        
        response = requests.get(metadata_url, headers=headers, timeout=5)
        if response.status_code == 200:
            metadata = response.json()
            compute = metadata.get("compute", {})
            
            # Extract subscription ID and resource group from metadata
            subscription_id = compute.get("subscriptionId")
            resource_group = compute.get("resourceGroupName")
            
            if subscription_id and resource_group:
                # Try to find AKS cluster in the resource group
                credential = DefaultAzureCredential()
                resource_client = ResourceManagementClient(credential, subscription_id)
                
                # Look for AKS clusters in the resource group
                resources = resource_client.resources.list_by_resource_group(
                    resource_group_name=resource_group,
                    filter="resourceType eq 'Microsoft.ContainerService/managedClusters'"
                )
                
                for resource in resources:
                    # Return the first AKS cluster found
                    return resource.id
                    
    except Exception as e:
        logging.debug(f"Failed to get AKS cluster resource ID from metadata: {e}")
    
    try:
        # Fallback: Try to get cluster info from Kubernetes environment
        # Check if we're running in a Kubernetes pod with service account
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r") as f:
            namespace = f.read().strip()
            
        # This is a best effort - we're in Kubernetes but need to determine the cluster
        # We'll need to use Azure Resource Graph to find clusters
        logging.debug("Running in Kubernetes, attempting to find AKS cluster via Azure Resource Graph")
        
        # Use Azure Resource Graph to find AKS clusters
        credential = DefaultAzureCredential()
        
        # Get all subscriptions the credential has access to
        subscriptions = get_accessible_subscriptions(credential)
        
        for subscription_id in subscriptions:
            try:
                resource_client = ResourceManagementClient(credential, subscription_id)
                resources = resource_client.resources.list(
                    filter="resourceType eq 'Microsoft.ContainerService/managedClusters'"
                )
                
                for resource in resources:
                    # Return the first AKS cluster found
                    # In a real scenario, we might need better logic to identify the correct cluster
                    return resource.id
                    
            except Exception as e:
                logging.debug(f"Failed to query subscription {subscription_id}: {e}")
                continue
                
    except Exception as e:
        logging.debug(f"Failed to get AKS cluster resource ID from Kubernetes: {e}")
    
    return None


def get_accessible_subscriptions(credential) -> list[str]:
    """
    Get list of subscription IDs that the credential has access to.
    
    Args:
        credential: Azure credential object
        
    Returns:
        list[str]: List of subscription IDs
    """
    try:
        # This is a simplified approach - in practice you might want to use
        # the Azure Management SDK to get subscriptions
        from azure.mgmt.resource import ResourceManagementClient
        
        # For now, we'll try to get the default subscription
        # This would need to be enhanced for multi-subscription scenarios
        import os
        subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
        if subscription_id:
            return [subscription_id]
            
        # If no explicit subscription, try to get from Azure CLI config
        try:
            import subprocess
            result = subprocess.run(
                ["az", "account", "show", "--query", "id", "-o", "tsv"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return [result.stdout.strip()]
        except Exception:
            pass
            
    except Exception as e:
        logging.debug(f"Failed to get accessible subscriptions: {e}")
    
    return []


def extract_cluster_name_from_resource_id(resource_id: str) -> Optional[str]:
    """
    Extract the cluster name from an Azure resource ID.
    
    Args:
        resource_id: Full Azure resource ID
        
    Returns:
        str: Cluster name if extracted successfully, None otherwise
    """
    try:
        # Azure resource ID format:
        # /subscriptions/{subscription-id}/resourceGroups/{resource-group}/providers/Microsoft.ContainerService/managedClusters/{cluster-name}
        parts = resource_id.split("/")
        if len(parts) >= 9 and parts[-2] == "managedClusters":
            return parts[-1]
    except Exception as e:
        logging.debug(f"Failed to extract cluster name from resource ID {resource_id}: {e}")
    
    return None


def check_if_running_in_aks() -> bool:
    """
    Check if the current environment is running inside an AKS cluster.
    
    Returns:
        bool: True if running in AKS, False otherwise
    """
    try:
        # Check for Kubernetes service account
        if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
            # Check if we can access Azure Instance Metadata Service
            metadata_url = "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
            headers = {"Metadata": "true"}
            
            response = requests.get(metadata_url, headers=headers, timeout=5)
            if response.status_code == 200:
                metadata = response.json()
                # Check if we're running on Azure (which combined with Kubernetes suggests AKS)
                if metadata.get("compute", {}).get("provider") == "Microsoft.Compute":
                    return True
    except Exception as e:
        logging.debug(f"Failed to check if running in AKS: {e}")
    
    return False


def execute_azure_resource_graph_query(query: str, subscription_id: str) -> Optional[Dict]:
    """
    Execute an Azure Resource Graph query.
    
    Args:
        query: The Azure Resource Graph query to execute
        subscription_id: The subscription ID to query
        
    Returns:
        dict: Query results if successful, None otherwise
    """
    try:
        from azure.mgmt.resourcegraph import ResourceGraphClient
        from azure.mgmt.resourcegraph.models import QueryRequest
        
        credential = DefaultAzureCredential()
        
        # Create Resource Graph client
        graph_client = ResourceGraphClient(credential)
        
        # Create the query request
        query_request = QueryRequest(
            query=query,
            subscriptions=[subscription_id]
        )
        
        # Execute the query
        query_response = graph_client.resources(query_request)
        
        if query_response and hasattr(query_response, 'data'):
            return {
                "data": query_response.data,
                "total_records": getattr(query_response, 'total_records', 0),
                "count": getattr(query_response, 'count', 0)
            }
            
    except ImportError:
        logging.warning("azure-mgmt-resourcegraph package not available. Install it with: pip install azure-mgmt-resourcegraph")
        return None
    except AzureError as e:
        logging.error(f"Azure error executing Resource Graph query: {e}")
    except Exception as e:
        logging.error(f"Unexpected error executing Resource Graph query: {e}")
    
    return None


def get_azure_monitor_workspace_for_cluster(cluster_resource_id: str) -> Optional[Dict]:
    """
    Get Azure Monitor workspace details for a given AKS cluster using Azure Resource Graph.
    
    Args:
        cluster_resource_id: Full Azure resource ID of the AKS cluster
        
    Returns:
        dict: Azure Monitor workspace details if found, None otherwise
    """
    try:
        # Extract subscription ID from cluster resource ID
        parts = cluster_resource_id.split("/")
        if len(parts) >= 3:
            subscription_id = parts[2]
        else:
            logging.error(f"Invalid cluster resource ID format: {cluster_resource_id}")
            return None
        
        # The ARG query from the requirements, parameterized
        query = f"""
        resources 
        | where type == "microsoft.insights/datacollectionrules"
        | extend ma = properties.destinations.monitoringAccounts
        | extend flows = properties.dataFlows
        | mv-expand flows
        | where flows.streams contains "Microsoft-PrometheusMetrics"
        | mv-expand ma
        | where array_index_of(flows.destinations, tostring(ma.name)) != -1
        | project dcrId = tolower(id), azureMonitorWorkspaceResourceId=tolower(tostring(ma.accountResourceId))
        | join (insightsresources | extend clusterId = split(tolower(id), '/providers/microsoft.insights/datacollectionruleassociations/')[0] | where clusterId =~ "{cluster_resource_id.lower()}" | project clusterId = tostring(clusterId), dcrId = tolower(tostring(parse_json(properties).dataCollectionRuleId)), dcraName = name) on dcrId
        | join kind=leftouter (resources | where type == "microsoft.monitor/accounts" | extend prometheusQueryEndpoint=tostring(properties.metrics.prometheusQueryEndpoint) | extend amwLocation = location | project azureMonitorWorkspaceResourceId=tolower(id), prometheusQueryEndpoint, amwLocation) on azureMonitorWorkspaceResourceId
        | project-away dcrId1, azureMonitorWorkspaceResourceId1
        | join kind=leftouter (resources | where type == "microsoft.dashboard/grafana" | extend amwIntegrations = properties.grafanaIntegrations.azureMonitorWorkspaceIntegrations | mv-expand amwIntegrations | extend azureMonitorWorkspaceResourceId = tolower(tostring(amwIntegrations.azureMonitorWorkspaceResourceId)) | where azureMonitorWorkspaceResourceId != "" | extend grafanaObject = pack("grafanaResourceId", tolower(id), "grafanaWorkspaceName", name, "grafanaEndpoint", properties.endpoint) | summarize associatedGrafanas=make_list(grafanaObject) by azureMonitorWorkspaceResourceId) on azureMonitorWorkspaceResourceId
        | extend amwToGrafana = pack("azureMonitorWorkspaceResourceId", azureMonitorWorkspaceResourceId, "prometheusQueryEndpoint", prometheusQueryEndpoint, "amwLocation", amwLocation, "associatedGrafanas", associatedGrafanas)
        | summarize amwToGrafanas=make_list(amwToGrafana) by dcrResourceId = dcrId, dcraName
        | order by dcrResourceId
        """
        
        result = execute_azure_resource_graph_query(query, subscription_id)
        
        if result and result.get("data"):
            data = result["data"]
            if isinstance(data, list) and len(data) > 0:
                # Take the first result
                first_result = data[0]
                amw_to_grafanas = first_result.get("amwToGrafanas", [])
                
                if amw_to_grafanas and len(amw_to_grafanas) > 0:
                    # Take the first Azure Monitor workspace
                    amw_info = amw_to_grafanas[0]
                    
                    prometheus_endpoint = amw_info.get("prometheusQueryEndpoint")
                    if prometheus_endpoint:
                        return {
                            "prometheus_query_endpoint": prometheus_endpoint,
                            "azure_monitor_workspace_resource_id": amw_info.get("azureMonitorWorkspaceResourceId"),
                            "location": amw_info.get("amwLocation"),
                            "associated_grafanas": amw_info.get("associatedGrafanas", [])
                        }
        
        logging.info(f"No Azure Monitor workspace found for cluster {cluster_resource_id}")
        return None
        
    except Exception as e:
        logging.error(f"Failed to get Azure Monitor workspace for cluster {cluster_resource_id}: {e}")
        return None


def enhance_promql_with_cluster_filter(promql_query: str, cluster_name: str) -> str:
    """
    Enhance a PromQL query to include cluster filtering.
    
    Args:
        promql_query: Original PromQL query
        cluster_name: Name of the cluster to filter by
        
    Returns:
        str: Enhanced PromQL query with cluster filtering
    """
    try:
        # Simple approach: add cluster label filter to metric selectors
        # This is a basic implementation - a more sophisticated parser might be needed for complex queries
        
        # Find metric selectors (text before { or at the start)
        # Pattern to match metric names and their label selectors
        pattern = r'([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(\{[^}]*\})?'
        
        def add_cluster_filter(match):
            metric_name = match.group(1)
            existing_labels = match.group(2) or "{}"
            
            # If no existing labels, add cluster filter
            if existing_labels == "{}":
                return f'{metric_name}{{cluster="{cluster_name}"}}'
            else:
                # If existing labels, add cluster filter to them
                labels_content = existing_labels[1:-1]  # Remove { and }
                if labels_content.strip():
                    return f'{metric_name}{{cluster="{cluster_name}",{labels_content}}}'
                else:
                    return f'{metric_name}{{cluster="{cluster_name}"}}'
        
        enhanced_query = re.sub(pattern, add_cluster_filter, promql_query)
        
        logging.debug(f"Enhanced PromQL query: {promql_query} -> {enhanced_query}")
        return enhanced_query
        
    except Exception as e:
        logging.warning(f"Failed to enhance PromQL query with cluster filter: {e}")
        # Return original query if enhancement fails
        return promql_query


import os
