from flask import Flask, request, jsonify
import requests
import json
import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

# Strands Framework imports
from strands import Agent
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
import boto3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
TEAMS_WEBHOOK_URL = os.getenv('TEAMS_WEBHOOK_URL', 'YOUR_TEAMS_WEBHOOK_URL_HERE')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
BEDROCK_MODEL = os.getenv('BEDROCK_MODEL', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
EKS_CLUSTER_NAME = os.getenv('EKS_CLUSTER_NAME', 'default-cluster')

class EKSDeploymentTroubleshooter:
    """Strands-based EKS deployment troubleshooter with smart remediation"""
    
    def __init__(self):
        self.eks_mcp_client = None
        self.agent = None
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup Strands agent with EKS MCP server connection"""
        try:
            # Create MCP client for EKS server
            self.eks_mcp_client = MCPClient(
                StdioServerParameters(
                    command="uvx",
                    args=["awslabs.eks-mcp-server@latest"],
                    env={
                        "AWS_REGION": AWS_REGION,
                        "AWS_DEFAULT_REGION": AWS_REGION
                    }
                )
            )
            
            # Define the EKS troubleshooting agent
            self.agent = Agent(
                model=f"bedrock:{BEDROCK_MODEL}",
                tools=[self.eks_mcp_client],
                system_prompt=self._get_system_prompt()
            )
            
            logger.info("✅ EKS Deployment Troubleshooter initialized with Strands agent")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup EKS agent: {str(e)}")
            raise
    
    def _get_system_prompt(self):
        """System prompt for the EKS troubleshooting agent"""
        return """
You are an expert Kubernetes DevOps engineer specializing in Amazon EKS troubleshooting and automated remediation.

Your role:
1. Analyze Splunk alerts about EKS deployment issues
2. Investigate the deployment state using EKS MCP tools
3. Determine appropriate remediation actions
4. Execute fixes when safe and beneficial
5. Provide clear summaries for Teams notifications

Available EKS MCP Tools:
- list_k8s_resources: List deployments, pods, services in a namespace
- manage_k8s_resource: Restart deployments, scale replicas, update configurations
- get_pod_logs: Retrieve logs from pods in a deployment
- apply_yaml: Apply Kubernetes manifests for fixes
- search_eks_troubleshoot_guide: Search AWS EKS troubleshooting documentation

Decision Guidelines:
- Memory/OOM issues: Restart deployment immediately
- CrashLoopBackOff: Check logs first, then restart if config issue
- Connection/timeout issues: Restart deployment
- High CPU: Check if scaling needed, restart if process-related
- Pod scheduling issues: Check node capacity and resource limits
- Production namespaces: Be more conservative, always check logs first
- Non-production: More aggressive remediation is acceptable

Always:
1. First investigate current state (list resources, check pod status)
2. Get recent logs to understand the issue
3. Make remediation decision based on issue type and environment
4. Execute fixes when appropriate
5. Verify results after remediation
6. Provide structured summary with actions taken

Be concise but thorough. Focus on actionable insights and clear next steps.
"""
    
    async def analyze_and_remediate(self, alert_data):
        """Main method to analyze alert and perform remediation"""
        try:
            # Extract deployment info from alert
            namespace = alert_data.get('kubernetes_namespace', 'default')
            deployment_name = alert_data.get('kubernetes_deployment_name', 
                                           alert_data.get('kubernetes_container_name', 'unknown'))
            alert_type = self._classify_alert_type(alert_data)
            severity = alert_data.get('severity', 'medium').lower()
            
            logger.info(f"🔍 Analyzing {severity} alert for deployment: {namespace}/{deployment_name}")
            
            # Create investigation prompt for the agent
            investigation_prompt = f"""
ALERT ANALYSIS REQUEST:

Alert Details:
- Deployment: {namespace}/{deployment_name}
- Alert Name: {alert_data.get('search_name', 'Unknown Alert')}
- Severity: {severity}
- Alert Type: {alert_type}
- Raw Log: {alert_data.get('_raw', 'No raw log available')[:500]}
- Timestamp: {alert_data.get('trigger_time', datetime.now().isoformat())}

TASK: Investigate this EKS deployment issue and perform appropriate remediation.

Steps:
1. Check current deployment status and pod health
2. Retrieve recent logs to understand the issue
3. Determine if remediation is needed based on issue type and severity
4. Execute appropriate fixes (restart, scale, config update)
5. Verify the remediation was successful
6. Provide a structured summary

Focus on deployment-level operations. If this is a {severity} severity issue in {namespace} namespace, 
{'be aggressive with remediation' if severity in ['high', 'critical'] else 'be conservative and investigate thoroughly'}.
"""
            
            # Run the agent
            logger.info("🤖 Executing Strands EKS agent analysis...")
            response = await self.agent.run(investigation_prompt)
            
            # Parse agent response into structured format
            summary = self._parse_agent_response(response, alert_data)
            
            logger.info(f"✅ EKS agent completed: {summary['status']}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ EKS agent analysis failed: {str(e)}")
            return {
                'status': 'error',
                'analysis': f"EKS troubleshooting failed: {str(e)}",
                'recommended_actions': [
                    f"Manual investigation required for {namespace}/{deployment_name}",
                    "Check EKS MCP server connectivity",
                    "Verify AWS credentials and EKS access permissions"
                ],
                'remediation_executed': False,
                'error': str(e)
            }
    
    def _classify_alert_type(self, alert_data):
        """Classify the type of alert for context"""
        alert_name = alert_data.get('search_name', '').lower()
        raw_log = alert_data.get('_raw', '').lower()
        combined_text = alert_name + ' ' + raw_log
        
        if any(keyword in combined_text for keyword in ['memory', 'oom', 'heap', 'leak']):
            return 'memory_issue'
        elif any(keyword in combined_text for keyword in ['crashloop', 'crash', 'failed', 'error']):
            return 'crash_issue'
        elif any(keyword in combined_text for keyword in ['connection', 'timeout', 'hang']):
            return 'connection_issue'
        elif any(keyword in combined_text for keyword in ['cpu', 'load', 'performance']):
            return 'performance_issue'
        elif any(keyword in combined_text for keyword in ['pending', 'scheduling', 'node']):
            return 'scheduling_issue'
        else:
            return 'general_issue'
    
    def _parse_agent_response(self, agent_response, alert_data):
        """Parse agent response into structured format for Teams notification"""
        try:
            response_text = str(agent_response)
            
            # Determine if remediation was executed based on response content
            remediation_keywords = ['restarted', 'scaled', 'applied', 'updated', 'fixed', 'remediated']
            remediation_executed = any(keyword in response_text.lower() for keyword in remediation_keywords)
            
            # Determine status
            if 'error' in response_text.lower() or 'failed' in response_text.lower():
                status = 'error'
            elif remediation_executed:
                status = 'remediated'
            else:
                status = 'investigated'
            
            # Extract key actions (look for bullet points or numbered lists)
            actions = []
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith(('•', '-', '*')) or any(line.startswith(f'{i}.') for i in range(1, 10)):
                    actions.append(line.lstrip('•-*0123456789. '))
            
            # If no structured actions found, create generic ones
            if not actions:
                if remediation_executed:
                    actions = [
                        f"✅ Automated remediation executed for {alert_data.get('kubernetes_namespace')}/{alert_data.get('kubernetes_deployment_name')}",
                        "Monitor deployment for stability",
                        "Check if alert conditions have cleared"
                    ]
                else:
                    actions = [
                        f"🔍 Investigation completed for {alert_data.get('kubernetes_namespace')}/{alert_data.get('kubernetes_deployment_name')}",
                        "Review analysis results above",
                        "Consider manual intervention if needed"
                    ]
            
            return {
                'status': status,
                'analysis': response_text[:1000],  # Truncate for Teams card
                'recommended_actions': actions[:5],  # Limit to 5 actions
                'remediation_executed': remediation_executed,
                'deployment_target': f"{alert_data.get('kubernetes_namespace')}/{alert_data.get('kubernetes_deployment_name')}",
                'agent_used': 'strands_eks_agent'
            }
            
        except Exception as e:
            logger.error(f"Failed to parse agent response: {str(e)}")
            return {
                'status': 'error',
                'analysis': f"Response parsing failed: {str(e)}",
                'recommended_actions': ["Manual review required"],
                'remediation_executed': False,
                'error': str(e)
            }

# Global troubleshooter instance
eks_troubleshooter = None

def get_troubleshooter():
    """Get or create the EKS troubleshooter instance"""
    global eks_troubleshooter
    if eks_troubleshooter is None:
        eks_troubleshooter = EKSDeploymentTroubleshooter()
    return eks_troubleshooter

# Flask Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test if troubleshooter can be initialized
        troubleshooter = get_troubleshooter()
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "strands-eks-deployment-troubleshooter",
            "eks_agent": "ready",
            "aws_region": AWS_REGION
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/splunk-alert', methods=['POST'])
def handle_splunk_alert():
    """Main endpoint for Splunk webhooks with Strands EKS analysis"""
    try:
        alert_data = request.json
        
        if not alert_data:
            logger.warning("Received empty alert data")
            return jsonify({"status": "error", "message": "No alert data provided"}), 400
        
        # Validate required fields for EKS troubleshooting
        namespace = alert_data.get('kubernetes_namespace')
        deployment = alert_data.get('kubernetes_deployment_name') or alert_data.get('kubernetes_container_name')
        
        if not namespace or not deployment:
            logger.warning("Missing kubernetes_namespace or deployment name in alert")
            return jsonify({
                "status": "error", 
                "message": "Alert must include kubernetes_namespace and kubernetes_deployment_name"
            }), 400
        
        logger.info(f"🚨 Received Splunk alert for: {namespace}/{deployment}")
        
        # Process alert with Strands EKS agent
        troubleshooter = get_troubleshooter()
        analysis_result = asyncio.run(troubleshooter.analyze_and_remediate(alert_data))
        
        # Send to Teams
        teams_success = send_alert_to_teams(alert_data, analysis_result)
        
        if teams_success:
            logger.info("✅ Alert processed and sent to Teams successfully")
            return jsonify({
                "status": "success",
                "message": "Alert processed with Strands EKS agent and sent to Teams",
                "analysis_status": analysis_result.get("status"),
                "remediation_executed": analysis_result.get("remediation_executed", False),
                "deployment_target": analysis_result.get("deployment_target")
            }), 200
        else:
            logger.error("❌ Failed to send to Teams")
            return jsonify({
                "status": "partial_success",
                "message": "Alert analyzed but failed to send to Teams",
                "analysis_result": analysis_result
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing alert: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Alert processing failed: {str(e)}"
        }), 500

def send_alert_to_teams(alert_data, analysis_result):
    """Send alert with EKS analysis to Teams using Adaptive Cards"""
    try:
        # Extract alert details
        alert_name = alert_data.get('search_name', 'Unknown Alert')
        severity = alert_data.get('severity', 'Medium')
        host = alert_data.get('host', 'Unknown Host')
        timestamp = alert_data.get('trigger_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        result_count = alert_data.get('result_count', 0)
        splunk_url = alert_data.get('view_link', '#')
        
        # Extract Kubernetes info
        k8s_namespace = alert_data.get('kubernetes_namespace')
        k8s_deployment = alert_data.get('kubernetes_deployment_name') or alert_data.get('kubernetes_container_name')
        guid = alert_data.get('guid')
        raw_log = alert_data.get('_raw')
        
        # Extract analysis results
        analysis_status = analysis_result.get('status', 'unknown')
        analysis_text = analysis_result.get('analysis', 'No analysis available')
        recommended_actions = analysis_result.get('recommended_actions', [])
        remediation_executed = analysis_result.get('remediation_executed', False)
        deployment_target = analysis_result.get('deployment_target', f"{k8s_namespace}/{k8s_deployment}")
        
        # Determine colors and emojis
        severity_colors = {
            'critical': 'Attention',
            'high': 'Warning', 
            'medium': 'Good',
            'low': 'Accent'
        }
        alert_color = severity_colors.get(severity.lower(), 'Good')
        
        status_emojis = {
            'remediated': "✅",
            'investigated': "🔍", 
            'error': "❌"
        }
        status_emoji = status_emojis.get(analysis_status, "⚠️")
        
        # Create adaptive card
        adaptive_card = {
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "Container",
                            "style": "emphasis",
                            "items": [{
                                "type": "ColumnSet",
                                "columns": [
                                    {
                                        "type": "Column",
                                        "width": "auto",
                                        "items": [{
                                            "type": "TextBlock",
                                            "text": "🚨",
                                            "size": "Large"
                                        }]
                                    },
                                    {
                                        "type": "Column",
                                        "width": "stretch",
                                        "items": [{
                                            "type": "TextBlock",
                                            "text": "EKS Deployment Alert + AI Remediation",
                                            "weight": "Bolder",
                                            "size": "Large",
                                            "color": alert_color
                                        }]
                                    }
                                ]
                            }]
                        },
                        {
                            "type": "TextBlock",
                            "text": alert_name,
                            "weight": "Bolder",
                            "size": "Medium",
                            "spacing": "Medium"
                        },
                        {
                            "type": "FactSet",
                            "facts": [
                                {"title": "Severity", "value": severity.upper()},
                                {"title": "Namespace", "value": k8s_namespace},
                                {"title": "Deployment", "value": k8s_deployment},
                                {"title": "Host", "value": host},
                                {"title": "Results", "value": str(result_count)},
                                {"title": "Time", "value": timestamp},
                                {"title": "Remediation", "value": "Yes" if remediation_executed else "No"},
                                {"title": "Status", "value": analysis_status.title()}
                            ] + ([{"title": "GUID", "value": guid}] if guid else [])
                        },
                        {
                            "type": "Container",
                            "style": "good" if analysis_status == "remediated" else "attention" if analysis_status == "error" else "default",
                            "items": [
                                {
                                    "type": "TextBlock",
                                    "text": f"{status_emoji} **Strands EKS Agent Analysis**",
                                    "weight": "Bolder",
                                    "size": "Medium"
                                },
                                {
                                    "type": "TextBlock",
                                    "text": f"**Target:** {deployment_target}",
                                    "weight": "Bolder",
                                    "spacing": "Small"
                                },
                                {
                                    "type": "TextBlock",
                                    "text": analysis_text,
                                    "wrap": True,
                                    "spacing": "Small"
                                }
                            ]
                        }
                    ],
                    "actions": [{
                        "type": "Action.OpenUrl",
                        "title": "🔍 View in Splunk",
                        "url": splunk_url
                    }]
                }
            }]
        }
        
        # Add recommended actions
        if recommended_actions:
            actions_text = "\n".join([f"• {action}" for action in recommended_actions])
            adaptive_card["attachments"][0]["content"]["body"].append({
                "type": "TextBlock",
                "text": "🎯 **Recommended Actions:**",
                "weight": "Bolder",
                "spacing": "Medium"
            })
            adaptive_card["attachments"][0]["content"]["body"].append({
                "type": "TextBlock",
                "text": actions_text,
                "wrap": True,
                "spacing": "Small"
            })
        
        # Add raw log if available (truncated)
        if raw_log:
            truncated_log = raw_log[:300] + "..." if len(raw_log) > 300 else raw_log
            adaptive_card["attachments"][0]["content"]["body"].extend([
                {
                    "type": "TextBlock",
                    "text": "📝 **Raw Log Data:**",
                    "weight": "Bolder",
                    "spacing": "Medium"
                },
                {
                    "type": "TextBlock",
                    "text": truncated_log,
                    "wrap": True,
                    "fontType": "Monospace",
                    "spacing": "Small",
                    "separator": True
                }
            ])
        
        # Send to Teams
        response = requests.post(TEAMS_WEBHOOK_URL, json=adaptive_card, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Successfully sent alert to Teams")
            return True
        else:
            logger.error(f"❌ Teams webhook failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error sending to Teams: {str(e)}")
        return False

@app.route('/test-deployment-alert', methods=['POST'])
def test_deployment_alert():
    """Test endpoint for deployment-based alerts"""
    try:
        test_data = {
            "search_name": "EKS Deployment Memory Leak Alert",
            "severity": "High",
            "host": "eks-node-01",
            "trigger_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "result_count": 5,
            "view_link": "https://splunk.company.com/test",
            "description": "Memory usage spike in payment service deployment",
            "kubernetes_namespace": "production-api",
            "kubernetes_deployment_name": "payment-service",
            "guid": "test-guid-67890",
            "_raw": "OutOfMemoryError: Java heap space in payment-service deployment, multiple pods affected, restart recommended"
        }
        
        # Override with POST data if provided
        if request.json:
            test_data.update(request.json)
        
        logger.info("🧪 Running test deployment alert with Strands EKS agent...")
        
        # Process with Strands EKS agent
        troubleshooter = get_troubleshooter()
        analysis_result = asyncio.run(troubleshooter.analyze_and_remediate(test_data))
        
        # Send to Teams
        teams_success = send_alert_to_teams(test_data, analysis_result)
        
        if teams_success:
            return jsonify({
                "status": "success",
                "message": "Test deployment alert processed with Strands EKS agent",
                "analysis_status": analysis_result.get("status"),
                "remediation_executed": analysis_result.get("remediation_executed"),
                "deployment_target": analysis_result.get("deployment_target"),
                "analysis_preview": analysis_result.get("analysis", "")[:200] + "..."
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to send test alert to Teams"
            }), 500
            
    except Exception as e:
        logger.error(f"Test deployment alert failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/test-basic', methods=['POST'])
def test_basic():
    """Basic test without EKS (for troubleshooting)"""
    try:
        test_data = {
            "search_name": "Basic Test Alert - No EKS",
            "severity": "Medium",
            "host": "test-server-01",
            "trigger_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "description": "Basic test without EKS interaction"
        }
        
        if request.json:
            test_data.update(request.json)
        
        # Mock analysis
        mock_analysis = {
            "status": "mock",
            "analysis": "This is a basic test alert without real EKS analysis",
            "recommended_actions": ["Manual investigation recommended"],
            "remediation_executed": False,
            "deployment_target": "test/deployment",
            "agent_used": "mock_agent"
        }
        
        teams_success = send_alert_to_teams(test_data, mock_analysis)
        
        if teams_success:
            return jsonify({
                "status": "success",
                "message": "Basic test alert sent to Teams"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to send basic test alert"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Validate configuration
    if TEAMS_WEBHOOK_URL == 'YOUR_TEAMS_WEBHOOK_URL_HERE':
        logger.warning("⚠️  Teams webhook URL not configured!")
        logger.warning("⚠️  Set TEAMS_WEBHOOK_URL environment variable")
    else:
        logger.info("✅ Teams webhook URL configured")
    
    # Check AWS credentials
    try:
        boto3.Session().get_credentials()
        logger.info("✅ AWS credentials found")
    except Exception as e:
        logger.warning(f"⚠️  AWS credentials issue: {str(e)}")
    
    logger.info("🚀 Starting Strands EKS Deployment Troubleshooter...")
    logger.info("📡 Available endpoints:")
    logger.info("   POST /splunk-alert              - Main Splunk webhook (with Strands EKS agent)")
    logger.info("   POST /test-deployment-alert     - Test with deployment troubleshooting")
    logger.info("   POST /test-basic               - Basic test without EKS")
    logger.info("   GET  /health                   - Health check")
    logger.info(f"🤖 Agent: Strands Framework + AWS EKS MCP Server")
    logger.info(f"🌍 AWS Region: {AWS_REGION}")
    logger.info(f"🧠 Model: {BEDROCK_MODEL}")
    logger.info("🎯 Focus: Kubernetes Deployment troubleshooting and remediation")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
    )
