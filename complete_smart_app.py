from flask import Flask, request, jsonify
import requests
import json
import os
import logging
import asyncio
import subprocess
import re
from datetime import datetime
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration - set via environment variables
TEAMS_WEBHOOK_URL = os.getenv('TEAMS_WEBHOOK_URL', 'YOUR_TEAMS_WEBHOOK_URL_HERE')
MCP_SERVER_COMMAND = os.getenv('MCP_SERVER_COMMAND', 'node')
MCP_SERVER_PATH = os.getenv('MCP_SERVER_PATH', 'C:\\Users\\sbagh\\MCP\\mcp-server-kubernetes\\dist\\index.js')
MCP_TIMEOUT = int(os.getenv('MCP_TIMEOUT', '60'))

class SimpleMCPClient:
    """MCP Client that communicates with Node.js MCP server via stdio"""
    
    def __init__(self):
        self.server_command = MCP_SERVER_COMMAND
        self.server_path = MCP_SERVER_PATH
        self.timeout = MCP_TIMEOUT
    
    async def call_mcp_tool(self, tool_name, params):
        """Call specific tool on MCP server"""
        
        mcp_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            }
        }
        
        try:
            result = await self._call_mcp_server(mcp_request)
            return result
        except Exception as e:
            logger.error(f"MCP tool call failed: {tool_name} - {str(e)}")
            raise
    
    async def _call_mcp_server(self, mcp_request):
        """Call MCP server via subprocess"""
        try:
            # Convert request to JSON
            request_json = json.dumps(mcp_request)
            
            # Start MCP server process
            process = await asyncio.create_subprocess_exec(
                self.server_command,
                self.server_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send request and get response
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=request_json.encode()),
                timeout=self.timeout
            )
            
            if stderr:
                logger.warning(f"MCP server stderr: {stderr.decode()}")
            
            # Parse response
            if stdout:
                response = json.loads(stdout.decode())
                if 'result' in response:
                    return response['result']
                elif 'error' in response:
                    raise Exception(f"MCP server error: {response['error']}")
                else:
                    return response
            else:
                raise Exception("No response from MCP server")
                
        except asyncio.TimeoutError:
            logger.error("MCP server call timed out")
            raise Exception("MCP server timeout")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MCP response: {str(e)}")
            raise Exception("Invalid MCP response format")
        except Exception as e:
            logger.error(f"MCP server call failed: {str(e)}")
            raise

class SmartToolMapper:
    """Rule-based tool discovery and mapping for MCP servers"""
    
    def __init__(self):
        self.available_tools = []
        self.tool_capabilities = {}
        
        # Define what we're looking for
        self.intent_signatures = {
            'pod_status': {
                'name_patterns': [r'describe.*pod', r'get.*pod.*status', r'pod.*info', r'pod.*detail'],
                'desc_patterns': [r'describe.*pod', r'pod.*status', r'pod.*detail', r'pod.*information'],
                'required_words': ['pod'],
                'exclude_words': ['delete', 'create', 'update', 'list'],
                'weight': {'name': 0.7, 'desc': 0.3}
            },
            'pod_logs': {
                'name_patterns': [r'get.*log', r'log', r'tail'],
                'desc_patterns': [r'log', r'tail', r'output'],
                'required_words': ['log'],
                'exclude_words': ['delete', 'create'],
                'weight': {'name': 0.8, 'desc': 0.2}
            },
            'pod_restart': {
                'name_patterns': [r'delete.*pod', r'restart.*pod', r'kill.*pod', r'remove.*pod'],
                'desc_patterns': [r'delete.*pod', r'remove.*pod', r'restart', r'kill'],
                'required_words': ['pod'],
                'exclude_words': ['get', 'list', 'describe'],
                'weight': {'name': 0.9, 'desc': 0.1}
            },
            'pod_list': {
                'name_patterns': [r'list.*pod', r'get.*pods'],
                'desc_patterns': [r'list.*pod', r'pods.*namespace'],
                'required_words': ['pod'],
                'exclude_words': ['delete', 'create', 'describe'],
                'weight': {'name': 0.8, 'desc': 0.2}
            }
        }
        
        # Parameter mapping rules
        self.param_mappings = {
            'pod_name': {
                'candidates': ['name', 'pod_name', 'podName', 'pod', 'resource_name'],
                'type': 'string',
                'required': True
            },
            'namespace': {
                'candidates': ['namespace', 'ns', 'nameSpace', 'project'],
                'type': 'string',
                'required': True
            },
            'container_name': {
                'candidates': ['container', 'container_name', 'containerName', 'container'],
                'type': 'string',
                'required': False
            },
            'log_lines': {
                'candidates': ['tail', 'lines', 'num_lines', 'limit', 'count'],
                'type': 'number',
                'required': False,
                'default': 50
            },
            'resource_type': {
                'candidates': ['resourceType', 'resource_type', 'type'],
                'type': 'string',
                'required': False,
                'default': 'pod'
            }
        }
    
    async def discover_capabilities(self, mcp_client):
        """Discover available tools and map them to capabilities"""
        try:
            # Get all available tools from MCP server
            tools_response = await mcp_client.call_mcp_tool('tools/list', {})
            
            if 'tools' in tools_response:
                self.available_tools = tools_response['tools']
            elif isinstance(tools_response, list):
                self.available_tools = tools_response
            else:
                logger.error(f"Unexpected tools response format: {tools_response}")
                return False
            
            logger.info(f"Discovered {len(self.available_tools)} tools from MCP server")
            
            # Map tools to capabilities
            for intent in self.intent_signatures.keys():
                best_match = self._find_best_tool_for_intent(intent)
                if best_match:
                    self.tool_capabilities[intent] = best_match
                    logger.info(f"Mapped {intent} → {best_match['tool']['name']} (score: {best_match['score']:.2f})")
                else:
                    logger.warning(f"No tool found for intent: {intent}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to discover MCP capabilities: {str(e)}")
            return False
    
    def _find_best_tool_for_intent(self, intent):
        """Find the best matching tool for a given intent"""
        
        if intent not in self.intent_signatures:
            return None
        
        signature = self.intent_signatures[intent]
        candidates = []
        
        for tool in self.available_tools:
            score = self._calculate_match_score(tool, signature)
            if score > 0:
                candidates.append({
                    'tool': tool,
                    'score': score,
                    'intent': intent
                })
        
        # Sort by score and return best match
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Only return if confidence is reasonable
        if candidates and candidates[0]['score'] > 0.3:
            return candidates[0]
        
        return None
    
    def _calculate_match_score(self, tool, signature):
        """Calculate how well a tool matches an intent signature"""
        
        tool_name = tool['name'].lower()
        tool_desc = tool.get('description', '').lower()
        
        # Check required words (must have at least one)
        has_required = any(word in tool_name or word in tool_desc 
                          for word in signature['required_words'])
        if not has_required:
            return 0.0
        
        # Check exclude words (must not have any)
        has_excluded = any(word in tool_name or word in tool_desc 
                          for word in signature['exclude_words'])
        if has_excluded:
            return 0.0
        
        # Score name patterns
        name_score = 0.0
        for pattern in signature['name_patterns']:
            if re.search(pattern, tool_name):
                name_score += 1.0
        name_score = min(name_score / len(signature['name_patterns']), 1.0)
        
        # Score description patterns
        desc_score = 0.0
        for pattern in signature['desc_patterns']:
            if re.search(pattern, tool_desc):
                desc_score += 1.0
        desc_score = min(desc_score / len(signature['desc_patterns']), 1.0)
        
        # Weighted final score
        final_score = (name_score * signature['weight']['name'] + 
                      desc_score * signature['weight']['desc'])
        
        return final_score
    
    def adapt_parameters(self, intent_params, tool_schema):
        """Adapt our standard parameters to tool-specific format"""
        
        adapted = {}
        missing_required = []
        tool_properties = tool_schema.get('inputSchema', {}).get('properties', {})
        required_params = tool_schema.get('inputSchema', {}).get('required', [])
        
        # Map our parameters to tool parameters
        for our_param, value in intent_params.items():
            mapped_param = self._find_matching_parameter(our_param, tool_properties)
            if mapped_param:
                converted_value = self._convert_parameter_value(value, tool_properties[mapped_param])
                adapted[mapped_param] = converted_value
            else:
                logger.warning(f"Could not map parameter {our_param}")
        
        # Add defaults for missing required parameters
        for req_param in required_params:
            if req_param not in adapted:
                default_value = self._get_default_value(req_param, tool_properties.get(req_param, {}))
                if default_value is not None:
                    adapted[req_param] = default_value
                    logger.info(f"Added default value for {req_param}: {default_value}")
                else:
                    missing_required.append(req_param)
        
        return {
            'params': adapted,
            'missing': missing_required,
            'success': len(missing_required) == 0
        }
    
    def _find_matching_parameter(self, our_param, tool_properties):
        """Find matching parameter name in tool schema"""
        
        if our_param not in self.param_mappings:
            return None
        
        candidates = self.param_mappings[our_param]['candidates']
        
        # Try exact matches first
        for candidate in candidates:
            if candidate in tool_properties:
                return candidate
        
        # Try fuzzy matches
        for tool_param in tool_properties.keys():
            for candidate in candidates:
                if (candidate.lower() in tool_param.lower() or 
                    tool_param.lower() in candidate.lower()):
                    return tool_param
        
        return None
    
    def _convert_parameter_value(self, value, param_schema):
        """Convert value to expected parameter type"""
        
        param_type = param_schema.get('type', 'string')
        
        try:
            if param_type == 'number':
                return int(value) if isinstance(value, str) and value.isdigit() else float(value)
            elif param_type == 'boolean':
                if isinstance(value, str):
                    return value.lower() in ['true', '1', 'yes', 'on']
                return bool(value)
            elif param_type == 'array' and not isinstance(value, list):
                return [value]
            else:
                return str(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {value} to {param_type}, using as string")
            return str(value)
    
    def _get_default_value(self, param_name, param_schema):
        """Get default value for a parameter"""
        
        # Check if parameter has default in schema
        if 'default' in param_schema:
            return param_schema['default']
        
        # Use our predefined defaults
        param_defaults = {
            'resourceType': 'pod',
            'resource_type': 'pod',
            'type': 'pod',
            'ignoreNotFound': False,
            'ignore_not_found': False,
            'tail': 50,
            'lines': 50,
            'timestamps': False
        }
        
        return param_defaults.get(param_name)
    
    async def execute_smart_call(self, mcp_client, intent, intent_params):
        """Execute a smart tool call using discovered capabilities"""
        
        if intent not in self.tool_capabilities:
            raise Exception(f"No tool available for intent: {intent}")
        
        capability = self.tool_capabilities[intent]
        tool = capability['tool']
        
        # Adapt parameters
        param_result = self.adapt_parameters(intent_params, tool)
        
        if not param_result['success']:
            missing = ', '.join(param_result['missing'])
            raise Exception(f"Cannot call {tool['name']}: missing required parameters: {missing}")
        
        # Execute the call
        logger.info(f"Executing {tool['name']} with params: {param_result['params']}")
        
        try:
            result = await mcp_client.call_mcp_tool(tool['name'], param_result['params'])
            return {
                'success': True,
                'result': result,
                'tool_used': tool['name'],
                'adapted_params': param_result['params']
            }
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'tool_used': tool['name'],
                'adapted_params': param_result['params']
            }
    
    def get_capabilities_summary(self):
        """Get a summary of discovered capabilities"""
        summary = {}
        for intent, capability in self.tool_capabilities.items():
            summary[intent] = {
                'tool_name': capability['tool']['name'],
                'confidence': capability['score'],
                'description': capability['tool'].get('description', 'No description')
            }
        return summary

class SmartPodRestartOrchestrator:
    """Orchestrator with dynamic tool discovery and smart mapping"""
    
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.tool_mapper = SmartToolMapper()
        self.capabilities_discovered = False
    
    async def analyze_and_respond(self, alert_data):
        """Main workflow with smart tool discovery"""
        
        # Discover capabilities if not done yet
        if not self.capabilities_discovered:
            await self._discover_capabilities()
        
        # Classification (same as before)
        issue_type = self._classify_alert(alert_data)
        logger.info(f"Alert classified as: {issue_type['type']}")
        
        # Smart investigation using discovered tools
        investigation = await self._smart_investigate(alert_data)
        
        # Restart decision (same logic as before)
        restart_decision = self._should_restart_pod(issue_type, investigation, alert_data)
        
        if restart_decision['should_restart']:
            restart_result = await self._smart_restart_pod(alert_data)
            verification = await self._smart_verify_restart(alert_data)
            
            summary = self._generate_restart_summary(
                issue_type, investigation, restart_result, verification, restart_decision
            )
        else:
            summary = self._generate_investigation_summary(
                issue_type, investigation, restart_decision
            )
        
        return summary
    
    async def _discover_capabilities(self):
        """Discover and map MCP server capabilities"""
        try:
            logger.info("🔍 Discovering MCP server capabilities...")
            
            success = await self.tool_mapper.discover_capabilities(self.mcp_client)
            
            if success:
                capabilities = self.tool_mapper.get_capabilities_summary()
                logger.info("✅ MCP capabilities discovered:")
                for intent, cap in capabilities.items():
                    logger.info(f"   {intent} → {cap['tool_name']} (confidence: {cap['confidence']:.2f})")
                
                self.capabilities_discovered = True
            else:
                logger.error("❌ Failed to discover MCP capabilities")
                # Continue with limited functionality
                
        except Exception as e:
            logger.error(f"Error during capability discovery: {str(e)}")
            # Continue anyway - individual calls will handle missing tools
    
    async def _smart_investigate(self, alert_data):
        """Investigate using discovered tools"""
        
        namespace = alert_data.get('kubernetes_namespace')
        container = alert_data.get('kubernetes_container_name')
        
        investigation = {
            'timestamp': datetime.now().isoformat(),
            'target': {'namespace': namespace, 'container': container},
            'results': {},
            'tools_used': []
        }
        
        # Try to get pod status using smart mapping
        try:
            logger.info("🔍 Getting pod status via smart mapping...")
            
            result = await self.tool_mapper.execute_smart_call(
                self.mcp_client,
                intent='pod_status',
                intent_params={
                    'pod_name': container,
                    'namespace': namespace
                }
            )
            
            if result['success']:
                investigation['results']['pod_status'] = result['result']
                investigation['tools_used'].append(f"pod_status → {result['tool_used']}")
                logger.info(f"✅ Pod status retrieved via {result['tool_used']}")
            else:
                investigation['results']['pod_status'] = f"Error: {result['error']}"
                logger.warning(f"⚠️ Pod status failed: {result['error']}")
                
        except Exception as e:
            investigation['results']['pod_status'] = f"Smart mapping failed: {str(e)}"
            logger.warning(f"⚠️ Pod status smart mapping failed: {e}")
        
        # Try to get pod logs using smart mapping
        try:
            logger.info("🔍 Getting pod logs via smart mapping...")
            
            result = await self.tool_mapper.execute_smart_call(
                self.mcp_client,
                intent='pod_logs',
                intent_params={
                    'pod_name': container,
                    'namespace': namespace,
                    'log_lines': 50
                }
            )
            
            if result['success']:
                investigation['results']['logs'] = result['result']
                investigation['tools_used'].append(f"pod_logs → {result['tool_used']}")
                logger.info(f"✅ Pod logs retrieved via {result['tool_used']}")
            else:
                investigation['results']['logs'] = f"Error: {result['error']}"
                logger.warning(f"⚠️ Pod logs failed: {result['error']}")
                
        except Exception as e:
            investigation['results']['logs'] = f"Smart mapping failed: {str(e)}"
            logger.warning(f"⚠️ Pod logs smart mapping failed: {e}")
        
        # Try to list pods for context
        try:
            logger.info("🔍 Listing pods for context...")
            
            result = await self.tool_mapper.execute_smart_call(
                self.mcp_client,
                intent='pod_list',
                intent_params={
                    'namespace': namespace
                }
            )
            
            if result['success']:
                investigation['results']['pod_list'] = result['result']
                investigation['tools_used'].append(f"pod_list → {result['tool_used']}")
                logger.info(f"✅ Pod list retrieved via {result['tool_used']}")
            else:
                investigation['results']['pod_list'] = f"Error: {result['error']}"
                
        except Exception as e:
            investigation['results']['pod_list'] = f"Smart mapping failed: {str(e)}"
            logger.warning(f"⚠️ Pod list smart mapping failed: {e}")
        
        return investigation
    
    async def _smart_restart_pod(self, alert_data):
        """Execute pod restart using smart mapping"""
        
        namespace = alert_data.get('kubernetes_namespace')
        container = alert_data.get('kubernetes_container_name')
        
        restart_result = {
            'timestamp': datetime.now().isoformat(),
            'target': f"{namespace}/{container}",
            'success': False,
            'method': 'smart_mapping'
        }
        
        try:
            logger.info(f"🔄 Restarting pod via smart mapping: {namespace}/{container}")
            
            result = await self.tool_mapper.execute_smart_call(
                self.mcp_client,
                intent='pod_restart',
                intent_params={
                    'pod_name': container,
                    'namespace': namespace
                }
            )
            
            if result['success']:
                restart_result['success'] = True
                restart_result['details'] = result['result']
                restart_result['tool_used'] = result['tool_used']
                restart_result['message'] = f"Successfully restarted {namespace}/{container} using {result['tool_used']}"
                
                logger.info(f"✅ Pod restart completed via {result['tool_used']}")
            else:
                restart_result['success'] = False
                restart_result['error'] = result['error']
                restart_result['tool_used'] = result['tool_used']
                restart_result['message'] = f"Failed to restart via {result['tool_used']}: {result['error']}"
                
                logger.error(f"❌ Pod restart failed via {result['tool_used']}: {result['error']}")
            
        except Exception as e:
            restart_result['success'] = False
            restart_result['error'] = str(e)
            restart_result['message'] = f"Smart restart mapping failed: {str(e)}"
            
            logger.error(f"❌ Smart restart failed: {e}")
        
        return restart_result
    
    async def _smart_verify_restart(self, alert_data):
        """Verify restart using smart mapping"""
        
        namespace = alert_data.get('kubernetes_namespace')
        container = alert_data.get('kubernetes_container_name')
        
        verification = {
            'timestamp': datetime.now().isoformat(),
            'target': f"{namespace}/{container}",
            'checks': []
        }
        
        # Wait for restart to complete
        logger.info("⏳ Waiting 30 seconds for restart to complete...")
        await asyncio.sleep(30)
        
        # Check if pods are running again
        try:
            result = await self.tool_mapper.execute_smart_call(
                self.mcp_client,
                intent='pod_list',
                intent_params={
                    'namespace': namespace
                }
            )
            
            if result['success']:
                verification['checks'].append({
                    'check': 'pod_list_after_restart',
                    'success': True,
                    'tool_used': result['tool_used'],
                    'result': result['result']
                })
                logger.info(f"✅ Post-restart verification via {result['tool_used']}")
            else:
                verification['checks'].append({
                    'check': 'pod_list_after_restart',
                    'success': False,
                    'error': result['error'],
                    'tool_used': result['tool_used']
                })
                
        except Exception as e:
            verification['checks'].append({
                'check': 'pod_list_after_restart',
                'success': False,
                'error': str(e)
            })
        
        return verification
    
    def _classify_alert(self, alert_data):
        """Classify alert type for restart decision"""
        alert_name = alert_data.get('search_name', '').lower()
        raw_log = alert_data.get('_raw', '').lower()
        
        if any(keyword in alert_name + raw_log for keyword in ['memory', 'oom', 'heap', 'leak']):
            return {
                'type': 'memory_issue',
                'restart_likelihood': 'very_high',
                'reason': 'Memory issues typically resolved by pod restart'
            }
        
        if any(keyword in alert_name + raw_log for keyword in ['connection', 'timeout', 'hang', 'stuck']):
            return {
                'type': 'connection_issue',
                'restart_likelihood': 'very_high',
                'reason': 'Connection issues usually resolved by pod restart'
            }
        
        if any(keyword in alert_name + raw_log for keyword in ['error', 'exception', 'crash', 'failed']):
            return {
                'type': 'application_error',
                'restart_likelihood': 'high',
                'reason': 'Application errors often resolved by fresh start'
            }
        
        if any(keyword in alert_name + raw_log for keyword in ['pod', 'container', 'restart', 'crashloop']):
            return {
                'type': 'pod_issue',
                'restart_likelihood': 'very_high',
                'reason': 'Direct pod issues indicate restart needed'
            }
        
        if any(keyword in alert_name + raw_log for keyword in ['cpu', 'load', 'processor']):
            return {
                'type': 'cpu_issue',
                'restart_likelihood': 'medium',
                'reason': 'CPU issues may be helped by restart if process-related'
            }
        
        return {
            'type': 'generic_issue',
            'restart_likelihood': 'medium',
            'reason': 'General issue that might benefit from restart'
        }
    
    def _should_restart_pod(self, issue_type, investigation, alert_data):
        """Decide if pod restart is appropriate"""
        namespace = alert_data.get('kubernetes_namespace', '').lower()
        severity = alert_data.get('severity', '').lower()
        
        is_production = any(prod in namespace for prod in ['prod', 'production', 'live'])
        is_critical = severity in ['critical', 'high']
        restart_likelihood = issue_type['restart_likelihood']
        
        if restart_likelihood == 'very_high':
            should_restart = True
            reason = f"{issue_type['reason']} - restart highly recommended"
        elif restart_likelihood == 'high':
            if is_critical or not is_production:
                should_restart = True
                reason = f"{issue_type['reason']} - restart recommended for {severity} severity"
            else:
                should_restart = False
                reason = "Production safety: manual approval needed for non-critical issues"
        elif restart_likelihood == 'medium':
            if is_critical:
                should_restart = True
                reason = f"Critical severity justifies restart for {issue_type['type']}"
            else:
                should_restart = False
                reason = "Medium likelihood + non-critical severity = investigation only"
        else:
            should_restart = False
            reason = "Low restart likelihood - investigation only"
        
        return {
            'should_restart': should_restart,
            'reason': reason,
            'safety_checks': {
                'is_production': is_production,
                'is_critical': is_critical,
                'restart_likelihood': restart_likelihood
            }
        }
    
    def _generate_restart_summary(self, issue_type, investigation, restart_result, verification, decision):
        """Enhanced summary with smart mapping info"""
        restart_success = restart_result['success']
        verification_success = len([c for c in verification['checks'] if c['success']]) > 0
        
        status = "success" if restart_success and verification_success else "partial"
        
        analysis_parts = [
            f"Issue classified as: {issue_type['type'].replace('_', ' ').title()}",
            f"Restart decision: {decision['reason']}"
        ]
        
        # Add smart mapping info
        if 'tools_used' in investigation:
            tools_info = ", ".join(investigation['tools_used'])
            analysis_parts.append(f"Tools discovered and used: {tools_info}")
        
        if restart_success:
            tool_used = restart_result.get('tool_used', 'unknown')
            analysis_parts.append(f"✅ Pod restart executed successfully via {tool_used}")
        else:
            analysis_parts.append("❌ Pod restart failed")
        
        if verification_success:
            analysis_parts.append("✅ Post-restart verification completed")
        
        actions = []
        if restart_success:
            actions.append("✅ Pod restarted successfully using smart tool mapping")
            actions.append("Monitor application for stability")
            if status == "partial":
                actions.append("Manual verification recommended")
        else:
            actions.append("❌ Restart failed - manual intervention required")
            actions.append("Check pod logs and status manually")
        
        return {
            'status': status,
            'analysis': ". ".join(analysis_parts),
            'recommended_actions': actions,
            'details': {
                'issue_type': issue_type,
                'restart_executed': restart_success,
                'investigation': investigation,
                'restart_result': restart_result,
                'verification': verification,
                'smart_mapping_used': True
            }
        }
    
    def _generate_investigation_summary(self, issue_type, investigation, decision):
        """Enhanced investigation summary with smart mapping info"""
        analysis_parts = [
            f"Issue classified as: {issue_type['type'].replace('_', ' ').title()}",
            "Investigation completed using smart tool discovery",
            f"Restart decision: {decision['reason']}"
        ]
        
        # Add tools used info
        if 'tools_used' in investigation:
            tools_info = ", ".join(investigation['tools_used'])
            analysis_parts.append(f"Tools used: {tools_info}")
        
        actions = [
            "🔍 Investigation completed using smart tool mapping",
            "Review investigation results above",
            "Manual restart can be performed if needed",
            "Monitor alert conditions for changes"
        ]
        
        return {
            'status': 'investigated',
            'analysis': ". ".join(analysis_parts),
            'recommended_actions': actions,
            'details': {
                'issue_type': issue_type,
                'restart_executed': False,
                'investigation': investigation,
                'decision': decision,
                'smart_mapping_used': True
            }
        }

# Flask Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "smart-splunk-teams-mcp-forwarder"
    }), 200

@app.route('/splunk-alert', methods=['POST'])
def handle_splunk_alert():
    """Main endpoint for Splunk webhooks with smart MCP integration"""
    try:
        alert_data = request.json
        
        if not alert_data:
            logger.warning("Received empty alert data")
            return jsonify({"status": "error", "message": "No alert data provided"}), 400
        
        logger.info(f"Received Splunk alert: {alert_data.get('search_name', 'Unknown')}")
        
        # Process alert with smart MCP orchestration
        mcp_analysis = asyncio.run(process_alert_with_smart_mcp(alert_data))
        
        # Send to Teams
        teams_success = send_alert_to_teams(alert_data, mcp_analysis)
        
        if teams_success:
            logger.info("Alert processed and sent to Teams successfully")
            return jsonify({
                "status": "success",
                "message": "Alert processed and forwarded to Teams with smart MCP analysis",
                "mcp_status": mcp_analysis.get("status"),
                "restart_executed": mcp_analysis.get("restart_executed", False),
                "tools_discovered": mcp_analysis.get("tools_discovered", False)
            }), 200
        else:
            logger.error("Failed to send to Teams")
            return jsonify({
                "status": "error",
                "message": "Alert processed but failed to send to Teams"
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing alert: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Alert processing failed: {str(e)}"
        }), 500

async def process_alert_with_smart_mcp(alert_data):
    """Process alert using smart MCP tool discovery and mapping"""
    try:
        # Create MCP client
        mcp_client = SimpleMCPClient()
        
        # Create smart orchestrator with tool discovery
        orchestrator = SmartPodRestartOrchestrator(mcp_client)
        
        # Execute intelligent analysis with dynamic tool mapping
        summary = await orchestrator.analyze_and_respond(alert_data)
        
        logger.info(f"Smart MCP orchestration completed: {summary['status']}")
        
        return {
            "status": summary['status'],
            "analysis": summary['analysis'],
            "recommended_actions": summary['recommended_actions'],
            "restart_executed": summary['details'].get('restart_executed', False),
            "investigation_completed": True,
            "tools_discovered": summary['details'].get('smart_mapping_used', False),
            "source": "smart_mcp_orchestrator"
        }
        
    except Exception as e:
        logger.error(f"Smart MCP orchestration failed: {str(e)}")
        
        # Fallback to basic analysis
        return {
            "status": "error",
            "analysis": f"Smart MCP analysis failed: {str(e)}. This suggests either MCP server connectivity issues or tool discovery problems. Recommend manual investigation of {alert_data.get('kubernetes_namespace')}/{alert_data.get('kubernetes_container_name')}",
            "recommended_actions": [
                "Check MCP server connectivity",
                "Verify kubectl access from MCP server location", 
                f"Manually investigate pod: {alert_data.get('kubernetes_container_name')}",
                "Consider manual restart if needed"
            ],
            "restart_executed": False,
            "investigation_completed": False,
            "tools_discovered": False,
            "source": "fallback_after_smart_mapping_failure"
        }

def send_alert_to_teams(alert_data, mcp_analysis):
    """Send alert with MCP analysis to Teams"""
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
        k8s_container = alert_data.get('kubernetes_container_name')
        guid = alert_data.get('guid')
        raw_log = alert_data.get('_raw')
        
        # Extract MCP analysis
        mcp_status = mcp_analysis.get('status', 'unknown')
        analysis_text = mcp_analysis.get('analysis', 'No analysis available')
        recommended_actions = mcp_analysis.get('recommended_actions', [])
        restart_executed = mcp_analysis.get('restart_executed', False)
        tools_discovered = mcp_analysis.get('tools_discovered', False)
        
        # Determine colors and emojis
        color_map = {
            'critical': 'Attention',
            'high': 'Warning', 
            'medium': 'Good',
            'low': 'Accent'
        }
        alert_color = color_map.get(severity.lower(), 'Good')
        
        if mcp_status == 'success':
            mcp_emoji = "✅"
        elif mcp_status == 'partial':
            mcp_emoji = "⚠️"
        else:
            mcp_emoji = "❌"
        
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
                                            "text": "Splunk Alert with Smart AI Analysis",
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
                                {"title": "Host", "value": host},
                                {"title": "Results", "value": str(result_count)},
                                {"title": "Time", "value": timestamp}
                            ] + (
                                [{"title": "K8s Namespace", "value": k8s_namespace}] if k8s_namespace else []
                            ) + (
                                [{"title": "K8s Container", "value": k8s_container}] if k8s_container else []
                            ) + (
                                [{"title": "GUID", "value": guid}] if guid else []
                            ) + (
                                [{"title": "Pod Restarted", "value": "Yes" if restart_executed else "No"}]
                            ) + (
                                [{"title": "Smart Tools", "value": "Yes" if tools_discovered else "No"}]
                            )
                        },
                        {
                            "type": "Container",
                            "style": "good" if mcp_status == "success" else "attention",
                            "items": [
                                {
                                    "type": "TextBlock",
                                    "text": f"{mcp_emoji} **Smart AI Analysis & Actions**",
                                    "weight": "Bolder",
                                    "size": "Medium"
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
        
        # Add description if available
        if alert_data.get('description'):
            adaptive_card["attachments"][0]["content"]["body"].append({
                "type": "TextBlock",
                "text": f"**Description:** {alert_data['description']}",
                "wrap": True,
                "spacing": "Medium"
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
            logger.info("Successfully sent alert to Teams")
            return True
        else:
            logger.error(f"Teams webhook failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending to Teams: {str(e)}")
        return False

@app.route('/test-smart-mapping', methods=['POST'])
def test_smart_mapping():
    """Test the smart tool discovery and mapping"""
    try:
        namespace = request.json.get('namespace', 'staging-api') if request.json else 'staging-api'
        pod_name = request.json.get('pod_name', 'payment-service') if request.json else 'payment-service'
        
        async def run_smart_test():
            # Create MCP client and tool mapper
            mcp_client = SimpleMCPClient()
            tool_mapper = SmartToolMapper()
            
            # Discover capabilities
            discovery_success = await tool_mapper.discover_capabilities(mcp_client)
            
            results = {
                'discovery_success': discovery_success,
                'capabilities': tool_mapper.get_capabilities_summary() if discovery_success else {},
                'test_calls': {}
            }
            
            if discovery_success:
                # Test each capability
                test_params = {
                    'pod_name': pod_name,
                    'namespace': namespace,
                    'log_lines': 10
                }
                
                for intent in ['pod_status', 'pod_logs', 'pod_list']:
                    try:
                        result = await tool_mapper.execute_smart_call(
                            mcp_client, intent, test_params
                        )
                        results['test_calls'][intent] = {
                            'success': result['success'],
                            'tool_used': result.get('tool_used'),
                            'error': result.get('error'),
                            'has_result': bool(result.get('result'))
                        }
                    except Exception as e:
                        results['test_calls'][intent] = {
                            'success': False,
                            'error': str(e)
                        }
            
            return results
        
        # Run the async test
        results = asyncio.run(run_smart_test())
        
        return jsonify({
            'status': 'success',
            'message': 'Smart mapping test completed',
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/test-alert', methods=['POST'])
def test_alert():
    """Test endpoint with smart MCP analysis"""
    try:
        test_data = {
            "search_name": "Test Alert - Memory Leak",
            "severity": "High",
            "host": "test-server-01",
            "trigger_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "result_count": 3,
            "view_link": "https://splunk.company.com/test",
            "description": "Test alert for smart MCP integration",
            "kubernetes_namespace": "staging-api",
            "kubernetes_container_name": "nginx-frontend",
            "guid": "test-guid-12345",
            "_raw": "OutOfMemoryError detected in nginx-frontend container, pod restart recommended"
        }
        
        # Override with POST data if provided
        if request.json:
            test_data.update(request.json)
        
        logger.info("Running test alert with smart MCP analysis...")
        
        # Process with smart MCP
        mcp_analysis = asyncio.run(process_alert_with_smart_mcp(test_data))
        
        # Send to Teams
        teams_success = send_alert_to_teams(test_data, mcp_analysis)
        
        if teams_success:
            return jsonify({
                "status": "success",
                "message": "Test alert sent to Teams with smart MCP analysis",
                "mcp_status": mcp_analysis.get("status"),
                "restart_executed": mcp_analysis.get("restart_executed"),
                "tools_discovered": mcp_analysis.get("tools_discovered"),
                "analysis_preview": mcp_analysis.get("analysis", "")[:200] + "..."
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to send test alert to Teams"
            }), 500
            
    except Exception as e:
        logger.error(f"Test alert failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/test-alert-basic', methods=['POST'])
def test_alert_basic():
    """Basic test without MCP (for troubleshooting)"""
    try:
        test_data = {
            "search_name": "Basic Test Alert",
            "severity": "Medium",
            "host": "test-server-01",
            "trigger_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "description": "Basic test without MCP"
        }
        
        if request.json:
            test_data.update(request.json)
        
        # Mock analysis
        mock_analysis = {
            "status": "mock",
            "analysis": "This is a basic test alert without real MCP analysis",
            "recommended_actions": ["Manual investigation recommended"],
            "restart_executed": False,
            "tools_discovered": False,
            "source": "basic_test"
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
    
    if os.path.exists(MCP_SERVER_PATH):
        logger.info(f"✅ MCP server found: {MCP_SERVER_PATH}")
    else:
        logger.warning(f"⚠️  MCP server not found: {MCP_SERVER_PATH}")
        logger.warning("⚠️  Set MCP_SERVER_PATH environment variable")
    
    logger.info("🚀 Starting Smart Splunk-Teams-MCP Integration Server...")
    logger.info("📡 Available endpoints:")
    logger.info("   POST /splunk-alert        - Main Splunk webhook endpoint (with smart MCP)")
    logger.info("   POST /test-alert          - Test with smart MCP analysis")
    logger.info("   POST /test-alert-basic    - Basic test without MCP")
    logger.info("   POST /test-smart-mapping  - Test smart tool discovery")
    logger.info("   GET  /health              - Health check")
    logger.info(f"🤖 MCP Server: {MCP_SERVER_COMMAND} {MCP_SERVER_PATH}")
    logger.info(f"⏱️  Timeout: {MCP_TIMEOUT} seconds")
    logger.info("🧠 Smart tool discovery enabled - adapts to any MCP server!")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
    )