# import streamlit as st
# import requests
# import json
# import uuid
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from typing import List, Dict, Any, Optional
# import openai
# import re

# # Page configuration
# st.set_page_config(
#     page_title="Mortgage Rule Generator",
#     page_icon="🏦",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize session state
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'client_id' not in st.session_state:
#     st.session_state.client_id = 307
# if 'data_sources' not in st.session_state:
#     st.session_state.data_sources = {}
# if 'last_generated_rule' not in st.session_state:
#     st.session_state.last_generated_rule = None
# if 'synthetic_data' not in st.session_state:
#     st.session_state.synthetic_data = None
# if 'show_synthetic_data' not in st.session_state:
#     st.session_state.show_synthetic_data = False
# if 'confirmed_structure' not in st.session_state:
#     st.session_state.confirmed_structure = None
# if 'rule_visualization' not in st.session_state:
#     st.session_state.rule_visualization = None
# if 'awaiting_confirmation' not in st.session_state:
#     st.session_state.awaiting_confirmation = False
# if 'logical_options' not in st.session_state:
#     st.session_state.logical_options = []
# if 'selected_option' not in st.session_state:
#     st.session_state.selected_option = None
# if 'show_structure_options' not in st.session_state:
#     st.session_state.show_structure_options = False

# # Get secrets from Streamlit secrets
# def get_secrets():
#     secrets = {
#         'external_api_base_url': st.secrets.get("EXTERNAL_API_BASE_URL", "https://lmsdev-external-distributor-api.pulseid.com"),
#         'external_api_key': st.secrets.get("EXTERNAL_API_KEY", ""),
#         'external_api_secret': st.secrets.get("EXTERNAL_API_SECRET", ""),
#         'default_client_id': st.secrets.get("DEFAULT_CLIENT_ID", 307),
#         'openai_api_key': st.secrets.get("OPENAI_API_KEY", "")
#     }
#     return secrets

# secrets = get_secrets()

# # Initialize OpenAI client
# try:
#     client = openai.OpenAI(api_key=secrets['openai_api_key'])
# except Exception as e:
#     st.error(f"Failed to initialize OpenAI client: {str(e)}")
#     client = None

# # ---------- Helper Functions ----------
# def parse_logical_options(logical_structure: str) -> List[str]:
#     """Parse logical structure string to extract multiple options"""
#     options = []
    
#     # Check if it's multiple options format
#     if "Option 1:" in logical_structure:
#         # Split by "Option X:" pattern
#         pattern = r'Option \d+:'
#         parts = re.split(pattern, logical_structure)
        
#         for i, part in enumerate(parts[1:], 1):  # Skip first empty part
#             option_text = f"Option {i}: {part.strip()}"
#             # Clean up the option text
#             option_text = option_text.replace('\\n', ' ').replace('\n', ' ').strip()
#             options.append(option_text)
    
#     # If not in Option X format, check for numbered options
#     elif "1." in logical_structure or "2." in logical_structure:
#         lines = logical_structure.split('\n')
#         for line in lines:
#             line = line.strip()
#             if re.match(r'^\d+[\.\)]', line):
#                 options.append(line)
    
#     # If single structure
#     else:
#         options = [logical_structure.strip()]
    
#     # Clean up options
#     cleaned_options = []
#     for opt in options:
#         # Remove any trailing connector text
#         opt = re.sub(r'\s*(Do you agree with any of this structure.*)', '', opt)
#         opt = opt.strip()
#         if opt:
#             cleaned_options.append(opt)
    
#     return cleaned_options if cleaned_options else [logical_structure]

# def extract_last_user_message():
#     """Extract the last user message from chat history"""
#     for msg in reversed(st.session_state.chat_history):
#         if msg["role"] == "user":
#             return msg["content"]
#     return None

# # ---------- Visualization Functions ----------
# def create_rule_visualization(rules_data: Dict[str, Any]) -> Dict[str, Any]:
#     """Create a structured visualization of the rule for display"""
    
#     def process_rule_item(rule_item, level=0):
#         """Recursively process rule items for visualization"""
#         item_type = rule_item.get("ruleType", "condition")
        
#         if item_type == "conditionGroup":
#             # Process condition group
#             group_data = {
#                 "type": "group",
#                 "id": rule_item.get("id", str(uuid.uuid4())),
#                 "group_connector": rule_item.get("groupConnector", "AND"),
#                 "conditions": [],
#                 "connector": rule_item.get("connector"),
#                 "level": level
#             }
            
#             # Process nested conditions
#             for condition in rule_item.get("conditions", []):
#                 processed_condition = process_rule_item(condition, level + 1)
#                 group_data["conditions"].append(processed_condition)
            
#             return group_data
        
#         else:  # Individual condition
#             return {
#                 "type": "condition",
#                 "id": rule_item.get("id", str(uuid.uuid4())),
#                 "dataSource": rule_item.get("dataSource", "Unknown"),
#                 "dataSourceId": rule_item.get("dataSourceId", "N/A"),
#                 "field": rule_item.get("field", "Unknown"),
#                 "fieldId": rule_item.get("fieldId", "N/A"),
#                 "eligibilityPeriod": rule_item.get("eligibilityPeriod", "n_a"),
#                 "function": rule_item.get("function", "n_a"),
#                 "operator": rule_item.get("operator", "equal"),
#                 "value": rule_item.get("value", ""),
#                 "connector": rule_item.get("connector"),
#                 "level": level
#             }
    
#     # Start processing from top level
#     visualization = {
#         "rules": [],
#         "topLevelConnector": rules_data.get("topLevelConnector", "AND"),
#         "logical_structure": st.session_state.confirmed_structure or ""
#     }
    
#     for rule in rules_data.get("rules", []):
#         processed_rule = process_rule_item(rule)
#         visualization["rules"].append(processed_rule)
    
#     return visualization

# def display_condition(condition_data: Dict[str, Any], container):
#     """Display a single condition in the UI"""
#     with container:
#         cols = st.columns([2, 2, 1, 1, 2])
        
#         with cols[0]:
#             st.selectbox(
#                 "Data Source",
#                 [condition_data["dataSource"]],
#                 key=f"ds_{condition_data['id']}",
#                 disabled=True,
#                 label_visibility="collapsed"
#             )
        
#         with cols[1]:
#             st.selectbox(
#                 "Field",
#                 [condition_data["field"]],
#                 key=f"field_{condition_data['id']}",
#                 disabled=True,
#                 label_visibility="collapsed"
#             )
        
#         with cols[2]:
#             period_value = condition_data["eligibilityPeriod"] if condition_data["eligibilityPeriod"] != "n_a" else "N/A"
#             st.text_input(
#                 "Period",
#                 value=period_value,
#                 key=f"period_{condition_data['id']}",
#                 disabled=True,
#                 label_visibility="collapsed"
#             )
        
#         with cols[3]:
#             func_map = {
#                 "n_a": "N/A",
#                 "sum": "Sum",
#                 "count": "Count",
#                 "average": "Average",
#                 "max": "Max",
#                 "min": "Min"
#             }
#             func_display = func_map.get(condition_data["function"], condition_data["function"])
#             st.text_input(
#                 "Function",
#                 value=func_display,
#                 key=f"func_{condition_data['id']}",
#                 disabled=True,
#                 label_visibility="collapsed"
#             )
        
#         with cols[4]:
#             op_col, val_col = st.columns(2)
#             with op_col:
#                 op_map = {
#                     "equal": "=",
#                     "greater_than": ">",
#                     "less_than": "<",
#                     "greater_than_or_equal": "≥",
#                     "less_than_or_equal": "≤",
#                     "contains": "contains",
#                     "between": "between"
#                 }
#                 op_display = op_map.get(condition_data["operator"], condition_data["operator"])
#                 st.text_input(
#                     "Operator",
#                     value=op_display,
#                     key=f"op_{condition_data['id']}",
#                     disabled=True,
#                     label_visibility="collapsed"
#                 )
#             with val_col:
#                 st.text_input(
#                     "Value",
#                     value=str(condition_data["value"]),
#                     key=f"val_{condition_data['id']}",
#                     disabled=True,
#                     label_visibility="collapsed"
#                 )

# def display_condition_group(group_data: Dict[str, Any], container, level=0):
#     """Display a condition group in the UI"""
#     with container:
#         indent = "  " * level
#         group_title = f"{indent}Condition Group ({group_data['group_connector']})"
        
#         with st.expander(group_title, expanded=True):
#             for i, condition in enumerate(group_data["conditions"]):
#                 if condition["type"] == "group":
#                     display_condition_group(condition, st.container(), level + 1)
#                 else:
#                     display_condition(condition, st.container())
                
#                 if i < len(group_data["conditions"]) - 1:
#                     st.markdown(f"<div style='margin-left: {20 * (level + 1)}px; color: #666;'><i>{group_data['group_connector']}</i></div>", 
#                               unsafe_allow_html=True)

# def display_rule_visualization(visualization_data: Dict[str, Any]):
#     """Main function to display the rule visualization"""
    
#     st.markdown("---")
#     st.subheader("📋 Generated Rule Structure")
    
#     # Display confirmed logical structure
#     if st.session_state.confirmed_structure:
#         with st.expander("✅ Confirmed Logical Structure", expanded=True):
#             st.success(st.session_state.confirmed_structure)
    
#     # Display top-level rules
#     for i, rule in enumerate(visualization_data["rules"]):
#         if rule["type"] == "group":
#             display_condition_group(rule, st.container())
#         else:
#             display_condition(rule, st.container())
        
#         if i < len(visualization_data["rules"]) - 1:
#             connector = visualization_data.get("topLevelConnector", "AND")
#             st.markdown(f"<div style='color: #666; margin: 10px 0;'><i>{connector}</i></div>", 
#                       unsafe_allow_html=True)
    
#     # Action buttons
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         if st.button("📋 Copy Rule JSON", use_container_width=True, key="copy_json"):
#             rule_json = json.dumps(st.session_state.last_generated_rule, indent=2)
#             st.code(rule_json)
#             st.success("Rule JSON copied to clipboard!")
    
#     with col2:
#         if st.button("📊 Generate Sample Data", use_container_width=True, key="gen_data"):
#             st.session_state.show_synthetic_data = True
#             st.rerun()
    
#     with col3:
#         if st.button("🔄 Start New Rule", use_container_width=True, key="new_rule"):
#             st.session_state.last_generated_rule = None
#             st.session_state.rule_visualization = None
#             st.session_state.show_synthetic_data = False
#             st.rerun()

# # ---------- External API Functions ----------
# def convert_static_to_rich_format(static_data: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
#     rich_format = {}
#     for source_name, fields in static_data.items():
#         rich_fields = []
#         for i, field in enumerate(fields):
#             rich_fields.append({
#                 "field": field,
#                 "field_id": f"static_{i}",
#                 "type": "string",
#                 "description": f"{field} field from {source_name}",
#                 "data_source_id": f"static_{source_name}"
#             })
#         rich_format[source_name] = rich_fields
#     return rich_format

# def fetch_data_sources(client_id: int = None):
#     if client_id is None:
#         client_id = secrets['default_client_id']
    
#     try:
#         headers = {
#             "x-api-key": secrets['external_api_key'],
#             "x-api-secret": secrets['external_api_secret']
#         }
        
#         params = {
#             "clientId": client_id,
#             "page": 1,
#             "limit": 10,
#             "status": "exact:ACTIVE"
#         }
        
#         response = requests.get(
#             f"{secrets['external_api_base_url']}/data-sources",
#             headers=headers,
#             params=params,
#             timeout=10
#         )
        
#         if response.status_code == 200:
#             data = response.json()
#             if data.get("success"):
#                 sources_mapping = {}
                
#                 for source in data.get("data", []):
#                     source_name = source.get("sourceName")
#                     source_id = source.get("id")
#                     mapping_data = source.get("mapping", {}).get("mappingData", {})
#                     mapping_list = mapping_data.get("mapping", [])
                    
#                     if source_name and mapping_list:
#                         fields = []
#                         for mapping_item in mapping_list:
#                             mapped_field = mapping_item.get("mappedField")
#                             field_id = mapping_item.get("id")
#                             field_type = mapping_item.get("mappingType", "string")
#                             field_description = mapping_item.get("description", "")
                            
#                             if mapped_field:
#                                 fields.append({
#                                     "field": mapped_field,
#                                     "field_id": field_id,
#                                     "type": field_type,
#                                     "description": field_description or f"{mapped_field} field",
#                                     "data_source_id": source_id
#                                 })
                        
#                         if fields:
#                             sources_mapping[source_name] = fields
                
#                 if sources_mapping:
#                     return sources_mapping
        
#         # Fallback to static data
#         CSV_STRUCTURES = {
#             "sample_mortgage_accounts.csv": [
#                 "customer_id", "product_type", "account_status", "loan_open_date", "loan_balance"
#             ],
#             "sample_loan_repayments.csv": [
#                 "repayment_id", "customer_id", "loan_account_number", "repayment_date",
#                 "repayment_amount", "installment_number", "payment_method_status", "loan_type",
#                 "interest_component", "principal_component", "remaining_balance"
#             ],
#             "sample_telco_billing.csv": [
#                 "billing_id", "customer_id", "bill_date", "bill_amount", "plan_type",
#                 "data_used_gb", "voice_minutes", "sms_count", "channel"
#             ],
#             "sample_product_enrollments.csv": [
#                 "enrollment_id", "customer_id", "product_type", "product_name", "enrollment_date", "status"
#             ],
#             "sample_customer_profiles.csv": [
#                 "customer_id", "name", "email", "phone", "dob", "gender",
#                 "region", "segment", "household_id", "is_primary"
#             ],
#             "sample_savings_account_transactions.csv": [
#                 "transaction_id", "account_id", "customer_id", "amount", "date", "transaction_type"
#             ],
#             "sample_credit_card_transactions.csv": [
#                 "customer_id", "card_number", "transaction_date", "transaction_amount", "transaction_type"
#             ]
#         }
        
#         return convert_static_to_rich_format(CSV_STRUCTURES)
        
#     except Exception as e:
#         st.error(f"Error fetching data sources: {e}")
#         return convert_static_to_rich_format(CSV_STRUCTURES)

# def create_field_mapping(data_sources: Dict[str, List[Dict[str, str]]]) -> Dict[str, str]:
#     field_mapping = {}
#     for source_name, fields in data_sources.items():
#         for field_info in fields:
#             field_name = field_info["field"]
#             field_id = field_info["field_id"]
#             field_mapping[field_name] = field_id
#     return field_mapping

# def detect_response_type(parsed_json: Dict[str, Any]) -> str:
#     if "rules" in parsed_json:
#         return "rule"
#     elif "logical_structure" in parsed_json and "user_message" in parsed_json:
#         return "confirmation"
#     elif "message" in parsed_json and len(parsed_json) == 1:
#         return "general"
#     else:
#         return "unknown"

# def generate_rule_with_openai(user_input: str, client_id: Optional[int] = None, context: str = ""):
#     """Generate rule using OpenAI with optional context"""
    
#     # Fetch data sources
#     data_sources = fetch_data_sources(client_id or secrets['default_client_id'])
    
#     # Create field mapping
#     field_mapping = create_field_mapping(data_sources)
    
#     # Format available data
#     available_data_lines = []
#     for source_name, fields in data_sources.items():
#         data_source_id = fields[0].get("data_source_id", "N/A") if fields else "N/A"
#         available_data_lines.append(f"- {source_name} (ID: {data_source_id}):")
#         for field_info in fields:
#             field_name = field_info["field"]
#             field_id = field_info["field_id"]
#             field_type = field_info["type"]
#             field_desc = field_info["description"]
#             available_data_lines.append(f"  * {field_name} (ID: {field_id}, {field_type}): {field_desc}")
    
#     available_data = "\n".join(available_data_lines)
    
#     # Create field mapping string
#     field_mapping_lines = []
#     for field_name, field_id in field_mapping.items():
#         field_mapping_lines.append(f"  * {field_name} -> {field_id}")
#     field_mapping_str = "\n".join(field_mapping_lines)
    
#     current_date = datetime.today().strftime("%Y-%m-%d")
    
#     # Add context if provided (for re-generating with confirmed structure)
#     full_prompt = user_input
#     if context:
#         full_prompt = f"{user_input}\n\nContext: {context}"
    
#     system_prompt = f"""You are a rule generation assistant. Create rules based on this confirmed logical structure:

#         You mostly generate 3 things. 
#         1. Logical structures
#         2. Rule
#         3. General message

#         ### Communication Style:
#         1. Be polite, patient, and professional
#         2. Use clear, simple language (avoid jargon unless necessary)
#         3. Maintain a helpful and approachable tone
#         4. Confirm understanding before proceeding
#         5. Provide explanations when asked

#         ### Core Responsibilities - Rule Generation:
#         # Logical Structure Handling
#         1. When the request contains ONLY AND operators or ONLY OR operators:
#            - Generate the single correct logical structure without giving options to select.
#            - Present it to the user for confirmation
#            - Example: "(A AND B AND C)" or "(X OR Y OR Z)"

#         2. When the request contains a MIX of AND/OR operators:
#            - MUST propose ALL possible logical structures (typically 3 options)
#            - Present them as clearly numbered options (Option 1, Option 2, Option 3)
#            - Example formats:
#                Option 1: (A AND B) OR C
#                Option 2: A AND (B OR C)
#                Option 3: (A OR B) AND C
#            - Ask user to confirm which option matches their intent


#         For before generate the rule you will be given a logical structure(s) and get the confirmation of the logical structure. (YES or NO)

#         ### Your Flow before generating the rule:
#         1. Before generating the rule, you will be given a logical structure.
#         2. and you have to ask confirmation of the logical structure with Yes or No. simple question
#         3. if the user says yes, then you will generate the rule.
#         4. if the user says no, then you will suggest another boolean logical structure for it.
#         5. after confirming you will generate the rule

#         ### While generating the rule dont add any explanation just generate the rule with ONLY JSON output.

#         OUTPUT confirmation message format When the request contains ONLY AND operators or ONLY OR operators
#         {{
#             "message": "Logical structure confirmed",
#             "logical_structure": "(Customer spends over $2500 on a credit card in a month) AND (has an active mortgage) AND (loan balance is more than $1000)",
#             "user_message": "Do you agree with this structure, please suggest your requirement (agree or suggest another structure)"
#         }}       

#         OUTPUT confirmation message format When the request contains a MIX of AND/OR operators:
#         {{
#             "message": "Logical structure confirmed",
#             "logical_structure": "Option 1: (Customer spends over $2500 on a credit card in a month) OR (has an active mortgage AND loan balance is more than $1000),
#                 Option 2: Customer spends over $2500 on a credit card in a month OR (has an active mortgage AND loan balance is more than $1000),
#                 Option 3: (Customer spends over $2500 on a credit card in a month OR has an active mortgage) AND loan balance is more than $1000",
#             "user_message": "Do you agree with any of this structure, please suggest your requirement (agree or suggest another structure)"
#         }}

#         OUTPUT general message format
#         {{
#             "message": <general response message for other messages and any general message>,
#         }}  

        
#         If it is (A AND B) OR C, Output JSON matching this type of schema: PLease change the position of condintional groups according to th brackets.
#         {{
#     "rules": [
#         {{
#                     "id": <id>,
#                     "priority": null,
#                     "ruleType": "conditionGroup",
#                     "conditions": [
#                         {{
#                             "id": <id>,
#                             "dataSource": <data source name>,
#                             "dataSourceId": <data source id>,
#                             "field": <field name>,
#                             "fieldId": <field id>,
#                             "eligibilityPeriod": "Rolling 30 days",
#                             "function": <function> example: "sum", "count", "average", "max" etc,
#                             "operator": <operator> example: "greater_than", "equal", "less_than" etc,
#                             "value": "2500",
#                             "connector": "AND"
#                         }},
#                         {{
#                             "id": <id>,
#                             "dataSource": <data source name>,
#                             "dataSourceId": <data source id>,
#                             "field": <field name>,
#                             "fieldId": <field id>,
#                             "eligibilityPeriod": "n_a",
#                             "function": "n_a",
#                             "operator": <operator> example: "greater_than", "equal", "less_than" etc,
#                             "value": "1000"
#                         }}
#                     ],
#                     "connector": "OR"
#                 }},
#                 {{
#                     "id": <id>,
#                     "dataSource": <data source name>,
#                     "dataSourceId": <data source id>,
#                     "field": <field name>,
#                     "fieldId": <field id>,
#                     "eligibilityPeriod": "n_a",
#                     "function": "n_a",
#                     "operator": <operator> example: "greater_than", "equal", "less_than" etc,
#                     "value": "active",
#                     "priority": null,
#                     "ruleType": "condition"
#                 }}
#             ]
#         }}
#         If it is A AND (B OR C), Output JSON matching this type of schema:
#         {{
#             "rules": [
#                 {{
#                     "id": <id>,
#                     "dataSource": <data source name>,
#                     "dataSourceId": <data source id>,
#                     "field": <field name>,
#                     "fieldId": <field id>,
#                     "eligibilityPeriod": "Rolling 30 days",
#                     "function": <function> example: "sum", "count", "average", "max" etc,
#                     "operator": <operator> example: "greater_than", "equal", "less_than" etc,
#                     "value": "2500",
#                     "priority": null,
#                     "ruleType": "condition",
#                     "connector": "AND"
#                 }},
#                 {{
#                     "id": <id>,
#                     "priority": null,
#                     "ruleType": "conditionGroup",
#                     "conditions": [
#                         {{
#                             "id": <id>,
#                             "dataSource": <data source name>,
#                             "dataSourceId": <data source id>,
#                             "field": <field name>,
#                             "fieldId": <field id>,
#                             "eligibilityPeriod": "n_a",
#                             "function": "n_a",
#                             "operator": <operator> example: "greater_than", "equal", "less_than" etc,
#                             "value": "active",
#                             "connector": "OR"
#                         }},
#                         {{
#                             "id": <id>,
#                             "dataSource": <data source name>,
#                             "dataSourceId": <data source id>,
#                             "field": <field name>,
#                             "fieldId": <field id>,
#                             "eligibilityPeriod": "n_a",
#                             "function": "n_a",
#                             "operator": <operator> example: "greater_than", "equal", "less_than" etc,
#                             "value": "1000"
#                         }}
#                     ]
#                 }}
#             ]
#         }}



        
#         CRITICAL INSTRUCTIONS:
#         1. Use ONLY the exact column names from these data sources
#         2. Use the exact fieldId from the field mapping above for each field (e.g., if using "customer_id", use the corresponding fieldId from the mapping)
#         3. Use the exact dataSourceId from the available data sources for each data source
#         4. Use ONLY the operator VALUES (e.g., ">" -> "greater_than", "=" -> "equal", "contains" -> "contains") not the labels
#         5. Use ONLY the function VALUES (e.g., "sum" -> "sum", "count" -> "count", "N/A" -> "n_a") not the labels
#         6. Follow the logical structure EXACTLY as provided
#         7.When generating eligibilityPeriod:
#         If the condition explicitly mentions a **date range** (e极, "between February and March", "from 2025-02-01 to 2025-03-31", "between 9 AM and 6 PM"), set eligibilityPeriod = 'n_a'.
#         If the condition mentions 'last month" or "up to this month', set eligibilityPeriod to "Last X days" (e极, "Last 30 days").
#         If the condition mentions 'every month', set eligibilityPeriod to "Rolling X days".
#         Never swap these values.
#         Replace X with the configured number of days."
#         8. For amount aggregations, use "sum" function

#         Respond ONLY with the JSON output.

#         available data sources:
#         {available_data}

#         Field Mapping:
#         {field_mapping_str}

#         While using operators, use the value, not the label:
#         Don't use operators not mentioned below. strictly bind with this operators
#             const operators = [
#           {{ value: 'equal', label: '=' }},
#           {{ value: 'not_equal', label: '≠' }},
#           {{ value: 'greater_than', label: '>' }},
#           {{ value: 'less_than', label: '<' }},
#           {{ value: 'greater_than_or_equal', label: '≥' }},
#           {{ value: 'less_than_or_equal', label: '≤' }},
#           {{ value: 'between', label: 'Between' }},
#           {{ value: 'not_between', label: 'Not Between' }},
#           {{ value: 'contains', label: 'Contains' }},
#           {{ value: 'begins_with', label: 'Begins With' }},
#           {{ value: 'ends_with', label: 'Ends With' }},
#           {{ value: 'does_not_contain', label: 'Does Not Contain' }}
#         ];
        
#         Whenever using operators "=" is "equal"

#         While using functions, use the value, not the label:
#         Don't use functions not mentioned below. strictly bind with this functions
        
#         const functions = [
#           {{ value: 'n_a', label: 'N/A' }},
#           {{ value: 'sum', label: 'Sum' }},
#           {{ value: 'count', label: 'Count' }},
#           {{ value: 'average', label: 'Average' }},
#           {{ value: 'max', label: 'Maximum' }},
#           {{ value: 'min', label: 'Minimum' }},
#           {{ value: 'exact_match', label: 'Exact Match' }},
#           {{ value: 'change_detected', label: 'Change Detected' }},
#           {{ value: 'exists', label: 'Exists' }},
#           {{ value: 'consecutive', label: 'Consecutive' }},
#           {{ value: 'streak_count', label: 'Streak Count' }},
#           {{ value: 'first_time', label: 'First Time' }},
#           {{ value: 'nth_time', label: 'Nth Time' }},
#           {{ value: 'recent_change', label: 'Recent Change' }}
#         ];

#         # Strictly use the value, not the label.
#         # Don't use operators and functions not mentioned above. strictly bind with this operators
        
#         How LOGICAL structure:
#         Input: "A and B or C"
#         Output:
#         1. (A AND B) OR C
#         2. A AND (B OR C)

#         ### One-shot example of Logical structure:

#         user_input:
#         User spends over $2500 on a credit card in a month OR has an active mortgage AND loan balance is more than $1000

#         Logical structure for this user_input (2 variations right):
#         (User spends over $2500 on a credit card in a month) OR (has an active mortgage AND loan balance is more than $1000)
#         User spends over $2500 on a credit card in a month OR (has an active mortgage AND loan balance is more than $1000)
#         """
    
#     try:
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": full_prompt}
#         ]
        
#         response = client.chat.completions.create(
#             model="gpt-4o-2024-08-06",
#             messages=messages,
#             temperature=0.3,  # Lower temperature for more consistent structure
#             response_format={"type": "json_object"}
#         )
        
#         response_content = response.choices[0].message.content
#         json_response = json.loads(response_content)
        
#         # Post-process to add fieldId if missing
#         if "rules" in json_response:
#             def add_field_ids(rules):
#                 for rule in rules:
#                     if rule.get("ruleType") == "condition":
#                         if "field" in rule and "fieldId" not in rule:
#                             field_name = rule["field"]
#                             if field_name in field_mapping:
#                                 rule["fieldId"] = field_mapping[field_name]
#                         if "dataSource" in rule and "dataSourceId" not in rule:
#                             data_source_name = rule["dataSource"]
#                             for source_name, fields in data_sources.items():
#                                 if source_name == data_source_name and fields:
#                                     rule["dataSourceId"] = fields[0].get("data_source_id", "N/A")
#                                     break
#                     elif rule.get("ruleType") == "conditionGroup":
#                         if "conditions" in rule:
#                             add_field_ids(rule["conditions"])
            
#             add_field_ids(json_response["rules"])
        
#         return json_response
        
#     except Exception as e:
#         st.error(f"Error generating rule with OpenAI: {str(e)}")
#         return {"message": f"Error: {str(e)}"}

# def generate_confirmed_rule():
#     """Generate rule based on confirmed logical structure"""
#     with st.spinner("Generating rule based on confirmed structure..."):
#         try:
#             last_user_message = extract_last_user_message()
#             if not last_user_message:
#                 st.error("No user message found")
#                 return
            
#             # Add the confirmed structure as context
#             context = f"Confirmed logical structure: {st.session_state.confirmed_structure}"
#             result = generate_rule_with_openai(last_user_message, st.session_state.client_id, context)
            
#             if "rules" in result:
#                 st.session_state.last_generated_rule = result
#                 st.session_state.rule_visualization = create_rule_visualization(result)
                
#                 # Add success message to chat
#                 success_msg = {
#                     "message": f"✅ Rule generated successfully based on your selected structure!",
#                     "rules_generated": True
#                 }
#                 st.session_state.chat_history.append({
#                     "role": "assistant", 
#                     "content": json.dumps(success_msg, indent=2)
#                 })
#             else:
#                 error_msg = {
#                     "message": "Failed to generate rule. Please try again with a clearer description."
#                 }
#                 st.session_state.chat_history.append({
#                     "role": "assistant", 
#                     "content": json.dumps(error_msg, indent=2)
#                 })
        
#         except Exception as e:
#             error_msg = {
#                 "message": f"Error generating rule: {str(e)}"
#             }
#             st.session_state.chat_history.append({
#                 "role": "assistant", 
#                 "content": json.dumps(error_msg, indent=2)
#             })

# # ---------- UI Components ----------
# def display_sidebar():
#     with st.sidebar:
#         st.title("⚙️ Configuration")
        
#         st.session_state.client_id = st.number_input(
#             "Client ID",
#             value=st.session_state.client_id,
#             min_value=1,
#             help="Enter the client ID for data source retrieval"
#         )
        
#         st.subheader("Data Sources")
#         if st.button("🔄 Refresh Data Sources", use_container_width=True):
#             with st.spinner("Fetching data sources..."):
#                 st.session_state.data_sources = fetch_data_sources(st.session_state.client_id)
        
#         if st.session_state.data_sources:
#             st.success(f"✅ Loaded {len(st.session_state.data_sources)} data sources")
#             with st.expander("View Data Sources"):
#                 for source_name, fields in st.session_state.data_sources.items():
#                     st.write(f"**{source_name}** ({len(fields)} fields)")
#                     for field in fields[:3]:
#                         st.write(f"  - {field['field']}")
#                     if len(fields) > 3:
#                         st.write(f"  ... and {len(fields) - 3} more")
#         else:
#             st.info("Click 'Refresh Data Sources' to load available data sources")
        
#         st.subheader("Chat Management")
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("🗑️ Clear Chat", use_container_width=True):
#                 st.session_state.chat_history = []
#                 st.session_state.last_generated_rule = None
#                 st.session_state.synthetic_data = None
#                 st.session_state.show_synthetic_data = False
#                 st.session_state.confirmed_structure = None
#                 st.session_state.rule_visualization = None
#                 st.session_state.awaiting_confirmation = False
#                 st.session_state.logical_options = []
#                 st.session_state.selected_option = None
#                 st.session_state.show_structure_options = False
#                 st.rerun()
        
#         with col2:
#             if st.button("📥 Export Chat", use_container_width=True):
#                 export_chat()

# def export_chat():
#     chat_data = {
#         "client_id": st.session_state.client_id,
#         "timestamp": datetime.now().isoformat(),
#         "chat_history": st.session_state.chat_history,
#         "last_generated_rule": st.session_state.last_generated_rule,
#         "confirmed_structure": st.session_state.confirmed_structure
#     }
    
#     json_str = json.dumps(chat_data, indent=2)
    
#     st.download_button(
#         label="Download Chat JSON",
#         data=json_str,
#         file_name=f"chat_history_{st.session_state.client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#         mime="application/json"
#     )

# def display_structure_options():
#     """Display multiple logical structure options for user to choose"""
#     st.markdown("---")
#     st.subheader("🤔 Choose a Logical Structure")
#     st.info("Please select which logical structure matches your intent:")
    
#     # Display each option with radio button
#     selected_index = st.radio(
#         "Select an option:",
#         range(len(st.session_state.logical_options)),
#         format_func=lambda i: st.session_state.logical_options[i]
#     )
    
#     st.session_state.selected_option = selected_index
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if st.button("✅ Confirm Selection", use_container_width=True, type="primary"):
#             selected_structure = st.session_state.logical_options[selected_index]
#             st.session_state.confirmed_structure = selected_structure
#             st.session_state.show_structure_options = False
#             st.session_state.awaiting_confirmation = False
            
#             # Add confirmation to chat
#             confirmation_msg = {
#                 "message": f"Confirmed: {selected_structure}",
#                 "selected_option": selected_index + 1
#             }
#             st.session_state.chat_history.append({
#                 "role": "assistant", 
#                 "content": json.dumps(confirmation_msg, indent=2)
#             })
            
#             # Generate the rule
#             generate_confirmed_rule()
#             st.rerun()
    
#     with col2:
#         if st.button("🔄 Suggest Other Options", use_container_width=True):
#             # Clear current options and ask for new ones
#             st.session_state.logical_options = []
#             st.session_state.selected_option = None
#             st.session_state.show_structure_options = False
            
#             # Ask user to suggest different structure
#             suggestion_msg = {
#                 "message": "Please suggest a different logical structure for your rule.",
#                 "request": "suggest_different_structure"
#             }
#             st.session_state.chat_history.append({
#                 "role": "assistant", 
#                 "content": json.dumps(suggestion_msg, indent=2)
#             })
#             st.rerun()

# def display_single_structure_confirmation():
#     """Display single logical structure for confirmation"""
#     st.markdown("---")
#     st.subheader("✅ Confirm Logical Structure")
    
#     if st.session_state.logical_options:
#         logical_structure = st.session_state.logical_options[0]
#         st.info(f"**Proposed Logical Structure:**\n\n{logical_structure}")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             if st.button("✅ Yes, Generate Rule", use_container_width=True, type="primary"):
#                 st.session_state.confirmed_structure = logical_structure
#                 st.session_state.awaiting_confirmation = False
                
#                 # Add confirmation to chat
#                 confirmation_msg = {
#                     "message": f"Confirmed: {logical_structure}",
#                     "status": "confirmed"
#                 }
#                 st.session_state.chat_history.append({
#                     "role": "assistant", 
#                     "content": json.dumps(confirmation_msg, indent=2)
#                 })
                
#                 generate_confirmed_rule()
#                 st.rerun()
        
#         with col2:
#             if st.button("🔄 Suggest Different Structure", use_container_width=True):
#                 st.session_state.awaiting_confirmation = False
#                 st.session_state.logical_options = []
                
#                 # Ask for different structure
#                 suggestion_msg = {
#                     "message": "Please suggest a different logical structure for your rule.",
#                     "request": "suggest_different_structure"
#                 }
#                 st.session_state.chat_history.append({
#                     "role": "assistant", 
#                     "content": json.dumps(suggestion_msg, indent=2)
#                 })
#                 st.rerun()

# def display_chat_history():
#     st.title("🏦 Mortgage Rule Generator")
#     st.subheader("Natural Language to Rule Conversion")
    
#     # Display chat messages
#     chat_container = st.container()
    
#     with chat_container:
#         for message in st.session_state.chat_history:
#             if message["role"] == "user":
#                 with st.chat_message("user"):
#                     st.write(message["content"])
#             else:
#                 with st.chat_message("assistant"):
#                     try:
#                         content_data = json.loads(message["content"])
#                         if "message" in content_data:
#                             st.write(content_data["message"])
#                             if "logical_structure" in content_data:
#                                 st.info(f"**Logical Structure:**\n{content_data['logical_structure']}")
#                         else:
#                             st.json(content_data)
#                     except:
#                         st.write(message["content"])
    
#     # If showing structure options
#     if st.session_state.show_structure_options and st.session_state.logical_options:
#         display_structure_options()
#         return
    
#     # If awaiting single structure confirmation
#     if st.session_state.awaiting_confirmation and st.session_state.logical_options:
#         display_single_structure_confirmation()
#         return
    
#     # Chat input (only show if not in confirmation mode)
#     user_input = st.chat_input("Describe your rule in natural language...")
    
#     if user_input:
#         # Add user message to chat
#         st.session_state.chat_history.append({"role": "user", "content": user_input})
        
#         # Check if user is responding to structure request
#         last_assistant_msg = None
#         for msg in reversed(st.session_state.chat_history):
#             if msg["role"] == "assistant":
#                 last_assistant_msg = msg["content"]
#                 break
        
#         if last_assistant_msg and "suggest_different_structure" in last_assistant_msg:
#             # User is suggesting a different structure
#             with st.spinner("Analyzing your suggested structure..."):
#                 result = generate_rule_with_openai(user_input, st.session_state.client_id)
#                 handle_ai_response(result)
#         else:
#             # New rule request
#             with st.spinner("Processing your rule request..."):
#                 if len(st.session_state.data_sources) == 0:
#                     st.session_state.data_sources = fetch_data_sources(st.session_state.client_id)
                
#                 result = generate_rule_with_openai(user_input, st.session_state.client_id)
#                 handle_ai_response(result)
        
#         st.rerun()

# def handle_ai_response(result):
#     """Handle AI response and update session state accordingly"""
#     response_type = detect_response_type(result)
    
#     if response_type == "rule":
#         st.session_state.last_generated_rule = result
#         st.session_state.rule_visualization = create_rule_visualization(result)
        
#         response_content = json.dumps({
#             "message": "✅ Rule generated successfully!",
#             "rules_available": True
#         }, indent=2)
        
#         st.session_state.chat_history.append({"role": "assistant", "content": response_content})
    
#     elif response_type == "confirmation":
#         logical_structure = result.get("logical_structure", "")
#         user_message = result.get("user_message", "")
        
#         # Parse the logical structure to get options
#         options = parse_logical_options(logical_structure)
        
#         if len(options) > 1:
#             # Multiple options - show selection UI
#             st.session_state.logical_options = options
#             st.session_state.show_structure_options = True
            
#             # Store the AI response in chat
#             ai_response = {
#                 "message": "I've identified multiple possible logical structures for your rule.",
#                 "logical_structure": logical_structure,
#                 "user_message": "Please select which structure matches your intent."
#             }
#             st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(ai_response, indent=2)})
        
#         else:
#             # Single option - show confirmation
#             st.session_state.logical_options = options
#             st.session_state.awaiting_confirmation = True
            
#             ai_response = {
#                 "message": "I've identified a logical structure for your rule.",
#                 "logical_structure": logical_structure,
#                 "user_message": user_message or "Do you agree with this structure?"
#             }
#             st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(ai_response, indent=2)})
    
#     elif response_type == "general":
#         st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(result, indent=2)})
    
#     else:
#         st.session_state.chat_history.append({
#             "role": "assistant", 
#             "content": json.dumps({"message": "I'm not sure how to process that. Please try rephrasing your rule description."}, indent=2)
#         })

# def generate_synthetic_dataset_with_openai(rules: List[Dict[str, Any]], data_sources: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
#     # Extract exact fields mentioned in the rules
#     rule_fields = set()
    
#     def extract_fields_from_rules(rules_list):
#         for rule in rules_list:
#             if rule.get("ruleType") == "condition":
#                 field = rule.get("field")
#                 if field:
#                     rule_fields.add(field)
#             elif rule.get("ruleType") == "conditionGroup" and "conditions" in rule:
#                 extract_fields_from_rules(rule["conditions"])
    
#     extract_fields_from_rules(rules)
    
#     current_date = datetime.today().strftime("%Y-%m-%d")
    
#     system_prompt = f"""Generate synthetic data for testing a rule. Include ONLY fields mentioned in the rule.
#     Today's date: {current_date}
    
#     Generate 10 records:
#     - Include customer_id and matches_rule fields
#     - Include ONLY the rule fields: {list(rule_fields)}
#     - 5 records should match ALL conditions (matches_rule: true)
#     - 5 records should NOT match (matches_rule: false)
    
#     Return JSON format:
#     {{
#       "synthetic_dataset": [
#         {{
#           "customer_id": "CUST001",
#           "field1": "value1",
#           "matches_rule": true
#         }}
#       ]
#     }}
#     """
    
#     user_prompt = f"Generate synthetic data for this rule:\n{json.dumps(rules, indent=2)}"
    
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-2024-08-06",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=0.7,
#             response_format={"type": "json_object"}
#         )
        
#         response_content = response.choices[0].message.content
#         synthetic_data = json.loads(response_content)
        
#         if "synthetic_dataset" in synthetic_data:
#             return synthetic_data
#         else:
#             return generate_fallback_rule_dataset(rules, rule_fields)
        
#     except Exception as e:
#         st.error(f"Error generating synthetic data: {e}")
#         return generate_fallback_rule_dataset(rules, rule_fields)

# def generate_fallback_rule_dataset(rules: List[Dict[str, Any]], rule_fields: set) -> Dict[str, Any]:
#     dataset = {
#         "synthetic_dataset": []
#     }
    
#     for i in range(10):
#         customer_id = f"CUST{100 + i}"
#         matches_rule = i < 5
        
#         record = {
#             "customer_id": customer_id,
#             "matches_rule": matches_rule
#         }
        
#         for field in rule_fields:
#             if field == "applicationDate":
#                 if matches_rule:
#                     record[field] = (datetime.now() - timedelta(days=np.random.randint(100, 365))).isoformat()[:10]
#                 else:
#                     record[field] = (datetime.now() - timedelta(days=np.random.randint(366, 730))).isoformat()[:10]
#             elif field == "pricePaid" or "price" in field.lower():
#                 if matches_rule:
#                     record[field] = round(np.random.uniform(1001, 5000), 2)
#                 else:
#                     record[field] = round(np.random.uniform(100, 999), 2)
#             elif field == "purchaseAmount":
#                 if matches_rule:
#                     record[field] = round(np.random.uniform(1501, 3000), 2)
#                 else:
#                     record[field] = round(np.random.uniform(500, 1499), 2)
#             elif field == "lastRepaymentDate":
#                 if matches_rule:
#                     record[field] = (datetime.now() - timedelta(days=np.random.randint(1, 59))).isoformat()[:10]
#                 else:
#                     record[field] = (datetime.now() - timedelta(days=np.random.randint(60, 120))).isoformat()[:10]
#             else:
#                 record[field] = f"value_{np.random.randint(1, 100)}"
        
#         dataset["synthetic_dataset"].append(record)
    
#     return dataset

# def display_synthetic_data():
#     if st.session_state.synthetic_data and st.session_state.show_synthetic_data:
#         with st.expander("📊 Synthetic Data Preview", expanded=True):
#             data = st.session_state.synthetic_data["synthetic_dataset"]
#             df = pd.DataFrame(data)
            
#             total_records = len(df)
#             matching_records = df[df["matches_rule"]].shape[0]
#             match_percentage = (matching_records / total_records * 100) if total_records > 0 else 0
            
#             col1, col2, col3 = st.columns(3)
#             col1.metric("Total Records", total_records)
#             col2.metric("Matching Records", matching_records)
#             col3.metric("Match Rate", f"{match_percentage:.1f}%")
            
#             st.dataframe(df, use_container_width=True)
            
#             csv = df.to_csv(index=False)
#             st.download_button(
#                 label="📥 Download as CSV",
#                 data=csv,
#                 file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                 mime="text/csv",
#                 use_container_width=True
#             )

# def main():
#     # Custom CSS
#     st.markdown("""
#     <style>
#     .stChatMessage {
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-bottom: 1rem;
#     }
#     .stChatMessage[data-testid="user"] {
#         background-color: #f0f2f6;
#     }
#     .stChatMessage[data-testid="assistant"] {
#         background-color: #e6f7ff;
#     }
#     div[data-testid="stExpander"] div[role="button"] p {
#         font-size: 1.1rem;
#         font-weight: 600;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     # Display sidebar
#     display_sidebar()
    
#     # Main content area
#     col1, col2 = st.columns([3, 1])
    
#     with col1:
#         display_chat_history()
    
#     with col2:
#         st.info("💡 **Tips:**\n\n"
#                 "1. Be specific with conditions\n"
#                 "2. Mention AND/OR logic clearly\n"
#                 "3. Example: 'Customers who spent >$1000 AND have active mortgage'\n"
#                 "4. You'll get to confirm the structure before rule generation")
    
#     # Display rule visualization if available
#     if st.session_state.rule_visualization and st.session_state.last_generated_rule:
#         display_rule_visualization(st.session_state.rule_visualization)
        
#         # Show synthetic data generation button
#         if not st.session_state.show_synthetic_data:
#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.button("📊 Generate Sample Data", use_container_width=True, type="secondary", key="gen_sample"):
#                     with st.spinner("Generating synthetic data..."):
#                         data_sources = fetch_data_sources(st.session_state.client_id)
#                         synthetic_data = generate_synthetic_dataset_with_openai(
#                             st.session_state.last_generated_rule["rules"], 
#                             data_sources
#                         )
#                         st.session_state.synthetic_data = synthetic_data
#                         st.session_state.show_synthetic_data = True
#                         st.rerun()
    
#     # Display synthetic data if available
#     display_synthetic_data()

# if __name__ == "__main__":
#     if not secrets['openai_api_key']:
#         st.error("OpenAI API key not found in secrets. Please add it to .streamlit/secrets.toml")
#         st.stop()
    
#     if not secrets['external_api_key'] or not secrets['external_api_secret']:
#         st.warning("External API credentials not found. Using static data sources only.")
    
#     main()












import streamlit as st
import requests
import json
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import openai
import re

# Page configuration
st.set_page_config(
    page_title="Mortgage Rule Generator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'client_id' not in st.session_state:
    st.session_state.client_id = 307
if 'data_sources' not in st.session_state:
    st.session_state.data_sources = {}
if 'last_generated_rule' not in st.session_state:
    st.session_state.last_generated_rule = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'show_synthetic_data' not in st.session_state:
    st.session_state.show_synthetic_data = False
if 'confirmed_structure' not in st.session_state:
    st.session_state.confirmed_structure = None
if 'rule_visualization' not in st.session_state:
    st.session_state.rule_visualization = None
if 'awaiting_confirmation' not in st.session_state:
    st.session_state.awaiting_confirmation = False
if 'logical_options' not in st.session_state:
    st.session_state.logical_options = []
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None
if 'show_structure_options' not in st.session_state:
    st.session_state.show_structure_options = False

# Get secrets from Streamlit secrets
def get_secrets():
    secrets = {
        'external_api_base_url': st.secrets.get("EXTERNAL_API_BASE_URL", "https://lmsdev-external-distributor-api.pulseid.com"),
        'external_api_key': st.secrets.get("EXTERNAL_API_KEY", ""),
        'external_api_secret': st.secrets.get("EXTERNAL_API_SECRET", ""),
        'default_client_id': st.secrets.get("DEFAULT_CLIENT_ID", 307),
        'openai_api_key': st.secrets.get("OPENAI_API_KEY", "")
    }
    return secrets

secrets = get_secrets()

# Initialize OpenAI client
try:
    client = openai.OpenAI(api_key=secrets['openai_api_key'])
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {str(e)}")
    client = None

# ---------- Helper Functions ----------
def parse_logical_options(logical_structure: str) -> List[str]:
    """Parse logical structure string to extract multiple options"""
    options = []
    
    # Check if it's multiple options format
    if "Option 1:" in logical_structure:
        # Split by "Option X:" pattern
        pattern = r'Option \d+:'
        parts = re.split(pattern, logical_structure)
        
        for i, part in enumerate(parts[1:], 1):
            option_text = f"Option {i}: {part.strip()}"
            option_text = option_text.replace('\\n', ' ').replace('\n', ' ').strip()
            option_text = re.sub(r'\s*(Do you agree with any of this structure.*)', '', option_text)
            option_text = re.sub(r'\s*(Please select.*)', '', option_text)
            if option_text and len(option_text) > 10:
                options.append(option_text)
    
    # If not in Option X format, check for numbered options
    elif "1." in logical_structure or "2." in logical_structure or "3." in logical_structure:
        lines = logical_structure.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+[\.\)]', line):
                clean_line = re.sub(r'\s*(Do you agree with any of this structure.*)', '', line)
                clean_line = re.sub(r'\s*(Please select.*)', '', clean_line)
                if clean_line and len(clean_line) > 10:
                    options.append(clean_line)
    
    # If single structure
    else:
        clean_structure = re.sub(r'\s*(Do you agree with this structure.*)', '', logical_structure)
        clean_structure = re.sub(r'\s*(Are you agree with.*)', '', clean_structure)
        clean_structure = clean_structure.strip()
        if clean_structure:
            options = [clean_structure]
    
    # Return options, or if empty, return the original structure
    return options if options else [logical_structure]


def extract_last_user_message():
    """Extract the last user message from chat history"""
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "user":
            return msg["content"]
    return None

# ---------- Visualization Functions ----------
# Update the create_rule_visualization function to ensure unique IDs
def create_rule_visualization(rules_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a structured visualization of the rule for display"""
    
    def process_rule_item(rule_item, level=0):
        """Recursively process rule items for visualization"""
        item_type = rule_item.get("ruleType", "condition")
        
        # Ensure unique ID
        if "id" not in rule_item:
            rule_item["id"] = str(uuid.uuid4())
        
        if item_type == "conditionGroup":
            # Process condition group
            group_data = {
                "type": "group",
                "id": rule_item["id"],
                "group_connector": rule_item.get("groupConnector", "AND"),
                "conditions": [],
                "connector": rule_item.get("connector"),
                "level": level
            }
            
            # Process nested conditions
            for condition in rule_item.get("conditions", []):
                processed_condition = process_rule_item(condition, level + 1)
                group_data["conditions"].append(processed_condition)
            
            return group_data
        
        else:  # Individual condition
            return {
                "type": "condition",
                "id": rule_item["id"],
                "dataSource": rule_item.get("dataSource", "Unknown"),
                "dataSourceId": rule_item.get("dataSourceId", "N/A"),
                "field": rule_item.get("field", "Unknown"),
                "fieldId": rule_item.get("fieldId", "N/A"),
                "eligibilityPeriod": rule_item.get("eligibilityPeriod", "n_a"),
                "function": rule_item.get("function", "n_a"),
                "operator": rule_item.get("operator", "equal"),
                "value": rule_item.get("value", ""),
                "connector": rule_item.get("connector"),
                "level": level
            }
    
    # Start processing from top level
    visualization = {
        "rules": [],
        "topLevelConnector": rules_data.get("topLevelConnector", "AND"),
        "logical_structure": st.session_state.confirmed_structure or ""
    }
    
    for i, rule in enumerate(rules_data.get("rules", [])):
        processed_rule = process_rule_item(rule)
        visualization["rules"].append(processed_rule)
    
    return visualization

def display_condition(condition_data: Dict[str, Any], container, unique_key: str):
    """Display a single condition in the UI"""
    with container:
        cols = st.columns([2, 2, 1, 1, 2])
        
        with cols[0]:
            st.selectbox(
                "Data Source",
                [condition_data["dataSource"]],
                key=f"ds_{unique_key}",
                disabled=True,
                label_visibility="collapsed"
            )
        
        with cols[1]:
            st.selectbox(
                "Field",
                [condition_data["field"]],
                key=f"field_{unique_key}",
                disabled=True,
                label_visibility="collapsed"
            )
        
        with cols[2]:
            period_value = condition_data["eligibilityPeriod"] if condition_data["eligibilityPeriod"] != "n_a" else "N/A"
            st.text_input(
                "Period",
                value=period_value,
                key=f"period_{unique_key}",
                disabled=True,
                label_visibility="collapsed"
            )
        
        with cols[3]:
            func_map = {
                "n_a": "N/A",
                "sum": "Sum",
                "count": "Count",
                "average": "Average",
                "max": "Max",
                "min": "Min"
            }
            func_display = func_map.get(condition_data["function"], condition_data["function"])
            st.text_input(
                "Function",
                value=func_display,
                key=f"func_{unique_key}",
                disabled=True,
                label_visibility="collapsed"
            )
        
        with cols[4]:
            op_col, val_col = st.columns(2)
            with op_col:
                op_map = {
                    "equal": "=",
                    "greater_than": ">",
                    "less_than": "<",
                    "greater_than_or_equal": "≥",
                    "less_than_or_equal": "≤",
                    "contains": "contains",
                    "between": "between"
                }
                op_display = op_map.get(condition_data["operator"], condition_data["operator"])
                st.text_input(
                    "Operator",
                    value=op_display,
                    key=f"op_{unique_key}",
                    disabled=True,
                    label_visibility="collapsed"
                )
            with val_col:
                st.text_input(
                    "Value",
                    value=str(condition_data["value"]),
                    key=f"val_{unique_key}",
                    disabled=True,
                    label_visibility="collapsed"
                )

def display_condition_group(group_data: Dict[str, Any], container, level=0, parent_key=""):
    """Display a condition group in the UI"""
    with container:
        indent = "  " * level
        group_id = group_data.get("id", str(uuid.uuid4()))
        group_key = f"{parent_key}_group_{group_id}"
        group_title = f"{indent}Condition Group ({group_data['group_connector']})"
        
        with st.expander(group_title, expanded=True):
            for i, condition in enumerate(group_data["conditions"]):
                condition_id = condition.get("id", str(uuid.uuid4()))
                condition_key = f"{group_key}_cond_{condition_id}_{i}"
                
                if condition["type"] == "group":
                    display_condition_group(condition, st.container(), level + 1, group_key)
                else:
                    display_condition(condition, st.container(), condition_key)
                
                if i < len(group_data["conditions"]) - 1:
                    st.markdown(f"<div style='margin-left: {20 * (level + 1)}px; color: #666;'><i>{group_data['group_connector']}</i></div>", 
                              unsafe_allow_html=True)

def display_rule_visualization(visualization_data: Dict[str, Any]):
    """Main function to display the rule visualization"""
    
    st.markdown("---")
    st.subheader("📋 Generated Rule Structure")
    
    # Display confirmed logical structure
    if st.session_state.confirmed_structure:
        with st.expander("✅ Confirmed Logical Structure", expanded=True):
            st.success(st.session_state.confirmed_structure)
    
    # Display top-level rules
    for i, rule in enumerate(visualization_data["rules"]):
        rule_id = rule.get("id", str(uuid.uuid4()))
        
        if rule["type"] == "group":
            display_condition_group(rule, st.container(), parent_key=f"rule_{i}_{rule_id}")
        else:
            display_condition(rule, st.container(), f"rule_{i}_{rule_id}_cond")
        
        if i < len(visualization_data["rules"]) - 1:
            connector = visualization_data.get("topLevelConnector", "AND")
            st.markdown(f"<div style='color: #666; margin: 10px 0;'><i>{connector}</i></div>", 
                      unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Copy Rule JSON", use_container_width=True, key="copy_json"):
            rule_json = json.dumps(st.session_state.last_generated_rule, indent=2)
            st.code(rule_json)
            st.success("Rule JSON copied to clipboard!")
    
    with col2:
        if st.button("📊 Generate Sample Data", use_container_width=True, key="gen_data"):
            st.session_state.show_synthetic_data = True
            st.rerun()
    
    with col3:
        if st.button("🔄 Start New Rule", use_container_width=True, key="new_rule"):
            st.session_state.last_generated_rule = None
            st.session_state.rule_visualization = None
            st.session_state.show_synthetic_data = False
            st.rerun()

# ---------- External API Functions ----------
def convert_static_to_rich_format(static_data: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
    rich_format = {}
    for source_name, fields in static_data.items():
        rich_fields = []
        for i, field in enumerate(fields):
            rich_fields.append({
                "field": field,
                "field_id": f"static_{i}",
                "type": "string",
                "description": f"{field} field from {source_name}",
                "data_source_id": f"static_{source_name}"
            })
        rich_format[source_name] = rich_fields
    return rich_format

def fetch_data_sources(client_id: int = None):
    if client_id is None:
        client_id = secrets['default_client_id']
    
    try:
        headers = {
            "x-api-key": secrets['external_api_key'],
            "x-api-secret": secrets['external_api_secret']
        }
        
        params = {
            "clientId": client_id,
            "page": 1,
            "limit": 10,
            "status": "exact:ACTIVE"
        }
        
        response = requests.get(
            f"{secrets['external_api_base_url']}/data-sources",
            headers=headers,
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                sources_mapping = {}
                
                for source in data.get("data", []):
                    source_name = source.get("sourceName")
                    source_id = source.get("id")
                    mapping_data = source.get("mapping", {}).get("mappingData", {})
                    mapping_list = mapping_data.get("mapping", [])
                    
                    if source_name and mapping_list:
                        fields = []
                        for mapping_item in mapping_list:
                            mapped_field = mapping_item.get("mappedField")
                            field_id = mapping_item.get("id")
                            field_type = mapping_item.get("mappingType", "string")
                            field_description = mapping_item.get("description", "")
                            
                            if mapped_field:
                                fields.append({
                                    "field": mapped_field,
                                    "field_id": field_id,
                                    "type": field_type,
                                    "description": field_description or f"{mapped_field} field",
                                    "data_source_id": source_id
                                })
                        
                        if fields:
                            sources_mapping[source_name] = fields
                
                if sources_mapping:
                    return sources_mapping
        
        # Fallback to static data
        CSV_STRUCTURES = {
            "sample_mortgage_accounts.csv": [
                "customer_id", "product_type", "account_status", "loan_open_date", "loan_balance"
            ],
            "sample_loan_repayments.csv": [
                "repayment_id", "customer_id", "loan_account_number", "repayment_date",
                "repayment_amount", "installment_number", "payment_method_status", "loan_type",
                "interest_component", "principal_component", "remaining_balance"
            ],
            "sample_telco_billing.csv": [
                "billing_id", "customer_id", "bill_date", "bill_amount", "plan_type",
                "data_used_gb", "voice_minutes", "sms_count", "channel"
            ],
            "sample_product_enrollments.csv": [
                "enrollment_id", "customer_id", "product_type", "product_name", "enrollment_date", "status"
            ],
            "sample_customer_profiles.csv": [
                "customer_id", "name", "email", "phone", "dob", "gender",
                "region", "segment", "household_id", "is_primary"
            ],
            "sample_savings_account_transactions.csv": [
                "transaction_id", "account_id", "customer_id", "amount", "date", "transaction_type"
            ],
            "sample_credit_card_transactions.csv": [
                "customer_id", "card_number", "transaction_date", "transaction_amount", "transaction_type"
            ]
        }
        
        return convert_static_to_rich_format(CSV_STRUCTURES)
        
    except Exception as e:
        st.error(f"Error fetching data sources: {e}")
        return convert_static_to_rich_format(CSV_STRUCTURES)

def create_field_mapping(data_sources: Dict[str, List[Dict[str, str]]]) -> Dict[str, str]:
    field_mapping = {}
    for source_name, fields in data_sources.items():
        for field_info in fields:
            field_name = field_info["field"]
            field_id = field_info["field_id"]
            field_mapping[field_name] = field_id
    return field_mapping

def detect_response_type(parsed_json: Dict[str, Any]) -> str:
    if "rules" in parsed_json:
        return "rule"
    elif "logical_structure" in parsed_json and "user_message" in parsed_json:
        return "confirmation"
    elif "message" in parsed_json and len(parsed_json) == 1:
        return "general"
    else:
        return "unknown"

def generate_rule_with_openai(user_input: str, client_id: Optional[int] = None):
    """Generate rule using OpenAI"""
    
    # Fetch data sources
    data_sources = fetch_data_sources(client_id or secrets['default_client_id'])
    
    # Create field mapping
    field_mapping = create_field_mapping(data_sources)
    
    # Format available data
    available_data_lines = []
    for source_name, fields in data_sources.items():
        data_source_id = fields[0].get("data_source_id", "N/A") if fields else "N/A"
        available_data_lines.append(f"- {source_name} (ID: {data_source_id}):")
        for field_info in fields:
            field_name = field_info["field"]
            field_id = field_info["field_id"]
            field_type = field_info["type"]
            field_desc = field_info["description"]
            available_data_lines.append(f"  * {field_name} (ID: {field_id}, {field_type}): {field_desc}")
    
    available_data = "\n".join(available_data_lines)
    
    # Create field mapping string
    field_mapping_lines = []
    for field_name, field_id in field_mapping.items():
        field_mapping_lines.append(f"  * {field_name} -> {field_id}")
    field_mapping_str = "\n".join(field_mapping_lines)
    
    current_date = datetime.today().strftime("%Y-%m-%d")
    
    system_prompt = f"""You are a rule generation assistant. Create rules based on this confirmed logical structure:

        You mostly generate 3 things. 
        1. Logical structures
        2. Rule
        3. General message

        ### Communication Style:
        1. Be polite, patient, and professional
        2. Use clear, simple language (avoid jargon unless necessary)
        3. Maintain a helpful and approachable tone
        4. Confirm understanding before proceeding
        5. Provide explanations when asked

        ### Core Responsibilities - Rule Generation:
        # Logical Structure Handling
        1. When the request contains ONLY AND operators or ONLY OR operators:
           - Generate the single correct logical structure without giving options to select.
           - Present it to the user for confirmation
           - Example: "(A AND B AND C)" or "(X OR Y OR Z)"

        2. When the request contains a MIX of AND/OR operators:
           - MUST propose ALL possible logical structures (typically 3 options)
           - Present them as clearly numbered options (Option 1, Option 2, Option 3)
           - Example formats:
               Option 1: (A AND B) OR C
               Option 2: A AND (B OR C)
               Option 3: (A OR B) AND C
           - Ask user to confirm which option matches their intent


        For before generate the rule you will be given a logical structure(s) and get the confirmation of the logical structure. (YES or NO)

        ### Your Flow before generating the rule:
        1. Before generating the rule, you will be given a logical structure.
        2. and you have to ask confirmation of the logical structure with Yes or No. simple question
        3. if the user says yes, then you will generate the rule.
        4. if the user says no, then you will suggest another boolean logical structure for it.
        5. after confirming you will generate the rule

        ### While generating the rule dont add any explanation just generate the rule with ONLY JSON output.

        OUTPUT confirmation message format When the request contains ONLY AND operators or ONLY OR operators
        {{
            "message": "Logical structure confirmed",
            "logical_structure": "(Customer spends over $2500 on a credit card in a month) AND (has an active mortgage) AND (loan balance is more than $1000)",
            "user_message": "Do you agree with this structure, please suggest your requirement (agree or suggest another structure)"
        }}       

        OUTPUT confirmation message format When the request contains a MIX of AND/OR operators:
        {{
            "message": "Logical structure confirmed",
            "logical_structure": "Option 1: (Customer spends over $2500 on a credit card in a month) OR (has an active mortgage AND loan balance is more than $1000),
                Option 2: Customer spends over $2500 on a credit card in a month OR (has an active mortgage AND loan balance is more than $1000),
                Option 3: (Customer spends over $2500 on a credit card in a month OR has an active mortgage) AND loan balance is more than $1000",
            "user_message": "Do you agree with any of this structure, please suggest your requirement (agree or suggest another structure)"
        }}

        OUTPUT general message format
        {{
            "message": <general response message for other messages and any general message>,
        }}  

        
        If it is (A AND B) OR C, Output JSON matching this type of schema: PLease change the position of condintional groups according to th brackets.
        {{
    "rules": [
        {{
                    "id": <id>,
                    "priority": null,
                    "ruleType": "conditionGroup",
                    "conditions": [
                        {{
                            "id": <id>,
                            "dataSource": <data source name>,
                            "dataSourceId": <data source id>,
                            "field": <field name>,
                            "fieldId": <field id>,
                            "eligibilityPeriod": "Rolling 30 days",
                            "function": <function> example: "sum", "count", "average", "max" etc,
                            "operator": <operator> example: "greater_than", "equal", "less_than" etc,
                            "value": "2500",
                            "connector": "AND"
                        }},
                        {{
                            "id": <id>,
                            "dataSource": <data source name>,
                            "dataSourceId": <data source id>,
                            "field": <field name>,
                            "fieldId": <field id>,
                            "eligibilityPeriod": "n_a",
                            "function": "n_a",
                            "operator": <operator> example: "greater_than", "equal", "less_than" etc,
                            "value": "1000"
                        }}
                    ],
                    "connector": "OR"
                }},
                {{
                    "id": <id>,
                    "dataSource": <data source name>,
                    "dataSourceId": <data source id>,
                    "field": <field name>,
                    "fieldId": <field id>,
                    "eligibilityPeriod": "n_a",
                    "function": "n_a",
                    "operator": <operator> example: "greater_than", "equal", "less_than" etc,
                    "value": "active",
                    "priority": null,
                    "ruleType": "condition"
                }}
            ]
        }}
        If it is A AND (B OR C), Output JSON matching this type of schema:
        {{
            "rules": [
                {{
                    "id": <id>,
                    "dataSource": <data source name>,
                    "dataSourceId": <data source id>,
                    "field": <field name>,
                    "fieldId": <field id>,
                    "eligibilityPeriod": "Rolling 30 days",
                    "function": <function> example: "sum", "count", "average", "max" etc,
                    "operator": <operator> example: "greater_than", "equal", "less_than" etc,
                    "value": "2500",
                    "priority": null,
                    "ruleType": "condition",
                    "connector": "AND"
                }},
                {{
                    "id": <id>,
                    "priority": null,
                    "ruleType": "conditionGroup",
                    "conditions": [
                        {{
                            "id": <id>,
                            "dataSource": <data source name>,
                            "dataSourceId": <data source id>,
                            "field": <field name>,
                            "fieldId": <field id>,
                            "eligibilityPeriod": "n_a",
                            "function": "n_a",
                            "operator": <operator> example: "greater_than", "equal", "less_than" etc,
                            "value": "active",
                            "connector": "OR"
                        }},
                        {{
                            "id": <id>,
                            "dataSource": <data source name>,
                            "dataSourceId": <data source id>,
                            "field": <field name>,
                            "fieldId": <field id>,
                            "eligibilityPeriod": "n_a",
                            "function": "n_a",
                            "operator": <operator> example: "greater_than", "equal", "less_than" etc,
                            "value": "1000"
                        }}
                    ]
                }}
            ]
        }}



        
        CRITICAL INSTRUCTIONS:
        1. Use ONLY the exact column names from these data sources
        2. Use the exact fieldId from the field mapping above for each field (e.g., if using "customer_id", use the corresponding fieldId from the mapping)
        3. Use the exact dataSourceId from the available data sources for each data source
        4. Use ONLY the operator VALUES (e.g., ">" -> "greater_than", "=" -> "equal", "contains" -> "contains") not the labels
        5. Use ONLY the function VALUES (e.g., "sum" -> "sum", "count" -> "count", "N/A" -> "n_a") not the labels
        6. Follow the logical structure EXACTLY as provided
        7.When generating eligibilityPeriod:
        If the condition explicitly mentions a **date range** (e极, "between February and March", "from 2025-02-01 to 2025-03-31", "between 9 AM and 6 PM"), set eligibilityPeriod = 'n_a'.
        If the condition mentions 'last month" or "up to this month', set eligibilityPeriod to "Last X days" (e极, "Last 30 days").
        If the condition mentions 'every month', set eligibilityPeriod to "Rolling X days".
        Never swap these values.
        Replace X with the configured number of days."
        8. For amount aggregations, use "sum" function

        Respond ONLY with the JSON output.

        available data sources:
        {available_data}

        Field Mapping:
        {field_mapping_str}

        While using operators, use the value, not the label:
        Don't use operators not mentioned below. strictly bind with this operators
            const operators = [
          {{ value: 'equal', label: '=' }},
          {{ value: 'not_equal', label: '≠' }},
          {{ value: 'greater_than', label: '>' }},
          {{ value: 'less_than', label: '<' }},
          {{ value: 'greater_than_or_equal', label: '≥' }},
          {{ value: 'less_than_or_equal', label: '≤' }},
          {{ value: 'between', label: 'Between' }},
          {{ value: 'not_between', label: 'Not Between' }},
          {{ value: 'contains', label: 'Contains' }},
          {{ value: 'begins_with', label: 'Begins With' }},
          {{ value: 'ends_with', label: 'Ends With' }},
          {{ value: 'does_not_contain', label: 'Does Not Contain' }}
        ];
        
        Whenever using operators "=" is "equal"

        While using functions, use the value, not the label:
        Don't use functions not mentioned below. strictly bind with this functions
        
        const functions = [
          {{ value: 'n_a', label: 'N/A' }},
          {{ value: 'sum', label: 'Sum' }},
          {{ value: 'count', label: 'Count' }},
          {{ value: 'average', label: 'Average' }},
          {{ value: 'max', label: 'Maximum' }},
          {{ value: 'min', label: 'Minimum' }},
          {{ value: 'exact_match', label: 'Exact Match' }},
          {{ value: 'change_detected', label: 'Change Detected' }},
          {{ value: 'exists', label: 'Exists' }},
          {{ value: 'consecutive', label: 'Consecutive' }},
          {{ value: 'streak_count', label: 'Streak Count' }},
          {{ value: 'first_time', label: 'First Time' }},
          {{ value: 'nth_time', label: 'Nth Time' }},
          {{ value: 'recent_change', label: 'Recent Change' }}
        ];

        # Strictly use the value, not the label.
        # Don't use operators and functions not mentioned above. strictly bind with this operators
        
        How LOGICAL structure:
        Input: "A and B or C"
        Output:
        1. (A AND B) OR C
        2. A AND (B OR C)

        ### One-shot example of Logical structure:

        user_input:
        User spends over $2500 on a credit card in a month OR has an active mortgage AND loan balance is more than $1000

        Logical structure for this user_input (2 variations right):
        (User spends over $2500 on a credit card in a month) OR (has an active mortgage AND loan balance is more than $1000)
        User spends over $2500 on a credit card in a month OR (has an active mortgage AND loan balance is more than $1000)
        """
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        json_response = json.loads(response_content)
        
        # Post-process to add missing IDs
        if "rules" in json_response:
            def add_missing_ids(rules):
                for rule in rules:
                    # Ensure id exists
                    if "id" not in rule:
                        rule["id"] = str(uuid.uuid4())
                    
                    if rule.get("ruleType") == "condition":
                        # Add fieldId if missing
                        if "field" in rule and "fieldId" not in rule:
                            field_name = rule["field"]
                            if field_name in field_mapping:
                                rule["fieldId"] = field_mapping[field_name]
                            else:
                                rule["fieldId"] = f"unknown_{field_name}"
                        
                        # Add dataSourceId if missing
                        if "dataSource" in rule and "dataSourceId" not in rule:
                            data_source_name = rule["dataSource"]
                            for source_name, fields in data_sources.items():
                                if source_name == data_source_name and fields:
                                    rule["dataSourceId"] = fields[0].get("data_source_id", "N/A")
                                    break
                            if "dataSourceId" not in rule:
                                rule["dataSourceId"] = "N/A"
                        
                        # Ensure other required fields
                        if "eligibilityPeriod" not in rule:
                            rule["eligibilityPeriod"] = "n_a"
                        if "function" not in rule:
                            rule["function"] = "n_a"
                        if "operator" not in rule:
                            rule["operator"] = "equal"
                        if "value" not in rule:
                            rule["value"] = ""
                        if "connector" not in rule:
                            rule["connector"] = None
                        if "priority" not in rule:
                            rule["priority"] = None
                    
                    elif rule.get("ruleType") == "conditionGroup":
                        if "conditions" not in rule:
                            rule["conditions"] = []
                        if "groupConnector" not in rule:
                            rule["groupConnector"] = "AND"
                        if "conditions" in rule:
                            add_missing_ids(rule["conditions"])
            
            add_missing_ids(json_response["rules"])
        
        return json_response
        
    except Exception as e:
        st.error(f"OpenAI API Error: {str(e)}")
        return {"message": f"Error: {str(e)}"}

def generate_confirmed_rule():
    """Generate rule based on confirmed logical structure"""
    with st.spinner("Generating rule based on confirmed structure..."):
        try:
            last_user_message = extract_last_user_message()
            if not last_user_message:
                error_msg = {
                    "message": "No user message found. Please start over.",
                    "suggestion": "Clear chat and try again."
                }
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": json.dumps(error_msg, indent=2)
                })
                return
            
            # Create a clear prompt for rule generation
            enhanced_prompt = f"""
            User's original request: "{last_user_message}"
            
            Confirmed logical structure to use: "{st.session_state.confirmed_structure}"
            
            Generate a complete rule JSON that implements this logical structure.
            Use appropriate fields from available data sources.
            Include fieldId and dataSourceId for each field.
"""
            
            result = generate_rule_with_openai(enhanced_prompt, st.session_state.client_id)
            
            if result and "rules" in result and len(result["rules"]) > 0:
                st.session_state.last_generated_rule = result
                st.session_state.rule_visualization = create_rule_visualization(result)
                
                # Add success message to chat
                success_msg = {
                    "message": f"✅ Rule generated successfully based on your selected structure!",
                    "rules_generated": True,
                    "confirmed_structure": st.session_state.confirmed_structure,
                    "rule_count": len(result["rules"])
                }
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": json.dumps(success_msg, indent=2)
                })
            else:
                # Try alternative approach - ask user to specify fields
                alternative_msg = {
                    "message": "I need more specific information to generate the rule.",
                    "suggestion": "Please specify which data fields you want to use. Example: 'Use customer age field for age > 30 and account_status field for active accounts'",
                    "confirmed_structure": st.session_state.confirmed_structure
                }
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": json.dumps(alternative_msg, indent=2)
                })
        
        except Exception as e:
            error_msg = {
                "message": f"Error generating rule: {str(e)}",
                "suggestion": "Please try again with a simpler rule description."
            }
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": json.dumps(error_msg, indent=2)
            })

# ---------- UI Components ----------
def display_sidebar():
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        st.session_state.client_id = st.number_input(
            "Client ID",
            value=st.session_state.client_id,
            min_value=1,
            help="Enter the client ID for data source retrieval"
        )
        
        st.subheader("Data Sources")
        if st.button("🔄 Refresh Data Sources", use_container_width=True):
            with st.spinner("Fetching data sources..."):
                st.session_state.data_sources = fetch_data_sources(st.session_state.client_id)
        
        if st.session_state.data_sources:
            st.success(f"✅ Loaded {len(st.session_state.data_sources)} data sources")
            with st.expander("View Data Sources"):
                for source_name, fields in st.session_state.data_sources.items():
                    st.write(f"**{source_name}** ({len(fields)} fields)")
                    for field in fields[:3]:
                        st.write(f"  - {field['field']}")
                    if len(fields) > 3:
                        st.write(f"  ... and {len(fields) - 3} more")
        else:
            st.info("Click 'Refresh Data Sources' to load available data sources")
        
        st.subheader("Chat Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.last_generated_rule = None
                st.session_state.synthetic_data = None
                st.session_state.show_synthetic_data = False
                st.session_state.confirmed_structure = None
                st.session_state.rule_visualization = None
                st.session_state.awaiting_confirmation = False
                st.session_state.logical_options = []
                st.session_state.selected_option = None
                st.session_state.show_structure_options = False
                st.rerun()
        
        with col2:
            if st.button("📥 Export Chat", use_container_width=True):
                export_chat()

def export_chat():
    chat_data = {
        "client_id": st.session_state.client_id,
        "timestamp": datetime.now().isoformat(),
        "chat_history": st.session_state.chat_history,
        "last_generated_rule": st.session_state.last_generated_rule,
        "confirmed_structure": st.session_state.confirmed_structure
    }
    
    json_str = json.dumps(chat_data, indent=2)
    
    st.download_button(
        label="Download Chat JSON",
        data=json_str,
        file_name=f"chat_history_{st.session_state.client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def display_structure_options():
    """Display multiple logical structure options for user to choose"""
    st.markdown("---")
    st.subheader("🤔 Choose a Logical Structure")
    st.info("Please select which logical structure matches your intent:")
    
    # Display each option with radio button
    option_texts = []
    for i, option in enumerate(st.session_state.logical_options):
        option_texts.append(option)
    
    selected_index = st.radio(
        "Select an option:",
        range(len(option_texts)),
        format_func=lambda i: option_texts[i]
    )
    
    st.session_state.selected_option = selected_index
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Confirm Selection", use_container_width=True, type="primary"):
            selected_structure = st.session_state.logical_options[selected_index]
            st.session_state.confirmed_structure = selected_structure
            st.session_state.show_structure_options = False
            st.session_state.awaiting_confirmation = False
            
            # Add confirmation to chat
            confirmation_msg = {
                "message": f"✅ Confirmed Option {selected_index + 1}: {selected_structure}",
                "selected_option": selected_index + 1,
                "status": "confirmed"
            }
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": json.dumps(confirmation_msg, indent=2)
            })
            
            # Generate the rule
            generate_confirmed_rule()
            st.rerun()
    
    with col2:
        if st.button("🔄 Suggest Other Options", use_container_width=True):
            # Clear current options and ask for new ones
            st.session_state.logical_options = []
            st.session_state.selected_option = None
            st.session_state.show_structure_options = False
            
            # Ask user to suggest different structure
            suggestion_msg = {
                "message": "Please suggest a different logical structure for your rule.",
                "request": "suggest_different_structure"
            }
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": json.dumps(suggestion_msg, indent=2)
            })
            st.rerun()

def display_single_structure_confirmation():
    """Display single logical structure for confirmation"""
    st.markdown("---")
    st.subheader("✅ Confirm Logical Structure")
    
    if st.session_state.logical_options:
        logical_structure = st.session_state.logical_options[0]
        st.info(f"**Proposed Logical Structure:**\n\n{logical_structure}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Yes, Generate Rule", use_container_width=True, type="primary"):
                st.session_state.confirmed_structure = logical_structure
                st.session_state.awaiting_confirmation = False
                
                # Add confirmation to chat
                confirmation_msg = {
                    "message": f"✅ Confirmed: {logical_structure}",
                    "status": "confirmed"
                }
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": json.dumps(confirmation_msg, indent=2)
                })
                
                generate_confirmed_rule()
                st.rerun()
        
        with col2:
            if st.button("🔄 Suggest Different Structure", use_container_width=True):
                st.session_state.awaiting_confirmation = False
                st.session_state.logical_options = []
                
                # Ask for different structure
                suggestion_msg = {
                    "message": "Please suggest a different logical structure for your rule.",
                    "request": "suggest_different_structure"
                }
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": json.dumps(suggestion_msg, indent=2)
                })
                st.rerun()

def display_chat_history():
    st.title("🏦 Mortgage Rule Generator")
    st.subheader("Natural Language to Rule Conversion")
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    try:
                        content_data = json.loads(message["content"])
                        if "message" in content_data:
                            st.write(content_data["message"])
                            if "logical_structure" in content_data:
                                st.info(f"**Logical Structure:**\n{content_data['logical_structure']}")
                            if "suggestion" in content_data:
                                st.warning(content_data["suggestion"])
                        else:
                            st.json(content_data)
                    except:
                        st.write(message["content"])
    
    # If showing structure options
    if st.session_state.show_structure_options and st.session_state.logical_options:
        display_structure_options()
        return
    
    # If awaiting single structure confirmation
    if st.session_state.awaiting_confirmation and st.session_state.logical_options:
        display_single_structure_confirmation()
        return
    
    # Chat input (only show if not in confirmation mode)
    user_input = st.chat_input("Describe your rule in natural language...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Check if user is responding to structure request
        last_assistant_msg = None
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "assistant":
                last_assistant_msg = msg["content"]
                break
        
        if last_assistant_msg and "suggest_different_structure" in last_assistant_msg:
            # User is suggesting a different structure
            with st.spinner("Analyzing your suggested structure..."):
                result = generate_rule_with_openai(user_input, st.session_state.client_id)
                handle_ai_response(result)
        else:
            # New rule request
            with st.spinner("Processing your rule request..."):
                if len(st.session_state.data_sources) == 0:
                    st.session_state.data_sources = fetch_data_sources(st.session_state.client_id)
                
                result = generate_rule_with_openai(user_input, st.session_state.client_id)
                handle_ai_response(result)
        
        st.rerun()

def handle_ai_response(result):
    """Handle AI response and update session state accordingly"""
    
    # Check if result is valid
    if not result:
        error_msg = {
            "message": "No response from AI service.",
            "suggestion": "Please check your API configuration and try again."
        }
        st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(error_msg, indent=2)})
        return
    
    if "message" in result and "Error:" in result["message"]:
        error_msg = {
            "message": "AI service error occurred.",
            "suggestion": "Please try again in a moment or simplify your request."
        }
        st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(error_msg, indent=2)})
        return
    
    response_type = detect_response_type(result)
    
    if response_type == "rule":
        # Validate that rules are present and have minimum structure
        if "rules" in result and len(result["rules"]) > 0:
            st.session_state.last_generated_rule = result
            st.session_state.rule_visualization = create_rule_visualization(result)
            
            response_content = json.dumps({
                "message": "✅ Rule generated successfully!",
                "rules_available": True,
                "rule_count": len(result["rules"])
            }, indent=2)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response_content})
        else:
            error_msg = {
                "message": "Rule was generated but no valid conditions were found.",
                "suggestion": "Please try rephrasing your rule with specific field names. Example: 'Customers with age greater than 30 AND account_status equals active'"
            }
            st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(error_msg, indent=2)})
    
    elif response_type == "confirmation":
        logical_structure = result.get("logical_structure", "")
        user_message = result.get("user_message", "")
        
        if not logical_structure:
            error_msg = {
                "message": "Could not determine logical structure.",
                "suggestion": "Please specify your rule with clear AND/OR conditions."
            }
            st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(error_msg, indent=2)})
            return
        
        # Parse the logical structure to get options
        options = parse_logical_options(logical_structure)
        
        if len(options) > 1:
            # Multiple options - show selection UI
            st.session_state.logical_options = options
            st.session_state.show_structure_options = True
            
            # Store the AI response in chat
            ai_response = {
                "message": "I've identified multiple possible logical structures for your rule.",
                "logical_structure": logical_structure,
                "user_message": "Please select which structure matches your intent."
            }
            st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(ai_response, indent=2)})
        
        else:
            # Single option - show confirmation
            st.session_state.logical_options = options
            st.session_state.awaiting_confirmation = True
            
            ai_response = {
                "message": "I've identified a logical structure for your rule.",
                "logical_structure": logical_structure,
                "user_message": user_message or "Do you agree with this structure?"
            }
            st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(ai_response, indent=2)})
    
    elif response_type == "general":
        st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(result, indent=2)})
    
    else:
        # Unknown response type
        error_msg = {
            "message": "I'm not sure how to process that request.",
            "suggestion": "Please try rephrasing your rule description. Example: 'Customers who spent over $1000 AND have active status'"
        }
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": json.dumps(error_msg, indent=2)
        })

def generate_synthetic_dataset_with_openai(rules: List[Dict[str, Any]], data_sources: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """Generate synthetic dataset with ONLY fields referenced in the rule"""
    
    # Extract exact fields mentioned in the rules
    rule_fields = set()
    
    def extract_fields_from_rules(rules_list):
        for rule in rules_list:
            if rule.get("ruleType") == "condition":
                field = rule.get("field")
                if field:
                    rule_fields.add(field)
            elif rule.get("ruleType") == "conditionGroup" and "conditions" in rule:
                extract_fields_from_rules(rule["conditions"])
    
    extract_fields_from_rules(rules)
    
    current_date = datetime.today().strftime("%Y-%m-%d")
    
    system_prompt = f"""Generate synthetic data for testing a mortgage rule. Today's date: {current_date}
    
    Generate 10 customer records with these fields: {list(rule_fields)}
    
    Requirements:
    1. Include customer_id and matches_rule fields
    2. 5 records should match ALL conditions (matches_rule: true)
    3. 5 records should NOT match (matches_rule: false)
    4. Use realistic values for each field type
    5. Output ONLY JSON with this format:
    
    {{
      "synthetic_dataset": [
        {{
          "customer_id": "CUST001",
          "field1": "value1",
          "field2": "value2",
          "matches_rule": true
        }}
      ]
    }}
    """
    
    user_prompt = f"Generate synthetic data for this mortgage rule:\n{json.dumps(rules, indent=2)}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        synthetic_data = json.loads(response_content)
        
        if "synthetic_dataset" in synthetic_data:
            return synthetic_data
        else:
            return generate_fallback_rule_dataset(rules, rule_fields)
        
    except Exception as e:
        st.error(f"Error generating synthetic data: {e}")
        return generate_fallback_rule_dataset(rules, rule_fields)

def generate_fallback_rule_dataset(rules: List[Dict[str, Any]], rule_fields: set) -> Dict[str, Any]:
    """Fallback dataset generation"""
    dataset = {
        "synthetic_dataset": []
    }
    
    for i in range(10):
        customer_id = f"CUST{100 + i}"
        matches_rule = i < 5
        
        record = {
            "customer_id": customer_id,
            "matches_rule": matches_rule
        }
        
        for field in rule_fields:
            if "date" in field.lower() or "Date" in field:
                if matches_rule:
                    record[field] = (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat()[:10]
                else:
                    record[field] = (datetime.now() - timedelta(days=np.random.randint(60, 365))).isoformat()[:10]
            
            elif "amount" in field.lower() or "price" in field.lower() or "balance" in field.lower():
                if matches_rule:
                    record[field] = round(np.random.uniform(1000, 5000), 2)
                else:
                    record[field] = round(np.random.uniform(100, 999), 2)
            
            elif "age" in field.lower():
                if matches_rule:
                    record[field] = np.random.randint(30, 60)
                else:
                    record[field] = np.random.randint(18, 29)
            
            elif "status" in field.lower():
                record[field] = "active" if matches_rule else "inactive"
            
            else:
                record[field] = f"value_{np.random.randint(1, 100)}"
        
        dataset["synthetic_dataset"].append(record)
    
    return dataset

def display_synthetic_data():
    if st.session_state.synthetic_data and st.session_state.show_synthetic_data:
        with st.expander("📊 Synthetic Data Preview", expanded=True):
            data = st.session_state.synthetic_data["synthetic_dataset"]
            df = pd.DataFrame(data)
            
            total_records = len(df)
            matching_records = df[df["matches_rule"]].shape[0]
            match_percentage = (matching_records / total_records * 100) if total_records > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", total_records)
            col2.metric("Matching Records", matching_records)
            col3.metric("Match Rate", f"{match_percentage:.1f}%")
            
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stChatMessage[data-testid="user"] {
        background-color: #f0f2f6;
    }
    .stChatMessage[data-testid="assistant"] {
        background-color: #e6f7ff;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_chat_history()
    
    with col2:
        st.info("💡 **Tips for successful rules:**\n\n"
                "1. **Be specific**: 'age > 30' not 'old customers'\n"
                "2. **Use AND/OR clearly**: 'A AND B' or 'A OR B'\n"
                "3. **Example**: 'Customers with age > 30 AND account_status = active'\n"
                "4. **Simple first**: Start with 1-2 conditions")
    
    # Display rule visualization if available
    if st.session_state.rule_visualization and st.session_state.last_generated_rule:
        display_rule_visualization(st.session_state.rule_visualization)
        
        # Show synthetic data generation button
        if not st.session_state.show_synthetic_data:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Generate Sample Data", use_container_width=True, type="secondary", key="gen_sample"):
                    with st.spinner("Generating synthetic data..."):
                        data_sources = fetch_data_sources(st.session_state.client_id)
                        synthetic_data = generate_synthetic_dataset_with_openai(
                            st.session_state.last_generated_rule["rules"], 
                            data_sources
                        )
                        st.session_state.synthetic_data = synthetic_data
                        st.session_state.show_synthetic_data = True
                        st.rerun()
    
    # Display synthetic data if available
    display_synthetic_data()

if __name__ == "__main__":
    if not secrets['openai_api_key']:
        st.error("❌ OpenAI API key not found in secrets. Please add it to .streamlit/secrets.toml")
        st.stop()
    
    if not client:
        st.error("❌ Failed to initialize OpenAI client. Check your API key.")
        st.stop()
    
    if not secrets['external_api_key'] or not secrets['external_api_secret']:
        st.warning("⚠️ External API credentials not found. Using static data sources only.")
    
    main()
