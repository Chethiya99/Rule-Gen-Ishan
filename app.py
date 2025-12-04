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
        
        for i, part in enumerate(parts[1:], 1):  # Skip first empty part
            option_text = f"Option {i}: {part.strip()}"
            # Clean up the option text
            option_text = option_text.replace('\\n', ' ').replace('\n', ' ').strip()
            options.append(option_text)
    
    # If not in Option X format, check for numbered options
    elif "1." in logical_structure or "2." in logical_structure:
        lines = logical_structure.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+[\.\)]', line):
                options.append(line)
    
    # If single structure
    else:
        options = [logical_structure.strip()]
    
    # Clean up options
    cleaned_options = []
    for opt in options:
        # Remove any trailing connector text
        opt = re.sub(r'\s*(Do you agree with any of this structure.*)', '', opt)
        opt = opt.strip()
        if opt:
            cleaned_options.append(opt)
    
    return cleaned_options if cleaned_options else [logical_structure]

def extract_last_user_message():
    """Extract the last user message from chat history"""
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "user":
            return msg["content"]
    return None

# ---------- Visualization Functions ----------
def create_rule_visualization(rules_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a structured visualization of the rule for display"""
    
    def process_rule_item(rule_item, level=0):
        """Recursively process rule items for visualization"""
        item_type = rule_item.get("ruleType", "condition")
        
        if item_type == "conditionGroup":
            # Process condition group
            group_data = {
                "type": "group",
                "id": rule_item.get("id", str(uuid.uuid4())),
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
                "id": rule_item.get("id", str(uuid.uuid4())),
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
    
    for rule in rules_data.get("rules", []):
        processed_rule = process_rule_item(rule)
        visualization["rules"].append(processed_rule)
    
    return visualization

def display_condition(condition_data: Dict[str, Any], container):
    """Display a single condition in the UI"""
    with container:
        cols = st.columns([2, 2, 1, 1, 2])
        
        with cols[0]:
            st.selectbox(
                "Data Source",
                [condition_data["dataSource"]],
                key=f"ds_{condition_data['id']}",
                disabled=True,
                label_visibility="collapsed"
            )
        
        with cols[1]:
            st.selectbox(
                "Field",
                [condition_data["field"]],
                key=f"field_{condition_data['id']}",
                disabled=True,
                label_visibility="collapsed"
            )
        
        with cols[2]:
            period_value = condition_data["eligibilityPeriod"] if condition_data["eligibilityPeriod"] != "n_a" else "N/A"
            st.text_input(
                "Period",
                value=period_value,
                key=f"period_{condition_data['id']}",
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
                key=f"func_{condition_data['id']}",
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
                    key=f"op_{condition_data['id']}",
                    disabled=True,
                    label_visibility="collapsed"
                )
            with val_col:
                st.text_input(
                    "Value",
                    value=str(condition_data["value"]),
                    key=f"val_{condition_data['id']}",
                    disabled=True,
                    label_visibility="collapsed"
                )

def display_condition_group(group_data: Dict[str, Any], container, level=0):
    """Display a condition group in the UI"""
    with container:
        indent = "  " * level
        group_title = f"{indent}Condition Group ({group_data['group_connector']})"
        
        with st.expander(group_title, expanded=True):
            for i, condition in enumerate(group_data["conditions"]):
                if condition["type"] == "group":
                    display_condition_group(condition, st.container(), level + 1)
                else:
                    display_condition(condition, st.container())
                
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
        if rule["type"] == "group":
            display_condition_group(rule, st.container())
        else:
            display_condition(rule, st.container())
        
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

def generate_rule_with_openai(user_input: str, client_id: Optional[int] = None, context: str = ""):
    """Generate rule using OpenAI with optional context"""
    
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
    
    # Add context if provided (for re-generating with confirmed structure)
    full_prompt = user_input
    if context:
        full_prompt = f"{user_input}\n\nContext: {context}"
    
    system_prompt = f"""You are a rule generation assistant. Create rules based on logical structures.

    You generate 3 types of responses:
    1. Logical structure options (for user confirmation)
    2. Complete rule JSON (after confirmation)
    3. General messages

    ### When user provides a rule description:
    1. If it contains ONLY AND or ONLY OR operators:
       - Generate ONE logical structure
       - Ask for confirmation
    
    2. If it contains MIXED AND/OR operators:
       - Generate 3 possible logical structures
       - Number them as Option 1, Option 2, Option 3
       - Ask user to choose one

    ### After user confirms a structure:
    - Generate the complete rule JSON
    - Use ONLY the fields from available data sources
    - Include fieldId and dataSourceId from mapping

    Available data sources:
    {available_data}

    Field Mapping:
    {field_mapping_str}

    Respond ONLY in JSON format.
    """
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent structure
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        json_response = json.loads(response_content)
        
        # Post-process to add fieldId if missing
        if "rules" in json_response:
            def add_field_ids(rules):
                for rule in rules:
                    if rule.get("ruleType") == "condition":
                        if "field" in rule and "fieldId" not in rule:
                            field_name = rule["field"]
                            if field_name in field_mapping:
                                rule["fieldId"] = field_mapping[field_name]
                        if "dataSource" in rule and "dataSourceId" not in rule:
                            data_source_name = rule["dataSource"]
                            for source_name, fields in data_sources.items():
                                if source_name == data_source_name and fields:
                                    rule["dataSourceId"] = fields[0].get("data_source_id", "N/A")
                                    break
                    elif rule.get("ruleType") == "conditionGroup":
                        if "conditions" in rule:
                            add_field_ids(rule["conditions"])
            
            add_field_ids(json_response["rules"])
        
        return json_response
        
    except Exception as e:
        st.error(f"Error generating rule with OpenAI: {str(e)}")
        return {"message": f"Error: {str(e)}"}

def generate_confirmed_rule():
    """Generate rule based on confirmed logical structure"""
    with st.spinner("Generating rule based on confirmed structure..."):
        try:
            last_user_message = extract_last_user_message()
            if not last_user_message:
                st.error("No user message found")
                return
            
            # Add the confirmed structure as context
            context = f"Confirmed logical structure: {st.session_state.confirmed_structure}"
            result = generate_rule_with_openai(last_user_message, st.session_state.client_id, context)
            
            if "rules" in result:
                st.session_state.last_generated_rule = result
                st.session_state.rule_visualization = create_rule_visualization(result)
                
                # Add success message to chat
                success_msg = {
                    "message": f"✅ Rule generated successfully based on your selected structure!",
                    "rules_generated": True
                }
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": json.dumps(success_msg, indent=2)
                })
            else:
                error_msg = {
                    "message": "Failed to generate rule. Please try again with a clearer description."
                }
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": json.dumps(error_msg, indent=2)
                })
        
        except Exception as e:
            error_msg = {
                "message": f"Error generating rule: {str(e)}"
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
    selected_index = st.radio(
        "Select an option:",
        range(len(st.session_state.logical_options)),
        format_func=lambda i: st.session_state.logical_options[i]
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
                "message": f"Confirmed: {selected_structure}",
                "selected_option": selected_index + 1
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
                    "message": f"Confirmed: {logical_structure}",
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
    response_type = detect_response_type(result)
    
    if response_type == "rule":
        st.session_state.last_generated_rule = result
        st.session_state.rule_visualization = create_rule_visualization(result)
        
        response_content = json.dumps({
            "message": "✅ Rule generated successfully!",
            "rules_available": True
        }, indent=2)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response_content})
    
    elif response_type == "confirmation":
        logical_structure = result.get("logical_structure", "")
        user_message = result.get("user_message", "")
        
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
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": json.dumps({"message": "I'm not sure how to process that. Please try rephrasing your rule description."}, indent=2)
        })

def generate_synthetic_dataset_with_openai(rules: List[Dict[str, Any]], data_sources: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
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
    
    system_prompt = f"""Generate synthetic data for testing a rule. Include ONLY fields mentioned in the rule.
    Today's date: {current_date}
    
    Generate 10 records:
    - Include customer_id and matches_rule fields
    - Include ONLY the rule fields: {list(rule_fields)}
    - 5 records should match ALL conditions (matches_rule: true)
    - 5 records should NOT match (matches_rule: false)
    
    Return JSON format:
    {{
      "synthetic_dataset": [
        {{
          "customer_id": "CUST001",
          "field1": "value1",
          "matches_rule": true
        }}
      ]
    }}
    """
    
    user_prompt = f"Generate synthetic data for this rule:\n{json.dumps(rules, indent=2)}"
    
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
            if field == "applicationDate":
                if matches_rule:
                    record[field] = (datetime.now() - timedelta(days=np.random.randint(100, 365))).isoformat()[:10]
                else:
                    record[field] = (datetime.now() - timedelta(days=np.random.randint(366, 730))).isoformat()[:10]
            elif field == "pricePaid" or "price" in field.lower():
                if matches_rule:
                    record[field] = round(np.random.uniform(1001, 5000), 2)
                else:
                    record[field] = round(np.random.uniform(100, 999), 2)
            elif field == "purchaseAmount":
                if matches_rule:
                    record[field] = round(np.random.uniform(1501, 3000), 2)
                else:
                    record[field] = round(np.random.uniform(500, 1499), 2)
            elif field == "lastRepaymentDate":
                if matches_rule:
                    record[field] = (datetime.now() - timedelta(days=np.random.randint(1, 59))).isoformat()[:10]
                else:
                    record[field] = (datetime.now() - timedelta(days=np.random.randint(60, 120))).isoformat()[:10]
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
    </style>
    """, unsafe_allow_html=True)
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_chat_history()
    
    with col2:
        st.info("💡 **Tips:**\n\n"
                "1. Be specific with conditions\n"
                "2. Mention AND/OR logic clearly\n"
                "3. Example: 'Customers who spent >$1000 AND have active mortgage'\n"
                "4. You'll get to confirm the structure before rule generation")
    
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
        st.error("OpenAI API key not found in secrets. Please add it to .streamlit/secrets.toml")
        st.stop()
    
    if not secrets['external_api_key'] or not secrets['external_api_secret']:
        st.warning("External API credentials not found. Using static data sources only.")
    
    main()
