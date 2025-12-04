import streamlit as st
import requests
import json
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import openai
import time
from collections import defaultdict

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
        "logical_structure": rules_data.get("logical_structure", "")
    }
    
    for rule in rules_data.get("rules", []):
        processed_rule = process_rule_item(rule)
        visualization["rules"].append(processed_rule)
    
    return visualization

def display_condition(condition_data: Dict[str, Any], container):
    """Display a single condition in the UI"""
    with container:
        col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 2])
        
        with col1:
            # Data source dropdown (read-only)
            st.selectbox(
                "Data Source",
                [condition_data["dataSource"]],
                key=f"ds_{condition_data['id']}",
                disabled=True,
                label_visibility="collapsed"
            )
        
        with col2:
            # Field dropdown (read-only)
            st.selectbox(
                "Field",
                [condition_data["field"]],
                key=f"field_{condition_data['id']}",
                disabled=True,
                label_visibility="collapsed"
            )
        
        with col3:
            # Eligibility period (read-only)
            if condition_data["eligibilityPeriod"] != "n_a":
                st.text_input(
                    "Period",
                    value=condition_data["eligibilityPeriod"],
                    key=f"period_{condition_data['id']}",
                    disabled=True,
                    label_visibility="collapsed"
                )
            else:
                st.text_input(
                    "Period",
                    value="N/A",
                    key=f"period_{condition_data['id']}",
                    disabled=True,
                    label_visibility="collapsed"
                )
        
        with col4:
            # Function (read-only)
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
        
        with col5:
            # Operator and value
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
        # Group header
        indent = "  " * level
        group_title = f"{indent}Condition Group ({group_data['group_connector']})"
        
        with st.expander(group_title, expanded=True):
            # Display conditions in this group
            for i, condition in enumerate(group_data["conditions"]):
                if condition["type"] == "group":
                    # Nested group
                    display_condition_group(condition, st.container(), level + 1)
                else:
                    # Individual condition
                    display_condition(condition, st.container())
                
                # Show connector between conditions (except last one)
                if i < len(group_data["conditions"]) - 1:
                    connector = group_data["group_connector"]
                    st.markdown(f"<div style='margin-left: {20 * (level + 1)}px; color: #666; font-style: italic;'>{connector}</div>", unsafe_allow_html=True)

def display_rule_visualization(visualization_data: Dict[str, Any]):
    """Main function to display the rule visualization"""
    
    st.markdown("---")
    st.subheader("📋 Rule Structure")
    
    # Display logical structure if available
    if visualization_data.get("logical_structure"):
        with st.expander("Logical Structure", expanded=True):
            st.info(visualization_data["logical_structure"])
    
    # Display top-level rules
    for i, rule in enumerate(visualization_data["rules"]):
        if rule["type"] == "group":
            display_condition_group(rule, st.container())
        else:
            display_condition(rule, st.container())
        
        # Show connector between top-level rules (except last one)
        if i < len(visualization_data["rules"]) - 1:
            connector = visualization_data.get("topLevelConnector", "AND")
            st.markdown(f"<div style='color: #666; font-style: italic; margin: 10px 0;'>{connector}</div>", unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Copy Rule JSON", use_container_width=True):
            rule_json = json.dumps(st.session_state.last_generated_rule, indent=2)
            st.code(rule_json)
            st.success("Rule JSON copied to clipboard!")
    
    with col2:
        if st.button("📊 Generate Sample Data", use_container_width=True):
            st.session_state.show_synthetic_data = True
            st.rerun()
    
    with col3:
        if st.button("🔄 Edit Rule", use_container_width=True):
            st.info("Edit functionality coming soon!")

# ---------- External API Functions (same as before) ----------
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

def generate_rule_with_openai(user_input: str, client_id: Optional[int] = None, generate_synthetic_data: bool = False):
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

        ### Core Responsibilities - Rule Generation:
        # Logical Structure Handling
        1. When the request contains ONLY AND operators or ONLY OR operators:
           - Generate the single correct logical structure without giving options to select.
           - Present it to the user for confirmation

        2. When the request contains a MIX of AND/OR operators:
           - MUST propose ALL possible logical structures (typically 3 options)
           - Present them as clearly numbered options (Option 1, Option 2, Option 3)

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

        If it is (A AND B) OR C, Output JSON matching this type of schema:
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
                            "function": <function>,
                            "operator": <operator>,
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
                            "operator": <operator>,
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
                    "operator": <operator>,
                    "value": "active",
                    "priority": null,
                    "ruleType": "condition"
                }}
            ]
        }}

        CRITICAL INSTRUCTIONS:
        1. Use ONLY the exact column names from these data sources
        2. Use the exact fieldId from the field mapping above for each field
        3. Use the exact dataSourceId from the available data sources for each data source
        4. Use ONLY the operator VALUES (e.g., ">" -> "greater_than", "=" -> "equal")
        5. Use ONLY the function VALUES (e.g., "sum" -> "sum", "count" -> "count", "N/A" -> "n_a")
        6. Follow the logical structure EXACTLY as provided

        available data sources:
        {available_data}

        Field Mapping:
        {field_mapping_str}
        """
    
    try:
        # Build messages array
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        
        response_content = response.choices[0].message.content
        
        # Extract JSON from response
        json_str = response_content[response_content.find('{'):response_content.rfind('}')+1]
        json_response = json.loads(json_str)
        
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
    
    system_prompt = f"""You are a data generator in year 2025. Generate a synthetic dataset that includes ONLY the fields referenced in the provided rule. Always assume today's date is {current_date}.

    Generate 10 customer records with:
    1. ONLY the fields mentioned in the rule (plus customer_id)
    2. Realistic, natural-looking values appropriate for each field type
    3. Exactly 10 records total
    4. About half should match ALL rule conditions (matches_rule: true), half should not (matches_rule: false)
    5. Output in exact JSON format specified

    Return ONLY JSON in this exact format:
    {{
      "synthetic_dataset": [
        {{
          "customer_id": "CUST001",
          "field1": "value1",
          "field2": "value2",
          "matches_rule": true
        }},
        ...
      ]
    }}

    DO NOT include any other fields or metadata. ONLY the synthetic_dataset array.
    """

    user_prompt = f"""
    Generate synthetic data for this rule. Include ONLY these fields: {list(rule_fields)}

    RULE CONDITIONS:
    {json.dumps(rules, indent=2)}

    Generate 10 records with realistic values. About 5 should match ALL conditions, 5 should fail at least one condition.
    Output ONLY the JSON with synthetic_dataset array, no other text.
    """

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
        if st.button("Refresh Data Sources", use_container_width=True):
            with st.spinner("Fetching data sources..."):
                st.session_state.data_sources = fetch_data_sources(st.session_state.client_id)
        
        if st.session_state.data_sources:
            st.success(f"Loaded {len(st.session_state.data_sources)} data sources")
            with st.expander("View Data Sources"):
                for source_name, fields in st.session_state.data_sources.items():
                    st.write(f"**{source_name}** ({len(fields)} fields)")
                    for field in fields[:3]:
                        st.write(f"  - {field['field']} ({field['type']})")
                    if len(fields) > 3:
                        st.write(f"  ... and {len(fields) - 3} more")
        else:
            st.info("Click 'Refresh Data Sources' to load available data sources")
        
        st.subheader("Chat Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.last_generated_rule = None
                st.session_state.synthetic_data = None
                st.session_state.show_synthetic_data = False
                st.session_state.confirmed_structure = None
                st.session_state.rule_visualization = None
                st.session_state.awaiting_confirmation = False
                st.session_state.logical_options = []
                st.rerun()
        
        with col2:
            if st.button("Export Chat", use_container_width=True):
                export_chat()

def export_chat():
    chat_data = {
        "client_id": st.session_state.client_id,
        "timestamp": datetime.now().isoformat(),
        "chat_history": st.session_state.chat_history,
        "last_generated_rule": st.session_state.last_generated_rule
    }
    
    json_str = json.dumps(chat_data, indent=2)
    
    st.download_button(
        label="Download Chat JSON",
        data=json_str,
        file_name=f"chat_history_{st.session_state.client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def display_confirmation_options(logical_structure: str):
    """Display confirmation options for logical structure"""
    st.markdown("---")
    st.subheader("✅ Confirm Logical Structure")
    
    # Display the logical structure
    st.info(f"**Proposed Logical Structure:**\n\n{logical_structure}")
    
    # Confirmation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Agree & Generate Rule", use_container_width=True, type="primary"):
            st.session_state.confirmed_structure = logical_structure
            st.session_state.awaiting_confirmation = False
            # Generate the rule based on confirmed structure
            generate_confirmed_rule()
    
    with col2:
        if st.button("🔄 Suggest Another Structure", use_container_width=True):
            st.session_state.awaiting_confirmation = True
            st.rerun()
    
    with col3:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.awaiting_confirmation = False
            st.rerun()

def generate_confirmed_rule():
    """Generate rule after confirmation"""
    with st.spinner("Generating rule based on confirmed structure..."):
        try:
            # Use the last user input to generate the rule
            last_user_message = None
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "user":
                    last_user_message = msg["content"]
                    break
            
            if last_user_message:
                # Add confirmation context to the prompt
                prompt_with_confirmation = f"{last_user_message}\n\nConfirmed logical structure: {st.session_state.confirmed_structure}"
                
                result = generate_rule_with_openai(prompt_with_confirmation, st.session_state.client_id)
                
                if "rules" in result:
                    st.session_state.last_generated_rule = result
                    st.session_state.rule_visualization = create_rule_visualization(result)
                    
                    # Add assistant response to chat
                    confirmation_msg = f"Rule generated successfully based on confirmed structure: {st.session_state.confirmed_structure}"
                    st.session_state.chat_history.append({"role": "assistant", "content": confirmation_msg})
                else:
                    st.error("Failed to generate rule. Please try again.")
        
        except Exception as e:
            st.error(f"Error generating rule: {str(e)}")

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
                                st.info(f"**Logical Structure:** {content_data['logical_structure']}")
                                if "user_message" in content_data:
                                    st.write(f"**Question:** {content_data['user_message']}")
                        else:
                            st.json(content_data)
                    except:
                        st.write(message["content"])
    
    # If awaiting confirmation, show the confirmation UI
    if st.session_state.awaiting_confirmation:
        # Get the last logical structure from chat
        last_logical_structure = None
        for msg in reversed(st.session_state.chat_history):
            if msg["role"] == "assistant":
                try:
                    content_data = json.loads(msg["content"])
                    if "logical_structure" in content_data:
                        last_logical_structure = content_data["logical_structure"]
                        break
                except:
                    continue
        
        if last_logical_structure:
            display_confirmation_options(last_logical_structure)
            return  # Don't show chat input while confirming
    
    # Chat input (only show if not awaiting confirmation)
    user_input = st.chat_input("Describe your rule in natural language...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # If user is responding to a confirmation request
        if st.session_state.awaiting_confirmation:
            # Check if user agreed
            if user_input.lower() in ["yes", "agree", "yep", "confirm", "ok"]:
                st.session_state.confirmed_structure = st.session_state.logical_options[0] if st.session_state.logical_options else None
                st.session_state.awaiting_confirmation = False
                generate_confirmed_rule()
            elif user_input.lower() in ["no", "disagree", "nope"]:
                st.session_state.chat_history.append({"role": "assistant", "content": "Please suggest another logical structure."})
                st.session_state.awaiting_confirmation = False
            st.rerun()
            return
        
        # Generate response
        with st.spinner("Processing your request..."):
            try:
                if len(st.session_state.data_sources) == 0:
                    st.session_state.data_sources = fetch_data_sources(st.session_state.client_id)
                
                result = generate_rule_with_openai(user_input, st.session_state.client_id)
                response_type = detect_response_type(result)
                
                if response_type == "rule":
                    st.session_state.last_generated_rule = result
                    st.session_state.rule_visualization = create_rule_visualization(result)
                    
                    response_content = json.dumps({
                        "message": "Rule generated successfully!",
                        "rules_available": True
                    }, indent=2)
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": response_content})
                
                elif response_type == "confirmation":
                    # Store logical options and set awaiting confirmation
                    logical_structure = result.get("logical_structure", "")
                    st.session_state.logical_options = [logical_structure]
                    st.session_state.awaiting_confirmation = True
                    st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(result, indent=2)})
                
                elif response_type == "general":
                    st.session_state.chat_history.append({"role": "assistant", "content": json.dumps(result, indent=2)})
                
                else:
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": json.dumps({"message": "I'm not sure how to process that request. Please try rephrasing your rule description."}, indent=2)
                    })
                
                st.rerun()
                
            except Exception as e:
                error_message = f"Error generating rule: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                st.error(error_message)
                st.rerun()

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
    # Custom CSS for better styling
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
    .condition-group {
        border-left: 3px solid #4CAF50;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .condition-item {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
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
        st.info("💡 **Tips for best results:**\n\n"
                "1. Be specific with conditions\n"
                "2. Mention field names when possible\n"
                "3. Use clear AND/OR logic\n"
                "4. Example: 'Customers who spent over $1000 AND have an active mortgage'")
    
    # Display rule visualization if available
    if st.session_state.rule_visualization and st.session_state.last_generated_rule:
        display_rule_visualization(st.session_state.rule_visualization)
        
        # Show synthetic data generation button
        if not st.session_state.show_synthetic_data:
            col1, col2, col3 = st.columns(3)
            with col2:
                if st.button("📊 Generate Sample Data", use_container_width=True, type="secondary"):
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
