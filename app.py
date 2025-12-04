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
    st.session_state.client_id = 307  # Default client ID
if 'data_sources' not in st.session_state:
    st.session_state.data_sources = {}
if 'last_generated_rule' not in st.session_state:
    st.session_state.last_generated_rule = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'show_synthetic_data' not in st.session_state:
    st.session_state.show_synthetic_data = False

# Get secrets from Streamlit secrets
def get_secrets():
    """Get configuration from Streamlit secrets"""
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

# ---------- External API Functions ----------
def convert_static_to_rich_format(static_data: Dict[str, List[str]]) -> Dict[str, List[Dict[str, str]]]:
    """Convert static CSV_STRUCTURES format to rich format with metadata."""
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
    """Fetch data sources from external API or use static data"""
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
    """Create a mapping from field name to field ID across all data sources."""
    field_mapping = {}
    for source_name, fields in data_sources.items():
        for field_info in fields:
            field_name = field_info["field"]
            field_id = field_info["field_id"]
            field_mapping[field_name] = field_id
    return field_mapping

def detect_response_type(parsed_json: Dict[str, Any]) -> str:
    """Detect the type of response from the parsed JSON"""
    if "rules" in parsed_json:
        return "rule"
    elif "logical_structure" in parsed_json and "user_message" in parsed_json:
        return "confirmation"
    elif "message" in parsed_json and len(parsed_json) == 1:
        return "general"
    else:
        return "unknown"

def generate_rule_with_openai(user_input: str, client_id: Optional[int] = None, generate_synthetic_data: bool = False):
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
                        # Add dataSourceId if missing
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
            # Fallback to simple generation
            return generate_fallback_rule_dataset(rules, rule_fields)
        
    except Exception as e:
        st.error(f"Error generating synthetic data: {e}")
        return generate_fallback_rule_dataset(rules, rule_fields)

def generate_fallback_rule_dataset(rules: List[Dict[str, Any]], rule_fields: set) -> Dict[str, Any]:
    """Fallback dataset generation with only rule fields"""
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
    """Display sidebar with configuration"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Client ID selection
        st.session_state.client_id = st.number_input(
            "Client ID",
            value=st.session_state.client_id,
            min_value=1,
            help="Enter the client ID for data source retrieval"
        )
        
        # Data Sources section
        st.subheader("Data Sources")
        if st.button("Refresh Data Sources", use_container_width=True):
            with st.spinner("Fetching data sources..."):
                st.session_state.data_sources = fetch_data_sources(st.session_state.client_id)
        
        if st.session_state.data_sources:
            st.success(f"Loaded {len(st.session_state.data_sources)} data sources")
            with st.expander("View Data Sources"):
                for source_name, fields in st.session_state.data_sources.items():
                    st.write(f"**{source_name}** ({len(fields)} fields)")
                    for field in fields[:5]:  # Show first 5 fields
                        st.write(f"  - {field['field']} ({field['type']})")
                    if len(fields) > 5:
                        st.write(f"  ... and {len(fields) - 5} more fields")
        else:
            st.info("Click 'Refresh Data Sources' to load available data sources")
        
        # Chat Management
        st.subheader("Chat Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.last_generated_rule = None
                st.session_state.synthetic_data = None
                st.session_state.show_synthetic_data = False
                st.rerun()
        
        with col2:
            if st.button("Export Chat", use_container_width=True):
                export_chat()
        
        # Settings
        st.subheader("Settings")
        auto_refresh = st.checkbox("Auto-refresh data sources on chat", value=True)
        show_synthetic_by_default = st.checkbox("Show synthetic data by default", value=False)
        
        # Information
        st.divider()
        st.caption(f"API Base URL: {secrets['external_api_base_url']}")
        st.caption(f"Default Client ID: {secrets['default_client_id']}")

def export_chat():
    """Export chat history as JSON"""
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
        mime="application/json",
        use_container_width=True
    )

def display_chat_history():
    """Display chat history in main area"""
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
                    # Check if content is JSON or text
                    try:
                        content_data = json.loads(message["content"])
                        if "message" in content_data:
                            st.write(content_data["message"])
                            if "logical_structure" in content_data:
                                st.info(f"**Logical Structure:** {content_data['logical_structure']}")
                                st.write(f"**Question:** {content_data.get('user_message', '')}")
                        else:
                            st.json(content_data)
                    except:
                        st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Describe your rule in natural language...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Generate response
        with st.spinner("Generating rule..."):
            try:
                # Auto-refresh data sources if enabled
                if len(st.session_state.data_sources) == 0:
                    st.session_state.data_sources = fetch_data_sources(st.session_state.client_id)
                
                # Generate rule
                result = generate_rule_with_openai(user_input, st.session_state.client_id)
                response_type = detect_response_type(result)
                
                if response_type == "rule":
                    st.session_state.last_generated_rule = result
                    
                    # Check if user wants synthetic data
                    if "synthetic" in user_input.lower() or "sample" in user_input.lower() or "data" in user_input.lower():
                        with st.spinner("Generating synthetic data..."):
                            data_sources = fetch_data_sources(st.session_state.client_id)
                            synthetic_data = generate_synthetic_dataset_with_openai(result["rules"], data_sources)
                            st.session_state.synthetic_data = synthetic_data
                            st.session_state.show_synthetic_data = True
                            
                            # Count matches
                            matching_count = len([r for r in synthetic_data["synthetic_dataset"] if r["matches_rule"]])
                            total_count = len(synthetic_data["synthetic_dataset"])
                            
                            response_content = json.dumps({
                                "message": f"Generated rule successfully. Found {matching_count} matching records out of {total_count}.",
                                "rules": result["rules"],
                                "synthetic_data_available": True
                            }, indent=2)
                    else:
                        response_content = json.dumps({
                            "message": "Rule generated successfully. Would you like me to extract sample data as well?",
                            "rules": result["rules"]
                        }, indent=2)
                
                elif response_type == "confirmation":
                    response_content = json.dumps(result, indent=2)
                
                elif response_type == "general":
                    response_content = json.dumps(result, indent=2)
                
                else:
                    response_content = json.dumps({
                        "message": "I'm not sure how to process that request. Please try rephrasing your rule description."
                    }, indent=2)
                
                # Add assistant response to chat
                st.session_state.chat_history.append({"role": "assistant", "content": response_content})
                
                # Rerun to update display
                st.rerun()
                
            except Exception as e:
                error_message = f"Error generating rule: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                st.error(error_message)
                st.rerun()

def display_rule_details():
    """Display detailed rule information in expander"""
    if st.session_state.last_generated_rule:
        with st.expander("📋 Generated Rule Details", expanded=True):
            st.json(st.session_state.last_generated_rule)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Copy Rule JSON", use_container_width=True):
                    st.code(json.dumps(st.session_state.last_generated_rule, indent=2))
                    st.success("Rule JSON copied to clipboard!")
            
            with col2:
                if st.button("📊 Generate Synthetic Data", use_container_width=True):
                    with st.spinner("Generating synthetic data..."):
                        data_sources = fetch_data_sources(st.session_state.client_id)
                        synthetic_data = generate_synthetic_dataset_with_openai(
                            st.session_state.last_generated_rule["rules"], 
                            data_sources
                        )
                        st.session_state.synthetic_data = synthetic_data
                        st.session_state.show_synthetic_data = True
                        st.rerun()

def display_synthetic_data():
    """Display synthetic data if available"""
    if st.session_state.synthetic_data and st.session_state.show_synthetic_data:
        with st.expander("📊 Synthetic Data Preview", expanded=True):
            data = st.session_state.synthetic_data["synthetic_dataset"]
            df = pd.DataFrame(data)
            
            # Calculate statistics
            total_records = len(df)
            matching_records = df[df["matches_rule"]].shape[0]
            match_percentage = (matching_records / total_records * 100) if total_records > 0 else 0
            
            # Display stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", total_records)
            col2.metric("Matching Records", matching_records)
            col3.metric("Match Rate", f"{match_percentage:.1f}%")
            
            # Display data table
            st.dataframe(df, use_container_width=True)
            
            # Download options
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

def main():
    """Main application function"""
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
                "4. Example: 'Customers who spent over $1000 AND have an active mortgage'\n"
                "5. Add 'with sample data' to generate synthetic data")
    
    # Display rule details if available
    if st.session_state.last_generated_rule:
        display_rule_details()
    
    # Display synthetic data if available
    display_synthetic_data()

if __name__ == "__main__":
    if not secrets['openai_api_key']:
        st.error("OpenAI API key not found in secrets. Please add it to .streamlit/secrets.toml")
        st.stop()
    
    if not secrets['external_api_key'] or not secrets['external_api_secret']:
        st.warning("External API credentials not found. Using static data sources only.")
    
    main()
