#!/bin/bash
set -e

# Script parameters
PROJECT_ID="laure-430512"
REGION=${REGION:-"us-central1"}
SECURITY_POLICY_NAME="voice-service-policy"

# Verify we're using the correct GCP account
CURRENT_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
if [[ "$CURRENT_ACCOUNT" != "crb@liddy.ai" ]]; then
  echo "ERROR: You are currently authenticated as '$CURRENT_ACCOUNT'"
  echo "Please authenticate with the correct account using:"
  echo "gcloud auth login crb@liddy.ai"
  exit 1
fi
echo "Authenticated as $CURRENT_ACCOUNT. Proceeding with Cloud Armor setup."

# Make sure we're using the correct project
echo "Setting project to ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"

echo "Setting up Cloud Armor WAF protections for voice service..."

# Check if security policy exists
POLICY_EXISTS=$(gcloud compute security-policies list --filter="name=${SECURITY_POLICY_NAME}" --format="value(name)" 2>/dev/null || echo "")
if [ -z "$POLICY_EXISTS" ]; then
  echo "Creating Cloud Armor security policy ${SECURITY_POLICY_NAME}..."
  gcloud compute security-policies create ${SECURITY_POLICY_NAME} \
    --description "WAF rules for voice-service"
else
  echo "Security policy ${SECURITY_POLICY_NAME} already exists. Updating existing policy."
fi

# Function to create or update a rule
create_or_update_rule() {
  local priority=$1
  local expression=$2
  local action=$3
  local description=$4
  
  # Check if rule with this priority already exists
  RULE_EXISTS=$(gcloud compute security-policies rules describe ${priority} \
    --security-policy ${SECURITY_POLICY_NAME} --format="value(priority)" 2>/dev/null || echo "")
  
  if [ -z "$RULE_EXISTS" ]; then
    echo "Creating rule with priority ${priority}..."
    gcloud compute security-policies rules create ${priority} \
      --security-policy ${SECURITY_POLICY_NAME} \
      --expression "${expression}" \
      --action ${action} \
      --description "${description}"
  else
    echo "Updating existing rule with priority ${priority}..."
    gcloud compute security-policies rules update ${priority} \
      --security-policy ${SECURITY_POLICY_NAME} \
      --expression "${expression}" \
      --action ${action} \
      --description "${description}"
  fi
}

# Add SQL Injection protection
echo "Adding SQL Injection protection rule..."
create_or_update_rule 1000 "evaluatePreconfiguredWaf('sqli-v33-stable', {'sensitivity': 2})" "deny-403" "SQL Injection protection with sensitivity level 2"

# Add Cross-Site Scripting protection
echo "Adding Cross-Site Scripting protection rule..."
create_or_update_rule 1001 "evaluatePreconfiguredWaf('xss-v33-stable', {'sensitivity': 2})" "deny-403" "XSS protection with sensitivity level 2"

# Add Local File Inclusion protection
echo "Adding Local File Inclusion protection rule..."
create_or_update_rule 1002 "evaluatePreconfiguredWaf('lfi-v33-stable', {'sensitivity': 2})" "deny-403" "LFI protection with sensitivity level 2"

# Add Remote Code Execution protection
echo "Adding Remote Code Execution protection rule..."
create_or_update_rule 1003 "evaluatePreconfiguredWaf('rce-v33-stable', {'sensitivity': 2})" "deny-403" "RCE protection with sensitivity level 2"

# Use a simpler approach for rate limiting
echo "Note: Skipping rate limiting configuration for now. Consider manually adding rate limiting through the Google Cloud Console if needed."

# Add default rule to allow legitimate traffic
echo "Adding default rule to allow legitimate traffic..."
gcloud compute security-policies rules update 2147483647 \
  --security-policy ${SECURITY_POLICY_NAME} \
  --action allow \
  --description "Default rule: allow all legitimate traffic"

echo "Cloud Armor security policy setup complete!"
echo "To view the security policy, run:"
echo "gcloud compute security-policies describe ${SECURITY_POLICY_NAME}"
echo ""
echo "To attach this policy to your Cloud Run service, run:"
echo "gcloud run services update SERVICE_NAME --region=${REGION} --cloud-armor-security-policy=${SECURITY_POLICY_NAME}"
echo ""
echo "Note: The deploy-cloudrun.sh script has been updated to automatically attach this policy."