# Initialize git and create a README
git init
echo "# catalog-maintenance" > README.md

# Create your Python project structure
mkdir -p configs src tests
touch requirements.txt

# First commit
git add .
git commit -m "Initial project scaffold for Phase 1 ingestion & catalog maintenance"

# Create the GitHub repo and push
gh repo create hellolilly-labs/catalog-maintenance \
  --public \
  --description "Liddy Phase 1 Python ingestion & catalog maintenance pipeline" \
  --source . \
  --remote origin \
  --push