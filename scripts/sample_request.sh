#!/usr/bin/env bash
# Usage: scripts/sample_request.sh [base_url]
URL=${1:-http://localhost:8000}
curl -s "$URL/health"; echo
curl -s -X POST "$URL/predict" -H "Content-Type: application/json" \
  -d '{"features":{"age":58,"sex":1,"cp":4,"trestbps":130,"chol":250,"fbs":0,"restecg":0,"thalach":150,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":3}}'; echo
