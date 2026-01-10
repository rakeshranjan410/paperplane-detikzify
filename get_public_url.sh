#!/bin/bash

# Retrieve token for IMDSv2
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" -s)

# Get Public IP
PUBLIC_IP=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/public-ipv4)

echo "---------------------------------------------------"
echo "DeTikZify Service URL"
echo "---------------------------------------------------"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "Access the API documentation here:"
echo "http://$PUBLIC_IP:8000/docs"
echo ""
echo "Note: If this URL times out, check your AWS Security Group:"
echo "  1. Go to EC2 Console -> Instances -> Select your instance."
echo "  2. Click 'Security' tab -> Click the Security Group ID."
echo "  3. 'Edit inbound rules'."
echo "  4. Add Rule: Custom TCP, Port 8000, Source 0.0.0.0/0 (Anywhere)."
echo "---------------------------------------------------"
