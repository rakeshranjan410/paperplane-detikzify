#!/bin/bash
echo "============================================="
echo "       DISK SPACE CLEANUP & DIAGNOSTICS      "
echo "============================================="

echo "1. Current Disk Usage:"
df -h
echo ""

echo "2. Cleaning Hugging Face Cache..."
# Removing the cache directory entirely is the safest way to clear corrupted/partial downloads
# and old models that are taking up space.
rm -rf ~/.cache/huggingface
echo "   - Removed ~/.cache/huggingface"

echo "3. Cleaning Pip Cache..."
rm -rf ~/.cache/pip
echo "   - Removed ~/.cache/pip"

echo "4. Removing generic temp files..."
rm -rf /tmp/* 2>/dev/null
echo "   - Cleaned /tmp"

echo ""
echo "============================================="
echo "Disk Usage After Cleanup:"
df -h
echo "============================================="
echo "NOTE for EC2 Users:"
echo "If your 'Available' space is less than 20GB, downloading the 8B model might fail again."
echo "You may need to expand your EBS volume in the AWS Console."
echo "============================================="
