#!/usr/bin/env bash
# Helper script to deploy to Hugging Face Spaces

echo "=========================================="
echo "  Deploy to Hugging Face Spaces"
echo "=========================================="
echo ""
read -p "Enter your Hugging Face Username: " HF_USER
read -p "Enter your Space Name (default: synthid-remover): " SPACE_NAME

SPACE_NAME=${SPACE_NAME:-synthid-remover}
HF_URL="https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"

echo ""
echo "Adding remote 'hf' -> ${HF_URL}"
git remote remove hf 2>/dev/null
git remote add hf "${HF_URL}"

echo ""
echo "Pushing code to Hugging Face Space..."
echo "Note: When prompted for password, paste your Hugging Face Access Token (with 'write' permission)."
echo "      Get one here: https://huggingface.co/settings/tokens"
echo ""

git push -f hf master:main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Successfully pushed to Hugging Face Space!"
    echo "Your app will be live at: https://${HF_USER}-${SPACE_NAME}.hf.space"
else
    echo ""
    echo "❌ Push failed. Please check your username and access token."
fi

