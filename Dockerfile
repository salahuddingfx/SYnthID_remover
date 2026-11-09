FROM python:3.11-slim

# Create non-root user with UID 1000 (standard requirement for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy files with user ownership
COPY --chown=user . $HOME/app

# Install dependencies into user directory
RUN pip install --no-cache-dir --user -r requirements.txt

# Hugging Face Spaces exposes port 7860 by default
ENV PORT=7860
EXPOSE 7860

CMD ["python3", "app.py"]
