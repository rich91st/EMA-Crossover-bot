FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY bot.py .
COPY watchlist.json .

# Command to run the bot
CMD ["python", "bot.py"]